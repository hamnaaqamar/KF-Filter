import os
import socket
import time
import json
import logging
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
from openpyxl import Workbook
from dronekit import connect, VehicleMode, Vehicle
from pymavlink import mavutil

MODEL_IP = "127.0.0.1"
MODEL_PORT = 9000

DT = 100  # control loop period in ms

# IMPORTANT:
# These were present in your original script but not actually used.
# If the drone moves opposite to what you expect on any axis, toggle these.
# Based on latest logs, X correction was inverted (target x drifted to edge
# while commanding correction). Keep this False for current setup.
FLIP_X = False
FLIP_Y = True

CENTERED_DURATION = 4.0
DROPOUT_PREDICT_SEC = 0.7
MIN_COMMAND_SPEED = 0.01

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
TEXT_LOG_PATH = f"logs/localize_{RUN_ID}.log"
NUMERIC_LOG_PATH = f"logs/localize_{RUN_ID}_metrics.xlsx"

logger = logging.getLogger(__name__)


class KalmanFilter:
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        # State: [x, y, vx, vy]
        self.state = np.array([0.5, 0.5, 0.0, 0.0], dtype=float)
        self.covariance = np.eye(4) * 0.1

        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise

        self.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )
        self.last_time = time.time()

    def update(self, camera_x: Optional[float], camera_y: Optional[float]) -> Tuple[float, float]:
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Predict
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

        # No measurement => predicted state only
        if camera_x is None or camera_y is None:
            return float(self.state[0]), float(self.state[1])

        # Update
        try:
            measurement = np.array([float(camera_x), float(camera_y)], dtype=float)
            S = self.H @ self.covariance @ self.H.T + self.R
            K = self.covariance @ self.H.T @ np.linalg.inv(S)
            residual = measurement - (self.H @ self.state)
            self.state = self.state + K @ residual
            self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
            self.state[0] = np.clip(self.state[0], 0.0, 1.0)
            self.state[1] = np.clip(self.state[1], 0.0, 1.0)
        except Exception as exc:
            logger.warning(f"Kalman filter update error: {exc}")

        return float(self.state[0]), float(self.state[1])


def cut_throttle(vehicle: Vehicle, log: logging.Logger, throttle: float = 0.1) -> None:
    if vehicle.location.global_relative_frame.alt > 2:
        log.debug("Cutting throttle above 2m, returning")
        return
    if throttle > 0.2:
        log.debug("Throttle greater than 20%, returning")
        return

    msg = vehicle.message_factory.set_attitude_target_encode(
        0,
        1,
        1,
        0b00000111,
        [1, 0, 0, 0],
        0,
        0,
        0,
        throttle,
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def send_velocity(vehicle: Vehicle, log: logging.Logger, vx: float, vy: float, vz: float) -> None:
    """
    Send body-frame velocity command.
    Body NED: x=forward, y=right, z=down.
    """
    if vehicle is None:
        return
    mode_name = vehicle.mode.name if hasattr(vehicle.mode, "name") else str(vehicle.mode)
    if mode_name != "GUIDED":
        return

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b110111000111,  # velocity control only
        0,
        0,
        0,
        vx,  # FIX: do not swap vx/vy
        vy,
        vz,
        0,
        0,
        0,
        0,
        0,
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()
    log.debug(
        "Sending velocity: VX=%.4f, VY=%.4f VZ=%.4f. alt=%.3f",
        vx,
        vy,
        vz,
        vehicle.location.global_relative_frame.alt,
    )


def send_velocity_to_unity(sock: socket.socket, vx: float, vy: float, vz: float) -> None:
    try:
        command = f"VEL:{vx:.4f},{vy:.4f},{vz:.4f}\n"
        sock.sendall(command.encode())
    except Exception as exc:
        logger.error("Error sending velocity to Unity: %s", exc)


def receive_coordinates(sock: socket.socket, label_filter: Callable[[dict], bool]) -> Tuple[Optional[float], Optional[float]]:
    if sock is None:
        return None, None
    try:
        sock.sendall(b"detect\n")
        sock.settimeout(1.0)
        data_buffer = b""
        try:
            while True:
                chunk = sock.recv(256)
                if not chunk:
                    break
                data_buffer += chunk
                if b"\n" in chunk:
                    break
        except socket.timeout:
            pass
        finally:
            sock.settimeout(None)

        if not data_buffer:
            return None, None

        decoded = data_buffer.decode("utf-8", errors="ignore").strip()
        for line in decoded.split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "x" not in data or "y" not in data:
                continue
            if data["x"] is None or data["y"] is None:
                continue
            if not label_filter(data):
                continue
            return float(data["x"]), float(data["y"])
        return None, None
    except (socket.error, ConnectionError) as exc:
        logger.debug("Socket error: %s", exc)
        return None, None


def arm_and_takeoff(vehicle: Vehicle, target_altitude: float) -> None:
    logger.info("Basic pre-arm checks")
    while not vehicle.is_armable:
        logger.info(" Waiting for vehicle to initialise...")
        time.sleep(1)

    logger.info("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        logger.info(" Waiting for arming...")
        time.sleep(1)

    logger.info("Taking off to %.2fm!", target_altitude)
    vehicle.simple_takeoff(target_altitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        logger.info(" Altitude: %.2f", alt)
        if alt >= target_altitude * 0.95:
            logger.info("Reached target altitude")
            break
        time.sleep(1)


def control_from_image_error(
    x_error_raw: float,
    y_error_raw: float,
    kp_x: float,
    kp_y: float,
    max_speed: float,
) -> Tuple[float, float, float, float]:
    """
    Convert image error to body velocity.
    image x -> body y, image y -> body x (downward camera convention).
    FLIP_X / FLIP_Y correct camera sign mismatches.
    """
    x_error = -x_error_raw if FLIP_X else x_error_raw
    y_error = -y_error_raw if FLIP_Y else y_error_raw

    vx = kp_y * y_error
    vy = kp_x * x_error

    vx = float(np.clip(vx, -max_speed, max_speed))
    vy = float(np.clip(vy, -max_speed, max_speed))

    if abs(vx) < MIN_COMMAND_SPEED:
        vx = 0.0
    if abs(vy) < MIN_COMMAND_SPEED:
        vy = 0.0
    return vx, vy, x_error, y_error


def localize(
    vehicle: Optional[Vehicle],
    sock: socket.socket,
    log: logging.Logger,
    thresh: float = 0.08,
    descent_speed: float = 0.1,
    detection_alt: float = 4.0,
    landing_alt: float = 0.3,
    max_speed: float = 0.15,
    label_filter: Callable[[dict], bool] = lambda _: True,
    duty_cycle: int = DT,
    log_time_delay: float = 1.0,
    unity_drone_sock: Optional[socket.socket] = None,
) -> None:
    test_mode = vehicle is None
    if test_mode:
        log.warning("Vehicle is None — running UNITY TEST MODE.")
        simulated_alt = 5.0

    previous_wp_behaviour = None
    if not test_mode:
        previous_wp_behaviour = vehicle.parameters["WP_YAW_BEHAVIOR"]
        vehicle.parameters["WP_YAW_BEHAVIOR"] = 0

    # Simplified flow: center over target first, then descend.
    flight_phase = "centering"
    log_timer_start = time.time()

    kf = KalmanFilter(process_noise=0.02, measurement_noise=0.1)
    centered_start_time = None
    last_detection_time = time.time()

    # Separate gains can help if one axis is more sluggish.
    kp_x = 0.5
    kp_y = 0.5
    dead_zone = 0.05
    # Use stricter centering before initiating descent.
    descent_entry_thresh = max(0.05, min(thresh, 0.08))
    # During descent, pause/abort descent if target drifts too far.
    descent_track_thresh = max(descent_entry_thresh + 0.04, 0.12)
    descent_abort_thresh = max(descent_entry_thresh + 0.16, 0.22)

    if not test_mode:
        current_alt = vehicle.location.global_relative_frame.alt
        log.info("Starting at altitude: %.2fm", current_alt)
        log.info("Target detection altitude: %.2fm", detection_alt)
    else:
        current_alt = simulated_alt

    # Descend to detection altitude.
    if not test_mode and current_alt > detection_alt + 0.5:
        log.info("Descending to detection altitude %.2fm...", detection_alt)
        while vehicle.location.global_relative_frame.alt > detection_alt + 0.15:
            send_velocity(vehicle, log, 0.0, 0.0, abs(descent_speed))
            time.sleep(duty_cycle / 1000.0)
        log.info("Reached detection altitude")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "localization"
    sheet.append(
        [
            "timestamp",
            "elapsed_s",
            "phase",
            "tracking_source",
            "detection_valid",
            "alt_m",
            "x_raw",
            "y_raw",
            "x_est",
            "y_est",
            "x_err",
            "y_err",
            "vx",
            "vy",
            "vz",
            "centered",
            "ready_to_descend",
        ]
    )

    start_time = time.time()
    last_xlsx_save = start_time
    try:
        while True:
            now = time.time()
            if not test_mode:
                current_alt = vehicle.location.global_relative_frame.alt
            else:
                current_alt = simulated_alt

            x_raw, y_raw = receive_coordinates(sock, label_filter)
            detection_valid = x_raw is not None and y_raw is not None
            tracking_source = "lost"

            vx, vy, vz = 0.0, 0.0, 0.0
            centered = False
            ready_to_descend = False
            x_error = float("nan")
            y_error = float("nan")
            x_est = None
            y_est = None

            if detection_valid:
                last_detection_time = now
                x_est, y_est = kf.update(x_raw, y_raw)
                tracking_source = "detect"
            elif now - last_detection_time <= DROPOUT_PREDICT_SEC:
                x_est, y_est = kf.update(None, None)
                tracking_source = "predict"

            if x_est is not None and y_est is not None:
                x_error_raw = x_est - 0.5
                y_error_raw = y_est - 0.5
                vx, vy, x_error, y_error = control_from_image_error(
                    x_error_raw=x_error_raw,
                    y_error_raw=y_error_raw,
                    kp_x=kp_x,
                    kp_y=kp_y,
                    max_speed=max_speed,
                )
                centered = abs(x_error) < thresh and abs(y_error) < thresh
                ready_to_descend = abs(x_error) < descent_entry_thresh and abs(y_error) < descent_entry_thresh

                if abs(x_error) < dead_zone and abs(y_error) < dead_zone:
                    vx, vy = 0.0, 0.0

                if tracking_source == "detect":
                    log.debug(
                        "DETECT raw=(%.3f,%.3f) err=(%.3f,%.3f) vel=(%.3f,%.3f)",
                        x_est,
                        y_est,
                        x_error,
                        y_error,
                        vx,
                        vy,
                    )
                else:
                    log.debug(
                        "PREDICT est=(%.3f,%.3f) err=(%.3f,%.3f) vel=(%.3f,%.3f)",
                        x_est,
                        y_est,
                        x_error,
                        y_error,
                        vx,
                        vy,
                    )

                if flight_phase == "centering":
                    if ready_to_descend:
                        if centered_start_time is None:
                            centered_start_time = now
                            log.info("CENTERED - timer started")
                        elif now - centered_start_time >= CENTERED_DURATION:
                            flight_phase = "descending"
                            log.info("DESCENDING")
                    else:
                        centered_start_time = None
                elif flight_phase == "descending":
                    if current_alt <= landing_alt + 0.2:
                        flight_phase = "landed"
                        vx, vy, vz = 0.0, 0.0, 0.0
                        log.info("LANDED")
                    else:
                        abs_x, abs_y = abs(x_error), abs(y_error)
                        if abs_x > descent_abort_thresh or abs_y > descent_abort_thresh:
                            flight_phase = "centering"
                            centered_start_time = None
                            vz = 0.0
                            log.warning(
                                "DESCENT ABORTED: large offset (|ex|=%.3f, |ey|=%.3f). Re-centering.",
                                abs_x,
                                abs_y,
                            )
                        elif abs_x > descent_track_thresh or abs_y > descent_track_thresh:
                            vz = 0.0
                            log.debug(
                                "DESCENT PAUSED: offset too large for safe descend (|ex|=%.3f, |ey|=%.3f).",
                                abs_x,
                                abs_y,
                            )
                        else:
                            vz = abs(descent_speed)
                elif flight_phase == "landed":
                    if not test_mode:
                        cut_throttle(vehicle, log, 0.1)
                    break
            else:
                centered_start_time = None
                if flight_phase == "descending":
                    # Keep position and wait for detection/prediction again.
                    vz = 0.0

            if flight_phase == "descending" and current_alt <= 0.3:
                log.info("FORCE LAND")
                flight_phase = "landed"
                continue

            if not test_mode:
                send_velocity(vehicle, log, vx, vy, vz)
            else:
                if flight_phase == "descending":
                    simulated_alt -= vz * (duty_cycle / 1000.0)
                    if simulated_alt <= landing_alt:
                        simulated_alt = landing_alt
                        flight_phase = "landed"
                if unity_drone_sock is not None:
                    send_velocity_to_unity(unity_drone_sock, vx, vy, vz)

            if now - log_timer_start >= log_time_delay:
                log.debug(
                    "PHASE=%s SRC=%s ALT=%.2f V=(%.3f,%.3f,%.3f)",
                    flight_phase,
                    tracking_source,
                    current_alt,
                    vx,
                    vy,
                    vz,
                )
                log_timer_start = now

            sheet.append(
                [
                    datetime.now().isoformat(),
                    round(now - start_time, 3),
                    flight_phase,
                    tracking_source,
                    int(detection_valid),
                    round(float(current_alt), 4),
                    "" if x_raw is None else round(float(x_raw), 6),
                    "" if y_raw is None else round(float(y_raw), 6),
                    "" if x_est is None else round(float(x_est), 6),
                    "" if y_est is None else round(float(y_est), 6),
                    "" if np.isnan(x_error) else round(float(x_error), 6),
                    "" if np.isnan(y_error) else round(float(y_error), 6),
                    round(float(vx), 6),
                    round(float(vy), 6),
                    round(float(vz), 6),
                    int(centered),
                    int(ready_to_descend),
                ]
            )
            if now - last_xlsx_save >= 1.0:
                workbook.save(NUMERIC_LOG_PATH)
                last_xlsx_save = now

            time.sleep(duty_cycle / 1000.0)
    finally:
        workbook.save(NUMERIC_LOG_PATH)
        if not test_mode and previous_wp_behaviour is not None:
            vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
        log.info("Saved numeric telemetry to: %s", NUMERIC_LOG_PATH)
        log.info("Done")


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(TEXT_LOG_PATH),
            logging.StreamHandler(),
        ],
    )
    logger.info("=========== LOCALIZATION SCRIPT START ===========")
    print("\nStarting Unity Localization Script...")

    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect((MODEL_IP, MODEL_PORT))
        sock.settimeout(None)
        logger.info("Connected to Unity at %s:%s", MODEL_IP, MODEL_PORT)
    except Exception as exc:
        logger.error("Cannot connect to Unity (%s:%s): %s", MODEL_IP, MODEL_PORT, exc)
        if sock:
            sock.close()
        return

    connection_string = "192.168.56.1:14552"
    logger.info("Connecting to vehicle on %s", connection_string)

    try:
        vehicle = connect(connection_string, wait_ready=True, timeout=30)
        logger.info("Vehicle connected")

        target_altitude = 5.0
        logger.info("Taking off to %.1fm", target_altitude)
        arm_and_takeoff(vehicle, target_altitude)
        time.sleep(2)

        print("\nStarting localization and landing sequence...")
        print("Press Ctrl+C to stop\n")
        localize(
            vehicle=vehicle,
            sock=sock,
            log=logger,
            thresh=0.12,
            descent_speed=0.2,
            detection_alt=4.0,
            landing_alt=0.3,
            max_speed=0.3,
            log_time_delay=0.5,
        )

        if vehicle.armed:
            logger.info("Vehicle still armed, forcing disarm")
            vehicle.armed = False

        logger.info("Mission complete!")
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        if "vehicle" in locals() and vehicle.armed:
            logger.info("Emergency landing...")
            vehicle.mode = VehicleMode("LAND")
    except Exception as exc:
        logger.exception("Error in main: %s", exc)
    finally:
        if "vehicle" in locals():
            vehicle.close()
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        logger.info("Text log file: %s", TEXT_LOG_PATH)
        logger.info("Numeric telemetry file: %s", NUMERIC_LOG_PATH)
        logger.info("=========== LOCALIZATION SCRIPT END ===========")
        print("\nScript ended")


if __name__ == "__main__":
    main()