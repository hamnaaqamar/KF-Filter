import os
import socket
import time
import json
import logging
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
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

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/localize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
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


def debug_with_delay(to_print: Callable[[], None], timer_start: float, delay: float) -> float:
    # Log only when delay interval has elapsed (fixed from original).
    if time.time() - timer_start >= delay:
        to_print()
        return time.time()
    return timer_start


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


def _vehicle_mode_name(vehicle: Vehicle) -> str:
    mode = vehicle.mode
    if hasattr(mode, "name"):
        return mode.name
    return str(mode)


def send_velocity(vehicle: Vehicle, log: logging.Logger, vx: float, vy: float, vz: float) -> None:
    """
    Send body-frame velocity command.
    Body NED: x=forward, y=right, z=down.
    """
    if vehicle is None:
        return
    if _vehicle_mode_name(vehicle) != "GUIDED":
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

    flight_phase = "searching"
    log_timer_start = time.time()

    kf = KalmanFilter(process_noise=0.02, measurement_noise=0.1)

    centered_start_time = None
    consecutive_detections = 0
    min_detections_for_centering = 2
    last_detection_time = time.time()
    detection_timeout = 3.0

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

    def debug(msg: str) -> None:
        nonlocal log_timer_start
        log_timer_start = debug_with_delay(lambda: log.debug(msg), log_timer_start, log_time_delay)

    # Descend to detection altitude.
    if not test_mode and current_alt > detection_alt + 0.5:
        log.info("Descending to detection altitude %.2fm...", detection_alt)
        while vehicle.location.global_relative_frame.alt > detection_alt + 0.15:
            send_velocity(vehicle, log, 0.0, 0.0, abs(descent_speed))
            time.sleep(duty_cycle / 1000.0)
        log.info("Reached detection altitude")

    while True:
        vx = 0.0
        vy = 0.0
        vz = 0.0

        x_raw, y_raw = receive_coordinates(sock, label_filter)
        now = time.time()

        if not test_mode:
            current_alt = vehicle.location.global_relative_frame.alt
        else:
            current_alt = simulated_alt

        detection_valid = x_raw is not None and y_raw is not None

        if detection_valid:
            last_detection_time = now
            consecutive_detections += 1

            x_smooth, y_smooth = kf.update(x_raw, y_raw)
            x_error_raw = x_smooth - 0.5
            y_error_raw = y_smooth - 0.5

            vx, vy, x_error, y_error = control_from_image_error(
                x_error_raw=x_error_raw,
                y_error_raw=y_error_raw,
                kp_x=kp_x,
                kp_y=kp_y,
                max_speed=max_speed,
            )

            if abs(x_error) < dead_zone and abs(y_error) < dead_zone:
                vx = 0.0
                vy = 0.0
                log.debug("DEAD ZONE")

            centered = abs(x_error) < thresh and abs(y_error) < thresh
            ready_to_descend = abs(x_error) < descent_entry_thresh and abs(y_error) < descent_entry_thresh
            log.debug(
                "DETECT raw=(%.3f,%.3f) err_raw=(%.3f,%.3f) err_ctrl=(%.3f,%.3f) vel=(%.3f,%.3f)",
                x_smooth,
                y_smooth,
                x_error_raw,
                y_error_raw,
                x_error,
                y_error,
                vx,
                vy,
            )

            if flight_phase == "searching":
                if consecutive_detections >= min_detections_for_centering:
                    flight_phase = "centering"
                    centered_start_time = None
                    log.info("TARGET FOUND")

            elif flight_phase == "centering":
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
                    abs_x = abs(x_error)
                    abs_y = abs(y_error)
                    if abs_x > descent_abort_thresh or abs_y > descent_abort_thresh:
                        # Target drifted too far while descending. Recenter first.
                        flight_phase = "centering"
                        centered_start_time = None
                        vz = 0.0
                        log.warning(
                            "DESCENT ABORTED: large offset (|ex|=%.3f, |ey|=%.3f). Re-centering.",
                            abs_x,
                            abs_y,
                        )
                    elif abs_x > descent_track_thresh or abs_y > descent_track_thresh:
                        # Keep correcting laterally but pause vertical descent.
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
            consecutive_detections = 0
            time_since_detection = now - last_detection_time

            # Keep tracking briefly with prediction instead of hard "WAITING=0 velocity".
            if flight_phase in ("centering", "descending") and time_since_detection <= DROPOUT_PREDICT_SEC:
                x_pred, y_pred = kf.update(None, None)
                x_error_raw = x_pred - 0.5
                y_error_raw = y_pred - 0.5
                vx, vy, x_error, y_error = control_from_image_error(
                    x_error_raw=x_error_raw,
                    y_error_raw=y_error_raw,
                    kp_x=kp_x,
                    kp_y=kp_y,
                    max_speed=max_speed,
                )
                if flight_phase == "descending":
                    if abs(x_error) <= descent_track_thresh and abs(y_error) <= descent_track_thresh:
                        vz = abs(descent_speed)
                    else:
                        vz = 0.0
                log.debug(
                    "PREDICT track err_ctrl=(%.3f,%.3f) vel=(%.3f,%.3f)",
                    x_error,
                    y_error,
                    vx,
                    vy,
                )
            else:
                if flight_phase == "descending":
                    # Do not keep descending blind. Return to centering immediately.
                    flight_phase = "centering"
                    centered_start_time = None
                    log.warning("Detection missing during descent. Pausing descent and re-centering.")

                if time_since_detection > detection_timeout and flight_phase != "searching":
                    flight_phase = "searching"
                    log.info("TARGET LOST")

                if flight_phase == "searching":
                    # Simple square search.
                    search_step = int(now * 2) % 4
                    if search_step == 0:
                        vx, vy = 0.08, 0.0
                    elif search_step == 1:
                        vx, vy = 0.0, 0.08
                    elif search_step == 2:
                        vx, vy = -0.08, 0.0
                    else:
                        vx, vy = 0.0, -0.08
                    log.debug("SEARCH step=%d", search_step)
                else:
                    vx, vy, vz = 0.0, 0.0, 0.0

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

        debug(f"PHASE={flight_phase} ALT={current_alt:.2f} V=({vx:.3f},{vy:.3f},{vz:.3f})")
        time.sleep(duty_cycle / 1000.0)

    if not test_mode and previous_wp_behaviour is not None:
        vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
    log.info("Done")


def test_unity_connection() -> Optional[socket.socket]:
    print("\n" + "=" * 60)
    print("Testing Unity Connection...")
    print("=" * 60)

    sock = None
    try:
        print(f"Connecting to {MODEL_IP}:{MODEL_PORT}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect((MODEL_IP, MODEL_PORT))
        print("✓ Connected to Unity")

        for cmd_name, cmd_bytes in [
            ("detect", b"detect\n"),
            ("hello", b"hello\n"),
            ("status", b"status\n"),
            ("ping", b"ping\n"),
        ]:
            print(f"\nSending command: {cmd_name}")
            sock.sendall(cmd_bytes)
            try:
                response = sock.recv(1024)
                if response:
                    print(f"Response: {response}")
                    try:
                        print(f"Decoded: {response.decode('utf-8')}")
                    except Exception:
                        print("Could not decode as UTF-8")
                else:
                    print("No response (empty)")
            except socket.timeout:
                print("Timeout - no response")
            except Exception as exc:
                print(f"Error receiving: {exc}")
            time.sleep(0.5)
        return sock
    except Exception as exc:
        print(f"✗ Connection error: {exc}")
        if sock:
            sock.close()
        return None


def main() -> None:
    print("\nStarting Unity Localization Script...")
    sock = test_unity_connection()
    if sock is None:
        logger.error("Cannot connect to Unity. Make sure Unity is running.")
        return

    choice = input("\nDo you want to continue with localization? (y/n): ")
    if choice.strip().lower() != "y":
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
        print("\nScript ended")


if __name__ == "__main__":
    main()