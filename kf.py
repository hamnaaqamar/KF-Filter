import math
from typing import Callable
from typing_extensions import Literal
from dronekit import connect, VehicleMode, Vehicle
from pymavlink import mavutil
import socket
import time
import json
import logging
from datetime import datetime
import numpy as np

MODEL_IP = "127.0.0.1"
MODEL_PORT = 9000

GAIN = 0.8
DT = 100

FLIP_X = True
FLIP_Y = True
CENTERED_DURATION = 4

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/localize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KalmanFilter:
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # State: [x, y, vx, vy]
        self.state = np.array([0.5, 0.5, 0.0, 0.0])
        self.covariance = np.eye(4) * 0.1
        
        # Process and measurement noise
        self.Q = np.eye(4) * process_noise  # Process noise
        self.R = np.eye(2) * measurement_noise  # Measurement noise
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx stays same
            [0, 0, 0, 1]   # vy stays same
        ])
        
        # Measurement matrix (we only measure x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.last_time = time.time()
        
    def update(self, camera_x, camera_y):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Update state transition matrix with dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        
        # PREDICT step (always happens)
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # If no measurement, return predicted values
        if camera_x is None or camera_y is None:
            # Apply velocity damping when no measurement
            self.state[2] *= 0.9  # Damp velocity in x
            self.state[3] *= 0.9  # Damp velocity in y
            return float(self.state[0]), float(self.state[1])
        
        # UPDATE step (camera measurement available)
        try:
            measurement = np.array([float(camera_x), float(camera_y)])
            
            # Kalman gain
            S = self.H @ self.covariance @ self.H.T + self.R
            K = self.covariance @ self.H.T @ np.linalg.inv(S)
            
            # Update state
            y = measurement - (self.H @ self.state)
            self.state = self.state + K @ y
            self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
            
            # Clamp positions
            self.state[0] = max(0, min(1, self.state[0]))
            self.state[1] = max(0, min(1, self.state[1]))
            
        except Exception as e:
            print(f"Kalman filter update error: {e}")
        
        return float(self.state[0]), float(self.state[1])


def debug_with_delay(to_print: Callable[[], None], timer_start, delay): 
    if time.time() - timer_start <= delay: 
        to_print()
        return time.time()
    return timer_start

def get_landed_state(vehicle) -> str:
    """Get landed state using DroneKit attributes"""
    if vehicle.system_status == 'STANDBY':
        return "on_ground"
    elif vehicle.system_status == 'ACTIVE':
        if vehicle.armed:
            if vehicle.location.global_relative_frame.alt < 0.5:
                return "takeoff"
            else:
                return "in_air"
        else:
            return "on_ground"
    elif vehicle.system_status == 'LANDING':
        return "landing"
    else:
        return "undefined"

def cut_throttle(vehicle: Vehicle, logger: logging.Logger, throttle=0.1): 
    if vehicle.location.global_relative_frame.alt > 2: 
        logger.debug("Cutting throttle above 2m, returning")
        return

    if throttle > 0.2: 
        logger.debug("Throttle greater than 20%. returning")
        return

    msg = vehicle.message_factory.set_attitude_target_encode(
        0, 1, 1, 0b00000111, [1, 0, 0, 0], 0, 0, 0, throttle
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

def send_velocity(vehicle: Vehicle, logger: logging.Logger, vx, vy, vz):
    if vehicle is None: 
        return
    if vehicle.mode != "GUIDED": 
        return
    
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b110111000111, 0, 0, 0, vy, vx, vz, 0, 0, 0, 0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()
    logger.debug(f"Sending velocity: VX={vx:.4f}, VY={vy:.4f} VZ={vz:.4f}. alt={vehicle.location.global_relative_frame.alt:.3f}")

def send_velocity_to_unity(sock: socket.socket, vx: float, vy: float, vz: float):
    try:
        command = f"VEL:{vx:.4f},{vy:.4f},{vz:.4f}\n"
        sock.sendall(command.encode())
    except Exception as e:
        print(f">>> ERROR sending velocity to Unity: {e}")

def connect_to_unity(host='127.0.0.1', port=9000, max_retries=3):
    """Establish connection to Unity"""
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((host, port))
            logger.info(f"Connected to Unity on attempt {attempt+1}")
            return sock
        except Exception as e:
            logger.warning(f"Connection attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    raise ConnectionError(f"Failed to connect to Unity after {max_retries} attempts")

def receive_coordinates(sock, label_filter):
    """Receive coordinates from Unity"""
    if sock is None:
        return None, None
        
    try:
        sock.sendall(b"detect\n")
        sock.settimeout(1.0)
        
        # Receive response
        data_buffer = b""
        try:
            while True:
                chunk = sock.recv(256)
                if not chunk:
                    break
                data_buffer += chunk
                if b'\n' in chunk:
                    break
        except socket.timeout:
            pass
        
        sock.settimeout(None)
        
        if not data_buffer:
            return None, None
        
        # Parse response
        decoded = data_buffer.decode('utf-8', errors='ignore').strip()
        
        # Handle multiple JSON objects
        lines = decoded.split('\n')
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "x" in data and "y" in data:
                    if data["x"] is not None and data["y"] is not None:
                        if label_filter(data):
                            return float(data["x"]), float(data["y"])
            except json.JSONDecodeError:
                continue
        
        return None, None
        
    except (socket.error, ConnectionError) as e:
        logger.debug(f"Socket error: {e}")
        return None, None

def arm_and_takeoff(vehicle, target_altitude):
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

    logger.info(f"Taking off to {target_altitude}m!")
    vehicle.simple_takeoff(target_altitude)

    while True:
        logger.info(f" Altitude: {vehicle.location.global_relative_frame.alt:.2f}")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            logger.info("Reached target altitude")
            break
        time.sleep(1)

def localize(
        vehicle: Vehicle, 
        sock: socket.socket, 
        logger: logging.Logger, 
        thresh=0.05, 
        descent_speed=0.1,
        detection_alt=4.0,
        landing_alt=0.3,
        max_speed=0.2,
        label_filter=lambda x: True,
        duty_cycle=DT, 
        camera_flip=False,
        log_time_delay=1,
        unity_drone_sock=None
    ): 

    test_mode = vehicle is None
    if test_mode:
        logger.warning("Vehicle is None — running UNITY TEST MODE.")
        simulated_alt = 5.0

    if not test_mode:
        previous_wp_behaviour = vehicle.parameters["WP_YAW_BEHAVIOR"]
        vehicle.parameters["WP_YAW_BEHAVIOR"] = 0
    else:
        previous_wp_behaviour = None

    vx, vy, vz = 0, 0, 0
    flight_phase = "hover_detection"
    log_timer_start = time.time()
    
    kf = KalmanFilter()  
    centered_start_time = None
    centered_duration = 0
    last_coordinates_time = time.time()
    no_detection_timeout = 3.0

    if not test_mode:
        current_alt = vehicle.location.global_relative_frame.alt
        logger.info(f"Current altitude: {current_alt:.2f}m")
        logger.info(f"Will hover at {detection_alt}m for detection, then descend to land")
    else:
        current_alt = 5.0
        simulated_alt = current_alt

    def debug(string): 
        nonlocal log_timer_start
        log_timer_start = debug_with_delay(lambda: logger.debug(string), log_timer_start, log_time_delay)

    # Descend to detection altitude if needed
    if not test_mode and current_alt > detection_alt + 0.5:
        logger.info(f"Descending to detection altitude {detection_alt}m...")
        while vehicle.location.global_relative_frame.alt > detection_alt + 0.2:
            send_velocity(vehicle, logger, 0, 0, 0.1)
            time.sleep(duty_cycle / 1000)
        logger.info("Reached detection altitude")

    while True:
        # Get coordinates from Unity
        x_raw, y_raw = receive_coordinates(sock, label_filter)
        
        if x_raw is not None and y_raw is not None:
            last_coordinates_time = time.time()
        elif time.time() - last_coordinates_time > no_detection_timeout:
            debug("No detection for too long, resetting centering timer")
            centered_start_time = None
            centered_duration = 0

        if camera_flip and x_raw is not None:
            x_raw = 1 - x_raw
            y_raw = 1 - y_raw
        
        x_smooth, y_smooth = kf.update(x_raw, y_raw)
        
        if not test_mode:
            current_alt = vehicle.location.global_relative_frame.alt
        else:
            current_alt = simulated_alt
        
        if x_smooth is not None and y_smooth is not None:
            x_error = x_smooth - 0.5
            y_error = 0.5 - y_smooth
            
            vx = GAIN * x_error
            vy = GAIN * y_error
            
            vx = -vx if FLIP_X else vx
            vy = -vy if FLIP_Y else vy
            
            vx = max(min(max_speed, vx), -max_speed)
            vy = max(min(max_speed, vy), -max_speed)
            
            centered = (abs(x_smooth - 0.5) < thresh and abs(y_smooth - 0.5) < thresh)
            
            if flight_phase == "hover_detection":
                vz = 0
                
                if centered:
                    if centered_start_time is None:
                        centered_start_time = time.time()
                        logger.info("Target detected, starting centering timer")
                    else:
                        centered_duration = time.time() - centered_start_time
                    
                    if centered_duration >= CENTERED_DURATION:
                        flight_phase = "centered"
                        logger.info(f"Target centered for {centered_duration:.1f}s")
                else:
                    centered_start_time = None
                    centered_duration = 0
                    
            elif flight_phase == "centered":
                logger.info("Starting descent to landing")
                flight_phase = "descending"
                centered_start_time = None
                
            elif flight_phase == "descending":
                if current_alt <= landing_alt + 0.1:
                    flight_phase = "landed"
                    vz = 0
                    vx = 0
                    vy = 0
                    logger.info(f"Reached landing altitude: {current_alt:.2f}m")
                else:
                    vz = descent_speed
                    
            elif flight_phase == "landed":
                if not test_mode:
                    cut_throttle(vehicle, logger, 0.1)
                logger.info("Landed on target")
                break
                
            debug(
                f"Phase: {flight_phase}, "
                f"Alt: {current_alt:.2f}m, "
                f"Pos: ({x_smooth:.3f},{y_smooth:.3f}), "
                f"Centered: {centered_duration:.1f}s"
            )
            
        else:
            centered_start_time = None
            centered_duration = 0
            
            if flight_phase == "hover_detection":
                vx, vy, vz = 0, 0, 0
                debug(f"No detection - hovering at {current_alt:.2f}m")
            elif flight_phase == "descending":
                vx, vy = 0, 0
                vz = descent_speed * 0.5
                debug(f"No detection during descent - descending slowly")
        
        if not test_mode:
            send_velocity(vehicle, logger, vx, vy, vz)
            
            if flight_phase == "descending" and current_alt <= 0.2:
                logger.info("Very low altitude - forcing land")
                vehicle.mode = VehicleMode("LAND")
                break
        else:
            if flight_phase == "descending":
                simulated_alt -= vz * (duty_cycle / 1000)
                if simulated_alt <= landing_alt:
                    flight_phase = "landed"
                    simulated_alt = landing_alt
            
            if unity_drone_sock is not None:
                send_velocity_to_unity(unity_drone_sock, vx, vy, vz)
        
        time.sleep(duty_cycle / 1000)

    if not test_mode:
        vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
    logger.info("Localization and landing complete")

def test_unity_connection():
    """Test Unity connection and return a new socket"""
    print("\n" + "="*60)
    print("Testing Unity Connection...")
    print("="*60)
    
    sock = None
    try:
        print(f"Connecting to {MODEL_IP}:{MODEL_PORT}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect((MODEL_IP, MODEL_PORT))
        print("✓ Connected to Unity")
        
        # Test different commands
        test_commands = [
            ("detect", b"detect\n"),
            ("hello", b"hello\n"),
            ("status", b"status\n"),
            ("ping", b"ping\n")
        ]
        
        for cmd_name, cmd_bytes in test_commands:
            print(f"\nSending command: {cmd_name}")
            sock.sendall(cmd_bytes)
            
            try:
                response = sock.recv(1024)
                if response:
                    print(f"Response: {response}")
                    try:
                        decoded = response.decode('utf-8')
                        print(f"Decoded: {decoded}")
                    except:
                        print("Could not decode as UTF-8")
                else:
                    print("No response (empty)")
                    
            except socket.timeout:
                print("Timeout - no response")
            except Exception as e:
                print(f"Error receiving: {e}")
                
            time.sleep(0.5)
        
        return sock  # Return the connected socket for reuse
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        if sock:
            sock.close()
        return None

def main():
    print("\nStarting Unity Localization Script...")
    
    # Test connection and get a connected socket
    sock = test_unity_connection()
    if sock is None:
        logger.error("Cannot connect to Unity. Make sure Unity is running.")
        return
    
    choice = input("\nDo you want to continue with localization? (y/n): ")
    if choice.lower() != 'y':
        sock.close()
        return
    
    # Connect to the Vehicle
    connection_string = '192.168.56.1:14552'
    logger.info(f"Connecting to vehicle on {connection_string}")
    
    try:
        vehicle = connect(connection_string, wait_ready=True, timeout=30)
        logger.info("Vehicle connected")
        
        target_altitude = 5.0
        logger.info(f"Taking off to {target_altitude}m")
        arm_and_takeoff(vehicle, target_altitude)
        
        time.sleep(2)
        
        # Run localization with the same socket
        print("\nStarting localization and landing sequence...")
        print("Press Ctrl+C to stop\n")
        
        localize(
            vehicle,
            sock, 
            logger, 
            thresh=0.12,
            descent_speed=0.2,
            detection_alt=4.0,
            landing_alt=0.3,
            max_speed=0.3,
            log_time_delay=0.5
        )
        
        if vehicle.armed:
            logger.info("Vehicle still armed, forcing disarm")
            vehicle.armed = False
            
        logger.info("Mission complete!")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        if 'vehicle' in locals() and vehicle.armed:
            logger.info("Emergency landing...")
            vehicle.mode = VehicleMode("LAND")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'vehicle' in locals():
            vehicle.close()
        if sock:
            try:
                sock.close()
            except:
                pass
        print("\nScript ended")

if __name__ == "__main__":
    main()