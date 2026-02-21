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
        thresh=0.1,  # 10% from center is acceptable
        descent_speed=0.1,  # Slower descent
        detection_alt=4.0,
        landing_alt=0.3,
        max_speed=0.12,  # Much slower max speed (was 0.15)
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

    # State variables
    flight_phase = "searching"  # searching -> centering -> descending -> landed
    log_timer_start = time.time()
    
    # Kalman filter with conservative settings
    kf = KalmanFilter(process_noise=0.02, measurement_noise=0.15)
    
    # Centering variables
    centered_start_time = None
    centered_duration = 0
    
    # Detection tracking
    last_detection_time = time.time()
    detection_timeout = 3.0
    consecutive_detections = 0
    min_detections_for_centering = 2  # Need 2 detections to trust centering
    
    # Store last known good position
    last_good_x = 0.5
    last_good_y = 0.5

    # Simple P controller (no PID to keep it simple)
    Kp = 0.3  # Proportional gain (reduced from 0.8)
    
    # Dead zone - stop moving when very close
    dead_zone = 0.05  # 5% dead zone

    # Get current altitude
    if not test_mode:
        current_alt = vehicle.location.global_relative_frame.alt
        logger.info(f"Starting at altitude: {current_alt:.2f}m")
        logger.info(f"Target detection altitude: {detection_alt}m")
    else:
        current_alt = 5.0
        simulated_alt = current_alt

    def debug(string): 
        nonlocal log_timer_start
        log_timer_start = debug_with_delay(lambda: logger.debug(string), log_timer_start, log_time_delay)

    # Descend to detection altitude if needed
    if not test_mode and current_alt > detection_alt + 0.5:
        logger.info(f"Descending to detection altitude {detection_alt}m...")
        while vehicle.location.global_relative_frame.alt > detection_alt + 0.15:
            send_velocity(vehicle, logger, 0, 0, 0.1)
            time.sleep(duty_cycle / 1000)
        logger.info(f"Reached detection altitude")

    while True:
        # Get coordinates from Unity
        x_raw, y_raw = receive_coordinates(sock, label_filter)
        current_time = time.time()
        
        # Get current altitude
        if not test_mode:
            current_alt = vehicle.location.global_relative_frame.alt
        
        # ===== DETECTION HANDLING =====
        detection_valid = (x_raw is not None and y_raw is not None)
        
        if detection_valid:
            # We have a detection!
            last_detection_time = current_time
            consecutive_detections += 1
            
            # Apply Kalman filter
            x_smooth, y_smooth = kf.update(x_raw, y_raw)
            last_good_x, last_good_y = x_smooth, y_smooth
            
            # Log raw and smoothed values
            logger.debug(f"DETECT: raw=({x_raw:.3f},{y_raw:.3f}) smooth=({x_smooth:.3f},{y_smooth:.3f})")
            
            # Calculate errors from center (0.5, 0.5)
            x_error = x_smooth - 0.5  # Positive = target is RIGHT of center
            y_error = y_smooth - 0.5  # Positive = target is ABOVE center
            
            logger.debug(f"ERRORS: x_err={x_error:.3f}, y_err={y_error:.3f}")
            
            # SIMPLE CONTROL - just proportional
            # Move in the direction that reduces the error
            
            # Check if we're in dead zone (very close to center)
            if abs(x_error) < dead_zone and abs(y_error) < dead_zone:
                # In dead zone - stop moving
                vx = 0
                vy = 0
                logger.debug("DEAD ZONE: stopped")
            else:
                # Simple P controller
                vx = Kp * y_error   # Forward/back from vertical error
                vy = Kp * x_error   # Left/right from horizontal error
                
                # Limit speed
                vx = max(min(max_speed, vx), -max_speed)
                vy = max(min(max_speed, vy), -max_speed)
                
                logger.debug(f"VEL: vx={vx:.3f}, vy={vy:.3f}")
            
            # Determine if we're centered (using threshold, not dead zone)
            centered = (abs(x_error) < thresh and abs(y_error) < thresh)
            
            # STATE MACHINE
            if flight_phase == "searching":
                # Found target while searching
                if consecutive_detections >= min_detections_for_centering:
                    flight_phase = "centering"
                    centered_start_time = None
                    logger.info(f"TARGET FOUND - beginning centering (after {consecutive_detections} detections)")
                vz = 0  # Hold altitude
                
            elif flight_phase == "centering":
                vz = 0  # Hold altitude while centering
                
                if centered:
                    if centered_start_time is None:
                        centered_start_time = current_time
                        logger.info(f"Target centered - starting timer")
                    else:
                        centered_duration = current_time - centered_start_time
                        if centered_duration >= CENTERED_DURATION:
                            flight_phase = "descending"
                            logger.info(f"CENTERED for {centered_duration:.1f}s - starting descent")
                else:
                    centered_start_time = None
                    centered_duration = 0
                    
            elif flight_phase == "descending":
                # Maintain center while descending
                if current_alt <= landing_alt + 0.2:
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
                logger.info("LANDED on target")
                break
                
        else:
            # NO DETECTION
            consecutive_detections = 0
            time_since_detection = current_time - last_detection_time
            
            logger.debug(f"NO DETECT: {time_since_detection:.1f}s since last detection")
            
            # Check if we've lost the target for too long
            if time_since_detection > detection_timeout:
                if flight_phase != "searching":
                    flight_phase = "searching"
                    logger.info("TARGET LOST - starting search pattern")
            
            # Set velocities based on phase
            if flight_phase == "searching":
                # Simple back-and-forth search pattern
                # Move in a small square pattern to find the target
                search_step = int(current_time * 2) % 4
                
                if search_step == 0:
                    vx = 0.08  # Move forward
                    vy = 0
                elif search_step == 1:
                    vx = 0
                    vy = 0.08  # Move right
                elif search_step == 2:
                    vx = -0.08  # Move back
                else:
                    vx = 0
                    vy = -0.08  # Move left
                
                vz = 0
                logger.debug(f"SEARCH: step={search_step}")
                
            elif flight_phase == "centering":
                # Lost target during centering - stop and wait briefly
                if time_since_detection < 2.0:
                    vx, vy, vz = 0, 0, 0
                    logger.debug(f"Lost target during centering - waiting")
                else:
                    # After waiting, go back to search
                    flight_phase = "searching"
                    logger.info("Target lost during centering - switching to search")
                    vx, vy, vz = 0, 0, 0
                    
            elif flight_phase == "descending":
                # Lost target during descent - stop descending and hover
                vx, vy, vz = 0, 0, 0
                if time_since_detection > 1.0:
                    logger.warning(f"Lost target during descent - hovering")
            else:
                vx, vy, vz = 0, 0, 0
        
        # Safety check - if we're descending and very low, just land
        if flight_phase == "descending" and current_alt <= 0.3:
            logger.info("VERY LOW ALTITUDE - forcing land")
            flight_phase = "landed"
            continue
        
        # Send velocity commands
        if not test_mode:
            send_velocity(vehicle, logger, vx, vy, vz)
        else:
            if flight_phase == "descending":
                simulated_alt -= vz * (duty_cycle / 1000)
                if simulated_alt <= landing_alt:
                    flight_phase = "landed"
                    simulated_alt = landing_alt
            
            if unity_drone_sock is not None:
                send_velocity_to_unity(unity_drone_sock, vx, vy, vz)
        
        # Log current state periodically
        if int(current_time * 2) % 10 == 0:  # Log every ~5 seconds
            logger.info(
                f"PHASE: {flight_phase}, "
                f"ALT: {current_alt:.2f}m, "
                f"VEL: ({vx:.3f},{vy:.3f},{vz:.3f})"
            )
        else:
            debug(
                f"PHASE: {flight_phase}, "
                f"ALT: {current_alt:.2f}m, "
                f"VEL: ({vx:.3f},{vy:.3f},{vz:.3f})"
            )
        
        time.sleep(duty_cycle / 1000)

    if not test_mode:
        vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
    logger.info("Localization complete")
    
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