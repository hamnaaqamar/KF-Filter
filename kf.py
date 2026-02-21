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

GAIN = 2.0
DT = 100

FLIP_X = True
FLIP_Y = False
CENTERED_DURATION = 4

#print("Connecting to vehicle on: tcp:127.0.0.1:5762")
#vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/localize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def connect_to_ardupilot():
    """Connect to Ardupilot SITL"""
    print("Connecting to Ardupilot on: tcp:127.0.0.1:14550")
    try:
        vehicle = connect('tcp:127.0.0.1:14550', wait_ready=True, timeout=60)
        print(f"Connected to vehicle: {vehicle.version}")
        return vehicle
    except Exception as e:
        print(f"Failed to connect to Ardupilot: {e}")
        return None

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
"""   
class KalmanFilter:
    
    def __init__(self):
        self.state = np.array([0.5, 0.5, 0.0, 0.0])  # x, y, vx, vy
        self.last_time = time.time()
        self.trust_camera = 0.8 
        
    def update(self, camera_x, camera_y):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if camera_x is None: 
            # Standard KF prediction when no camera data
            self.state[0] += self.state[2] * dt  # x = x + vx*dt
            self.state[1] += self.state[3] * dt  # y = y + vy*dt
        else:  
            # Update with camera measurement
            predicted_x = self.state[0] + self.state[2] * dt
            predicted_y = self.state[1] + self.state[3] * dt
            
            # Fusion of prediction and measurement
            self.state[0] = predicted_x * (1 - self.trust_camera) + camera_x * self.trust_camera
            self.state[1] = predicted_y * (1 - self.trust_camera) + camera_y * self.trust_camera
            
            # Update velocities based on position change
            self.state[2] = (self.state[0] - predicted_x) / dt if dt > 0 else 0
            self.state[3] = (self.state[1] - predicted_y) / dt if dt > 0 else 0
        
        # Clamp positions between 0 and 1
        self.state[0] = max(0, min(1, self.state[0]))
        self.state[1] = max(0, min(1, self.state[1]))
        
        return self.state[0], self.state[1]
"""
def debug_with_delay(to_print: Callable[[], None], timer_start, delay): 
    if time.time() - timer_start <= delay: 
        to_print()
        return time.time()
    return timer_start

def get_landed_state(vehicle) -> Literal["undefined", "on_ground", "in_air", "takeoff", "landing"]:
    states = ["undefined", "on_ground", "in_air", "takeoff", "landing"]
    msg = vehicle.recv_match(type='EXTENDED_SYS_STATE', blocking=False)
    if msg:
        return states[msg.landed_state]
    return states[0] 

def cut_throttle(vehicle: Vehicle, logger: logging.Logger, throttle=0.1): 
    if vehicle.location.global_relative_frame.alt > 2: 
        logger.debug("Cutting throttle above 2m, returning")
        return

    if throttle > 0.2: 
        logger.debug("Throttle greater than 20%. returning")
        return

    msg = vehicle.message_factory.set_attitude_target_encode(
        0,     # time_boot_ms (ignored)
        1,     # target system
        1,     # target component
        0b00000111,  # type_mask: ignore attitude and body rates, ONLY thrust
        [1, 0, 0, 0],  # dummy quaternion (w=1, no rotation, ignored due to mask)
        0, 0, 0,       # body rates ignored
        throttle         # thrust 0.0-1.0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

def send_velocity(vehicle: Vehicle, logger: logging.Logger, vx, vy, vz):
    if vehicle is None: 
        return
    if vehicle.mode != "GUIDED": 
        return
    
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,                   
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b110111000111,        # velocity control mask
        0, 0, 0,                   # position (ignored)
        vy, vx, vz,                # velocities
        0, 0, 0,                   # acceleration (ignored)
        0, 0)                      # yaw/yaw_rate
    vehicle.send_mavlink(msg)
    vehicle.flush()
    logger.debug(f"Sending velocity: VX={vx:.4f}, VY={vy:.4f} VZ={vz:.4f}. alt={vehicle.location.global_relative_frame.alt}")

def send_velocity_to_unity(sock: socket.socket, vx: float, vy: float, vz: float):
    """Send velocity commands to Unity drone controller"""
    try:
        # Note: Unity might use different coordinate system
        # Adjust signs if needed based on your Unity setup
        command = f"VEL:{vx:.4f},{vy:.4f},{vz:.4f}\n"
        sock.sendall(command.encode())
    except Exception as e:
        print(f">>> ERROR sending velocity to Unity: {e}")
        
def yaw_to_angle(vehicle: Vehicle, logger: logging.Logger, yaw=0, duty_cycle=DT):
    if vehicle is None: 
        return

    previous_wp_behaviour = vehicle.parameters["WP_YAW_BEHAVIOR"]
    vehicle.parameters["WP_YAW_BEHAVIOR"] = 2
    time.sleep(1)

    vx = math.cos(yaw)
    vy = math.sin(yaw)
    
    while abs(vehicle.attitude.yaw - yaw) > 2: 
        send_velocity(vehicle, logger, vx, vy, 0)
        time.sleep(duty_cycle / 1000)
    
    vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
    time.sleep(1)

def calculate_velocity(xnorm, ynorm, max_speed, flip):
    x_error = float(xnorm) - 0.5  # Positive when target is LEFT
    y_error = 0.5 - float(ynorm)    # Positive when target is ABOVE
    
    vy = GAIN * y_error  # LEFT/RIGHT
    vx = GAIN * x_error  # UP/DOWN

    vx = -vx if flip else vx
    vy = -vy if flip else vy

    vx = max(min(max_speed, vx), -max_speed)
    vy = max(min(max_speed, vy), -max_speed)

    return vx, vy

def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False

def receive_coordinates(sock: socket.socket, label_filter):
    try:
        sock.settimeout(0.5)
        
        try:
            # Read until newline
            data = b""
            while True:
                chunk = sock.recv(1)
                if not chunk or chunk == b'\n':
                    break
                data += chunk
            
            if data:
                decoded = data.decode('utf-8').strip()
                print(f">>> RECEIVED: {decoded}")
                
                if decoded and is_valid_json(decoded):
                    data_dict = json.loads(decoded)
                    
                    x_val = data_dict.get("x")
                    y_val = data_dict.get("y")
                    
                    if x_val is None or y_val is None:
                        return None, None
                    
                    try:
                        return float(x_val), float(y_val)
                    except:
                        return None, None
                    
        except socket.timeout:
            return None, None
            
    except Exception as e:
        print(f">>> ERROR: {e}")
        return None, None
"""       
def localize(
        vehicle: Vehicle, 
        sock: socket.socket, 
        logger: logging.Logger, 
        thresh=0.05, 
        descent_speed=0.1,
        lowered_alt=1,  
        max_speed=0.2,
        label_filter=lambda x: True,
        duty_cycle=DT, 
        camera_flip=False,
        land=False, 
        log_time_delay=1,
        unity_drone_sock=None  # Add this parameter for Unity drone control
    ): 

    test_mode = vehicle is None
    if test_mode:
        logger.warning("Vehicle is None — running UNITY TEST MODE (WITH DRONE CONTROL).")
        simulated_alt = 10.0  # Starting simulated altitude for test mode
        
        # Check if we have Unity drone socket
        if unity_drone_sock is None:
            logger.warning("No Unity drone socket provided - drone won't move in Unity!")
        else:
            logger.debug("Unity drone control socket available")

    if not test_mode:
        previous_wp_behaviour = vehicle.parameters["WP_YAW_BEHAVIOR"]
        vehicle.parameters["WP_YAW_BEHAVIOR"] = 0
    else:
        previous_wp_behaviour = None

    vx, vy, vz = 0, 0, 0
    height_status = "not_lowering"  
    localize_status = "localizing"  
    log_timer_start = time.time()
    
    # Initialize Kalman Filter
    kf = KalmanFilter()  
    centered_start_time = None
    centered_duration = 0

    logger.debug(f"Localizing with parameters: thresh={thresh}, pickup_alt={lowered_alt}")

    def debug(string): 
        nonlocal log_timer_start
        log_timer_start = debug_with_delay(lambda: logger.debug(string), log_timer_start, log_time_delay)

    def baro_control(): 
        nonlocal vz, height_status, log_timer_start
        if test_mode:
            # Test mode handling - simulate altitude
            nonlocal simulated_alt
            if height_status == "lowering":
                simulated_alt -= descent_speed * (duty_cycle / 1000)
                if simulated_alt <= lowered_alt:
                    height_status = "lowered"
                    vz = 0
                    simulated_alt = lowered_alt
                    debug(f"TEST MODE: Lowered to {lowered_alt} meters")
        else:
            # Real drone handling
            if height_status == "lowering" and vehicle.location.global_relative_frame.alt <= lowered_alt: 
                height_status = "lowered"
                vz = 0
                debug(f"Lowered to {lowered_alt} meters")
    
    def land_control(): 
        nonlocal vz, height_status
        if not test_mode and height_status == "lowering" and get_landed_state(vehicle) == "on_ground": 
            height_status = 'lowered'
            debug(f"Landing complete")
            cut_throttle(vehicle, logger, 0.1)

    # Add debug for centering
    def check_centering(x_smooth, y_smooth):
        nonlocal centered_start_time, centered_duration, localize_status
        
        # Handle None values - can't center without coordinates
        if x_smooth is None or y_smooth is None:
            if centered_start_time is not None:
                debug(f"Lost centering after {time.time() - centered_start_time:.1f}s")
            centered_start_time = None
            centered_duration = 0
            localize_status = "localizing"
            return False
        
        x_error = abs(x_smooth - 0.5)
        y_error = abs(y_smooth - 0.5)
        centered = (x_error < thresh and y_error < thresh)
        
        if centered:
            if centered_start_time is None:
                centered_start_time = time.time()
                debug(f"CENTERING STARTED: x_error={x_error:.3f}, y_error={y_error:.3f}")
            else:
                centered_duration = time.time() - centered_start_time
                if centered_duration >= CENTERED_DURATION:
                    localize_status = "localized"
                    debug(f"CENTERED FOR {centered_duration:.1f}s - READY FOR DESCENT")
                else:
                    debug(f"Centering: {centered_duration:.1f}s (need {CENTERED_DURATION}s)")
        else:
            if centered_start_time is not None:
                debug(f"Lost centering after {time.time() - centered_start_time:.1f}s")
            centered_start_time = None
            centered_duration = 0
            localize_status = "localizing"
        
        return centered

    while True:
        x_raw, y_raw = receive_coordinates(sock, label_filter)

        if camera_flip and x_raw is not None:
            x_raw = 1 - x_raw
            y_raw = 1 - y_raw
        
        # APPLYING KF FILTER
        x_smooth, y_smooth = kf.update(x_raw, y_raw)
        
        if x_smooth is not None and y_smooth is not None:
            x_error = x_smooth - 0.5
            y_error = 0.5 - y_smooth
            
            vx = GAIN * x_error
            vy = GAIN * y_error
            
            # Apply flips
            vx = -vx if FLIP_X else vx
            vy = -vy if FLIP_Y else vy
            
            # Limit speed
            vx = max(min(max_speed, vx), -max_speed)
            vy = max(min(max_speed, vy), -max_speed)
            
            # Check if centered with detailed logging
            centered = check_centering(x_smooth, y_smooth)
            
            if centered_duration >= CENTERED_DURATION:
                vz = descent_speed
                debug(f"Starting descent at {descent_speed} m/s")
            
            # FIXED: Safe formatting for raw values
            raw_x_str = f"{x_raw:.3f}" if x_raw is not None else "None"
            raw_y_str = f"{y_raw:.3f}" if y_raw is not None else "None"
            
            debug(
                f"Position: Raw=({raw_x_str},{raw_y_str}), "
                f"Smooth=({x_smooth:.3f},{y_smooth:.3f}), "
                f"Error=({x_error:.3f},{y_error:.3f}), "
                f"Velocity=({vx:.3f},{vy:.3f},{vz:.3f})"
            )
        else:
            centered_start_time = None
            centered_duration = 0
            localize_status = "localizing"
            debug("No detection - maintaining last known position")
            vx, vy = 0, 0  # Stop horizontal movement when no detection
    
        if localize_status == "localized":
            height_status = "lowering"
        
        if land:
            land_control()
        else:
            baro_control()
            
        if height_status == "lowered" and localize_status == "localized": 
            debug("Localization and descent complete")
            break
    
        if not test_mode:
            send_velocity(vehicle, logger, vx, vy, vz)
        else:
            # Send velocity commands to Unity drone controller
            if unity_drone_sock is not None:
                try:
                    send_velocity_to_unity(unity_drone_sock, vx, vy, vz)
                    debug(f"Sent to Unity: VX={vx:.3f}, VY={vy:.3f}, VZ={vz:.3f}, Alt={simulated_alt:.1f}m")
                except Exception as e:
                    debug(f"Failed to send to Unity: {e}")
            else:
                # Fallback to logging only
                if vx != 0 or vy != 0 or vz != 0:
                    debug(f"TEST MODE (No Unity): Would send VX={vx:.3f}, VY={vy:.3f}, VZ={vz:.3f}")
        
        time.sleep(duty_cycle / 1000)

    if not test_mode:
        vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
    logger.debug("Localization complete")


def main():
    # Connect to Unity camera for detection
    camera_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    camera_sock.connect((MODEL_IP, MODEL_PORT))
    logger.debug("Connected to Unity Camera.")
    
    # Connect to Unity drone controller (port 9001)
    drone_sock = None
    try:
        drone_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        drone_sock.connect((MODEL_IP, 9001))  # Drone controller port
        logger.debug("Connected to Unity Drone Controller.")
        
        # Send control takeover command
        drone_sock.sendall(b"CONTROL\n")
        time.sleep(0.5)
        logger.debug("Control established with Unity drone")
        
    except Exception as e:
        logger.warning(f"Could not connect to Unity drone controller: {e}")
        drone_sock = None

    try:
        # Run localization with Unity control
        localize(None, camera_sock, logger, thresh=0.12, unity_drone_sock=drone_sock)
        
    except KeyboardInterrupt:
        logger.debug("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release control
        if drone_sock is not None:
            try:
                drone_sock.sendall(b"RELEASE\n")
            except:
                pass
            drone_sock.close()
        camera_sock.close()


"""

def localize(
        vehicle: Vehicle, 
        sock: socket.socket, 
        logger: logging.Logger, 
        thresh=0.05, 
        descent_speed=0.1,
        lowered_alt=1,  
        max_speed=0.2,
        label_filter=lambda x: True,
        duty_cycle=DT, 
        camera_flip=False,
        land=False, 
        log_time_delay=1
    ): 

    test_mode = vehicle is None
    if test_mode:
        logger.warning("Vehicle is None — running TEST MODE (NO DRONE CONTROL).")
        simulated_alt = 10.0  # Starting simulated altitude for test mode
        simulated_position = [0.63, 0.51]  # Simulated viewport position

    # Set WP_YAW_BEHAVIOR for real drone control
    if not test_mode:
        try:
            previous_wp_behaviour = vehicle.parameters["WP_YAW_BEHAVIOR"]
            vehicle.parameters["WP_YAW_BEHAVIOR"] = 0
            logger.debug("Set WP_YAW_BEHAVIOR to 0 for velocity control")
        except Exception as e:
            logger.warning(f"Could not set WP_YAW_BEHAVIOR: {e}")
            previous_wp_behaviour = None
    else:
        previous_wp_behaviour = None

    vx, vy, vz = 0, 0, 0
    height_status = "not_lowering"  
    localize_status = "localizing"  
    log_timer_start = time.time()
    
    # Initialize Kalman Filter
    kf = KalmanFilter()  
    centered_start_time = None
    centered_duration = 0

    logger.debug(f"Localizing with parameters: thresh={thresh}, pickup_alt={lowered_alt}")

    def debug(string): 
        nonlocal log_timer_start
        log_timer_start = debug_with_delay(lambda: logger.debug(string), log_timer_start, log_time_delay)

    def baro_control(): 
        nonlocal vz, height_status, log_timer_start
        if test_mode:
            # Test mode handling - simulate altitude
            nonlocal simulated_alt
            if height_status == "lowering":
                simulated_alt -= descent_speed * (duty_cycle / 1000)
                if simulated_alt <= lowered_alt:
                    height_status = "lowered"
                    vz = 0
                    simulated_alt = lowered_alt
                    debug(f"TEST MODE: Lowered to {lowered_alt} meters")
        else:
            # Real drone handling
            if height_status == "lowering" and vehicle.location.global_relative_frame.alt <= lowered_alt: 
                height_status = "lowered"
                vz = 0
                debug(f"Lowered to {lowered_alt} meters")
    
    def land_control(): 
        nonlocal vz, height_status
        if not test_mode and height_status == "lowering" and get_landed_state(vehicle) == "on_ground": 
            height_status = 'lowered'
            debug(f"Landing complete")
            cut_throttle(vehicle, logger, 0.1)

    # Add debug for centering
    def check_centering(x_smooth, y_smooth):
        nonlocal centered_start_time, centered_duration, localize_status
        
        # Handle None values - can't center without coordinates
        if x_smooth is None or y_smooth is None:
            if centered_start_time is not None:
                debug(f"Lost centering after {time.time() - centered_start_time:.1f}s")
            centered_start_time = None
            centered_duration = 0
            localize_status = "localizing"
            return False
        
        x_error = abs(x_smooth - 0.5)
        y_error = abs(y_smooth - 0.5)
        centered = (x_error < thresh and y_error < thresh)
        
        if centered:
            if centered_start_time is None:
                centered_start_time = time.time()
                debug(f"CENTERING STARTED: x_error={x_error:.3f}, y_error={y_error:.3f}")
            else:
                centered_duration = time.time() - centered_start_time
                if centered_duration >= CENTERED_DURATION:
                    localize_status = "localized"
                    debug(f"CENTERED FOR {centered_duration:.1f}s - READY FOR DESCENT")
                else:
                    debug(f"Centering: {centered_duration:.1f}s (need {CENTERED_DURATION}s)")
        else:
            if centered_start_time is not None:
                debug(f"Lost centering after {time.time() - centered_start_time:.1f}s")
            centered_start_time = None
            centered_duration = 0
            localize_status = "localizing"
        
        return centered

    # Update simulated position for test mode
    def update_simulated_position(vx, vy, dt):
        nonlocal simulated_position
        if test_mode and simulated_position is not None:
            # Simulate how the viewport coordinates would change
            # When drone moves left (negative vx), x coordinate should decrease toward 0.5
            simulated_position[0] += vx * dt * 0.1  # Scale factor for simulation
            simulated_position[1] += vy * dt * 0.1
            # Clamp to reasonable values
            simulated_position[0] = max(0.1, min(0.9, simulated_position[0]))
            simulated_position[1] = max(0.1, min(0.9, simulated_position[1]))

    while True:
        x_raw, y_raw = receive_coordinates(sock, label_filter)

        # In test mode with no camera data, use simulated position
        if test_mode and (x_raw is None or y_raw is None):
            x_raw, y_raw = simulated_position[0], simulated_position[1]
            debug(f"TEST MODE: Using simulated position ({x_raw:.3f}, {y_raw:.3f})")

        if camera_flip and x_raw is not None:
            x_raw = 1 - x_raw
            y_raw = 1 - y_raw
        
        # APPLYING KF FILTER
        x_smooth, y_smooth = kf.update(x_raw, y_raw)
        
        if x_smooth is not None and y_smooth is not None:
            x_error = x_smooth - 0.5
            y_error = 0.5 - y_smooth
            
            vx = GAIN * x_error
            vy = GAIN * y_error
            
            # Apply flips
            vx = -vx if FLIP_X else vx
            vy = -vy if FLIP_Y else vy
            
            # Limit speed
            vx = max(min(max_speed, vx), -max_speed)
            vy = max(min(max_speed, vy), -max_speed)
            
            # Check if centered with detailed logging
            centered = check_centering(x_smooth, y_smooth)
            
            if centered_duration >= CENTERED_DURATION:
                vz = descent_speed
                debug(f"Starting descent at {descent_speed} m/s")
            
            # Safe formatting for raw values
            raw_x_str = f"{x_raw:.3f}" if x_raw is not None else "None"
            raw_y_str = f"{y_raw:.3f}" if y_raw is not None else "None"
            
            debug(
                f"Position: Raw=({raw_x_str},{raw_y_str}), "
                f"Smooth=({x_smooth:.3f},{y_smooth:.3f}), "
                f"Error=({x_error:.3f},{y_error:.3f}), "
                f"Velocity=({vx:.3f},{vy:.3f},{vz:.3f})"
            )
            
            # Update simulated position in test mode
            if test_mode:
                update_simulated_position(vx, vy, duty_cycle/1000)
        else:
            centered_start_time = None
            centered_duration = 0
            localize_status = "localizing"
            debug("No detection - maintaining last known position")
            vx, vy = 0, 0  # Stop horizontal movement when no detection
    
        if localize_status == "localized":
            height_status = "lowering"
        
        if land:
            land_control()
        else:
            baro_control()
            
        if height_status == "lowered" and localize_status == "localized": 
            debug("Localization and descent complete")
            break
    
        if not test_mode:
            # Send velocity to REAL Ardupilot drone
            send_velocity(vehicle, logger, vx, vy, vz)
        else:
            # Test mode - just log
            if vx != 0 or vy != 0 or vz != 0:
                alt = simulated_alt if 'simulated_alt' in locals() else "N/A"
                debug(f"TEST MODE: Would send to Ardupilot - VX={vx:.3f}, VY={vy:.3f}, VZ={vz:.3f}, Alt={alt}m")
        
        time.sleep(duty_cycle / 1000)

    # Restore WP_YAW_BEHAVIOR for real drone
    if not test_mode and previous_wp_behaviour is not None:
        try:
            vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
            logger.debug(f"Restored WP_YAW_BEHAVIOR to {previous_wp_behaviour}")
        except Exception as e:
            logger.warning(f"Could not restore WP_YAW_BEHAVIOR: {e}")
            
    logger.debug("Localization complete")

def main():
    # Connect to Unity camera
    camera_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    camera_sock.connect((MODEL_IP, MODEL_PORT))
    logger.debug("Connected to Unity Camera.")
    
    # Connect to Ardupilot (NOT Unity drone control)
    vehicle = connect_to_ardupilot()
    if vehicle is None:
        logger.error("Could not connect to Ardupilot! Running in test mode.")
        vehicle = None  # Test mode
    
    try:
        # Run localization with REAL Ardupilot control
        localize(vehicle, camera_sock, logger, thresh=0.12)
        
    except KeyboardInterrupt:
        logger.debug("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera_sock.close()
        if vehicle is not None:
            vehicle.close()
        logger.debug("All connections closed")


if __name__ == "__main__":
    # Uncomment to run Kalman filter test
    # test_kalman_filter()
    
    # Run main localization with Unity
    main()