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

def receive_coordinates(sock: socket.socket, label_filter):
    try:
        print("\n>>> SENDING TO UNITY: detect")
        sock.sendall(b"detect\n")  # Add newline
        
        # Set timeout for receiving
        sock.settimeout(3.0)
        
        print(">>> Waiting for response from Unity...")
        
        # Try to receive data
        data_buffer = b""
        try:
            while True:
                chunk = sock.recv(256)
                if not chunk:
                    break
                data_buffer += chunk
                print(f">>> Received chunk: {chunk}")
        except socket.timeout:
            print(">>> Timeout while receiving from Unity")
        
        print(f">>> TOTAL DATA RECEIVED: {len(data_buffer)} bytes")
        
        if not data_buffer:
            print(">>> UNITY SENT NOTHING!")
            return None, None
        
        # Decode the data
        decoded = data_buffer.decode('utf-8', errors='ignore').strip()
        print(f">>> DECODED DATA: '{decoded}'")
        
        # Try to parse as JSON
        try:
            data = json.loads(decoded)
            print(f">>> PARSED JSON: {data}")
            
            if "x" not in data or "y" not in data:
                print(">>> JSON missing x or y keys")
                return None, None
            
            if not label_filter(data):
                print(">>> Label filter rejected", data)
                return None, None
            
            # Check if values are null or None
            if data["x"] is None or data["y"] is None:
                print(f">>> Null coordinates: x={data['x']}, y={data['y']}")
                return None, None
            
            # Convert to float
            try:
                x_val = float(data["x"])
                y_val = float(data["y"])
                print(f">>> RETURNING COORDINATES: ({x_val}, {y_val})")
                return x_val, y_val
            except (ValueError, TypeError) as e:
                print(f">>> Error converting to float: {e}")
                return None, None
                
        except json.JSONDecodeError as e:
            print(f">>> Not valid JSON: {e}")
            # Try to see if it's a simple string response
            if "error" in decoded.lower() or "no detection" in decoded.lower():
                print(f">>> Unity error message: {decoded}")
            return None, None
        
    except Exception as e:
        print(f">>> ERROR IN receive_coordinates: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

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
            
            # Check if centered
            centered = (abs(x_smooth - 0.5) < thresh and abs(y_smooth - 0.5) < thresh)
            
            if centered:
                if centered_start_time is None:
                    centered_start_time = time.time()
                else:
                    centered_duration = time.time() - centered_start_time
                
                if centered_duration >= CENTERED_DURATION:
                    localize_status = "localized"
                    vz = descent_speed
                    debug(f"Centered for {centered_duration:.1f}s, starting descent")
            else:
                centered_start_time = None
                centered_duration = 0
                localize_status = "localizing"
                
            debug(
                f"Position: Raw=({x_raw if x_raw is not None else 'None'},"
                f"{y_raw if y_raw is not None else 'None'}), "
                f"Smooth=({x_smooth:.3f},{y_smooth:.3f})"
            )
        else:
            centered_start_time = None
            centered_duration = 0
            debug("No detection")
            vx, vy = 0, 0  
    
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
            # Log what would happen in test mode
            if vx != 0 or vy != 0 or vz != 0:
                debug(f"TEST MODE: Would send velocity VX={vx:.3f}, VY={vy:.3f}, VZ={vz:.3f}")
        
        time.sleep(duty_cycle / 1000)

    if not test_mode:
        vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
    logger.debug("Localization complete")

