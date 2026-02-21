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
    """Get landed state using DroneKit attributes instead of pymavlink"""
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
        command = f"VEL:{vx:.4f},{vy:.4f},{vz:.4f}\n"
        sock.sendall(command.encode())
    except Exception as e:
        print(f">>> ERROR sending velocity to Unity: {e}")

def receive_coordinates(sock: socket.socket, label_filter):
    try:
        print("\n>>> SENDING TO UNITY: detect")
        sock.sendall(b"detect\n")
        
        # Set timeout for receiving
        sock.settimeout(1.0)  # Reduced timeout
        
        print(">>> Waiting for response from Unity...")
        
        # Try to receive data
        data_buffer = b""
        try:
            # Read until newline or timeout
            while True:
                chunk = sock.recv(256)
                if not chunk:
                    break
                data_buffer += chunk
                if b'\n' in chunk:  # Stop at newline
                    break
                print(f">>> Received chunk: {chunk}")
        except socket.timeout:
            print(">>> Timeout while receiving from Unity")
            # Don't close socket on timeout, just return None
            return None, None
        
        print(f">>> TOTAL DATA RECEIVED: {len(data_buffer)} bytes")
        
        if not data_buffer:
            print(">>> UNITY SENT NOTHING!")
            return None, None
        
        # Decode and parse
        decoded = data_buffer.decode('utf-8', errors='ignore').strip()
        print(f">>> DECODED DATA: '{decoded}'")
        
        # Parse JSON (handle multiple JSON objects)
        try:
            # Split by newlines and take the first valid JSON
            lines = decoded.split('\n')
            for line in lines:
                if line.strip():
                    data = json.loads(line)
                    print(f">>> PARSED JSON: {data}")
                    
                    if "x" in data and "y" in data and data["x"] is not None and data["y"] is not None:
                        if label_filter(data):
                            x_val = float(data["x"])
                            y_val = float(data["y"])
                            print(f">>> RETURNING COORDINATES: ({x_val}, {y_val})")
                            return x_val, y_val
                    else:
                        print(f">>> Invalid or null coordinates: {data}")
            
            return None, None
                
        except json.JSONDecodeError as e:
            print(f">>> Not valid JSON: {e}")
            return None, None
        
    except (socket.error, ConnectionError) as e:
        print(f">>> Socket error: {e}")
        return None, None

def arm_and_takeoff(vehicle, target_altitude):
    """
    Arms vehicle and fly to target_altitude.
    """
    logger.info("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        logger.info(" Waiting for vehicle to initialise...")
        time.sleep(1)

    logger.info("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        logger.info(" Waiting for arming...")
        time.sleep(1)

    logger.info("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    # Wait until the vehicle reaches a safe height
    while True:
        logger.info(f" Altitude: {vehicle.location.global_relative_frame.alt:.2f}")
        # Break and return from function just below target altitude
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
        lowered_alt=2,  # Changed to 2 meters (was 1)
        max_speed=0.2,
        label_filter=lambda x: True,
        duty_cycle=DT, 
        camera_flip=False,
        land=False, 
        log_time_delay=1,
        unity_drone_sock=None
    ): 

    test_mode = vehicle is None
    if test_mode:
        logger.warning("Vehicle is None — running UNITY TEST MODE (WITH DRONE CONTROL).")
        simulated_alt = 10.0

    if not test_mode:
        previous_wp_behaviour = vehicle.parameters["WP_YAW_BEHAVIOR"]
        vehicle.parameters["WP_YAW_BEHAVIOR"] = 0
    else:
        previous_wp_behaviour = None

    vx, vy, vz = 0, 0, 0
    # States: "searching" -> "centered" -> "lowering" -> "lowered"
    flight_phase = "searching"  # New state variable
    localize_status = "localizing"  
    log_timer_start = time.time()
    
    # Initialize Kalman Filter
    kf = KalmanFilter()  
    centered_start_time = None
    centered_duration = 0

    # Store takeoff altitude
    if not test_mode:
        takeoff_alt = vehicle.location.global_relative_frame.alt
        logger.info(f"Takeoff altitude: {takeoff_alt:.2f}m, Will descend to: {lowered_alt}m")
    else:
        takeoff_alt = 5.0

    logger.debug(f"Localizing with parameters: thresh={thresh}, pickup_alt={lowered_alt}")

    def debug(string): 
        nonlocal log_timer_start
        log_timer_start = debug_with_delay(lambda: logger.debug(string), log_timer_start, log_time_delay)

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
            
            # Calculate horizontal velocities based on flight phase
            if flight_phase == "searching" or flight_phase == "centered":
                # We're at takeoff altitude, trying to center
                vx = GAIN * x_error
                vy = GAIN * y_error
                
                # Apply flips
                vx = -vx if FLIP_X else vx
                vy = -vy if FLIP_Y else vy
                
                # Limit speed
                vx = max(min(max_speed, vx), -max_speed)
                vy = max(min(max_speed, vy), -max_speed)
                
                # No vertical movement while searching/centering
                vz = 0
            else:  # lowering phase
                # Maintain position while descending
                vx = GAIN * x_error
                vy = GAIN * y_error
                
                vx = -vx if FLIP_X else vx
                vy = -vy if FLIP_Y else vy
                
                vx = max(min(max_speed, vx), -max_speed)
                vy = max(min(max_speed, vy), -max_speed)
                
                # Keep descending at constant speed
                vz = descent_speed
            
            # Check if centered
            centered = (abs(x_smooth - 0.5) < thresh and abs(y_smooth - 0.5) < thresh)
            
            # State machine for flight phases
            if flight_phase == "searching":
                if centered:
                    if centered_start_time is None:
                        centered_start_time = time.time()
                        debug("Target detected, starting centering timer")
                    else:
                        centered_duration = time.time() - centered_start_time
                    
                    if centered_duration >= CENTERED_DURATION:
                        flight_phase = "centered"
                        logger.info(f"Target centered for {centered_duration:.1f}s, ready to descend")
                else:
                    centered_start_time = None
                    centered_duration = 0
                    
            elif flight_phase == "centered":
                # We're centered at takeoff altitude
                logger.info("Starting descent to pickup altitude")
                flight_phase = "lowering"
                
            elif flight_phase == "lowering":
                # Check if we've reached the target altitude
                if not test_mode:
                    current_alt = vehicle.location.global_relative_frame.alt
                    if current_alt <= lowered_alt:
                        flight_phase = "lowered"
                        vz = 0
                        logger.info(f"Reached pickup altitude: {current_alt:.2f}m")
                else:
                    # Test mode
                    if simulated_alt <= lowered_alt:
                        flight_phase = "lowered"
                        vz = 0
                        simulated_alt = lowered_alt
                        debug(f"TEST MODE: Lowered to {lowered_alt} meters")
                
            debug(
                f"Phase: {flight_phase}, "
                f"Position: ({x_smooth:.3f},{y_smooth:.3f}), "
                f"Vel: ({vx:.3f},{vy:.3f},{vz:.3f})"
            )
        else:
            # No detection
            centered_start_time = None
            centered_duration = 0
            
            if flight_phase == "searching":
                # Hover in place while searching
                vx, vy, vz = 0, 0, 0
                debug("No detection - hovering at takeoff altitude")
            elif flight_phase == "lowering":
                # Continue descending slowly even without detection
                vx, vy = 0, 0
                vz = descent_speed * 0.5  # Descend slower when no detection
                debug(f"No detection - descending slowly at {vz:.3f} m/s")
        
        # Break condition
        if flight_phase == "lowered":
            if land:
                # If land flag is True, we'll land after reaching pickup altitude
                logger.info("Starting landing sequence")
                if not test_mode:
                    vehicle.mode = VehicleMode("LAND")
                break
            else:
                # Just hover at pickup altitude
                logger.info("Reached pickup altitude, hovering")
                break
    
        # Send velocity commands
        if not test_mode:
            send_velocity(vehicle, logger, vx, vy, vz)
        else:
            if vx != 0 or vy != 0 or vz != 0:
                debug(f"TEST MODE: Would send velocity VX={vx:.3f}, VY={vy:.3f}, VZ={vz:.3f}")
            if unity_drone_sock is not None:
                send_velocity_to_unity(unity_drone_sock, vx, vy, vz)
        
        time.sleep(duty_cycle / 1000)

    if not test_mode:
        vehicle.parameters["WP_YAW_BEHAVIOR"] = previous_wp_behaviour
    logger.debug("Localization complete")

def test_unity_connection():
    """Simple test to check Unity connection"""
    print("\n" + "="*60)
    print("Testing Unity Connection...")
    print("="*60)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    
    try:
        print(f"Connecting to {MODEL_IP}:{MODEL_PORT}...")
        sock.connect((MODEL_IP, MODEL_PORT))
        print("✓ Connected to Unity")
        
        # Test different commands
        test_commands = [
            b"detect\n",
            b"hello\n",
            b"status\n",
            b"ping\n"
        ]
        
        for cmd in test_commands:
            print(f"\nSending command: {cmd.decode().strip()}")
            sock.sendall(cmd)
            
            try:
                response = sock.recv(1024)
                if response:
                    print(f"Response: {response}")
                    # Try to decode
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
                
            time.sleep(1)
            
    except ConnectionRefusedError:
        print("✗ Connection refused. Is Unity running?")
        print("Make sure:")
        print("1. Unity is running the simulation")
        print("2. The TCP server is listening on port 9000")
        print("3. No firewall is blocking the connection")
    except Exception as e:
        print(f"✗ Connection error: {e}")
    finally:
        sock.close()
        print("\n" + "="*60)
        print("Test complete")
        print("="*60)

def main():
    """Main function with better error handling"""
    print("\nStarting Unity Localization Script...")
    
    # First test the connection
    test_unity_connection()
    
    # Then try to run localization
    choice = input("\nDo you want to continue with localization? (y/n): ")
    if choice.lower() != 'y':
        return
    
    # Connect to the Vehicle through MAVProxy
    connection_string = '192.168.56.1:14552'
    logger.info(f"Connecting to vehicle on {connection_string}")
    
    try:
        vehicle = connect(connection_string, wait_ready=True, timeout=30)
        logger.info("Vehicle connected")
        
        # Check if vehicle is already flying
        current_alt = vehicle.location.global_relative_frame.alt
        logger.info(f"Current altitude: {current_alt:.2f}m")
        
        if current_alt < 1:  # If on ground or very low
            target_altitude = 10
            arm_and_takeoff(vehicle, target_altitude)
        else:
            logger.info("Vehicle already airborne, skipping takeoff")
            if vehicle.mode != "GUIDED":
                vehicle.mode = VehicleMode("GUIDED")
        
        # Give the drone a moment to stabilize
        logger.info("Stabilizing at altitude...")
        time.sleep(3)
        
        # Connect to Unity with retry logic
        sock = None
        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect(('127.0.0.1', MODEL_PORT))
                logger.debug(f"Connected to Unity Model on attempt {attempt+1}")
                break
            except Exception as e:
                logger.warning(f"Failed to connect to Unity (attempt {attempt+1}): {e}")
                if attempt == 2:
                    raise
                time.sleep(1)
        
        # Run localization
        print("\nStarting localization loop...")
        print("Press Ctrl+C to stop\n")
        
        localize(
            vehicle,
            sock, 
            logger, 
            thresh=0.12,           # How close to center needs to be
            lowered_alt=2,         # Descend to 2 meters (not ground)
            descent_speed=0.2,     # Descend at 0.2 m/s
            max_speed=0.3,         # Max horizontal speed
            land=False,            # Set to False to just hover at pickup altitude
            log_time_delay=0.5
        )
        
        # After localization completes, land if not already landed
        logger.info("Localization complete, landing...")
        if vehicle.armed and vehicle.location.global_relative_frame.alt > 0.5:
            vehicle.mode = VehicleMode("LAND")
            
        # Wait for landing
        timeout = time.time() + 30  # 30 second timeout
        while vehicle.armed and time.time() < timeout:
            logger.info(f" Landing... Altitude: {vehicle.location.global_relative_frame.alt:.2f}")
            time.sleep(1)
            
        if not vehicle.armed:
            logger.info("Landed successfully")
        else:
            logger.warning("Landing timeout - vehicle still armed")
        
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
        # Cleanup
        if 'vehicle' in locals():
            vehicle.close()
        if 'sock' in locals():
            try:
                sock.close()
            except:
                pass
        print("\nScript ended")

if __name__ == "__main__":
    main()