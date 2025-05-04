"""grocery controller."""

# Apr 1, 2025

from controller import Robot, Supervisor
import math
import numpy as np
from image_tools import ImageTools
import open3d as o3d
from ikpy.chain import Chain

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

ROBOT_MODE_TOGGLE_KEY = 'M' # Key to switch between drive and arm control
CARTESIAN_STEP = 0.015 # Step size for arm control in meters
ORIENTATION_STEP = 0.05 # Step size for arm orientation control in radians
KEY_COOLDOWN_CYCLES = 10 # Number of simulation steps for key press cooldown


# Start in drive mode
robot_mode = "drive" 
gripper_status="closed"
key_cooldown_timer = 0 # Initialize cooldown timer
mode_just_changed = False
print(f"Starting in '{robot_mode}' mode. Press '{ROBOT_MODE_TOGGLE_KEY}' to switch.")
# create the Robot instance.
robot = Robot()

image_tools = ImageTools(
    yolo_weight_path='./models/best.pt',
    fastsam_weight_path='./models/FastSAM-s.pt',
)

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())


END_EFFECTOR_CONTROL_KEYS = {"up":'E', "down":'Q', "left":'A', "right":'D', "forward":'W', "backward":'S',
                             "pitch_up":'I', "pitch_down":'K', "roll_left":'J', "roll_right":'L', "yaw_left":'U', "yaw_right":'O',
                             "orient_y":'Y', "orient_x":"X", "orient_z":"Z"}

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")


base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]

my_chain = Chain.from_urdf_file("robot_urdf.urdf", base_elements=base_elements)
# --- IK Setup ---
print("--- Setting up IK Chain ---")

try:
    # Define base elements for the camera chain (up to the head)
    camera_base_elements = base_elements[:-2] + ["head_1_joint", "head_1_link", "head_2_joint", "head_2_link"]
    camera_link_name = "rgb_camera_optical_frame"

    # Create the chain directly from the modified URDF file
    camera_chain = Chain.from_urdf_file(
        "robot_urdf.urdf",
        base_elements=camera_base_elements,
    )
    print(f"  Camera IK Chain created from URDF. Base: {camera_base_elements[0]}, Tip: {camera_link_name}")

    # You can now use camera_chain.forward_kinematics(joint_angles)
    # where joint_angles includes values for head_1_joint and head_2_joint
    # to get the transformation matrix from base_link to rgb_camera_optical_frame

except FileNotFoundError:
    print("!!! ERROR: robot_urdf.urdf not found. Cannot create camera chain.")
    camera_chain = None
except Exception as e:
    print(f"!!! ERROR creating camera IK chain: {e}")
    camera_chain = None


motor_dict = {}
# First pass: disable fixed links and any links not in part_names
for link_id in range(len(my_chain.links)):
    link = my_chain.links[link_id]
    # Disable fixed joints and joints not in our controllable part_names list
    # Keep base elements active even if not in part_names, ikpy handles them
    is_base_element = link.name in base_elements
    if link.joint_type == "fixed" or (link.name not in part_names and not is_base_element):
        print(f"  Disabling link: {link.name} (Type: {link.joint_type}, In parts: {link.name in part_names})")
        my_chain.active_links_mask[link_id] = False
        
# --- Lock specific joints for IK (e.g., torso_lift_joint) ---
print("--- Locking specific joints for IK ---")
disable_joint_names = ["torso_lift_joint", "gripper_right_finger_joint", "gripper_left_finger_joint"]

for joint_name in disable_joint_names:
    try:
        torso_link_index = [i for i, link in enumerate(my_chain.links) if link.name == joint_name][0]
        if my_chain.active_links_mask[torso_link_index]:
            my_chain.active_links_mask[torso_link_index] = False
            # print(f"  Locked '{joint_name}' (Link Index: {torso_link_index}) for IK calculations.")
        else:
            print(f"  Joint '{joint_name}' was already inactive for IK.")
    except IndexError:
        print(f"  Warning: Joint '{joint_name}' not found in the IK chain.")
print("--- Joint Locking Complete ---")


# Initialize the arm motors and link them to the IK chain.
print("--- Initializing Motors for IK ---")
for link_id in range(len(my_chain.links)):
    link = my_chain.links[link_id]
    # Only try to get motors for active, non-fixed links that are in part_names
    if my_chain.active_links_mask[link_id] and link.joint_type != "fixed" and link.name in part_names:
        try:
            motor = robot.getDevice(link.name)
            position_sensor = motor.getPositionSensor() # Assumes sensor exists if motor does

            if not position_sensor:
                print(f"  Warning: No position sensor found for motor '{link.name}', disabling link.")
                my_chain.active_links_mask[link_id] = False
                continue # Skip this motor if no sensor

            # Enable sensor if not already (should be covered later, but safe)
            if position_sensor.getSamplingPeriod() <= 0:
                position_sensor.enable(timestep)

            # Set appropriate velocity (adjust as needed)
            if link.name == "torso_lift_joint":
                motor.setVelocity(0.07)
            else:
                # Use a fraction of max velocity for smoother IK control
                motor.setVelocity(motor.getMaxVelocity() * 0.8) # Example: 80%

            motor_dict[link.name] = motor # Store motor, keyed by link name
            # print(f"  Enabled motor and sensor for IK link: {link.name}")

        except Exception as e:
            print(f"  Error getting device/sensor for link '{link.name}': {e}. Disabling link.")
            my_chain.active_links_mask[link_id] = False

print(f"--- IK Setup Complete. Active motors in motor_dict: {list(motor_dict.keys())} ---")
# --- End IK Setup ---

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)
upper_shelf = (0.0, 0.0, 0.35, 0.447, 0.651, -1.439, 2.035, 1.845, 0.816, 1.983, 'inf', 'inf', 0.045, 0.045)
above_basket = (0.0, 0.0, 0.35, 0.07, 0.619, -0.519, 2.290, 1.892, -1.353, 0.390, 'inf', 'inf', 0.045, 0.045)
target_pos = upper_shelf
# Initialize head angles from target_pos
current_head_yaw = target_pos[0]  # head_1_joint is the first element
current_head_tilt = target_pos[1] # head_2_joint is the second element
HEAD_TILT_STEP = 0.05
HEAD_TILT_MAX = 0.5
HEAD_TILT_MIN = -1.2
HEAD_YAW_STEP = 0.05 # Step size for head yaw
HEAD_YAW_MAX = 1.0   # Maximum head yaw angle
HEAD_YAW_MIN = -1.0  # Minimum head yaw angle

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    # Handle 'inf' for velocity-controlled wheels during setPosition
    if target_pos[i] != 'inf':
        robot_parts[part_name].setPosition(float(target_pos[i]))
    else:
        # For wheels, set position to infinity and velocity control
        robot_parts[part_name].setPosition(float('inf'))
        robot_parts[part_name].setVelocity(0.0) # Begin with wheels stopped

    # Set max velocity for all parts (wheels will be controlled by setVelocity later)
    # Reduce default velocity slightly for smoother movements if needed
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() * 0.5)

# Initialize and enable Position Sensors for joints that have them
print("--- Enabling Position Sensors ---")
robot_sensors = {}
for part_name in part_names:
    # Skip wheel joints
    if "wheel" in part_name:
        continue

    sensor_name = part_name + "_sensor"
    sensor = robot.getDevice(sensor_name)
    if sensor:
        sensor.enable(timestep)
        robot_sensors[part_name] = sensor # Store sensor keyed by part_name
        # print(f"  Enabled sensor: {sensor_name}")
    else:
        print(f"  Warning: Sensor '{sensor_name}' not found for joint '{part_name}'.")
print("--- Sensor Enabling Complete ---")

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Range Finder
range_finder = robot.getDevice('depth_camera')
range_finder.enable(timestep)

# Enable RGB Camera
rgb_camera = robot.getDevice('rgb_camera')
rgb_camera.enable(timestep)
# rgb_camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")
SAM_view_display = robot.getDevice("SAM view")


# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

# Camera Intrinsics
width = 640
height = 480
fov_x_rad = 1.49

# Calculate principal point
cx = width / 2
cy = height / 2

# Calculate focal length
fx = (width / 2) / np.tan(fov_x_rad / 2)
fy = fx # Assuming square pixels

o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Enable keyboard
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# ------------------------------------------------------------------
# Helper Functions
def handle_capture():
    # Get the raw image data as bytes
    print("Capturing image")
    color_image_data = rgb_camera.getImage()
    depth_image = range_finder.getRangeImage()

    
    # Convert bytes to numpy array (correct shape)
    color_image = np.frombuffer(color_image_data, np.uint8)
    # Reshape considering BGRA format (4 channels) that Webots uses
    color_image = color_image.reshape(height, width, 4)
    # Convert to RGB if needed
    color_image = color_image[:, :, :3]
    
    # Handle depth image
    depth_width = range_finder.getWidth()
    depth_height = range_finder.getHeight()
    depth_image = np.array(depth_image).reshape(depth_height, depth_width)
    
    image_tools.save_images(color_image, depth_image, '../../camera_data')
    # image_tools.save_sequential_color_image(color_image, '../../training_data')

def draw_detected_objects():
    SAM_view_display.setColor(0x000000)
    SAM_view_display.fillRectangle(0, 0, 640, 480)
    
    color_image = rgb_camera.getImage()
    depth_image = range_finder.getRangeImage()
    
    color_image = np.frombuffer(color_image, np.uint8)
    color_image = color_image.reshape(height, width, 4)
    color_image = color_image[:, :, :3]
    
    depth_image = np.array(depth_image).reshape(height, width)

    result_image, object_mask, detections = image_tools.process_and_extract_objects(color_image, depth_image, o3d_intrinsics, result_image=False)
    SAM_view_display.setColor(0xFFFFFF)
    if object_mask.any():
        for x in range(height):
            for y in range(width):
                if object_mask[x, y]:
                    SAM_view_display.drawPixel(y, x)
        return object_mask, depth_image, True
    else:
        return object_mask, depth_image, False
    
def convert_camera_coord_to_robot_coord(obj_pos):
    """Converts object coordinates from camera frame to robot base frame."""
    global camera_chain, robot_sensors, part_names

    if camera_chain is None:
        print("Error: Camera chain is not initialized.")
        return None

    try:
        # 1. Get current head joint angles
        head_1_angle = robot_sensors["head_1_joint"].getValue()
        head_2_angle = robot_sensors["head_2_joint"].getValue()

        # 2. Construct the joint state for the camera chain
        #    Indices depend on the structure defined in camera_base_elements + camera_link_name
        #    Need to map 'head_1_joint', 'head_2_joint' to their positions in camera_chain.links
        num_cam_chain_joints = len(camera_chain.links)
        current_joint_state = np.zeros(num_cam_chain_joints)

        # Find indices for head joints within the camera_chain
        head_1_idx = -1
        head_2_idx = -1
        for i, link in enumerate(camera_chain.links):
            if link.name == "head_1_joint":
                head_1_idx = i
            elif link.name == "head_2_joint":
                head_2_idx = i

        if head_1_idx != -1:
            current_joint_state[head_1_idx] = head_1_angle
        else:
            print("Warning: head_1_joint not found in camera_chain")
        if head_2_idx != -1:
            current_joint_state[head_2_idx] = head_2_angle
        else:
            print("Warning: head_2_joint not found in camera_chain")

        # 3. Calculate forward kinematics for the camera chain
        base_T_camera = camera_chain.forward_kinematics(current_joint_state)

        # 4. Convert object position to homogeneous coordinates
        obj_pos_camera_homogeneous = np.append(obj_pos, 1)

        # 5. Transform position to robot base frame
        obj_pos_robot_homogeneous = base_T_camera @ obj_pos_camera_homogeneous

        # 6. Extract Cartesian coordinates
        obj_pos_robot = obj_pos_robot_homogeneous[:3]

        # print(f"  Head Angles: [{head_1_angle:.3f}, {head_2_angle:.3f}]")
        # print(f"  Cam Chain State: {current_joint_state}")
        # print(f"  Base->Cam Transform:\n{base_T_camera}")
        # print(f"  Obj Pos (Cam Homo): {obj_pos_camera_homogeneous}")
        # print(f"  Obj Pos (Robot Homo): {obj_pos_robot_homogeneous}")
        # print(f"  Obj Pos (Robot): {obj_pos_robot}")

        return obj_pos_robot

    except KeyError as e:
        print(f"Error: Joint sensor key not found: {e}. Ensure head sensors are in robot_sensors.")
        return None
    except Exception as e:
        print(f"Error during coordinate transformation: {e}")
        return None

def joint_tester(tolerance=0.005, max_wait_steps=250):
    """
    Cycles each position-controlled joint through min -> max -> initial positions,
    waiting for completion using Position Sensors.
    Note: This function blocks execution while testing each joint.
    Requires Position Sensors to be enabled for all tested joints.

    Args:
        tolerance (float): Position tolerance in radians/meters to consider the target reached.
        max_wait_steps (int): Maximum simulation steps to wait before timing out.
    """
    print("=== Starting Joint Test (Sensor-Based Wait) ===")
    robot_parts["wheel_left_joint"].setVelocity(0)
    robot_parts["wheel_right_joint"].setVelocity(0)

    # Filter part_names to only include those with sensors (i.e., not wheels)
    testable_joints = [name for name in part_names if name in robot_sensors]

    if not testable_joints:
        print("No testable joints with enabled sensors found. Aborting test.")
        return

    initial_positions = {}
    # Retrieve initial positions only for testable joints
    for name in testable_joints:
        # Find the corresponding index in the original target_pos tuple
        try:
            original_index = part_names.index(name)
            initial_positions[name] = float(target_pos[original_index])
        except (ValueError, IndexError):
            print(f"Warning: Could not find initial position for {name} in global target_pos. Using 0.0 as fallback.")
            initial_positions[name] = 0.0 # Fallback, adjust if needed


    # --- Helper function for waiting ---
    def wait_for_position(robot_joint, sensor, target_position, target_name, part_name, speed_factor):
        print(f"  Moving {part_name} to {target_name} ({target_position:.3f})...")
        robot_joint.setPosition(target_position)
        robot_joint.setVelocity(robot_joint.getMaxVelocity() * speed_factor)
        steps_waited = 0
        while steps_waited < max_wait_steps:
            if robot.step(timestep) == -1:
                print("  Simulation stopped during wait.")
                return False # Indicate simulation stopped

            current_pos = sensor.getValue()
            if abs(current_pos - target_position) < tolerance:
                print(f"  Reached {target_name} position ({current_pos:.3f}) after {steps_waited} steps.")
                return True # Indicate success

            steps_waited += 1

        # Loop finished without reaching target (timeout)
        final_pos = sensor.getValue()
        print(f"  Warning: Timeout waiting for {part_name} to reach {target_name}. Steps: {steps_waited}, Current pos: {final_pos:.3f}")
        return True # Indicate timeout occurred but continue test
    # --- End Helper function ---


    for part_name in testable_joints:
        robot_joint = robot_parts[part_name]
        sensor = robot_sensors[part_name] # Assumes sensor exists from the filter above
        initial_pos = initial_positions[part_name]

        if not (hasattr(robot_joint, 'getMinPosition') and hasattr(robot_joint, 'getMaxPosition')):
            print(f"Skipping joint {part_name} (does not have getMinPosition/getMaxPosition methods).")
            continue

        min_pos = robot_joint.getMinPosition()
        max_pos = robot_joint.getMaxPosition()

        # Handle potentially infinite limits reported by Webots
        if min_pos < -1e10 or max_pos > 1e10:
            print(f"Skipping joint {part_name} due to potentially infinite limits ({min_pos}, {max_pos}).")
            continue

        print(f"\nTesting joint: {part_name} (Min: {min_pos:.3f}, Max: {max_pos:.3f}, Initial: {initial_pos:.3f})")

        # Move to Min
        if not wait_for_position(robot_joint, sensor, min_pos, "Min", part_name, 0.5): return # Exit if simulation stopped

        # Move to Max
        if not wait_for_position(robot_joint, sensor, max_pos, "Max", part_name, 0.5): return # Exit if simulation stopped

        # Return to Initial Position
        if not wait_for_position(robot_joint, sensor, initial_pos, "Initial", part_name, 0.5): return # Exit if simulation stopped

        print(f"Finished testing {part_name}.")

    print("\n=== Joint Test Complete ===")

def get_robot_joints():
    global part_names, robot_sensors
    current_pose_list = []
    for name in part_names:
        if "wheel" in name:
            current_pose_list.append("'inf'") # Add as string to match target_pos format
        elif name == "gripper_left_finger_joint":
            current_pose_list.append(f"{left_gripper_enc.getValue():.3f}")
        elif name == "gripper_right_finger_joint":
                current_pose_list.append(f"{right_gripper_enc.getValue():.3f}")
        elif name in robot_sensors:
            current_pose_list.append(f"{robot_sensors[name].getValue():.3f}")
            
    pose_str = "(" + ", ".join(current_pose_list) + ")"
    print(f"Current Pose: {pose_str}")

# --- IK Helper Functions ---

def get_current_ik_joint_state(chain, motors, sensor_bound_tolerance=0.01):
    """
    Retrieves the current joint positions for the active links in the IK chain.

    Reads sensor values, validates them, and clamps them within the URDF bounds
    to provide a valid initial guess for IK calculations.

    Args:
        chain: The ikpy Chain object representing the robot arm.
        motors: A dictionary mapping joint names to motor devices.
        sensor_bound_tolerance (float): Tolerance for checking if sensor values
                                         are significantly outside bounds.

    Returns:
        np.ndarray or None: A numpy array of the current joint angles for the chain,
                            or None if reading/validation fails.
    """
    num_links = len(chain.links)
    current_joint_state = np.zeros(num_links)
    valid_state = True

    for i in range(num_links):
        if chain.active_links_mask[i]:
            link = chain.links[i]
            link_name = link.name
            if link_name in motors:
                try:
                    sensor = motors[link_name].getPositionSensor()
                    if sensor is None:
                        print(f"!!! WARNING: No position sensor found for active motor '{link_name}' in motors dict. Skipping.")
                        # Decide if this should be a critical error (valid_state = False; break)
                        # or just skip this joint for the initial guess.
                        # For now, we skip, but the IK might be less accurate.
                        continue

                    sensor_value = sensor.getValue()

                    if not np.isfinite(sensor_value):
                        print(f"!!! ERROR: Non-finite sensor value ({sensor_value}) read for {link_name}. Cannot get current state.")
                        valid_state = False
                        break

                    lower_bound, upper_bound = link.bounds

                    if (lower_bound is not None and sensor_value < lower_bound - sensor_bound_tolerance) or \
                       (upper_bound is not None and sensor_value > upper_bound + sensor_bound_tolerance):
                        print(f"!!! WARNING: Sensor value {sensor_value:.4f} for {link_name} is significantly outside URDF bounds [{lower_bound}, {upper_bound}]. Clamping.")

                    if lower_bound is not None and upper_bound is not None:
                        current_joint_state[i] = np.clip(sensor_value, lower_bound, upper_bound)
                    elif lower_bound is not None:
                        current_joint_state[i] = max(sensor_value, lower_bound)
                    elif upper_bound is not None:
                        current_joint_state[i] = min(sensor_value, upper_bound)
                    else:
                        current_joint_state[i] = sensor_value

                except Exception as e:
                    print(f"Error reading sensor or getting bounds for {link_name}: {e}. Cannot get current state.")
                    valid_state = False
                    break
            # else: # Optional: Handle case where an active link doesn't have a motor in the dict
            #     print(f"Warning: Active link {link_name} not found in motor_dict.")

    if not valid_state:
        return None
    else:
        return current_joint_state

def rotation_matrix(axis, angle):
    """Creates a 3x3 rotation matrix for a given axis ('x', 'y', 'z') and angle (radians)."""
    c = np.cos(angle)
    s = np.sin(angle)
    # Assuming standard rotation matrices (adjust if your coordinate system differs)
    # Positive angle = counter-clockwise rotation looking from positive axis towards origin
    if axis == 'x': # Roll
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y': # Pitch
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z': # Yaw
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        # Return identity if axis is invalid, or raise error
        print(f"Warning: Invalid rotation axis '{axis}'. Returning identity matrix.")
        return np.identity(3)

def check_arm_at_position(ikResults, cutoff=0.01):
    """Checks if arm is close to the target position defined by ikResults."""
    global my_chain, motor_dict # Use the global motor_dict
    arm_error = 0
    count = 0
    for i in range(len(ikResults)):
        # Only check active links that have a corresponding motor
        if my_chain.active_links_mask[i]:
            link_name = my_chain.links[i].name
            if link_name in motor_dict:
                current_pos = motor_dict[link_name].getPositionSensor().getValue()
                arm_error += (current_pos - ikResults[i])**2
                count += 1

    if count > 0:
        arm_error = math.sqrt(arm_error / count)

    # Optional: print("Current arm error:", arm_error)
    return arm_error < cutoff

def move_arm_to_target(ikResults):
    """Commands the arm motors to the positions specified in ikResults."""
    global my_chain, motor_dict # Use the global motor_dict
    # Set the robot motors for active links only
    for i in range(len(ikResults)):
        if my_chain.active_links_mask[i]:
            link_name = my_chain.links[i].name
            if link_name in motor_dict:
                motor_dict[link_name].setPosition(ikResults[i])

def calculate_ik(target_position, initial_position, orient=False, orientation_mode="Y", target_orientation=None):
    """Calculates IK for a target position, using provided initial joint angles."""
    global my_chain # Use the global my_chain
    print(f"Calculating IK for target: {target_position}")

    try:
        # Calculate IK using the provided initial position
        ikResults = my_chain.inverse_kinematics(
            target_position,
            initial_position=initial_position, # Use the passed-in initial position
            target_orientation=target_orientation if orient else None,
            orientation_mode=orientation_mode if orient else None
        )

        # Optional: Validate result distance (can be computationally expensive)
        # final_fk = my_chain.forward_kinematics(ikResults)
        # squared_distance = np.linalg.norm(final_fk[:3, 3] - target_position)
        # print(f"IK solved. Target: {target_position}, Achieved: {final_fk[:3, 3]}, Distance: {squared_distance:.4f}")

        # Simple check: Ensure result is not wildly different if needed, or rely on ikpy exceptions
        # Note: ikpy usually handles basic joint limits based on URDF if defined there.
        # Add explicit limit enforcement here if URDF limits are unreliable or missing.
        return ikResults
    except ValueError as e:
        print(f"IK calculation failed: {e}")
        # Return None on failure (e.g., target unreachable)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during IK calculation: {e}")
        return None

# --- End IK Helper Functions ---

def wait_for_arm_movement(ik_results, robot, timestep, max_wait_steps=100, cutoff=0.02, description="position"):
    """Helper function to wait for arm to reach a target position.
    
    Args:
        ik_results: The IK solution to check against
        robot: The robot instance to step the simulation
        timestep: Simulation timestep
        max_wait_steps: Maximum steps to wait
        cutoff: Position tolerance
        description: Description of the movement for logging
        
    Returns:
        True if position reached, False if timeout
    """
    print(f"  Waiting for arm to reach {description}...")
    wait_steps = 0
    while wait_steps < max_wait_steps:
        robot.step(timestep)
        if check_arm_at_position(ik_results, cutoff=cutoff):
            print(f"  Arm reached {description}.")
            return True
        wait_steps += 1
    
    print(f"  Timeout waiting for arm to reach {description}.")
    return False

def move_arm_with_ik(target_position, initial_position, orientation=None, robot=None, timestep=None, 
                    orientation_mode="all", max_wait=100, cutoff=0.02, description="position", 
                    must_succeed=True):
    """Helper function to calculate IK, move arm, and wait for completion.
    
    Args:
        target_position: Target position for end effector
        initial_position: Initial joint state
        orientation: Target orientation matrix (if None, orientation not controlled)
        robot: Robot instance for simulation stepping
        timestep: Simulation timestep
        orientation_mode: Orientation control mode
        max_wait: Maximum wait steps
        cutoff: Position tolerance
        description: Description for logging
        must_succeed: If True, return False on failure
        
    Returns:
        (success, ik_results) tuple: success is True if move succeeded, 
                                    ik_results is the calculated IK solution or None
    """
    # Calculate IK
    ik_results = calculate_ik(
        target_position,
        initial_position=initial_position,
        orient=(orientation is not None),
        orientation_mode=orientation_mode if orientation is not None else None,
        target_orientation=orientation
    )
    
    if ik_results is None:
        print(f"  IK failed for {description}.")
        if must_succeed:
            return False, None
        else:
            return False, None
    
    # Move arm
    move_arm_to_target(ik_results)
    
    # Wait for completion if robot and timestep are provided
    if robot is not None and timestep is not None:
        success = wait_for_arm_movement(
            ik_results, robot, timestep, 
            max_wait_steps=max_wait, 
            cutoff=cutoff, 
            description=description
        )
        if not success and must_succeed:
            return False, ik_results
    
    return True, ik_results

def approach_and_grasp_object(object_mask, depth_image, o3d_intrinsics, my_chain, motor_dict, robot, timestep):
    """Calculates object position, moves arm nearby, then approaches for grasp."""
    # Need access for coordinate conversion, IK state, gripper, base motors, and constants
    global part_names, robot_sensors, gripper_status, right_gripper_enc, robot_parts, MAX_SPEED_MS, MAX_SPEED

    original_velocities = {} # Store original velocities to potentially restore later
    print("Attempting to approach and grasp detected object...")
    
    # --- Step 1: Get object position in robot frame ---
    obj_pos = image_tools.get_object_coord(object_mask, depth_image, o3d_intrinsics)
    if obj_pos is None or not np.all(np.isfinite(obj_pos)):
        print("Failed to get valid object coordinates.")
        return False
    print(f"Object Position (Camera Frame): {obj_pos}")

    robot_pos = convert_camera_coord_to_robot_coord(obj_pos)
    if robot_pos is None or not np.all(np.isfinite(robot_pos)):
        print("Failed to convert object coordinates to robot frame.")
        return False
    print(f"Object Position (Robot Frame): {robot_pos}")

    # Apply Z correction for head tilt
    if robot_pos is not None and np.all(np.isfinite(robot_pos)):
        try:
            current_tilt_angle = robot_sensors["head_2_joint"].getValue()
            Z_CORRECTION_FACTOR_PER_RADIAN = 0.12
            if current_tilt_angle != 0.0:
                z_correction = Z_CORRECTION_FACTOR_PER_RADIAN * current_tilt_angle
                # original_z = robot_pos[2]
                robot_pos[2] += z_correction
                print(f"  Head Tilt: {current_tilt_angle:.3f} rad, Z Correction: {z_correction:.4f}")
                print(f"  Position with correction: {robot_pos}")
        except Exception as e:
            print(f"Warning: Z correction failed: {e}")
    
    # Check reachability
    max_reach_distance = 1.3 # meters
    if np.linalg.norm(robot_pos) >= max_reach_distance:
        print(f"Object is too far ({np.linalg.norm(robot_pos):.2f}m > {max_reach_distance}m). Cannot reach.")
        return False

    # --- Step 2: Prepare arm movement ---
    approach_offset = 0.2 # meters back from the object for initial alignment
    print("Object is within reach. Calculating initial arm position...")
    
    # Get current arm state
    initial_position = get_current_ik_joint_state(my_chain, motor_dict)
    if initial_position is None:
        print("Failed to get current arm joint state.")
        return False

    # Get current end-effector position
    current_fk = my_chain.forward_kinematics(initial_position)
    current_pos_ee = current_fk[:3, 3]
    print(f"  Current end effector position: {current_pos_ee}")
    
    # Define orientations
    # Orientation for alignment phase
    alignment_orientation = np.array([
        [ 0.099267,  0.045,    -0.11156 ],
        [-0.11407,   0.05531,   -0.99193],
        [ 0.04000,   0.99800,   -0.06210]
    ])
    
    # Final orientation for grasp
    target_orientation_matrix = np.array([
        [ 0.091119,  0.044689, -0.99484 ],
        [-0.99582,   0.010419, -0.090741],
        [ 0.0063098, 0.99895,   0.045452]
    ])
    
    # --- Step 3: Retract arm if needed ---
    if current_pos_ee[0] > 0.35:
        print(f"  Arm extended past safety threshold (x={current_pos_ee[0]:.3f} > 0.35). Retracting first...")
        retract_pos = np.array([0.35, current_pos_ee[1], current_pos_ee[2]])
        
        # Try retraction with primary and fallback positions
        success, ik_results_retract = move_arm_with_ik(
            retract_pos, 
            initial_position, 
            orientation=current_fk[:3, :3],
            robot=robot, 
            timestep=timestep,
            description="retraction position", 
            must_succeed=False
        )
        
        if not success:
            print("  Trying alternative retraction...")
            retract_pos = np.array([0.35, current_pos_ee[1] * 0.8, current_pos_ee[2]])
            success, ik_results_retract = move_arm_with_ik(
                retract_pos, 
                initial_position, 
                orientation=current_fk[:3, :3],
                robot=robot, 
                timestep=timestep,
                description="alternative retraction position"
            )
            
            if not success:
                print("  All retraction attempts failed. Cannot proceed safely.")
                return False
        
        # Update position after retraction
        initial_position = get_current_ik_joint_state(my_chain, motor_dict)
        if initial_position is None:
            print("  Failed to get updated joint state after retraction.")
            return False
        current_fk = my_chain.forward_kinematics(initial_position)
        current_pos_ee = current_fk[:3, 3]
    
    # --- Step 4: Move to alignment position using checkpoints ---
    # Calculate Z-alignment position
    intermediate_pos_z_aligned = np.array([0.30, robot_pos[1], robot_pos[2] + 0.15])
    print(f"  Target Z-alignment position: {intermediate_pos_z_aligned}")
    
    # Create checkpoints for smooth movement
    num_checkpoints = 4
    checkpoints = []
    for i in range(1, num_checkpoints + 1):
        fraction = i / num_checkpoints
        checkpoint = current_pos_ee * (1 - fraction) + intermediate_pos_z_aligned * fraction
        checkpoint[0] = min(checkpoint[0], 0.35)  # Ensure x doesn't exceed 0.35
        checkpoints.append(checkpoint)
    
    # Move through checkpoints
    current_checkpoint_pos = initial_position
    for idx, checkpoint in enumerate(checkpoints):
        print(f"  Moving to checkpoint {idx+1}/{len(checkpoints)}: {checkpoint}")
        
        success, ik_results = move_arm_with_ik(
            checkpoint,
            current_checkpoint_pos,
            orientation=alignment_orientation,
            robot=robot,
            timestep=timestep,
            description=f"checkpoint {idx+1}",
            max_wait=80,
            cutoff=0.025,
            must_succeed=False
        )
        
        if success:
            current_checkpoint_pos = get_current_ik_joint_state(my_chain, motor_dict)
            if current_checkpoint_pos is None:
                print(f"  Failed to get updated joint state after checkpoint {idx+1}.")
                return False
        # If checkpoint fails, continue to next one
    
    # Final alignment position
    print("  Moving to final Z-alignment position...")
    success, ik_results_z_align = move_arm_with_ik(
        intermediate_pos_z_aligned,
        current_checkpoint_pos,
        orientation=alignment_orientation,
        robot=robot,
        timestep=timestep,
        description="final Z-aligned position"
    )
    
    if not success:
        return False
    
    # --- Step 5: Orient arm for grabbing ---
    print("Orienting arm to ready for item grabbing...")
    success, ik_results_orient = move_arm_with_ik(
        intermediate_pos_z_aligned,
        ik_results_z_align,
        orientation=target_orientation_matrix,
        robot=robot,
        timestep=timestep,
        description="orientation position"
    )
    
    if not success:
        return False
    
    # --- Step 6: Move to approach position (offset from object) ---
    state_after_z_align = get_current_ik_joint_state(my_chain, motor_dict)
    if state_after_z_align is None:
        print("Failed to get arm joint state after Z alignment.")
        return False

    # Get current end-effector position
    current_fk_approach = my_chain.forward_kinematics(state_after_z_align)
    current_pos_approach = current_fk_approach[:3, 3]
    print(f"  Current position before approach: {current_pos_approach}")

    # Apply offset for initial approach
    robot_pos_approach = robot_pos.copy()
    robot_pos_approach[0] -= approach_offset  # Stay back from object
    
    print(f"Stage 2: Moving arm to approach position: {robot_pos_approach}")
    
    # Create checkpoints for smooth approach movement
    num_approach_checkpoints = 4
    approach_checkpoints = []
    for i in range(1, num_approach_checkpoints + 1):
        fraction = i / num_approach_checkpoints
        checkpoint = current_pos_approach * (1 - fraction) + robot_pos_approach * fraction
        # Ensure Z doesn't dip below target
        checkpoint[2] = max(checkpoint[2], robot_pos_approach[2])
        approach_checkpoints.append(checkpoint)
    
    # Move through approach checkpoints
    current_checkpoint_pos = state_after_z_align
    for idx, checkpoint in enumerate(approach_checkpoints):
        print(f"  Moving to approach checkpoint {idx+1}/{len(approach_checkpoints)}: {checkpoint}")
        
        success, ik_results = move_arm_with_ik(
            checkpoint,
            current_checkpoint_pos,
            orientation=target_orientation_matrix,
            robot=robot,
            timestep=timestep,
            description=f"approach checkpoint {idx+1}",
            max_wait=80,
            cutoff=0.025,
            must_succeed=False
        )
        
        if success:
            current_checkpoint_pos = get_current_ik_joint_state(my_chain, motor_dict)
            if current_checkpoint_pos is None:
                print(f"  Failed to get updated joint state after approach checkpoint {idx+1}.")
                return False
    
    # Final approach position
    success, ik_results_approach = move_arm_with_ik(
        robot_pos_approach,
        current_checkpoint_pos,
        orientation=target_orientation_matrix,
        robot=robot,
        timestep=timestep,
        description="final approach position"
    )
    
    if not success:
        return False
        
    # --- Step 7: Final approach in small steps ---
    print(f"Stage 3: Performing final approach movement ({approach_offset}m)...")
    steps = 10
    step_size = approach_offset / steps
    
    for step in range(1, steps + 1):
        current_position_step = get_current_ik_joint_state(my_chain, motor_dict)
        if current_position_step is None:
            print("Failed to get current joint state during final approach step.")
            return False
        
        # Calculate incremental target
        step_target = robot_pos_approach.copy()
        step_target[0] += step * step_size
        
        # Move arm incrementally
        success, _ = move_arm_with_ik(
            step_target,
            current_position_step,
            orientation=target_orientation_matrix,
            robot=robot,
            timestep=timestep,
            description=f"approach step {step}/{steps}",
            max_wait=10,
            cutoff=0.03,
            must_succeed=True
        )
        
        if not success:
            print(f"IK failed during step {step} of final approach.")
            return False
    
    # Verify final position
    final_pos_check = get_current_ik_joint_state(my_chain, motor_dict)
    if final_pos_check is not None:
        final_fk = my_chain.forward_kinematics(final_pos_check)
        final_actual_pos = final_fk[:3, 3]
        final_dist = np.linalg.norm(final_actual_pos - robot_pos)
        print(f"Final approach complete. Distance: {final_dist:.3f}m")
        if final_dist > 0.1:
            print("End effector did not reach target position within tolerance.")
            return False
    else:
        print("Could not verify final arm position.")
    
    # --- Step 8: Close gripper ---
    print("Commanding gripper to close...")
    try:
        # Close gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.0)
        robot_parts["gripper_right_finger_joint"].setPosition(0.0)
        
        # Wait for gripper
        wait_duration_s = 2.0
        num_wait_steps = int(wait_duration_s * 1000 / timestep)
        for _ in range(num_wait_steps):
            if robot.step(timestep) == -1:
                print("Simulation stopped during gripper close wait.")
                return False
        
        gripper_status = "closed"
        print("  Gripper closed.")
    except KeyError as e:
        print(f"Error accessing gripper motor: {e}. Cannot close gripper.")
        return False
    
    # --- Step 9: Lift object ---
    print("Lifting object slightly...")
    current_state_before_lift = get_current_ik_joint_state(my_chain, motor_dict)
    if current_state_before_lift is None:
        print("Failed to get joint state before lifting.")
        return False
    
    try:
        # Calculate lift position
        current_fk_before_lift = my_chain.forward_kinematics(current_state_before_lift)
        current_pos_before_lift = current_fk_before_lift[:3, 3]
        current_orientation_before_lift = current_fk_before_lift[:3, :3]
        lift_target_pos = current_pos_before_lift + np.array([0.0, 0.0, 0.15])  # 15cm lift
        
        # Slow down arm for lift
        print("  Slowing down arm motors for lifting...")
        slow_down_factor = 0.2
        for i in range(len(my_chain.links)):
            if my_chain.active_links_mask[i]:
                link_name = my_chain.links[i].name
                if link_name in motor_dict and "wheel" not in link_name:
                    motor = motor_dict[link_name]
                    original_velocities[link_name] = motor.getVelocity()
                    motor.setVelocity(motor.getMaxVelocity() * slow_down_factor)
        
        # Move arm up
        success, _ = move_arm_with_ik(
            lift_target_pos,
            current_state_before_lift,
            orientation=current_orientation_before_lift,
            robot=robot,
            timestep=timestep,
            description="lift position",
            must_succeed=False
        )
        
        if not success:
            print("  Lift failed, but continuing with sequence.")
            
    except Exception as e:
        print(f"Error during lift: {e}")
        # Continue even if lift fails
    
    print("Approach and lift sequence completed.")
    
    # --- Step 10: Reverse robot base ---
    print("Reversing robot base...")
    reverse_distance = 0.5
    reverse_speed_factor = 0.25
    reverse_speed_ms = MAX_SPEED_MS * reverse_speed_factor
    
    if reverse_speed_ms > 0:
        duration_s = reverse_distance / reverse_speed_ms
        num_steps = int(duration_s * 1000 / timestep)
        reverse_velocity = -MAX_SPEED * reverse_speed_factor
        
        # Execute reverse movement
        robot_parts["wheel_left_joint"].setVelocity(reverse_velocity)
        robot_parts["wheel_right_joint"].setVelocity(reverse_velocity)
        
        for _ in range(num_steps):
            if robot.step(timestep) == -1:
                robot_parts["wheel_left_joint"].setVelocity(0)
                robot_parts["wheel_right_joint"].setVelocity(0)
                print("Simulation stopped during reverse.")
                return False
        
        # Stop wheels
        robot_parts["wheel_left_joint"].setVelocity(0)
        robot_parts["wheel_right_joint"].setVelocity(0)
    
    # --- Step 11: Move to final pose and open gripper ---
    print("Moving arm to final pose and releasing object...")
    final_target_pos = np.array([0.28548, 0.0, 0.33344])
    final_target_orient = np.array([
        [ 0.99527,  0.09564, -0.017219],
        [-0.01995,  0.027676, -0.99942 ],
        [-0.095108,  0.99503,  0.029453]
    ])
    
    # Get current arm state
    current_arm_pos = get_current_ik_joint_state(my_chain, motor_dict)
    if current_arm_pos is not None:
        # Move to final position
        success, _ = move_arm_with_ik(
            final_target_pos,
            current_arm_pos,
            orientation=final_target_orient,
            robot=robot,
            timestep=timestep,
            description="final pose",
            max_wait=300,
            must_succeed=False
        )
    
    # Open gripper regardless of arm movement success
    try:
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        
        # Wait for gripper to open
        open_wait_duration_s = 1.0
        num_open_wait_steps = int(open_wait_duration_s * 1000 / timestep)
        for _ in range(num_open_wait_steps):
            if robot.step(timestep) == -1:
                print("Simulation stopped during gripper open wait.")
                return False
        
        gripper_status = "open"
        print("  Gripper opened.")
    except KeyError as e:
        print(f"Error accessing gripper motor: {e}. Cannot open gripper.")
    
    # --- Restore arm speeds ---
    print("  Restoring original arm motor velocities...")
    for link_name, original_velocity in original_velocities.items():
        if link_name in motor_dict:
            motor_dict[link_name].setVelocity(original_velocity)
    
    print("Grasp sequence completed successfully.")
    return True

def arm_controller(key):
    """Handles keyboard input for moving the end effector in Cartesian space."""
    global my_chain, motor_dict, robot_sensors, CARTESIAN_STEP, ORIENTATION_STEP, END_EFFECTOR_CONTROL_KEYS

    key_char = chr(key).upper() # Use upper case for matching dictionary keys

    # 1. Get Current Joint Positions using the helper function
    initial_position = get_current_ik_joint_state(my_chain, motor_dict)

    if initial_position is None:
        print("Failed to get valid current joint state. Skipping arm control step.")
        return # Exit if initial positions couldn't be read reliably

    # 2. Forward Kinematics to find current end-effector pose
    try:
        current_fk = my_chain.forward_kinematics(initial_position) # Use state from helper function
        if not np.all(np.isfinite(current_fk)):
            # Enhanced logging for FK failure
            print("!!! ERROR: Non-finite values in FK result. Skipping step.")
            print(f"  Initial Position used: {initial_position}") # Log the state from helper
            return
        current_pos = current_fk[:3, 3]
        current_orientation_matrix = current_fk[:3, :3]
        print(f"Current Orientation Matrix:\n{current_orientation_matrix}")
        if not np.all(np.isfinite(current_pos)) or not np.all(np.isfinite(current_orientation_matrix)):
            print("!!! WARNING: Non-finite values in extracted pose after FK. Skipping step.")
            return

    except Exception as e:
        print(f"Forward kinematics calculation failed: {e}")
        print(f"  Initial Position used: {initial_position}") # Log the state from helper
        return

    # 3. Determine Desired Change (Position or Orientation)
    delta_pos = np.array([0.0, 0.0, 0.0])
    delta_orientation_matrix = np.identity(3) # Start with no orientation change
    position_change_requested = False
    orientation_change_requested = False
    action_key_pressed = True # Flag to check if any valid action key was pressed (assume true)

    # Check for Position Keys (Base frame: +X Forward, +Y Left, +Z Up)
    if key_char == END_EFFECTOR_CONTROL_KEYS["forward"]:
        delta_pos[0] = CARTESIAN_STEP
        position_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["backward"]:
        delta_pos[0] = -CARTESIAN_STEP
        position_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["left"]: # Move Left
        delta_pos[1] = CARTESIAN_STEP
        position_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["right"]: # Move Right
        delta_pos[1] = -CARTESIAN_STEP
        position_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["up"]:
        delta_pos[2] = CARTESIAN_STEP
        position_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["down"]:
        delta_pos[2] = -CARTESIAN_STEP
        position_change_requested = True

    # Check for Orientation Keys (End-effector frame rotations)
    elif key_char == END_EFFECTOR_CONTROL_KEYS["pitch_up"]: # Pitch Up (around local Y)
        delta_orientation_matrix = rotation_matrix('y', ORIENTATION_STEP)
        orientation_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["pitch_down"]: # Pitch Down (around local Y)
        delta_orientation_matrix = rotation_matrix('y', -ORIENTATION_STEP)
        orientation_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["roll_left"]: # Roll Left (around local X)
        delta_orientation_matrix = rotation_matrix('x', ORIENTATION_STEP)
        orientation_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["roll_right"]: # Roll Right (around local X)
        delta_orientation_matrix = rotation_matrix('x', -ORIENTATION_STEP)
        orientation_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["yaw_left"]: # Yaw Left (around local Z)
        delta_orientation_matrix = rotation_matrix('z', ORIENTATION_STEP)
        orientation_change_requested = True
    elif key_char == END_EFFECTOR_CONTROL_KEYS["yaw_right"]: # Yaw Right (around local Z)
        delta_orientation_matrix = rotation_matrix('z', -ORIENTATION_STEP)
        orientation_change_requested = True
    else:
        action_key_pressed = False
    print(f"delta_pos: {delta_pos}")
    # Exit if no relevant arm control key was pressed
    if not action_key_pressed:
        return

    # 4. Calculate Target Pose
    target_position = current_pos + delta_pos
    # Apply delta rotation relative to the current orientation
    target_orientation = np.dot(current_orientation_matrix, delta_orientation_matrix)

    # --- Add Check for target validity ---
    if not np.all(np.isfinite(target_position)) or not np.all(np.isfinite(target_orientation)):
        print("!!! WARNING: Non-finite values in calculated target pose. Skipping step.")
        return

    # 5. Calculate Target Joint Angles (IK)
    # manually set orientation
    orientation_change_requested = True
    print(f"Attempting IK: TargetPos={target_position}, OrientChange={orientation_change_requested}, OrientMode={'all' if orientation_change_requested else None}")
    # Pass the initial_position obtained from the helper function to IK
    ik_results = calculate_ik(
        target_position,
        initial_position=initial_position, # Use the state from the helper function
        orient=orientation_change_requested, # Pass True only if orientation key was pressed
        orientation_mode="all" if orientation_change_requested else None,
        target_orientation=target_orientation if orientation_change_requested else None # Pass target matrix only if needed
    )

    # 6. Command Joints
    if ik_results is not None:
        # Optional: Add check for non-finite values in ik_results before commanding
        if np.all(np.isfinite(ik_results)):
            move_arm_to_target(ik_results)
        else:
            print("!!! WARNING: Non-finite values detected in IK results. Not commanding motors.")
    else:
        # Enhanced logging for IK failure
        print("IK solver failed to find a solution for the target.")
        print(f"  Failed Target Position: {target_position}")
        if orientation_change_requested:
            print(f"  Failed Target Orientation:\n{target_orientation}")
        print(f"  Initial Guess used: {initial_position}") # Log the state from the helper function

# Main Loop
joint_test = False
steps_taken = 0
robot_pos = None
obj_pos = None
while robot.step(timestep) != -1:


    steps_taken += 1
    if steps_taken > 10:
        object_mask, depth_image, detected = draw_detected_objects()
        if detected:
            # if obj_pos is None:
            obj_pos = image_tools.get_object_coord(object_mask, depth_image, o3d_intrinsics)
            print(f"Object Position: {obj_pos}")
            robot_pos = convert_camera_coord_to_robot_coord(obj_pos)
            print(f"Robot Relative Position: {robot_pos}")
        steps_taken = 0
            
    if joint_test: 
        joint_tester()
        joint_test = False

    # --- Keyboard Input with Cooldown ---
    key_cooldown_timer = max(0, key_cooldown_timer - 1) # Decrement cooldown timer
    # bypass cooldown when driving
    if robot_mode == "drive" and not mode_just_changed: key_cooldown_timer = 0
    input_key = keyboard.getKey()
    
    key = -1 # Key that will trigger an action this cycle
    if input_key > -1 and key_cooldown_timer == 0:
        key = input_key
        key_cooldown_timer = KEY_COOLDOWN_CYCLES # Reset cooldown
        mode_just_changed = False

    # --- Mode Switching ---
    if key == ord(ROBOT_MODE_TOGGLE_KEY):
        if robot_mode == "drive":
            robot_mode = "arm"
            # Stop wheels when switching to arm mode
            robot_parts["wheel_left_joint"].setVelocity(0)
            robot_parts["wheel_right_joint"].setVelocity(0)
        else:
            robot_mode = "drive"
            # TODO: Reset arm to a default pose when switching to drive?
        print(f"Switched to '{robot_mode}' mode.")
        mode_just_changed = True
        key = -1 # Consume the mode switch key so it doesn't trigger actions

    # --- Mode-Specific Control ---
    if robot_mode == "drive":
        # Reset drive velocities at the start of drive mode loop iteration
        vL = 0
        vR = 0

        # Handle keyboard input for movement
        if key == ord('W'):  # Forward
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == ord('S'):  # Backward
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord('A'):  # Turn Left (Base movement, not arm)
            vL = -MAX_SPEED / 2
            vR = MAX_SPEED / 2
        elif key == ord('D'):  # Turn Right (Base movement, not arm)
            vL = MAX_SPEED / 2
            vR = -MAX_SPEED / 2
        # Handle other keyboard input in drive mode
        elif key == ord('C'):
            handle_capture()
        elif key == ord('P'):

            object_mask, depth_image, detected = draw_detected_objects()
            steps_taken = 0
            
            # Check if the robot is too close to any object
            if depth_image is not None and np.min(depth_image) < 0.38:
                print("The robot is too close to an object.")
            
            if detected:
                # Call the new function to handle approach and grasp
                success = approach_and_grasp_object(
                    object_mask, depth_image, o3d_intrinsics,
                    my_chain, motor_dict, robot, timestep
                )
                if success:
                    print("Full approach, grasp, and post-grasp sequence successful.")
                else:
                    print("Object grasp attempt failed.")
            else:
                robot_pos = None
                print("No object detected to approach.")

        elif key == keyboard.UP: # Head tilt up
            current_head_tilt = min(HEAD_TILT_MAX, current_head_tilt + HEAD_TILT_STEP)
            robot_parts["head_2_joint"].setPosition(current_head_tilt)
        elif key == keyboard.DOWN: # Head tilt down
            current_head_tilt = max(HEAD_TILT_MIN, current_head_tilt - HEAD_TILT_STEP)
            robot_parts["head_2_joint"].setPosition(current_head_tilt)
        elif key == keyboard.LEFT: # Head yaw left
            current_head_yaw = min(HEAD_YAW_MAX, current_head_yaw + HEAD_YAW_STEP)
            robot_parts["head_1_joint"].setPosition(current_head_yaw)
        elif key == keyboard.RIGHT: # Head yaw right
            current_head_yaw = max(HEAD_YAW_MIN, current_head_yaw - HEAD_YAW_STEP)
            robot_parts["head_1_joint"].setPosition(current_head_yaw)
        elif key == ord('G'):
            if gripper_status == "closed":
                # Code to open gripper
                gripper_status = "opening"
            elif gripper_status == "open":
                # Code to close gripper
                gripper_status = "closing"

        # Set wheel velocities only in drive mode
        robot_parts["wheel_left_joint"].setVelocity(vL)
        robot_parts["wheel_right_joint"].setVelocity(vR)

    elif robot_mode == "arm":
        # Ensure wheels are stopped in arm mode
        robot_parts["wheel_left_joint"].setVelocity(0)
        robot_parts["wheel_right_joint"].setVelocity(0)

        # Pass relevant keys to the arm controller
        arm_control_keys = [ord(k) for k in END_EFFECTOR_CONTROL_KEYS.values()] # Get ASCII values
        if key in arm_control_keys:
            arm_controller(key)
        # Handle other keys specific to arm mode if needed (e.g., gripper)
        elif key == ord('G'):
            if gripper_status == "closed":
                # Code to open gripper
                gripper_status = "opening"
            elif gripper_status == "open":
                # Code to close gripper
                gripper_status = "closing"
        # capture arm pose
        elif key == ord('C'):
            get_robot_joints()
            
    # Update gripper status based on sensors
    if gripper_status == "opening":
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue() >= 0.044: # Check sensor
            gripper_status = "open"
            print("Gripper Open")
    elif gripper_status == "closing":
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue() <= 0.005: # Check sensor
            gripper_status = "closed"
            print("Gripper Closed")


    # Remove the joint_tester() call from the main loop
    # 
