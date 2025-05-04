"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
from image_tools import ImageTools
import open3d as o3d
from arm_controller import ArmController

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
KEY_COOLDOWN_CYCLES = 10 # Number of simulation steps for key press cooldown

# Start in drive mode
robot_mode = "drive" 
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

# Create arm controller instance
arm_controller = ArmController(robot, timestep)

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# Define initial pose configurations
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf', 0.045, 0.045)
upper_shelf = (0.0, 0.0, 0.35, 0.447, 0.651, -1.439, 2.035, 1.845, 0.816, 1.983, 'inf', 'inf', 0.045, 0.045)
above_basket = (0.0, 0.0, 0.35, 0.07, 0.619, -0.519, 2.290, 1.892, -1.353, 0.390, 'inf', 'inf', 0.045, 0.045)

# Use upper_shelf as the initial position
initial_arm_pos = upper_shelf

# Have the arm controller set the initial position
arm_controller.set_arm_to_position(initial_arm_pos)

# Initialize head angles
current_head_yaw = initial_arm_pos[0]  # head_1_joint is the first element
current_head_tilt = initial_arm_pos[1] # head_2_joint is the second element
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
    if part_name in ["wheel_left_joint", "wheel_right_joint"]:
        # For wheels, set position to infinity and velocity control
        robot_parts[part_name].setPosition(float('inf'))
        robot_parts[part_name].setVelocity(0.0) # Begin with wheels stopped
    else:
        # The arm_controller has already set positions for arm joints
        pass

    # Set max velocity for all parts
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
    
# def joint_tester(tolerance=0.005, max_wait_steps=250):
#     """
#     Cycles each position-controlled joint through min -> max -> initial positions,
#     waiting for completion using Position Sensors.
#     Note: This function blocks execution while testing each joint.
#     Requires Position Sensors to be enabled for all tested joints.

#     Args:
#         tolerance (float): Position tolerance in radians/meters to consider the target reached.
#         max_wait_steps (int): Maximum simulation steps to wait before timing out.
#     """
#     print("=== Starting Joint Test (Sensor-Based Wait) ===")
#     robot_parts["wheel_left_joint"].setVelocity(0)
#     robot_parts["wheel_right_joint"].setVelocity(0)

#     # Filter part_names to only include those with sensors (i.e., not wheels)
#     testable_joints = [name for name in part_names if name in robot_sensors]

#     if not testable_joints:
#         print("No testable joints with enabled sensors found. Aborting test.")
#         return

#     initial_positions = {}
#     # Retrieve initial positions only for testable joints
#     for name in testable_joints:
#         # Find the corresponding index in the original target_pos tuple
#         try:
#             original_index = part_names.index(name)
#             initial_positions[name] = float(target_pos[original_index])
#         except (ValueError, IndexError):
#             print(f"Warning: Could not find initial position for {name} in global target_pos. Using 0.0 as fallback.")
#             initial_positions[name] = 0.0 # Fallback, adjust if needed


#     # --- Helper function for waiting ---
#     def wait_for_position(robot_joint, sensor, target_position, target_name, part_name, speed_factor):
#         print(f"  Moving {part_name} to {target_name} ({target_position:.3f})...")
#         robot_joint.setPosition(target_position)
#         robot_joint.setVelocity(robot_joint.getMaxVelocity() * speed_factor)
#         steps_waited = 0
#         while steps_waited < max_wait_steps:
#             if robot.step(timestep) == -1:
#                 print("  Simulation stopped during wait.")
#                 return False # Indicate simulation stopped

#             current_pos = sensor.getValue()
#             if abs(current_pos - target_position) < tolerance:
#                 print(f"  Reached {target_name} position ({current_pos:.3f}) after {steps_waited} steps.")
#                 return True # Indicate success

#             steps_waited += 1

#         # Loop finished without reaching target (timeout)
#         final_pos = sensor.getValue()
#         print(f"  Warning: Timeout waiting for {part_name} to reach {target_name}. Steps: {steps_waited}, Current pos: {final_pos:.3f}")
#         return True # Indicate timeout occurred but continue test
#     # --- End Helper function ---


#     for part_name in testable_joints:
#         robot_joint = robot_parts[part_name]
#         sensor = robot_sensors[part_name] # Assumes sensor exists from the filter above
#         initial_pos = initial_positions[part_name]

#         if not (hasattr(robot_joint, 'getMinPosition') and hasattr(robot_joint, 'getMaxPosition')):
#             print(f"Skipping joint {part_name} (does not have getMinPosition/getMaxPosition methods).")
#             continue

#         min_pos = robot_joint.getMinPosition()
#         max_pos = robot_joint.getMaxPosition()

#         # Handle potentially infinite limits reported by Webots
#         if min_pos < -1e10 or max_pos > 1e10:
#             print(f"Skipping joint {part_name} due to potentially infinite limits ({min_pos}, {max_pos}).")
#             continue

#         print(f"\nTesting joint: {part_name} (Min: {min_pos:.3f}, Max: {max_pos:.3f}, Initial: {initial_pos:.3f})")

#         # Move to Min
#         if not wait_for_position(robot_joint, sensor, min_pos, "Min", part_name, 0.5): return # Exit if simulation stopped

#         # Move to Max
#         if not wait_for_position(robot_joint, sensor, max_pos, "Max", part_name, 0.5): return # Exit if simulation stopped

#         # Return to Initial Position
#         if not wait_for_position(robot_joint, sensor, initial_pos, "Initial", part_name, 0.5): return # Exit if simulation stopped

#         print(f"Finished testing {part_name}.")

#     print("\n=== Joint Test Complete ===")

# def get_robot_joints():
#     global part_names, robot_sensors
#     current_pose_list = []
#     for name in part_names:
#         if "wheel" in name:
#             current_pose_list.append("'inf'") # Add as string to match target_pos format
#         elif name == "gripper_left_finger_joint":
#             current_pose_list.append(f"{left_gripper_enc.getValue():.3f}")
#         elif name == "gripper_right_finger_joint":
#                 current_pose_list.append(f"{right_gripper_enc.getValue():.3f}")
#         elif name in robot_sensors:
#             current_pose_list.append(f"{robot_sensors[name].getValue():.3f}")
            
#     pose_str = "(" + ", ".join(current_pose_list) + ")"
#     print(f"Current Pose: {pose_str}")

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
            robot_pos = arm_controller.convert_camera_coord_to_robot_coord(obj_pos)
            print(f"Robot Relative Position: {robot_pos}")
        steps_taken = 0
            
    # if joint_test: 
    #     joint_tester()
    #     joint_test = False

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
            print(f"Switched to '{robot_mode}' mode.")
        else:
            robot_mode = "drive"
            # When switching back to drive mode, make sure arm is in a safe position
            arm_controller.set_arm_to_position(initial_arm_pos)
            print(f"Switched to '{robot_mode}' mode. Reset arm to safe position.")
        
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
                # Call the arm controller's approach and grasp function
                success = arm_controller.approach_and_grasp_object(
                    object_mask, depth_image, o3d_intrinsics,
                    image_tools, MAX_SPEED_MS, MAX_SPEED
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
            # Toggle gripper using arm controller
            arm_controller.toggle_gripper()

        # Set wheel velocities only in drive mode
        robot_parts["wheel_left_joint"].setVelocity(vL)
        robot_parts["wheel_right_joint"].setVelocity(vR)

    elif robot_mode == "arm":
        # Ensure wheels are stopped in arm mode
        robot_parts["wheel_left_joint"].setVelocity(0)
        robot_parts["wheel_right_joint"].setVelocity(0)

        # Pass relevant keys to the arm controller
        arm_control_keys = [ord(k) for k in arm_controller.END_EFFECTOR_CONTROL_KEYS.values()] # Get ASCII values
        if key in arm_control_keys:
            arm_controller.handle_arm_control(key)
        # Handle other keys specific to arm mode if needed (e.g., gripper)
        elif key == ord('G'):
            # Toggle gripper using arm controller
            arm_controller.toggle_gripper()
        # capture arm pose
        elif key == ord('C'):
            arm_controller.get_current_joint_positions()
            
    # Update gripper status
    arm_controller.update_gripper_status()


    # Remove the joint_tester() call from the main loop
    # 
