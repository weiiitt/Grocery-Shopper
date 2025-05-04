"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
from image_tools import ImageTools
import open3d as o3d
from arm_controller import ArmController
from scipy.signal import convolve2d

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

# Map parameters
map = np.zeros(shape=[360, 360])
probability_step = 5e-3  # Small increment for probabilistic mapping
min_display_threshold = 0.1  # Threshold for displaying obstacles on map

# Map scale and offset parameters
MAP_SCALE = 12  # pixels per meter
MAP_OFFSET_X = 180  # Center of the map (x)
MAP_OFFSET_Y = 180  # Center of the map (y)

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
    
def world_to_map(world_x, world_y):
    """Convert world coordinates (meters) to map coordinates (pixels).
    World origin is mapped to the center of the map.
    """
    # Convert from world coordinates to map coordinates
    # Map origin is at the top-left corner
    map_x = int(MAP_OFFSET_X + world_x * MAP_SCALE)
    map_y = int(MAP_OFFSET_Y - world_y * MAP_SCALE)  # Y-axis is inverted
    
    # Ensure coordinates are within map boundaries
    map_x = max(0, min(359, map_x))
    map_y = max(0, min(359, map_y))
    
    return map_x, map_y

def map_to_world(map_x, map_y):
    """Convert map coordinates (pixels) to world coordinates (meters)."""
    world_x = (map_x - MAP_OFFSET_X) / MAP_SCALE
    world_y = (MAP_OFFSET_Y - map_y) / MAP_SCALE  # Y-axis is inverted
    return world_x, world_y

def load_map():
    """Load a saved map from disk."""
    global map
    try:
        loaded_map = np.load("map.npy")
        if loaded_map.shape == (360, 360):
            map = loaded_map.astype(np.float32)
            
            # Update the display with the loaded map
            display.setColor(0x000000)
            display.fillRectangle(0, 0, 360, 360)
            
            # Draw the actual map data
            for x in range(360):
                for y in range(360):
                    if map[x, y] > 0.5:  # Only draw significant obstacles
                        color = int(map[x, y] * 255)
                        display.setColor((color << 16) | (color << 8) | color)
                        display.drawPixel(x, y)
            
            print("Map loaded successfully")
        else:
            print("Error: Map dimensions don't match expected size (360x360)")
    except Exception as e:
        print(f"Error loading map: {e}")

def save_map():
    """Save the current map to disk."""
    global map
    try:
        np.save("map.npy", map)
        print("Map saved successfully")
    except Exception as e:
        print(f"Error saving map: {e}")

def rotate_y(x,y,z,theta):
    new_x = x*np.cos(theta) + y*np.sin(theta)
    new_z = z
    new_y = y*-np.sin(theta) + x*np.cos(theta)
    return [-new_x, new_y, new_z]

# Main Loop
joint_test = False
steps_taken = 0
robot_pos = None
obj_pos = None
# Track modified pixels to avoid redrawing the entire map
modified_pixels = set()
map_needs_redraw = True

while robot.step(timestep) != -1:

    steps_taken += 1
    
    # Only do object detection and map redraw every 50 steps
    should_update_display = (steps_taken % 50 == 0)
    
    if should_update_display:
        object_mask, depth_image, detected = draw_detected_objects()
        if detected:
            # if obj_pos is None:
            obj_pos = image_tools.get_object_coord(object_mask, depth_image, o3d_intrinsics)
            print(f"Object Position: {obj_pos}")
            robot_pos = arm_controller.convert_camera_coord_to_robot_coord(obj_pos)
            print(f"Robot Relative Position: {robot_pos}")
    
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

    # Get GPS values and update pose
    gps_values = gps.getValues()
    pose_x = gps_values[0]  # Update pose_x with actual GPS x value
    pose_y = gps_values[1]  # Update pose_y with actual GPS y value
    
    compass_values = compass.getValues()
    compass_angle = math.atan2(compass_values[0], compass_values[1])
    pose_theta = compass_angle  # Update pose_theta with actual compass angle
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings) - 83]
    
    # Always process LIDAR readings to update the map data structure
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]
        
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue
        
        # Calculate the obstacle position in robot's coordinate frame
        obs_x_rel = math.cos(alpha) * rho
        obs_y_rel = -math.sin(alpha) * rho
        
        # Convert to world coordinates by rotating and translating
        # based on robot's position and orientation
        obs_x = pose_x + obs_x_rel * math.cos(pose_theta) - obs_y_rel * math.sin(pose_theta)
        obs_y = pose_y + obs_x_rel * math.sin(pose_theta) + obs_y_rel * math.cos(pose_theta)
        
        # Convert world coordinates to map coordinates
        map_x, map_y = world_to_map(obs_x, obs_y)
        
        # Update map with obstacle information
        if 0 <= map_x < 360 and 0 <= map_y < 360:
            old_value = map[map_x, map_y]
            pixel_value = old_value
            if pixel_value < 1:
                pixel_value += probability_step
            pixel_value = min(1, pixel_value)
            map[map_x, map_y] = pixel_value
            
            if abs(pixel_value - old_value) > 0.01:
                map_needs_redraw = True
    
    # Only redraw the map every 50 steps or when map needs redraw after save/load
    if should_update_display or map_needs_redraw:
        # Clear the display to refresh the map view
        display.setColor(0x000000)
        display.fillRectangle(0, 0, 360, 360)
        
        # Calculate robot map position
        robot_map_x, robot_map_y = world_to_map(pose_x, pose_y)
        
        # Draw the full map, skipping the robot's position
        for x in range(360):
            for y in range(360):
                if (x == robot_map_x and y == robot_map_y):
                    continue
                    
                if map[x, y] > 0.1:  # Lower threshold to show more features
                    color = int(map[x, y] * 255)
                    display.setColor((color << 16) | (color << 8) | color)
                    display.drawPixel(x, y)
        
        # Draw robot position on map
        display.setColor(0xFF0000)  # Red color for robot
        display.drawPixel(robot_map_x, robot_map_y)
        
        # Draw robot heading indicator
        heading_length = 5  # pixels
        heading_x = robot_map_x + int(heading_length * math.cos(pose_theta))
        heading_y = robot_map_y - int(heading_length * math.sin(pose_theta))
        display.setColor(0x00FF00)  # Green color for heading
        if 0 <= heading_x < 360 and 0 <= heading_y < 360:
            display.drawLine(robot_map_x, robot_map_y, heading_x, heading_y)
            
        map_needs_redraw = False
    
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
        elif key == ord('M'):  # Save map
            save_map()
        elif key == ord('L'):  # Load map
            load_map()
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
