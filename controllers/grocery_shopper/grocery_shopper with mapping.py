"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
from scipy.signal import convolve2d
# from image_tools import ImageTools
# import open3d as o3d

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

# create the Robot instance.
robot = Robot()

# image_tools = ImageTools(
    # yolo_weight_path='./models/best.pt',
    # fastsam_weight_path='./models/FastSAM-s.pt',
# )

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)
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
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

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

# Map parameters
map = np.zeros(shape=[360, 360])
probability_step = 5e-3  # Small increment for probabilistic mapping
min_display_threshold = 0.1  # Threshold for displaying obstacles on map

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

# o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Enable keyboard
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Map scale and offset parameters - Adjusted for 30m x 16m world
MAP_SCALE = 12  # pixels per meter - reduced to fit entire world
MAP_OFFSET_X = 180  # Center of the map (x)
MAP_OFFSET_Y = 180  # Center of the map (y)

# ------------------------------------------------------------------
# Helper Functions
def rotate_y(x,y,z,theta):
    new_x = x*np.cos(theta) + y*np.sin(theta)
    new_z = z
    new_y = y*-np.sin(theta) + x*np.cos(theta)
    return [-new_x, new_y, new_z]
    
def handle_capture():
    # Get the raw image data as bytes
    color_image_data = rgb_camera.getImage()
    depth_image = range_finder.getRangeImage()

    
    # Convert bytes to numpy array (correct shape)
    color_image = np.frombuffer(color_image_data, np.uint8)
    # Reshape considering BGRA format (4 channels) that Webots uses
    color_image = color_image.reshape(height, width, 4)
    # Convert to RGB if needed
    color_image = color_image[:, :, :3]
    
    # Handle depth image
    # depth_width = range_finder.getWidth()
    # depth_height = range_finder.getHeight()
    # depth_image = np.array(depth_image).reshape(depth_height, depth_width)
    
    # image_tools.save_sequential_color_image(color_image, '../../training_data')

gripper_status="closed"

# Helper functions for mapping and navigation
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
    
# def create_configuration_space():
    # """Create the configuration space by dilating obstacles."""
    # global convolved_map
    # convolved_map = convolve2d(map, np.ones((19, 19)), mode="same", boundary="fill", fillvalue=0)
    # convolved_map = convolved_map > 0.5
    # convolved_map = np.transpose(convolved_map)
    # return convolved_map
    
# def normalize_angle(angle):
    # """Normalize angle to [-π, π]"""
    # while angle > math.pi:
        # angle -= 2 * math.pi
    # while angle < -math.pi:
        # angle += 2 * math.pi
    # return angle

def update_display():
    """Update the display with the current map"""
    # This function can be called to refresh the map display as needed
    for x in range(360):
        for y in range(360):
            if map[x, y] > 0.5:  # Only draw significant obstacles
                color = int(map[x, y] * 255)
                display.setColor((color << 16) | (color << 8) | color)
                display.drawPixel(x, y)

# Main Loop
while robot.step(timestep) != -1:
    SAM_view_display.setColor(0x000000)
    SAM_view_display.fillRectangle(0, 0, 640, 480)
    
    # Reset velocities at the start of each loop
    vL = 0
    vR = 0

    # Get keyboard input for this timestep
    key = keyboard.getKey() 

    # Handle keyboard input for movement
    if key == ord('W'):  # Forward
        vL = MAX_SPEED
        vR = MAX_SPEED
    elif key == ord('S'):  # Backward
        vL = -MAX_SPEED
        vR = -MAX_SPEED
    elif key == ord('A'):  # Turn Left
        vL = -MAX_SPEED / 2
        vR = MAX_SPEED / 2
    elif key == ord('D'):  # Turn Right
        vL = MAX_SPEED / 2
        vR = -MAX_SPEED / 2
    # Handle other keyboard input
    elif key == ord('C'):
        handle_capture()
    elif key == ord('M'):  # Save map
        save_map()
    elif key == ord('L'):  # Load map
        load_map()
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
    
    # Set wheel velocities based on the key pressed in this timestep
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    # Get GPS values and update pose
    gps_values = gps.getValues()
    pose_x = gps_values[0]  # Update pose_x with actual GPS x value
    pose_y = gps_values[1]  # Update pose_y with actual GPS y value
    
    compass_values = compass.getValues()
    compass_angle = math.atan2(compass_values[0], compass_values[1])
    pose_theta = compass_angle  # Update pose_theta with actual compass angle
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings) - 83]
    
    # Print GPS and compass information
    print(f"GPS: x={gps_values[0]:.2f}, y={gps_values[1]:.2f}, z={gps_values[2]:.2f}")
    print(f"Compass raw: x={compass_values[0]:.2f}, y={compass_values[1]:.2f}, z={compass_values[2]:.2f}")
    print(f"Compass angle (heading): {compass_angle:.2f} radians = {math.degrees(compass_angle):.2f} degrees")
    print("------")
    
    # Clear the display to refresh the map view
    display.setColor(0x000000)
    display.fillRectangle(0, 0, 360, 360)
    
    # Process LIDAR readings for mapping
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
            pixel_value = map[map_x, map_y]
            if pixel_value < 1:
                pixel_value += probability_step
            pixel_value = min(1, pixel_value)
            map[map_x, map_y] = pixel_value
            
            # Draw the obstacle on the display
            color = int(pixel_value * 255)
            display.setColor((color << 16) | (color << 8) | color)
            display.drawPixel(map_x, map_y)
    
    # After processing new LIDAR readings, redraw the entire map to prevent disappearing lines
    # This ensures all detected obstacles remain visible
    for x in range(360):
        for y in range(360):
            if map[x, y] > 0.1:  # Lower threshold to show more features
                color = int(map[x, y] * 255)
                display.setColor((color << 16) | (color << 8) | color)
                display.drawPixel(x, y)
    
    # Draw robot position on map with updated GPS coordinates
    robot_map_x, robot_map_y = world_to_map(pose_x, pose_y)
    display.setColor(0xFF0000)  # Red color for robot
    display.drawPixel(robot_map_x, robot_map_y)
    
    # Draw robot heading indicator (a short line in the direction the robot is facing)
    heading_length = 5  # pixels
    heading_x = robot_map_x + int(heading_length * math.cos(pose_theta))
    heading_y = robot_map_y - int(heading_length * math.sin(pose_theta))
    display.setColor(0x00FF00)  # Green color for heading
    if 0 <= heading_x < 360 and 0 <= heading_y < 360:
        display.drawLine(robot_map_x, robot_map_y, heading_x, heading_y)

# Gripper control could be added here if needed