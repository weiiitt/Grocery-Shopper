"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
from image_tools import ImageTools
import open3d as o3d
from arm_controller import ArmController
from scipy.signal import convolve2d
import queue

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
WHEEL_RADIUS = MAX_SPEED_MS / MAX_SPEED # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# --- Autonomous Navigation Parameters ---
ROBOT_RADIUS_M = 1 # Approximate radius of the robot base
OBSTACLE_THRESHOLD = 0.7 # Map probability threshold to consider a cell an obstacle for C-Space
CONFIG_SPACE_UPDATE_INTERVAL = 50 # Update config space every N steps
CHECK_FRONT_DISTANCE_M = 0.8 # How far ahead to check for obstacles in C-Space
STRAIGHT_TARGET_DISTANCE_M = 1.5 # How far to plan straight paths
TURN_TARGET_SIDE_OFFSET_M = 0.7 # How far sideways to aim for a turn target
TURN_TARGET_FORWARD_OFFSET_M = 1.0 # How far forward to aim for a turn target
PATH_FOLLOW_LOOKAHEAD_DIST = 0.5 # Lookahead distance for path follower (meters) - Not used in current simple follower
PATH_FOLLOW_GOAL_TOLERANCE = 0.30 # Tolerance to consider waypoint reached (meters)
PATH_FOLLOW_KP_ANGULAR = 1.0 # Proportional gain for angular velocity in path follower
PATH_FOLLOW_LOOKAHEAD_INDEX = 4 # How many waypoints ahead to look from the closest point
PATH_FOLLOW_ALPHA_HIGH = 0.5 # Radians (~28 deg). Above this, pure rotation.
PATH_FOLLOW_ALPHA_LOW = 0.1 # Radians (~5.7 deg). Below this, mostly forward.
PATH_FOLLOW_STEERING_GAIN = 0.3 # Gain for steering correction
PATH_FOLLOW_MAX_TURN_SPEED = MAX_SPEED / 4.0 # Max speed during pure rotation
PATH_FOLLOW_FORWARD_SPEED_FACTOR = 0.5 # Base speed factor when moving forward (can be adjusted)



robot_mode = "drive"
# Navigation States
NAV_STATE_IDLE = "IDLE"
NAV_STATE_MANUAL_DRIVE = "MANUAL_DRIVE"
NAV_STATE_EXPLORING = "EXPLORING" # Autonomous mode
NAV_STATE_ARM_CONTROL = "ARM_CONTROL"
NAV_STATE_PLANNING = "PLANNING" # Intermediate state while A* runs

# --- End Autonomous Navigation Parameters ---

ROBOT_MODE_TOGGLE_KEY = 'M' # Key to switch between drive/idle and arm control
EXPLORE_TOGGLE_KEY = 'E' # Key to toggle exploration mode
SAVE_MAP_KEY = 'K'
LOAD_MAP_KEY = 'L'

KEY_COOLDOWN_CYCLES = 10 # Number of simulation steps for key press cooldown

# Start in manual drive mode
navigation_state = NAV_STATE_MANUAL_DRIVE
key_cooldown_timer = 0 # Initialize cooldown timer
mode_just_changed = False # This might be removable later if not used by arm controller
print(f"Starting in '{navigation_state}' mode.")
print(f"Press '{ROBOT_MODE_TOGGLE_KEY}' to toggle Arm Control.")
print(f"Press '{EXPLORE_TOGGLE_KEY}' to toggle Exploration.")

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
cspace_display = robot.getDevice("cspace_display")

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

left_wheel_enc = robot_parts["wheel_left_joint"].getPositionSensor()
right_wheel_enc = robot_parts["wheel_right_joint"].getPositionSensor()
left_wheel_enc.enable(timestep)
right_wheel_enc.enable(timestep)

prev_left_enc = left_wheel_enc.getValue()
prev_right_enc = right_wheel_enc.getValue()

# Odometry
pose_x     = 0 # GPS X
pose_y     = 0 # GPS Y
pose_theta = 0 # Compass Theta

odom_pose_x = 0 # Odometry X
odom_pose_y = 0 # Odometry Y
odom_pose_theta = 0 # Odometry Theta

vL = 0
vR = 0

LIDAR_BIN_CUTOFF = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[LIDAR_BIN_CUTOFF:len(lidar_offsets)-LIDAR_BIN_CUTOFF] # Only keep lidar readings not blocked by robot chassis

# Map parameters
map = np.zeros(shape=[360, 360], dtype=np.float32)
probability_step = 5e-3  # Small increment for probabilistic mapping
min_display_threshold = 0.1  # Threshold for displaying obstacles on map

# Map scale and offset parameters
MAP_SCALE = 12  # pixels per meter
MAP_OFFSET_X = 180  # Center of the map (x)
MAP_OFFSET_Y = 180  # Center of the map (y)

# --- Navigation State Variables ---
ROBOT_RADIUS_PIXELS = int(ROBOT_RADIUS_M * MAP_SCALE) # Radius in map pixels
config_space = np.zeros_like(map, dtype=bool) # Boolean map for collision checking
current_path = [] # List of (world_x, world_y) waypoints
path_index = 0 # Index of the current target waypoint in current_path
last_config_space_update_step = -CONFIG_SPACE_UPDATE_INTERVAL # Ensure update on first relevant step
# --- End Navigation State Variables ---

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

# --- Autonomous Navigation Functions ---

def create_configuration_space(occupancy_map, robot_radius_pixels, obstacle_threshold):
    """Creates the configuration space map by dilating obstacles."""
    print(f"Updating C-Space (Radius: {robot_radius_pixels} px, Threshold: {obstacle_threshold})")
    # 1. Identify Obstacles based on threshold
    obstacles = occupancy_map >= obstacle_threshold

    # Convert boolean obstacles to integer (0 or 1)
    obstacles_int = obstacles.astype(int) # <-- ADDED conversion

    # 2. Convolution (Dilation)
    if robot_radius_pixels > 0:
        # Create a square kernel for approximation
        kernel_size = 2 * robot_radius_pixels + 1
        # Kernel should also be integer
        kernel = np.ones((kernel_size, kernel_size), dtype=int) # <-- Changed dtype to int
        # Convolve using integer arrays
        dilated_obstacles_int = convolve2d(obstacles_int, kernel, mode='same', boundary='fill', fillvalue=0) # <-- Use int arrays
        # Convert result back to boolean: True where convolution result > 0
        dilated_obstacles = dilated_obstacles_int > 0 # <-- Convert back to boolean
    else:
        dilated_obstacles = obstacles # No dilation if radius is 0

    # Ensure map boundaries are always obstacles in C-space
    dilated_obstacles[0, :] = True
    dilated_obstacles[-1, :] = True
    dilated_obstacles[:, 0] = True
    dilated_obstacles[:, -1] = True

    print("C-Space update complete.")
    return dilated_obstacles # True where robot center CANNOT go

def heuristic(a, b):
    """Calculate Manhattan distance heuristic for A*."""
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def path_planner(config_space_map, start_map, end_map):
    """
    Plans a path using A* algorithm on the configuration space.
    Args:
        config_space_map (np.array): Boolean map (True=Obstacle).
        start_map (tuple): (x, y) start coordinates in map pixels.
        end_map (tuple): (x, y) end coordinates in map pixels.
    Returns:
        list: A list of (x, y) map coordinates representing the path, or None if no path found.
    """
    print(f"Planning path from {start_map} to {end_map}...")
    rows, cols = config_space_map.shape
    start_map = (int(start_map[0]), int(start_map[1]))
    end_map = (int(end_map[0]), int(end_map[1]))

    # Bounds check for start/end
    if not (0 <= start_map[0] < rows and 0 <= start_map[1] < cols):
        print(f"Error: Start node {start_map} is out of bounds.")
        return None
    if not (0 <= end_map[0] < rows and 0 <= end_map[1] < cols):
        print(f"Error: End node {end_map} is out of bounds.")
        return None

    # Obstacle check for start/end
    if config_space_map[start_map]:
        print(f"Error: Start node {start_map} is in an obstacle.")
        return None
    if config_space_map[end_map]:
        print(f"Error: End node {end_map} is in an obstacle.")
        return None

    frontier = queue.PriorityQueue()
    frontier.put((0, start_map)) # Priority queue stores (priority, item)
    came_from = {start_map: None}
    cost_so_far = {start_map: 0}

    max_iterations = config_space_map.size * 2 # Safety break, allow more iterations
    iterations = 0

    while not frontier.empty() and iterations < max_iterations:
        iterations += 1
        current_priority, current_node = frontier.get()

        if current_node == end_map:
            print("Path found!")
            break # Goal reached

        # Explore neighbors (4-connectivity)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Add diagonals: , (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            next_node = (current_node[0] + dx, current_node[1] + dy)

            # Check bounds
            if not (0 <= next_node[0] < rows and 0 <= next_node[1] < cols):
                continue

            # Check obstacles in config_space
            if config_space_map[next_node]:
                continue

            # Cost: 1 for cardinal, sqrt(2) for diagonal (optional, use 1 for simplicity)
            move_cost = 1 # if dx == 0 or dy == 0 else math.sqrt(2)
            new_cost = cost_so_far[current_node] + move_cost
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(end_map, next_node)
                frontier.put((priority, next_node))
                came_from[next_node] = current_node
    else: # Loop finished without finding path or hit iteration limit
        if iterations >= max_iterations:
            print("Warning: A* hit max iterations.")
        elif frontier.empty():
             print("Error: Path not found (frontier empty).")
        else: # Path found, but loop exited via break
             pass # This case is handled below
        # If goal wasn't reached (loop finished naturally or max iterations)
        if current_node != end_map:
             return None


    # Reconstruct path
    path = []
    node = end_map
    while node is not None:
        path.append(node)
        if node in came_from:
             node = came_from[node]
        else: # Should only happen for start node
             break
    path.reverse() # Path is from start to end

    if not path or path[0] != start_map:
        print("Error: Path reconstruction failed.")
        return None

    print(f"Path length: {len(path)} nodes.")
    return path

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def find_closest_point_index(path_world_coords, pose_x, pose_y):
    """Finds the index of the closest waypoint in the path to the robot."""
    if not path_world_coords:
        return 0
    min_dist_sq = float('inf')
    closest_index = 0
    for i, (wx, wy) in enumerate(path_world_coords):
        dist_sq = (wx - pose_x)**2 + (wy - pose_y)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_index = i
    return closest_index

def turn_to_direction(pose_theta, target_theta, max_turn_speed):
    """Calculates wheel speeds for pure rotation towards a target angle."""
    angle_diff = normalize_angle(target_theta - pose_theta)

    # Determine turn direction and base speed
    # Positive angle_diff means turn left (vL negative, vR positive)
    # Negative angle_diff means turn right (vL positive, vR negative)
    turn_speed = max_turn_speed * np.sign(angle_diff)

    # Scale speed based on how close to target angle (optional, simple version first)
    # scale_factor = min(1.0, abs(angle_diff) / 0.1) # Example scaling
    # turn_speed *= scale_factor

    vL = -turn_speed
    vR = turn_speed

    # Optional: Add caster wheel drift correction if needed (like in lab5)
    # correction = 0.2 * np.sign(angle_diff)
    # vL += correction
    # vR -= correction

    return vL, vR


def follow_path_controller(pose_x, pose_y, pose_theta, path_world_coords, current_furthest_index):
    """
    Path following controller inspired by lab5_controller2 logic.
    Uses lookahead from the closest point on the path.
    Args:
        pose_x, pose_y, pose_theta: Current robot pose in world coordinates.
        path_world_coords (list): List of (x, y) world waypoints.
        current_furthest_index (int): Index of the furthest waypoint reached so far.
    Returns:
        tuple: (vL, vR, path_finished (bool), next_furthest_index (int))
    """
    if not path_world_coords:
        return 0, 0, True, 0 # Path finished or empty

    # Check distance to the *final* goal
    final_target_x, final_target_y = path_world_coords[-1]
    dist_to_final_target = math.sqrt((final_target_x - pose_x)**2 + (final_target_y - pose_y)**2)

    if dist_to_final_target < PATH_FOLLOW_GOAL_TOLERANCE:
        print("Follower: Reached final waypoint.")
        return 0, 0, True, len(path_world_coords) - 1 # Reached final waypoint

    # Find the closest point on the path to the robot
    closest_index = find_closest_point_index(path_world_coords, pose_x, pose_y)

    # Update the furthest point reached index
    next_furthest_index = max(current_furthest_index, closest_index)

    # Determine the lookahead target index
    target_index = min(next_furthest_index + PATH_FOLLOW_LOOKAHEAD_INDEX, len(path_world_coords) - 1)

    # Get the target coordinates
    target_x, target_y = path_world_coords[target_index]

    # Calculate distance and angle to the lookahead target
    rho = math.sqrt((target_x - pose_x)**2 + (target_y - pose_y)**2)
    desired_theta = math.atan2(target_y - pose_y, target_x - pose_x)

    # Calculate heading error
    angle_diff = normalize_angle(desired_theta - pose_theta)
    alpha = abs(angle_diff)

    vL_rads, vR_rads = 0.0, 0.0

    # Controller logic based on angle error (alpha)
    if alpha > PATH_FOLLOW_ALPHA_HIGH:
        # Large error - pure rotation
        vL_rads, vR_rads = turn_to_direction(pose_theta, desired_theta, PATH_FOLLOW_MAX_TURN_SPEED)
        # print(f"Follower: Pure Turn (alpha={alpha:.2f})")
    elif alpha > PATH_FOLLOW_ALPHA_LOW:
        # Moderate error - blend turning and forward motion
        blend_factor = 1.0 - (alpha - PATH_FOLLOW_ALPHA_LOW) / (PATH_FOLLOW_ALPHA_HIGH - PATH_FOLLOW_ALPHA_LOW)

        turn_vL, turn_vR = turn_to_direction(pose_theta, desired_theta, PATH_FOLLOW_MAX_TURN_SPEED)

        # Reduced forward speed during turns
        forward_speed_rads = MAX_SPEED * PATH_FOLLOW_FORWARD_SPEED_FACTOR * blend_factor

        # Combine turn and forward components
        # Steering correction is implicitly handled by turn_to_direction blend
        vL_rads = turn_vL * (1 - blend_factor) + forward_speed_rads * blend_factor
        vR_rads = turn_vR * (1 - blend_factor) + forward_speed_rads * blend_factor
        # print(f"Follower: Blended Turn (alpha={alpha:.2f}, blend={blend_factor:.2f})")
    else:
        # Small error - primarily forward motion with steering correction
        if rho > 0.05: # Only move if not too close to the lookahead point
            # Scale speed based on distance (optional, simple version first)
            # base_speed = min(MAX_SPEED, MAX_SPEED * (rho / 1.0) * 2)
            base_speed_rads = MAX_SPEED * PATH_FOLLOW_FORWARD_SPEED_FACTOR
            steering = PATH_FOLLOW_STEERING_GAIN * angle_diff * MAX_SPEED # Steering proportional to angle error

            vL_rads = base_speed_rads - steering
            vR_rads = base_speed_rads + steering
            # print(f"Follower: Forward (alpha={alpha:.2f}, rho={rho:.2f})")
        else:
            # Very close to lookahead point, slow down/stop turning
            vL_rads, vR_rads = 0.0, 0.0
            # print(f"Follower: Near Lookahead Target (rho={rho:.2f})")


    # Clamp wheel speeds to MAX_SPEED
    vL_clamped = max(-MAX_SPEED, min(MAX_SPEED, vL_rads))
    vR_clamped = max(-MAX_SPEED, min(MAX_SPEED, vR_rads))

    # print(f"Follower: TargetIdx={target_index}, Target=({target_x:.2f},{target_y:.2f}), Dist={rho:.2f}, AngleDiff={angle_diff:.2f}, vL={vL_clamped:.2f}, vR={vR_clamped:.2f}")
    return vL_clamped, vR_clamped, False, next_furthest_index


def check_front_obstacle(robot_map_x, robot_map_y, pose_theta, config_space, check_distance_m):
    """Checks for obstacles directly in front of the robot in config space."""
    check_dist_pixels = int(check_distance_m * MAP_SCALE)
    rows, cols = config_space.shape
    # Check a few points along the line
    # Step by roughly half the robot radius, but at least 1 pixel
    step_size = max(1, ROBOT_RADIUS_PIXELS // 2)
    for step in range(step_size, check_dist_pixels + step_size, step_size):
        check_x = robot_map_x + int(step * math.cos(pose_theta))
        check_y = robot_map_y - int(step * math.sin(pose_theta)) # Map Y inverted

        # Check bounds
        if not (0 <= check_x < rows and 0 <= check_y < cols):
            # print(f"  Front check blocked (Out of Bounds at step {step})")
            return True # Treat out of bounds as blocked

        # Check config space
        if config_space[check_x, check_y]:
            # print(f"  Front check blocked at map ({check_x}, {check_y}) step {step}")
            return True # Obstacle found

    # print("  Front clear.")
    return False # No obstacle found within check distance

def find_straight_target(robot_map_x, robot_map_y, pose_theta, config_space, forward_distance_m):
    """Finds a target point straight ahead, letting A* handle obstacles."""
    target_dist_pixels = int(forward_distance_m * MAP_SCALE)
    target_x = robot_map_x + int(target_dist_pixels * math.cos(pose_theta))
    target_y = robot_map_y - int(target_dist_pixels * math.sin(pose_theta)) # Map Y inverted

    # Basic bounds check
    target_x = max(0, min(config_space.shape[0] - 1, target_x))
    target_y = max(0, min(config_space.shape[1] - 1, target_y))

    # Simplification: Just return the projected point. A* will handle obstacles.
    # A more robust version would trace the line and stop before hitting config_space obstacle.
    if config_space[target_x, target_y]:
         print(f"Warning: Straight target map({target_x}, {target_y}) is in C-Space obstacle. A* might fail or find detour.")
         # Could try backtracking here, but let A* handle it for now.

    print(f"  Found straight target: map({target_x}, {target_y})")
    return (target_x, target_y)

def find_turn_target(robot_map_x, robot_map_y, pose_theta, config_space):
    """Finds a target point around a corner (prioritizes left)."""
    side_offset_pixels = int(TURN_TARGET_SIDE_OFFSET_M * MAP_SCALE)
    forward_offset_pixels = int(TURN_TARGET_FORWARD_OFFSET_M * MAP_SCALE)
    rows, cols = config_space.shape

    # Define relative angles for checking left/right turns
    turn_angles = {'left': math.pi / 2.0, 'right': -math.pi / 2.0}

    for direction, rel_angle in turn_angles.items():
        turn_angle = pose_theta + rel_angle

        # Calculate a potential target point:
        # Start slightly ahead of the robot to represent the 'corner'
        corner_check_dist_pixels = ROBOT_RADIUS_PIXELS * 1.5
        corner_x = robot_map_x + int(corner_check_dist_pixels * math.cos(pose_theta))
        corner_y = robot_map_y - int(corner_check_dist_pixels * math.sin(pose_theta))

        # Project sideways and then forward from that corner point
        target_x = corner_x + int(side_offset_pixels * math.cos(turn_angle)) + int(forward_offset_pixels * math.cos(pose_theta))
        target_y = corner_y - int(side_offset_pixels * math.sin(turn_angle)) - int(forward_offset_pixels * math.sin(pose_theta)) # Map Y inverted

        # Clamp target to map bounds
        target_x = max(0, min(rows - 1, target_x))
        target_y = max(0, min(cols - 1, target_y))

        # Check if the calculated target is valid (not in C-space)
        if not config_space[target_x, target_y]:
            print(f"  Found {direction} turn target: map({target_x}, {target_y})")
            return (target_x, target_y)
        else:
            print(f"  Proposed {direction} turn target map({target_x}, {target_y}) is in C-Space.")


    print("  Could not find suitable turn target.")
    return None # No suitable target found

def plan_new_path(start_map_coords, end_map_coords, config_space_map):
    """Plans and stores a new path, updating state."""
    global current_path, path_index, navigation_state, furthest_path_index # Added furthest_path_index

    # Ensure integer coordinates for planner
    start_map_coords = (int(start_map_coords[0]), int(start_map_coords[1]))
    end_map_coords = (int(end_map_coords[0]), int(end_map_coords[1]))

    # Check if start/end are the same
    if start_map_coords == end_map_coords:
        print("Planning skipped: Start and end points are the same.")
        current_path = []
        path_index = 0
        furthest_path_index = 0 # Reset furthest index
        navigation_state = NAV_STATE_EXPLORING # Go back to exploring state
        return

    print(f"Setting state to PLANNING for path from {start_map_coords} to {end_map_coords}")
    navigation_state = NAV_STATE_PLANNING # Indicate planning is in progress
    robot.step(timestep) # Allow simulation step for UI update if needed

    path_map_coords = path_planner(config_space_map, start_map_coords, end_map_coords)

    if path_map_coords and len(path_map_coords) > 1:
        # Convert map path to world path, skip the first point (current pos)
        current_path = [map_to_world(p[0], p[1]) for p in path_map_coords[1:]]
        path_index = 0 # Reset path index (might not be needed with new controller)
        furthest_path_index = 0 # Reset furthest index for the new path
        print(f"Successfully planned path with {len(current_path)} waypoints.")
        # Transition back to EXPLORING, path following will start on next cycle
        navigation_state = NAV_STATE_EXPLORING
    else:
        print("Path planning failed. Clearing path.")
        current_path = []
        path_index = 0
        furthest_path_index = 0 # Reset furthest index
        # Stay in PLANNING or switch to IDLE? Let's go back to EXPLORING to retry logic.
        navigation_state = NAV_STATE_EXPLORING # Will likely try to plan again immediately

def handle_exploration(pose_x, pose_y, pose_theta, config_space_map):
    """Handles the logic for autonomous exploration: follow path or plan new."""
    global current_path, path_index, navigation_state, furthest_path_index # Added furthest_path_index

    # --- Path Following ---
    if current_path:
        # Use the new controller, passing and receiving furthest_path_index
        vL, vR, path_finished, next_furthest_index = follow_path_controller(
            pose_x, pose_y, pose_theta, current_path, furthest_path_index
        )
        furthest_path_index = next_furthest_index # Update global furthest index

        if path_finished:
            print("Path finished.")
            current_path = [] # Clear the path
            path_index = 0    # Reset index (though not strictly used by new controller)
            furthest_path_index = 0 # Reset furthest index
            return 0, 0 # Stop briefly before deciding next move
        else:
            return vL, vR # Continue following path

    # --- Path Decision Making (No current path) ---
    else:
        robot_map_x, robot_map_y = world_to_map(pose_x, pose_y)

        # Check if C-Space is valid at current location (should not happen ideally)
        if config_space_map[robot_map_x, robot_map_y]:
             print("CRITICAL WARNING: Robot is inside C-Space obstacle! Stopping.")
             navigation_state = NAV_STATE_IDLE
             current_path = [] # Ensure path is cleared
             path_index = 0
             furthest_path_index = 0 # Reset furthest index
             return 0, 0

        # Check directly ahead in config_space
        is_front_blocked = check_front_obstacle(robot_map_x, robot_map_y, pose_theta, config_space_map, CHECK_FRONT_DISTANCE_M)

        if not is_front_blocked:
            # Plan to move straight
            print("Attempting to plan straight path...")
            target_map_coords = find_straight_target(robot_map_x, robot_map_y, pose_theta, config_space_map, STRAIGHT_TARGET_DISTANCE_M)
            if target_map_coords:
                plan_new_path((robot_map_x, robot_map_y), target_map_coords, config_space_map)
                return 0, 0 # Stop while planning
            else:
                # This case should ideally not happen with find_straight_target
                print("Warning: Front clear but cannot find straight target? Attempting turn.")
                target_map_coords = find_turn_target(robot_map_x, robot_map_y, pose_theta, config_space_map)
                if target_map_coords:
                    plan_new_path((robot_map_x, robot_map_y), target_map_coords, config_space_map)
                    return 0, 0
                else:
                    print("Error: Cannot find straight or turn target. Stopping.")
                    navigation_state = NAV_STATE_IDLE
                    return 0, 0
        else: # Front is blocked
            print("Front blocked. Attempting to plan turn path...")
            target_map_coords = find_turn_target(robot_map_x, robot_map_y, pose_theta, config_space_map)
            if target_map_coords:
                plan_new_path((robot_map_x, robot_map_y), target_map_coords, config_space_map)
                return 0, 0
            else:
                print("Error: Front blocked, cannot find turn target. Stopping.")
                # Implement recovery? e.g., try turning 180? For now, stop.
                navigation_state = NAV_STATE_IDLE
                return 0, 0

# --- End Autonomous Navigation Functions ---

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
        
        # only test relevant joints
        if part_name not in ["gripper_left_finger_joint","gripper_right_finger_joint"]:
            print(f"Skipping joint {part_name} cuz don't care.")
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

robot_gps_path = []
robot_odometry_path = []

#initialize odometry to start at same place as gps
gps_values = gps.getValues()
odom_pose_x = gps_values[0]
odom_pose_y = gps_values[1]

compass_values = compass.getValues()
compass_angle = math.atan2(compass_values[0], compass_values[1])
odom_pose_theta = compass_angle

# Initialize wheel velocities
vL = 0.0
vR = 0.0

while robot.step(timestep) != -1:

    steps_taken += 1
    
    if joint_test: 
        joint_tester()
        joint_test = False

    # Only do object detection and map redraw every 50 steps
    should_update_display = (steps_taken % 50 == 0)

    # --- Sensor Readings ---
    current_left_enc = left_wheel_enc.getValue()
    current_right_enc = right_wheel_enc.getValue()
    raw_scan = lidar.getRangeImage()

    # --- Odometry Calculation ---
    delta_left = (current_left_enc - prev_left_enc) * WHEEL_RADIUS
    delta_right = (current_right_enc - prev_right_enc) * WHEEL_RADIUS

    # Distance is the average distance traveled by the wheels
    distance_m = (delta_left + delta_right) / 2.0
    # Angle change in radians
    angle_rad = (delta_right - delta_left) / AXLE_LENGTH

    # Update odometry pose
    odom_pose_theta += angle_rad
    # Keep theta within [-pi, pi]
    while odom_pose_theta > math.pi:
        odom_pose_theta -= 2 * math.pi
    while odom_pose_theta < -math.pi:
        odom_pose_theta += 2 * math.pi

    odom_pose_x += distance_m * math.cos(odom_pose_theta)
    odom_pose_y += distance_m * math.sin(odom_pose_theta)

    # Update previous encoder values
    prev_left_enc = current_left_enc
    prev_right_enc = current_right_enc

    # Get GPS values and update pose (overwrites odometry-based pose for map updates)
    gps_values = gps.getValues()
    pose_x = gps_values[0]  # Update pose_x with actual GPS x value
    pose_y = gps_values[1]  # Update pose_y with actual GPS y value
    
    compass_values = compass.getValues()
    compass_angle = math.atan2(compass_values[0], compass_values[1])
    pose_theta = compass_angle  # Update pose_theta with actual compass angle

    # Store path points
    robot_gps_path.append((pose_x, pose_y))
    # robot_odometry_path.append((odom_pose_x, odom_pose_y))
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[LIDAR_BIN_CUTOFF:len(lidar_sensor_readings) - LIDAR_BIN_CUTOFF]
    
    # Always process LIDAR readings to update the map data structure
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]
        
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue
        
        # Calculate the obstacle position in robot's coordinate frame
        obs_x_rel = math.cos(alpha) * rho
        obs_y_rel = -math.sin(alpha) * rho
        
        # Convert to world coordinates by rotating and translating
        # based on robot's position and orientation (using GPS/Compass pose)
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
    
    # --- Update Configuration Space (Periodically) ---
    if steps_taken - last_config_space_update_step >= CONFIG_SPACE_UPDATE_INTERVAL:
        config_space = create_configuration_space(map, ROBOT_RADIUS_PIXELS, OBSTACLE_THRESHOLD)
        last_config_space_update_step = steps_taken
        # map_needs_redraw = True # C-Space update doesn't require map redraw unless visualizing C-Space

    # --- Keyboard Input ---
    key_cooldown_timer = max(0, key_cooldown_timer - 1)
    input_key = keyboard.getKey()
    key = -1
    if input_key > -1 and key_cooldown_timer == 0:
        key = input_key
        key_cooldown_timer = KEY_COOLDOWN_CYCLES
        mode_just_changed = False # Reset flag on new key press

    # --- State Machine Input Handling ---
    # Handle mode switching keys first
    if key == ord(ROBOT_MODE_TOGGLE_KEY):
        mode_just_changed = True
        key_cooldown_timer = KEY_COOLDOWN_CYCLES # Ensure cooldown after mode switch
        if navigation_state == NAV_STATE_ARM_CONTROL:
            navigation_state = NAV_STATE_IDLE # Switch back to IDLE from ARM
            print(f"Switched to '{navigation_state}' mode.")
            # Optional: Reset arm to safe pose?
        else:
            navigation_state = NAV_STATE_ARM_CONTROL
            print(f"Switched to '{navigation_state}' mode.")
            vL, vR = 0, 0 # Stop wheels
        key = -1 # Consume key

    elif key == ord(EXPLORE_TOGGLE_KEY):
        mode_just_changed = True
        key_cooldown_timer = KEY_COOLDOWN_CYCLES
        if navigation_state == NAV_STATE_EXPLORING:
            navigation_state = NAV_STATE_IDLE
            print(f"Switched to '{navigation_state}' mode. Exploration stopped.")
            vL, vR = 0, 0 # Stop wheels
            current_path = [] # Clear path
            path_index = 0
            furthest_path_index = 0 # Reset furthest index
        else:
            # Ensure C-Space is updated before starting exploration
            if steps_taken - last_config_space_update_step >= 0: # Check if C-space has been calculated at least once
                 config_space = create_configuration_space(map, ROBOT_RADIUS_PIXELS, OBSTACLE_THRESHOLD)
                 last_config_space_update_step = steps_taken
            else:
                 print("Warning: Starting exploration without initial C-Space calculation.")

            navigation_state = NAV_STATE_EXPLORING
            print(f"Switched to '{navigation_state}' mode.")
            current_path = [] # Clear any previous path
            path_index = 0
            furthest_path_index = 0 # Reset furthest index
        key = -1 # Consume key

    # --- State Machine Logic ---
    temp_vL, temp_vR = 0.0, 0.0 # Velocities determined by state logic

    if navigation_state == NAV_STATE_IDLE:
        temp_vL, temp_vR = 0.0, 0.0
        # Check for transition keys (handled above or below)
        if key in [ord('W'), ord('A'), ord('S'), ord('D')]:
             navigation_state = NAV_STATE_MANUAL_DRIVE
             print(f"Switched to '{navigation_state}' mode.")
             # Let MANUAL_DRIVE handle the key press this cycle

    if navigation_state == NAV_STATE_MANUAL_DRIVE:
        # Handle movement keys
        if input_key == ord('W'): temp_vL, temp_vR = MAX_SPEED, MAX_SPEED
        elif input_key == ord('S'): temp_vL, temp_vR = -MAX_SPEED, -MAX_SPEED
        elif input_key == ord('A'): temp_vL, temp_vR = -MAX_SPEED / 2, MAX_SPEED / 2
        elif input_key == ord('D'): temp_vL, temp_vR = MAX_SPEED / 2, -MAX_SPEED / 2
        elif input_key > 0: # Other key pressed, stop movement
             temp_vL, temp_vR = 0.0, 0.0
        else:
             temp_vL, temp_vR = 0.0, 0.0 

        # Transition out if no movement key is pressed *this cycle*
        if input_key == -1 and abs(temp_vL) < 0.01 and abs(temp_vR) < 0.01:
             # Only transition to IDLE if stopped and no key was pressed
             # navigation_state = NAV_STATE_IDLE # Keep in manual until mode switch
             pass # Stay in manual drive even when stopped

    elif navigation_state == NAV_STATE_ARM_CONTROL:
        temp_vL, temp_vR = 0.0, 0.0
        # Pass arm control keys
        arm_control_keys = [ord(k) for k in arm_controller.END_EFFECTOR_CONTROL_KEYS.values()]
        if key in arm_control_keys:
            arm_controller.handle_arm_control(key)
        elif key == ord('G'):
            arm_controller.toggle_gripper()
        elif key == ord('C'): # Capture pose in arm mode
            arm_controller.get_current_joint_positions()

    elif navigation_state == NAV_STATE_EXPLORING:
        # Autonomous logic determines velocities
        temp_vL, temp_vR = handle_exploration(pose_x, pose_y, pose_theta, config_space)
        # Check for manual override
        if key in [ord('W'), ord('A'), ord('S'), ord('D')]:
             navigation_state = NAV_STATE_MANUAL_DRIVE
             print(f"Switched to '{navigation_state}' mode (Manual Override).")
             # Let MANUAL_DRIVE handle the key press this cycle
             temp_vL, temp_vR = 0.0, 0.0 # Stop autonomous movement first

    elif navigation_state == NAV_STATE_PLANNING:
        # Robot waits while planning is happening (triggered in plan_new_path)
        temp_vL, temp_vR = 0.0, 0.0
        print("Waiting for path planner...")
        # State transition happens within plan_new_path

    # --- General Keyboard Commands (Available in IDLE/MANUAL) ---
    if navigation_state in [NAV_STATE_IDLE, NAV_STATE_MANUAL_DRIVE]:
        if key == ord('C'): handle_capture()
        elif key == ord(SAVE_MAP_KEY): save_map()
        elif key == ord(LOAD_MAP_KEY):
            load_map()
            map_needs_redraw = True # Force redraw after loading
            # Recalculate C-Space after loading map
            config_space = create_configuration_space(map, ROBOT_RADIUS_PIXELS, OBSTACLE_THRESHOLD)
            last_config_space_update_step = steps_taken
        elif key == ord('P'): # Grasp sequence (remains manual trigger for now)
             object_mask, depth_image, detected = draw_detected_objects()
             if detected:
                 success = arm_controller.approach_and_grasp_object(
                     object_mask, depth_image, o3d_intrinsics,
                     image_tools, MAX_SPEED_MS, MAX_SPEED
                 )
                 print("Grasp attempt finished.")
             else:
                 print("No object detected to grasp.")
        elif key == keyboard.UP:
            current_head_tilt = min(HEAD_TILT_MAX, current_head_tilt + HEAD_TILT_STEP)
            robot_parts["head_2_joint"].setPosition(current_head_tilt)
        elif key == keyboard.DOWN:
            current_head_tilt = max(HEAD_TILT_MIN, current_head_tilt - HEAD_TILT_STEP)
            robot_parts["head_2_joint"].setPosition(current_head_tilt)
        elif key == keyboard.LEFT:
            current_head_yaw = min(HEAD_YAW_MAX, current_head_yaw + HEAD_YAW_STEP)
            robot_parts["head_1_joint"].setPosition(current_head_yaw)
        elif key == keyboard.RIGHT:
            current_head_yaw = max(HEAD_YAW_MIN, current_head_yaw - HEAD_YAW_STEP)
            robot_parts["head_1_joint"].setPosition(current_head_yaw)
        elif key == ord('G'):
            arm_controller.toggle_gripper()


    # --- Set Final Wheel Velocities ---
    vL = temp_vL
    vR = temp_vR
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)

    # --- Display Update ---
    if should_update_display or map_needs_redraw:
        # --- Main Map Display ---
        display.setColor(0x000000)
        display.fillRectangle(0, 0, 360, 360)

        # Draw Occupancy Map
        for x in range(360):
            for y in range(360):
                if map[x, y] > min_display_threshold: # Use threshold
                    color_val = int(map[x, y] * 255)
                    color_val = max(0, min(255, color_val))
                    packed_color = (color_val << 16) | (color_val << 8) | color_val
                    display.setColor(packed_color)
                    display.drawPixel(x, y)

        # Draw GPS Path (Blue)
        display.setColor(0x0000FF)
        if len(robot_gps_path) > 1:
            for i in range(len(robot_gps_path) - 1):
                p1_x, p1_y = world_to_map(robot_gps_path[i][0], robot_gps_path[i][1])
                p2_x, p2_y = world_to_map(robot_gps_path[i+1][0], robot_gps_path[i+1][1])
                display.drawLine(p1_x, p1_y, p2_x, p2_y)

        # Draw Planned Path (Cyan) on Main Display
        if current_path:
            display.setColor(0x00FFFF) # Cyan
            path_map_coords = [world_to_map(wp[0], wp[1]) for wp in current_path]
            robot_map_x_main, robot_map_y_main = world_to_map(pose_x, pose_y) # Use different var names
            if path_map_coords:
                # Draw line from robot to the first *actual* waypoint being targeted (closest + lookahead)
                # This requires finding the target index again, or passing it out
                # For simplicity, just draw the full path for now.
                # A better visualization would highlight the current lookahead target.
                display.drawLine(robot_map_x_main, robot_map_y_main, path_map_coords[0][0], path_map_coords[0][1])
                for i in range(len(path_map_coords) - 1):
                    display.drawLine(path_map_coords[i][0], path_map_coords[i][1], path_map_coords[i+1][0], path_map_coords[i+1][1])

            # Highlight the furthest point reached (optional visualization)
            if furthest_path_index < len(path_map_coords):
                display.setColor(0xFFFF00) # Yellow for furthest reached
                f_map_x, f_map_y = path_map_coords[furthest_path_index]
                display.fillOval(f_map_x, f_map_y, 2, 2)

            # Highlight the lookahead target point (optional visualization)
            # Use current_path instead of path_world_coords which is not defined here
            lookahead_target_index = min(furthest_path_index + PATH_FOLLOW_LOOKAHEAD_INDEX, len(current_path) - 1)
            if lookahead_target_index < len(path_map_coords):
                display.setColor(0xFF00FF) # Magenta target waypoint
                target_map_x, target_map_y = path_map_coords[lookahead_target_index]

        # Draw Robot Position (Red) on Main Display
        robot_map_x_main, robot_map_y_main = world_to_map(pose_x, pose_y)
        display.setColor(0xFF0000)
        display.fillOval(robot_map_x_main, robot_map_y_main, 3, 3)

        # Draw Robot Heading (Green) on Main Display
        heading_length = 5
        heading_x_main = robot_map_x_main + int(heading_length * math.cos(pose_theta))
        heading_y_main = robot_map_y_main - int(heading_length * math.sin(pose_theta))
        display.setColor(0x00FF00)
        if 0 <= heading_x_main < 360 and 0 <= heading_y_main < 360:
            display.drawLine(robot_map_x_main, robot_map_y_main, heading_x_main, heading_y_main)


        # --- C-Space Display ---
        if cspace_display: # Check if the display exists
            cspace_display.setColor(0xFFFFFF) # White background for free space
            cspace_display.fillRectangle(0, 0, 360, 360)

            # Draw C-Space Obstacles (Gray)
            cspace_display.setColor(0x808080) # Gray
            # Ensure config_space has been calculated at least once
            if 'config_space' in globals() and config_space is not None and config_space.shape == (360, 360):
                for x in range(360):
                    for y in range(360):
                        if config_space[x, y]: # True means obstacle in C-Space
                            cspace_display.drawPixel(x, y)
            else:
                # Optionally draw a message if C-space not ready
                cspace_display.setColor(0x000000)
                cspace_display.setFont("Arial", 10, True)
                cspace_display.drawText("C-Space not calculated", 10, 10)


            # Draw Planned Path (Cyan) on C-Space Display
            if current_path:
                cspace_display.setColor(0x00FFFF) # Cyan
                path_map_coords_cspace = [world_to_map(wp[0], wp[1]) for wp in current_path] # Recalculate for clarity
                robot_map_x_cspace, robot_map_y_cspace = world_to_map(pose_x, pose_y)
                if path_map_coords_cspace:
                    cspace_display.drawLine(robot_map_x_cspace, robot_map_y_cspace, path_map_coords_cspace[0][0], path_map_coords_cspace[0][1])
                    for i in range(len(path_map_coords_cspace) - 1):
                        cspace_display.drawLine(path_map_coords_cspace[i][0], path_map_coords_cspace[i][1], path_map_coords_cspace[i+1][0], path_map_coords_cspace[i+1][1])
                if path_index < len(path_map_coords_cspace):
                    cspace_display.setColor(0xFF00FF) # Magenta target waypoint
                    target_map_x_c, target_map_y_c = path_map_coords_cspace[path_index]
                    cspace_display.fillOval(target_map_x_c, target_map_y_c, 3, 3)

            # Draw Robot Position (Red) on C-Space Display
            robot_map_x_cspace, robot_map_y_cspace = world_to_map(pose_x, pose_y)
            cspace_display.setColor(0xFF0000)
            cspace_display.fillOval(robot_map_x_cspace, robot_map_y_cspace, 3, 3)

            # Draw Robot Heading (Green) on C-Space Display
            heading_x_cspace = robot_map_x_cspace + int(heading_length * math.cos(pose_theta))
            heading_y_cspace = robot_map_y_cspace - int(heading_length * math.sin(pose_theta))
            cspace_display.setColor(0x00FF00)
            if 0 <= heading_x_cspace < 360 and 0 <= heading_y_cspace < 360:
                cspace_display.drawLine(robot_map_x_cspace, robot_map_y_cspace, heading_x_cspace, heading_y_cspace)


        map_needs_redraw = False # Reset flag after drawing both displays

    # --- Update Arm/Gripper Status ---
    arm_controller.update_gripper_status()

    # --- Object Detection Display (Optional) ---
    if should_update_display:
        object_mask, depth_image, detected = draw_detected_objects()
        # if detected: # Object position calculation moved inside handle_capture/grasp logic
        #     obj_pos = image_tools.get_object_coord(object_mask, depth_image, o3d_intrinsics)
        #     robot_pos = arm_controller.convert_camera_coord_to_robot_coord(obj_pos)
        #     # print(f"Object Position: {obj_pos}")
        #     # print(f"Robot Relative Position: {robot_pos}")


# End of main loop (robot.step returned -1)
print("Simulation ended.")