"""lab5 controller."""

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75  # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# Create the Robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

#import lab5_joint

# targets
target_item = "orange"
target_object_reference = None  # Store the target object once found

#verbose
vrb = True

PICK_STATE_APPROACH = 0
PICK_STATE_ALIGN = 1
PICK_STATE_CALCULATE_IK = 2
PICK_STATE_MOVE_ARM = 3
PICK_STATE_GRIP = 4
PICK_STATE_RAISE_ARM = 5
PICK_STATE_BACKUP = 6
PICK_STATE_DONE = 7

pick_state = PICK_STATE_APPROACH
last_target = None
ik_results = None
arm_move_start_time = 0
grip_start_time = 0
raise_arm_start_time = 0
target_found = False
objects_list = []
raised_arm_ik = None

## fix file paths
################ v [Begin] Do not modify v ##################

base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]

my_chain = Chain.from_urdf_file("tiago_urdf.urdf", base_elements=base_elements)

print(my_chain.links)

part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
            "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
            "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# Dictionary to keep track of motor objects by name
motor_dict = {}

# First pass: disable fixed links and any links not in part_names
for link_id in range(len(my_chain.links)):
    link = my_chain.links[link_id]
    if link.joint_type == "fixed" or link.name not in part_names:
        print("Disabling {}".format(link.name))
        my_chain.active_links_mask[link_id] = False
        
# Initialize the arm motors and encoders.
motors = []
for link_id in range(len(my_chain.links)):
    link = my_chain.links[link_id]
    if link.name in part_names and my_chain.active_links_mask[link_id]:
        try:
            motor = robot.getDevice(link.name)
            
            # Set appropriate velocity
            if link.name == "torso_lift_joint":
                motor.setVelocity(0.07)
            else:
                motor.setVelocity(1)
                
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(timestep)
            motors.append(motor)
            motor_dict[link.name] = motor
            print(f"Enabled motor for {link.name}")
        except:
            print(f"Could not get device for {link.name}")
            my_chain.active_links_mask[link_id] = False

print("Motors in dictionary:", list(motor_dict.keys()))

# ------------------------------------------------------------------
# Helper Functions

def rotate_y(x,y,z,theta):
    new_x = x*np.cos(theta) + y*np.sin(theta)
    new_z = z
    new_y = y*-np.sin(theta) + x*np.cos(theta)
    return [-new_x, new_y, new_z]

def lookForTarget(recognized_objects):
    """Look for target in recognized objects list.
    Returns True if target found and within range."""
    global target_object_reference
    
    if len(recognized_objects) > 0:
        for item in recognized_objects:
            if target_item in str(item.getModel()):
                target_object_reference = item  # Store the target object reference
                target = recognized_objects[0].getPosition()
                dist = abs(target[2])

                if dist < 5:
                    return True
    
    # Return None to preserve original behavior when target not found
    return None

def checkArmAtPosition(ikResults, cutoff=0.00005):
    '''Checks if arm at position, given ikResults'''
    
    # Calculate the arm error only for links with motors
    arm_error = 0
    count = 0
    
    for i in range(len(ikResults)):
        link_name = my_chain.links[i].name
        if link_name in motor_dict:
            current_pos = motor_dict[link_name].getPositionSensor().getValue()
            arm_error += (current_pos - ikResults[i])**2
            count += 1
            
    if count > 0:
        arm_error = math.sqrt(arm_error / count)
    
    if arm_error < cutoff:
        if vrb:
            print("Arm at position.")
        return True
    return False

def moveArmToTarget(ikResults):
    '''Moves arm given ikResults'''
    # Set the robot motors for active links only
    for i in range(len(ikResults)):
        link_name = my_chain.links[i].name
        if link_name in motor_dict:
            motor_dict[link_name].setPosition(ikResults[i])
            
def calculateIk(offset_target, orient=True, orientation_mode="Y", target_orientation=[0,0,1]):
    '''
    This will calculate the IK given a target in robot coords
    Parameters
    ----------
    param offset_target: a vector specifying the target position of the end effector
    param orient: whether or not to orient, default True
    param orientation_mode: either "X", "Y", or "Z", default "Y"
    param target_orientation: the target orientation vector, default [0,0,1]

    Returns
    ----------
    rtype: array
        returns: ikResults array of joint angles
    '''
    print(f"Calculating IK for target: {offset_target}")
    
    # Get the number of links in the chain
    num_links = len(my_chain.links)

    # Create initial position array with the correct size
    initial_position = np.zeros(num_links)

    # Fill initial positions from actual motor positions
    for i in range(num_links):
        link_name = my_chain.links[i].name
        if link_name in motor_dict:
            initial_position[i] = motor_dict[link_name].getPositionSensor().getValue()
    
    # Check and enforce joint limits before calling IK
    for i in range(num_links):
        link = my_chain.links[i]
        if hasattr(link, 'bounds'):
            lower_bound, upper_bound = link.bounds
            if lower_bound is not None and initial_position[i] < lower_bound:
                print(f"Joint {link.name} initial position {initial_position[i]} below bound {lower_bound}, adjusting")
                initial_position[i] = lower_bound
            if upper_bound is not None and initial_position[i] > upper_bound:
                print(f"Joint {link.name} initial position {initial_position[i]} above bound {upper_bound}, adjusting")
                initial_position[i] = upper_bound

    try:
        # Calculate IK
        ikResults = my_chain.inverse_kinematics(
            offset_target, 
            initial_position=initial_position,
            target_orientation=target_orientation if orient else None, 
            orientation_mode=orientation_mode if orient else None
        )

        # Validate result
        position = my_chain.forward_kinematics(ikResults)
        squared_distance = math.sqrt(
            (position[0, 3] - offset_target[0])**2 + 
            (position[1, 3] - offset_target[1])**2 + 
            (position[2, 3] - offset_target[2])**2
        )
        print(f"IK calculated with error - {squared_distance}")
        
        # Ensure the solution is within joint limits
        for i in range(num_links):
            link = my_chain.links[i]
            if hasattr(link, 'bounds'):
                lower_bound, upper_bound = link.bounds
                if lower_bound is not None and ikResults[i] < lower_bound:
                    ikResults[i] = lower_bound
                if upper_bound is not None and ikResults[i] > upper_bound:
                    ikResults[i] = upper_bound
        
        return ikResults
    except Exception as e:
        print(f"IK calculation failed: {e}")
        # Return a safe default position or None on failure
        return None

def getTargetFromObjects(recognized_objects):
    ''' Gets a target vector from a list of recognized objects '''
    # Use cached target if available, otherwise use first object
    target_obj = target_object_reference if target_object_reference is not None else recognized_objects[0]
    target = target_obj.getPosition()

    # Show the raw camera position before any transformations
    print(f"Raw camera position: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")

    # Convert camera coordinates to IK/Robot coordinates
    # These offsets apply transformations for the arm movement
    # offset_target = [target[0]-0.06, (target[1])+0.97+0.2, (target[2])-0.22]
    
    return target

def closeGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.0)

    # r_error = abs(robot.getDevice("gripper_right_finger_joint").getPositionSensor().getValue() - 0.01)
    # l_error = abs(robot.getDevice("gripper_left_finger_joint").getPositionSensor().getValue() - 0.01)
    
    # print("ERRORS")
    # print(r_error)
    # print(l_error)

    # if r_error+l_error > 0.0001:
    #     return False
    # else:
    #     return True

def openGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.045)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.045)

    # r_error = abs(robot.getDevice("gripper_right_finger_joint").getPositionSensor().getValue() - 0.045)
    # l_error = abs(robot.getDevice("gripper_left_finger_joint").getPositionSensor().getValue() - 0.045)

    # if r_error+l_error > 0.0001:
    #     return False
    # else:
    #     return True

def moveHeadDown(amount=0.1, message="Moving head down"):
    """Move the robot's head down by a specified amount.
    
    Args:
        amount: How much to move the head down (positive value)
        message: Message to print
    
    Returns:
        The new head position
    """
    current_head_pos = robot_parts[0].getPositionSensor().getValue()
    new_head_pos = current_head_pos - amount  # Move down (negative is down for this joint)
    new_head_pos = max(new_head_pos, -1.0)  # Don't exceed limit
    robot_parts[0].setPosition(new_head_pos)
    print(f"{message}: {new_head_pos}")
    return new_head_pos

################ v [Begin] Do not modify v ##################


# The Tiago robot has multiple motors, each identified by their names below
part_names = (
    "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
    "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", 
    "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint"
)

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, "inf", "inf")
robot_parts = []

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)
    # Enable position sensors
    position_sensor = robot_parts[i].getPositionSensor()
    position_sensor.enable(timestep)

# Set up the sensors
range_finder = robot.getDevice("range-finder")
range_finder.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map
display = robot.getDevice("display")

# Set up robot state variables
pose_x = 0
pose_y = 0
pose_theta = 0
vL = 0
vR = 0
furthest_point_so_far = 0
goal_reached = False
object_positions = [(-2.28, -9.85, 0.5), (-6.96, -6.14, 0.84)]
# object_positions = [(-2.22, -3.99, 0.81), (-6.96, -6.14, 0.84)]
start_ws = (0, 0)
end_ws = [(-1.4, -9.7), (-6.92, -5.23)]
# end_ws = [(-2.9, -3.99), (-6.92, -5.23)]
object_of_interest = 0
wait_timer = 0
backup_distance = 0
backup_phase = False

# Set up LIDAR
lidar_sensor_readings = []
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE / 2.0, +LIDAR_ANGLE_RANGE / 2.0, LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets) - 83]

# Set mode
mode = "picknplace"  # Options: "manual", "planner", "autonomous", "picknplace"

probability_step = 5e-3

# Initialize map
map = np.zeros(shape=[360, 360])
waypoints = []
convolved_map = None

class RobotController:
    @staticmethod
    def load_map():
        """Load a saved map from disk."""
        global map
        map = np.load("map.npy").astype(np.float32)
        map = np.transpose(map)
        
        for x in range(0, 360):
            for y in range(0, 360):
                if map[x, y] == 1:
                    display.setColor(0xFFFFFF)
                    display.drawPixel(x, y)
    
    @staticmethod
    def save_map():
        """Save the current map to disk."""
        global map
        filtered_map = map > 0.8
        np.save("map.npy", filtered_map)
        print("Map file saved")
    
    @staticmethod
    def world_to_map(point):
        """Convert world coordinates (meters) to map coordinates (pixels)."""
        x = 360 - abs(int(point[0] * 30))
        y = abs(int(point[1] * 30))
        return x, y
    
    @staticmethod
    def map_to_world(point):
        """Convert map coordinates (pixels) to world coordinates (meters)."""
        x = (point[0] / 30) - 12
        y = -(point[1] / 30)
        return x, y
    
    @staticmethod
    def waypoints_to_world(waypoints):
        waypoints_w = []
        for point in waypoints:
            world_x = (point[0] / 30) - 12  # x increases from left to right in both systems
            world_y = -(point[1] / 30)  # y increases downward in map but upward in world
            waypoints_w.append((world_x, world_y))
        return np.array(waypoints_w)
    
    @staticmethod
    def create_configuration_space():
        """Create the configuration space by dilating obstacles."""
        global convolved_map
        convolved_map = convolve2d(map, np.ones((19, 19)), mode="same", boundary="fill", fillvalue=0)
        convolved_map = convolved_map > 0.5
        convolved_map = np.transpose(convolved_map)
        return convolved_map
    
    @staticmethod
    def get_closest_valid_point(map, point):
        """Find the closest valid (non-obstacle) point to the given point on the map."""
        x, y = int(point[0]), int(point[1])
        
        # If the point is already valid, return it
        if 0 <= x < map.shape[1] and 0 <= y < map.shape[0] and map[y, x] == 0:
            return (x, y)
        
        # Search in expanding circles
        max_radius = 50
        for radius in range(1, max_radius):
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if abs(i) == radius or abs(j) == radius:
                        nx, ny = x + i, y + j
                        if (0 <= nx < map.shape[1] and 0 <= ny < map.shape[0] and 
                            map[ny, nx] == 0):
                            return (nx, ny)
        
        return None
    
    @staticmethod
    def path_planner(map, start, end):
        """Plan a path using A* algorithm."""
        # Check if start or end coordinates are out of bounds
        height, width = map.shape
        
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            print(f"Start position {start} is out of bounds for map of size {width}x{height}")
            return []
        if not (0 <= end[0] < width and 0 <= end[1] < height):
            print(f"End position {end} is out of bounds for map of size {width}x{height}")
            return []
        
        # Check if start or end is in an obstacle
        if map[start[1], start[0]] > 0:
            print(f"Start position {start} is in an obstacle")
            return []
        if map[end[1], end[0]] > 0:
            print(f"End position {end} is in an obstacle")
            return []
        
        # A* algorithm implementation
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        
        # Define possible movements (8-connected grid)
        movements = [
            (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1),  # 4-connected
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)  # diagonals
        ]
        
        open_set = {start}
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            for dx, dy, cost in movements:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds
                if (neighbor[0] < 0 or neighbor[0] >= width or 
                    neighbor[1] < 0 or neighbor[1] >= height):
                    continue
                
                # Skip if in closed set
                if neighbor in closed_set:
                    continue
                
                # Skip if it's an obstacle
                if map[neighbor[1], neighbor[0]] > 0:
                    continue
                
                tentative_g_score = g_score[current] + cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float("inf")):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
        
        print("No path found")
        return []
    
    @staticmethod
    def initialise_path(map, start_ws, end_ws):
        """Initialize a path from start to end in world coordinates."""
        start = RobotController.world_to_map(start_ws)
        end = RobotController.world_to_map(end_ws)
        
        end_point = RobotController.get_closest_valid_point(map, end)
        
        if end_point is None:
            print("No valid end point found")
            return [], []
        
        path = RobotController.path_planner(map, start, end_point)
        if len(path) == 0:
            print("No path found")
            return [], []
        
        waypoints = np.array(path)
        waypoints_w = RobotController.waypoints_to_world(waypoints)
        
        # Draw the path on the map
        initial_color = 0xA00000
        for i, point in enumerate(waypoints):
            display.setColor(initial_color + i * 2)
            display.drawPixel(int(point[0]), int(point[1]))
        
        return waypoints, waypoints_w
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    @staticmethod
    def clip_angle(angle):
        """Clip angle to [0, 2π]"""
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < - math.pi:
            angle += 2 * math.pi
        return angle
    
    @staticmethod
    def find_closest_point_in_path(path, pose_x, pose_y):
        """Find the index of the closest waypoint."""
        min_distance = float("inf")
        index = 0
        
        for i, point in enumerate(path):
            distance = math.sqrt((point[0] - pose_x) ** 2 + (point[1] - pose_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                index = i
        
        return index
    
    @staticmethod
    def turn_to_direction(pose_theta, target_theta, speed):
        """Turn to face a specific direction."""
        # Calculate angle difference
        angle_diff = target_theta - pose_theta
        
        # Normalize the angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Determine turn direction
        if angle_diff > 0:  # Need to turn left
            vL = -speed
            vR = speed
        else:  # Need to turn right
            vL = speed
            vR = -speed
        
        # Scale speed based on how close to target angle
        scale_factor = min(1.0, abs(angle_diff) / 0.1)
        vL *= scale_factor
        vR *= scale_factor
        
        return vL, vR
    
    @staticmethod
    def follow_path_controller(pose_x, pose_y, pose_theta, waypoints_w, furthest_point_so_far):
        """Path following controller."""
        vL, vR = 0, 0
        max_turn_speed = MAX_SPEED / 4
        max_speed = MAX_SPEED
        
        # Look ahead for smoother trajectory
        lookahead = 4
        
        # Find next waypoint
        index = np.clip(
            RobotController.find_closest_point_in_path(waypoints_w, pose_x, pose_y) + lookahead,
            0,
            len(waypoints_w) - 1
        )
        
        if index > furthest_point_so_far:
            furthest_point_so_far = index
            
        closest_point = waypoints_w[furthest_point_so_far]
        
        # Calculate error
        rho = np.linalg.norm(np.array(closest_point) - np.array([pose_x, pose_y]))
        
        dx = closest_point[0] - pose_x
        dy = closest_point[1] - pose_y
        desired_theta = RobotController.clip_angle(math.atan2(dy, dx) - np.pi / 2)
        
        # Calculate angle difference
        angle_diff = desired_theta - pose_theta
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        alpha = abs(angle_diff)
        
        # print(f"dx: {dx}, dy: {dy}")
        # print(f"Desired theta: {np.degrees(desired_theta)} degrees")
        # print(f"Current theta: {np.degrees(pose_theta)} degrees")
        # print(f"Alpha: {np.degrees(alpha)} degrees")
        # print(f"Distance to closest point: {rho}")
        
        # Controller logic
        if alpha > 0.5:  # Large error - pure rotation
            vL, vR = RobotController.turn_to_direction(pose_theta, desired_theta, max_turn_speed)
            
            # Add correction for caster wheel drift
            correction = 0.2 * np.sign(angle_diff)
            vL += correction
            vR -= correction
        elif alpha > 0.1:  # Moderate error - blend turning and forward motion
            blend_factor = 1.0 - (alpha - 0.1) / 0.4
            
            turn_vL, turn_vR = RobotController.turn_to_direction(pose_theta, desired_theta, max_turn_speed)
            
            forward_speed = max_speed * 0.5 * blend_factor
            
            vL = turn_vL * (1 - blend_factor) + (forward_speed - 0.3 * angle_diff) * blend_factor
            vR = turn_vR * (1 - blend_factor) + (forward_speed + 0.3 * angle_diff) * blend_factor
        else:  # Well-aligned - forward motion with steering
            if rho > 0.05:
                base_speed = min(max_speed, max_speed * (rho / 1.0) * 2)
                steering = 0.3 * angle_diff
                
                vL = base_speed - steering
                vR = base_speed + steering
        
        return vL, vR, furthest_point_so_far
    
    @staticmethod
    def picknplace_sequence(waypoints_w, object_pos, pose_theta):
        """Pick and place sequence controller with state machine."""
        global wait_timer, backup_distance, backup_phase, pick_state
        global last_target, ik_results, arm_move_start_time, grip_start_time, raise_arm_start_time
        global target_found, objects_list, raised_arm_ik
        
        # robot_parts[0].setPosition(-0.75)  # Set head_2_joint to look downwards
        
        # debug_coordinates()
        
        # State machine for pick and place
        if pick_state == PICK_STATE_APPROACH:
            # Calculate angle to goal
            dx = object_pos[0] - waypoints_w[-1][0]
            dy = object_pos[1] - waypoints_w[-1][1]
            angle_to_goal = math.atan2(dy, dx) - np.pi / 2
            angle_to_goal = RobotController.clip_angle(angle_to_goal)
            
            alpha = abs(RobotController.clip_angle(angle_to_goal - pose_theta))
            print(f"Approach state, alpha: {alpha}")
            
            # If aligned with object, transition to ALIGN state
            if alpha < 0.05:
                pick_state = PICK_STATE_ALIGN
                return 0, 0, False
            
            # Turn to face object
            new_vL, new_vR = RobotController.turn_to_direction(pose_theta, angle_to_goal, MAX_SPEED / 4)
            return new_vL, new_vR, False
        
        elif pick_state == PICK_STATE_ALIGN:
            waiting_complete = False
            if wait_timer < 50:
                wait_timer += 1
            else:
                wait_timer = 0
                waiting_complete = True
                
            if not waiting_complete:
                return 0, 0, False
            
            # Check if we can see the target object
            if camera.getRecognitionNumberOfObjects() > 0:
                objects_list = camera.getRecognitionObjects()
                target_found = lookForTarget(objects_list)
                
                if target_found:
                    # Get vertical position of object in camera frame
                    # We know target_object_reference is valid here
                    target_z = target_object_reference.getPosition()[2]
                    
                    # Calculate how centered the target is (0 would be perfectly centered)
                    # The camera frame typically has y values from -1 to 1
                    target_offset = -target_z

                    print(f"target_offset: {target_offset}")
                    
                    current_head_pos = robot_parts[0].getPositionSensor().getValue()
                    
                    # Accept a much wider range of positions - as long as object is reasonably in frame
                    if abs(target_offset) < 0.11:
                        # Target is sufficiently in view
                        pick_state = PICK_STATE_CALCULATE_IK
                        print("Target in view, calculating IK")
                        return 0, 0, False
                    else:
                        # Adjust head position to keep target in frame
                        adjustment = -target_offset * 0.1  # Gentler adjustment
                        new_head_pos = current_head_pos + adjustment
                        new_head_pos = max(min(new_head_pos, 0.0), -1.0)  # Limit range
                        robot_parts[0].setPosition(new_head_pos)
                        print(f"Adjusting view: offset={target_offset}, adjusting head to {new_head_pos}")
                        return 0, 0, False
                else:
                    # No target found, try moving head more downward
                    moveHeadDown(0.1, "Target not found, moving head down to")
                    return 0, 0, False
            else:
                # No objects detected, try moving head more downward
                moveHeadDown(0.1, "No objects detected, moving head down to")
                return 0, 0, False
            
            # This return is now unreachable and redundant
            # return 0, 0, False
        
        elif pick_state == PICK_STATE_CALCULATE_IK:
            print(f"target_found: {target_found}, objects_list: {objects_list}")

            if target_found and objects_list:
                target = getTargetFromObjects(objects_list)

                # Convert target from robot coordinates to world coordinates
                # Get current robot position and orientation
                robot_pos = [pose_x, pose_y]
                robot_theta = pose_theta
                
                print(f"Robot position: {robot_pos}, orientation: {robot_theta:.3f} rad ({math.degrees(robot_theta):.1f}°)")
                
                # ==================== CAMERA TO WORLD TRANSFORM ====================
                # The camera target is in the robot's local coordinate frame
                # But different from the standard orientation
                
                # 1. First convert the robot coordinate frame to a standard orientation
                # In this robot, the camera/target frame has:
                # - X axis pointing forward from the robot
                # - Z axis pointing to the right
                # - Y axis pointing upward
                
                # Print the raw target for debugging
                print(f"Raw target (robot frame): x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f}")
                
                # 2. Calculate the robot's heading in world frame
                # The compass gives us the negative of the angle between the robot's x-axis and world's x-axis
                # But our rotations need to be relative to world coordinates
                
                # Determine rotation angle based on robot orientation in world frame
                world_heading = pose_theta + np.pi/2  # Adjust by π/2 because robot forward is along x-axis
                
                # 3. Perform the coordinate transformation
                # The target is relative to robot origin, so we need to:
                # a. Perform proper rotation based on robot's heading in world frame
                # b. Add the robot's world position
            
                head_tilt = robot_parts[0].getPositionSensor().getValue()  # Get head tilt angle
                cos_tilt = math.cos(head_tilt)
                sin_tilt = math.sin(head_tilt)

                # First rotate based on tilt (around x-axis)
                target_rotated_x = target[0]
                target_rotated_y = target[1] * cos_tilt - target[2] * sin_tilt
                target_rotated_z = target[1] * sin_tilt + target[2] * cos_tilt

                # Then rotate based on robot heading (around z-axis/yaw)
                x_world = robot_pos[0] + target_rotated_x * math.cos(world_heading) - target_rotated_z * math.sin(world_heading)
                y_world = robot_pos[1] + target_rotated_x * math.sin(world_heading) + target_rotated_z * math.cos(world_heading)
                
                # For height, use expected height from the object_positions
                # This works better than computed height which has significant error
                z_world = object_positions[object_of_interest][2]
                
                # Store calculated world position
                target_world_x = x_world
                target_world_y = y_world
                target_world_z = z_world
                
                # Print diagnostics
                print(f"World heading: {world_heading:.3f} rad ({math.degrees(world_heading):.1f}°)")
                print(f"Calculated world coordinates: ({target_world_x:.3f}, {target_world_y:.3f}, {target_world_z:.3f})")
                print(f"Expected object position: {object_positions[object_of_interest]}")
                
                # Calculate error between computed and expected position for debugging
                expected_pos = object_positions[object_of_interest]
                position_error = math.sqrt((target_world_x - expected_pos[0])**2 + 
                                         (target_world_y - expected_pos[1])**2)
                print(f"Position error: {position_error:.3f} meters")
                
                # 4. Continue with IK calculation using the original target
                # The IK function already properly handles the robot-frame coordinates
                if last_target is None or np.linalg.norm(np.array(target) - np.array(last_target)) > 0.01:
                    print(f"Calculating IK for target: {target}")
                    ik_results = calculateIk(target)
                    last_target = target
                
                print(f"IK results: {ik_results}")

                pick_state = PICK_STATE_MOVE_ARM
                arm_move_start_time = robot.getTime()

                # exit()
                
                return 0, 0, False
            else:
                raise Exception("Reached this state but no target found")
            # Only calculate IK once
            # objects_list = camera.getRecognitionObjects()
            # if lookForTarget(objects_list):
            #     # Get current target using existing function which handles coordinate transforms
            #     target = getTargetFromObjects(objects_list)
                
            #     # Only recalculate if target changed significantly
            #     if last_target is None or np.linalg.norm(np.array(target) - np.array(last_target)) > 0.01:
            #         print(f"Calculating IK for target: {target}")
            #         ik_results = calculateIk(target)
            #         last_target = target
                
            #     pick_state = PICK_STATE_MOVE_ARM
            #     arm_move_start_time = robot.getTime()
            
            # return 0, 0, False
        
        elif pick_state == PICK_STATE_MOVE_ARM:
            # Execute arm movement if we have IK results
            if ik_results is not None:
                moveArmToTarget(ik_results)
                
                # Check if arm is at position or if timeout reached (3 seconds)
                if checkArmAtPosition(ik_results) or (robot.getTime() - arm_move_start_time > 3.0):
                    print("Arm in position, moving to grip state")
                    pick_state = PICK_STATE_GRIP
                    grip_start_time = robot.getTime()
            
            return 0, 0, False
        
        elif pick_state == PICK_STATE_GRIP:
            # Close gripper
            closeGrip()
            
            # Wait a bit for gripper to close (1 second)
            current_time = robot.getTime()
            time_in_grip = current_time - grip_start_time
            print(f"In GRIP state, current time: {current_time:.2f}, start time: {grip_start_time:.2f}, elapsed: {time_in_grip:.2f} seconds")
            
            if time_in_grip > 1.0:
                print(f"Transitioning from GRIP to RAISE_ARM state after {time_in_grip:.2f} seconds")
                pick_state = PICK_STATE_RAISE_ARM
                raise_arm_start_time = robot.getTime()
                # We'll calculate the raised arm position in the RAISE_ARM state
            
            return 0, 0, False
        
        elif pick_state == PICK_STATE_RAISE_ARM:
            # Move the arm to the raised position
            if raised_arm_ik is None:
                # Calculate a safer arm position for backing up
                # Start with current joint positions
                joint_positions = []
                for link_id in range(len(my_chain.links)):
                    link_name = my_chain.links[link_id].name
                    if link_name in motor_dict:
                        joint_positions.append(motor_dict[link_name].getPositionSensor().getValue())
                    else:
                        joint_positions.append(0.0)  # Default for non-motor joints
                
                # Modify key joints to a safer position
                for i in range(len(my_chain.links)):
                    link_name = my_chain.links[i].name
                    if link_name == "torso_lift_joint":
                        joint_positions[i] = 0.35  # Maximum height for torso
                    elif link_name == "arm_1_joint":
                        joint_positions[i] = 0.07  # Default arm position
                    elif link_name == "arm_2_joint":
                        joint_positions[i] = 0.6  # Tucked in position
                    elif link_name == "arm_3_joint":
                        joint_positions[i] = -2.5  # More vertical
                
                # Store the new position
                raised_arm_ik = joint_positions
                print("Calculated safer arm position for backing up")
            
            print("Moving arm to safer position for backing up")
            moveArmToTarget(raised_arm_ik)
            
            # Check if the arm is at the raised position or if timeout (3 seconds)
            current_time = robot.getTime()
            time_raising = current_time - raise_arm_start_time
            print(f"Adjusting arm position, elapsed time: {time_raising:.2f} seconds")
            
            if checkArmAtPosition(raised_arm_ik, cutoff=0.05) or time_raising > 3.0:
                print("Arm in safe position, transitioning to BACKUP state")
                pick_state = PICK_STATE_BACKUP
                backup_distance = 0
            
            return 0, 0, False
        
        elif pick_state == PICK_STATE_BACKUP:
            # Backup from object - use a stronger negative speed to ensure backward motion
            backup_speed = -MAX_SPEED * 0.75  # Increase backup speed
            print(f"In BACKUP state, distance backed up: {backup_distance:.2f} meters, speed: {backup_speed}")
            
            # Calculate exact distance increment based on timestep
            distance_increment = abs(backup_speed) / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0
            backup_distance += distance_increment
            print(f"Distance increment: {distance_increment:.5f}, new total: {backup_distance:.2f}")
            
            # Check if we've backed up enough
            if backup_distance >= 0.5:
                print("Backup complete, transitioning to DONE state")
                pick_state = PICK_STATE_DONE
                return 0, 0, False
            
            # Make sure we return negative speeds for both wheels to move backward
            # Return the exact backup_speed value rather than relying on later processing
            print(f"Returning wheel velocities: {backup_speed}, {backup_speed}")
            return backup_speed, backup_speed, False
        
        elif pick_state == PICK_STATE_DONE:
            # Reset state for next object
            pick_state = PICK_STATE_APPROACH
            last_target = None
            ik_results = None
            raised_arm_ik = None
            backup_distance = 0
            return 0, 0, True
        
        # Default behavior - should not reach here
        return 0, 0, False

def debug_coordinates():
    """Debug function to show the relationships between different coordinate systems"""
    print("\n======== COORDINATE SYSTEMS DEBUGGING ========")
    
    # 1. World coordinates (from GPS)
    world_pos = gps.getValues()
    print(f"Robot position (world/GPS coords): {world_pos}")

    # 2. Robot heading
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2])) - math.pi / 2)
    print(f"Robot heading: {math.degrees(rad)} degrees")
    
    # 3. End effector position in robot coordinates - works even if ik_results is None
    # Get current joint positions from motors
    joint_positions = []
    for link_id in range(len(my_chain.links)):
        link_name = my_chain.links[link_id].name
        if link_name in motor_dict:
            joint_positions.append(motor_dict[link_name].getPositionSensor().getValue())
        else:
            joint_positions.append(0.0)  # Default for non-motor joints
    
    # Get end effector position from forward kinematics
    end_effector_matrix = my_chain.forward_kinematics(joint_positions)
    end_effector_pos = [end_effector_matrix[0, 3], end_effector_matrix[1, 3], end_effector_matrix[2, 3]]
    print(f"End effector position (robot/IK coords): {end_effector_pos}")
    
    # Calculate world position of end effector
    world_x = world_pos[0] + end_effector_pos[0] * math.cos(rad) - end_effector_pos[1] * math.sin(rad)
    world_y = world_pos[1] + end_effector_pos[0] * math.sin(rad) + end_effector_pos[1] * math.cos(rad)
    world_z = world_pos[2] + end_effector_pos[2]
    print(f"End effector position (world coords): [{world_x}, {world_y}, {world_z}]")
    
    # 4. Target object position
    if camera.getRecognitionNumberOfObjects() > 0:
        objects_list = camera.getRecognitionObjects()
        if lookForTarget(objects_list):
            # Camera coordinates - explicitly get values
            target_obj = objects_list[0]
            target_cam = [target_obj.getPosition()[0], target_obj.getPosition()[1], target_obj.getPosition()[2]]
            print(f"Target position (camera coords): {target_cam}")
            
            # Robot/IK coordinates after transformation
            target_robot = getTargetFromObjects(objects_list)
            print(f"Target position (robot/IK coords): {target_robot}")
            
            # Estimate world coordinates of target
            target_world_x = world_pos[0] + target_robot[0] * math.cos(rad) - target_robot[1] * math.sin(rad)
            target_world_y = world_pos[1] + target_robot[0] * math.sin(rad) + target_robot[1] * math.cos(rad)
            target_world_z = world_pos[2] + target_robot[2]
            print(f"Target position (world coords): [{target_world_x}, {target_world_y}, {target_world_z}]")
            
            # Compare to object_positions
            obj_pos = object_positions[object_of_interest]
            print(f"Target goal position (from object_positions): {obj_pos}")
            
            # Distance between end effector and target
            distance_robot = math.sqrt(
                sum((a - b)**2 for a, b in zip(end_effector_pos, target_robot))
            )
            print(f"Distance between end effector and target (robot coords): {distance_robot}")
            
            # Distance in world coordinates
            distance_world = math.sqrt(
                (world_x - target_world_x)**2 + 
                (world_y - target_world_y)**2 + 
                (world_z - target_world_z)**2
            )
            print(f"Distance between end effector and target (world coords): {distance_world}")
    
    print("============================================\n")
    
# Initialize for chosen mode
if mode in ["autonomous", "picknplace"]:
    RobotController.load_map()
    
    # Wait for valid GPS data
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    while np.isnan(pose_x) or np.isnan(pose_y):
        print("Waiting for GPS data...")
        robot.step(timestep)
        pose_x = gps.getValues()[0]
        pose_y = gps.getValues()[1]
    
    # Create configuration space
    convolved_map = RobotController.create_configuration_space()
    
    if mode == "picknplace":
        start_ws = (pose_x, pose_y)
        end_point_ws = end_ws[object_of_interest]
        waypoints, waypoints_w = RobotController.initialise_path(convolved_map, start_ws, end_point_ws)
    elif mode == "autonomous":
        # Find a random valid goal
        while True:
            end = (np.random.randint(0, 360), np.random.randint(0, 360))
            if convolved_map[end[1], end[0]] == 0:
                break
        
        end_w = RobotController.map_to_world(end)
        start_w = (pose_x, pose_y)
        waypoints, waypoints_w = RobotController.initialise_path(convolved_map, start_w, end_w)
elif mode == "planner":
    RobotController.load_map()
    
    # Wait for valid GPS data
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    while np.isnan(pose_x) or np.isnan(pose_y):
        print("Waiting for GPS data...")
        robot.step(timestep)
        pose_x = gps.getValues()[0]
        pose_y = gps.getValues()[1]
    
    # Create configuration space
    convolved_map = RobotController.create_configuration_space()
    
    # Set start position
    start_w = (pose_x, pose_y)
    start = RobotController.world_to_map(start_w)
    
    # Randomly sample a valid end point
    while True:
        end = (np.random.randint(0, 360), np.random.randint(0, 360))
        if convolved_map[end[1], end[0]] == 0:
            break
    
    # Plan path
    display.setColor(0xFF0000)
    display.drawPixel(start[0], start[1])
    display.drawPixel(end[0], end[1])
    
    path = RobotController.path_planner(convolved_map, start, end)
    waypoints = np.array(path)
    np.save("path.npy", waypoints)
    
    # Display path
    for point in waypoints:
        display.setColor(0x00A000)
        display.drawPixel(int(point[0]), int(point[1]))
    
    # Run simulation
    while robot.step(timestep) != -1:
        display.setColor(0x00FF00)
        display.drawPixel(int(end[0]), int(end[1]))

# Main control loop
while robot.step(timestep) != -1 and mode != "planner":
    # Update robot pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    offset_ = math.pi / 2
    rad = -((math.atan2(n[0], n[2])) - offset_)
    pose_theta = rad
    
    # Update map with LIDAR readings
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings) - 83]
    
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]
        
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue
        
        # Convert to robot-centric coordinates
        rx = math.cos(alpha) * rho
        ry = -math.sin(alpha) * rho
        
        # Convert to world coordinates
        t = pose_theta + np.pi / 2.0
        wx = math.cos(t) * rx - math.sin(t) * ry + pose_x
        wy = math.sin(t) * rx + math.cos(t) * ry + pose_y
        
        # Handle boundary conditions
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Update map
            pixel = (
                max(0, min(359, 360 - abs(int(wx * 30)))),
                max(0, min(359, abs(int(wy * 30))))
            )
            
            pixel_value = map[pixel[0], pixel[1]]
            if pixel_value < 1:
                pixel_value += probability_step
            pixel_value = min(1, pixel_value)
            map[pixel[0], pixel[1]] = pixel_value
            
            # Calculate color value properly
            color = int((pixel_value * 256**2 + pixel_value * 256 + pixel_value) * 255)
            display.setColor(color)
            display.drawPixel(pixel[0], pixel[1])
    
    # Draw robot position on map
    display.setColor(0xFF0000)
    display.drawPixel(360 - abs(int(pose_x * 30)), abs(int(pose_y * 30)))
    
    # Controller logic based on mode
    if mode == "manual":
        key = keyboard.getKey()
        while keyboard.getKey() != -1:
            pass
        
        if key == keyboard.LEFT:
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(" "):
            vL = 0
            vR = 0
        elif key == ord("S"):
            RobotController.save_map()
        elif key == ord("L"):
            RobotController.load_map()
            print("Map loaded")
        else:  # slow down
            vL *= 0.75
            vR *= 0.75
    else:
        # For autonomous and picknplace modes
        distance_to_goal = np.linalg.norm(
            np.array(waypoints_w[-1]) - np.array([pose_x, pose_y])
        )
        
        if goal_reached:
            distance_to_goal = 0
        
        if distance_to_goal < 0.05 and mode != "picknplace":
            vL = 0
            vR = 0
            print("Reached goal")
            break
        elif distance_to_goal < 0.05 and mode == "picknplace":
            goal_reached = True
            
            new_vL, new_vR, sequence_finished = RobotController.picknplace_sequence(
                waypoints_w, object_positions[object_of_interest], pose_theta
            )
            
            print(f"Received from picknplace_sequence: vL={new_vL}, vR={new_vR}, finished={sequence_finished}")
            
            if sequence_finished:
                print(f"Sequence finished for object {object_of_interest}")
                object_of_interest += 1
                if object_of_interest >= len(object_positions):
                    print("Reached all objects")
                    vL = vR = 0
                    break
                
                goal_reached = False
                furthest_point_so_far = 0
                
                # Clear display and reload map
                display.setColor(0x000000)
                for i in range(360):
                    for j in range(360):
                        display.drawPixel(i, j)
                
                RobotController.load_map()
                
                # Plan new path to next object
                waypoints, waypoints_w = RobotController.initialise_path(
                    convolved_map,
                    (pose_x, pose_y),
                    end_ws[object_of_interest]
                )
            
            vL, vR = new_vL, new_vR
            print(f"Setting wheel velocities: vL={vL}, vR={vR}")
        else:
            vL, vR, furthest_point_so_far = RobotController.follow_path_controller(
                pose_x, pose_y, pose_theta, waypoints_w, furthest_point_so_far
            )
    
    # Apply velocity limits
    vL = np.clip(vL, -MAX_SPEED, MAX_SPEED)
    vR = np.clip(vR, -MAX_SPEED, MAX_SPEED)
    
    # Odometry update (even though we use GPS, this is for future use)
    pose_x += (vL + vR) / 2 / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0 * math.cos(pose_theta)
    pose_y -= (vL + vR) / 2 / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0 * math.sin(pose_theta)
    pose_theta += (vR - vL) / AXLE_LENGTH / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0
    
    # Normalize pose_theta
    pose_theta = RobotController.normalize_angle(pose_theta)
    
    # Send commands to motors
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

# Keep controller running to avoid Webots bug on Windows
while robot.step(timestep) != -1:
    pass