from controller import Robot
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import ikpy.utils.plot as plot_utils
import math


# Initialize the robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

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
CAM_POS = (-0.013797,0.137,0.326805)
CAM_WIDTH = 240
CAM_HEIGHT = 135


# targets
target_item_list = ["orange"]


vrb = True
# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

## fix file paths
################ v [Begin] Do not modify v ##################

base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]
my_chain = Chain.from_urdf_file("tiago_urdf.urdf", base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])

print(my_chain.links)

part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
            "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
            "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

for link_id in range(len(my_chain.links)):

    # This is the actual link object
    link = my_chain.links[link_id]
    
    # I've disabled "torso_lift_joint" manually as it can cause
    # the TIAGO to become unstable.
    if link.name not in part_names or  link.name =="torso_lift_joint":
        print("Disabling {}".format(link.name))
        my_chain.active_links_mask[link_id] = False
        
# Initialize the arm motors and encoders.
motors = []
for link in my_chain.links:
    if link.name in part_names and link.name != "torso_lift_joint":
        motor = robot.getDevice(link.name)

        # Make sure to account for any motors that
        # require a different maximum velocity!
        if link.name == "torso_lift_joint":
            motor.setVelocity(0.07)
        else:
            motor.setVelocity(1)
            
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        motors.append(motor)

# ------------------------------------------------------------------
# Helper Functions

def rotate_y(x,y,z,theta):
    new_x = x*np.cos(theta) + y*np.sin(theta)
    new_z = z
    new_y = y*-np.sin(theta) + x*np.cos(theta)
    return [-new_x, new_y, new_z]

def lookForTarget(recognized_objects):
    if len(recognized_objects) > 0:

        for item in recognized_objects:
            if target_item in str(item.get_model()):

                target = recognized_objects[0].get_position()
                dist = abs(target[2])

                if dist < 5:
                    return True

def checkArmAtPosition(ikResults, cutoff=0.00005):
    '''Checks if arm at position, given ikResults'''
    
    # Get the initial position of the motors
    initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]

    # Calculate the arm
    arm_error = 0
    for item in range(14):
        arm_error += (initial_position[item] - ikResults[item])**2
    arm_error = math.sqrt(arm_error)

    if arm_error < cutoff:
        if vrb:
            print("Arm at position.")
        return True
    return False

def moveArmToTarget(ikResults):
    '''Moves arm given ikResults'''
    # Set the robot motors
    for res in range(len(ikResults)):
        if my_chain.links[res].name in part_names:
            # This code was used to wait for the trunk, but now unnecessary.
            # if abs(initial_position[2]-ikResults[2]) < 0.1 or res == 2:
            robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
            if vrb:
                print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))

def calculateIk(offset_target,  orient=True, orientation_mode="Y", target_orientation=[0,0,1]):
    '''
    This will calculate the iK given a target in robot coords
    Parameters
    ----------
    param offset_target: a vector specifying the target position of the end effector
    param orient: whether or not to orient, default True
    param orientation_mode: either "X", "Y", or "Z", default "Y"
    param target_orientation: the target orientation vector, default [0,0,1]

    Returns
    ----------
    rtype: bool
        returns: whether or not the arm is at the target
    '''
    # Get the number of links in the chain
    num_links = len(my_chain.links)

    # Create initial position array with the correct size
    initial_position = [0] * num_links

    # Map each motor to its corresponding link position
    motor_idx = 0
    for i in range(num_links):
        link_name = my_chain.links[i].name
        if link_name in part_names and link_name != "torso_lift_joint":
            if motor_idx < len(motors):
                initial_position[i] = motors[motor_idx].getPositionSensor().getValue()
                motor_idx += 1

    # Calculate IK
    ikResults = my_chain.inverse_kinematics(
        offset_target, 
        initial_position=initial_position,
        target_orientation=target_orientation, 
        orientation_mode=orientation_mode
    )

    # Validate result
    position = my_chain.forward_kinematics(ikResults)
    squared_distance = math.sqrt(
        (position[0, 3] - offset_target[0])**2 + 
        (position[1, 3] - offset_target[1])**2 + 
        (position[2, 3] - offset_target[2])**2
    )
    print(f"IK calculated with error - {squared_distance}")

    return ikResults

    # Legacy code for visualizing
        # import matplotlib.pyplot
        # from mpl_toolkits.mplot3d import Axes3D
        # ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')

        # my_chain.plot(ikResults, ax, target=ikTarget)
        # matplotlib.pyplot.show()
        
def getTargetFromObject(recognized_objects):
    ''' Gets a target vector from a list of recognized objects '''

    # Get the first valid target
    target = recognized_objects[0].get_position()

    # Convert camera coordinates to IK/Robot coordinates
    # offset_target = [-(target[2])+0.22, -target[0]+0.08, (target[1])+0.97+0.2]
    offset_target = [-(target[2])+0.22, -target[0]+0.06, (target[1])+0.97+0.2]

    return offset_target

def reachArm(target, previous_target, ikResults, cutoff=0.00005):
    '''
    This code is used to reach the arm over an object and pick it up.
    '''

    # Calculate the error using the ikTarget
    error = 0
    ikTargetCopy = previous_target

    # Make sure ikTarget is defined
    if previous_target is None:
        error = 100
    else:
        for item in range(3):
            error += (target[item] - previous_target[item])**2
        error = math.sqrt(error)

    
    # If error greater than margin
    if error > 0.05:
        print("Recalculating IK, error too high {}...".format(error))
        ikResults = calculateIk(target)
        ikTargetCopy = target
        moveArmToTarget(ikResults)

    # Exit Condition
    if checkArmAtPosition(ikResults, cutoff=cutoff):
        if vrb:
            print("NOW SWIPING")
        return [True, ikTargetCopy, ikResults]
    else:
        if vrb:
            print("ARM NOT AT POSITION")

    # Return ikResults
    return [False, ikTargetCopy, ikResults]

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
################ v [Begin] Do not modify v ##################
