from ikpy.chain import Chain
import numpy as np
import math

class ArmController:
    def __init__(self, robot, timestep):
        """Initialize the arm controller with robot and configuration."""
        self.robot = robot
        self.timestep = timestep
        
        # Constants for arm control
        self.CARTESIAN_STEP = 0.015  # Step size for arm control in meters
        self.ORIENTATION_STEP = 0.05  # Step size for arm orientation control in radians
        
        # End effector control keys
        self.END_EFFECTOR_CONTROL_KEYS = {
            "up": 'E', "down": 'Q', "left": 'A', "right": 'D', 
            "forward": 'W', "backward": 'S',
            "pitch_up": 'I', "pitch_down": 'K', "roll_left": 'J', 
            "roll_right": 'L', "yaw_left": 'U', "yaw_right": 'O',
            "orient_y": 'Y', "orient_x": "X", "orient_z": "Z"
        }
        
        # Initialize robot arm components
        self.motor_dict = {}
        self.robot_sensors = {}
        self.gripper_status = "opening" # Start as closed, then open
        
        # Set up the IK chain and related components
        self.setup_ik_chain()
        self.setup_motors_and_sensors()
        
        # Initialize gripper to open position and wait for it
        print("Initializing gripper to open...")
        if self.open_gripper(): # Start opening
            # Wait a bit for the gripper to open
            init_wait_steps = 0
            max_init_wait_steps = 150 # Wait up to 50 simulation steps
            while self.gripper_status != "open" and init_wait_steps < max_init_wait_steps:
                self.robot.step(self.timestep)
                self.update_gripper_status() # Check sensors and update status
                init_wait_steps += 1
            if self.gripper_status == "open":
                print("Gripper initialized to open.")
            else:
                print("Warning: Gripper did not fully open during initialization.")
        else:
            print("Error: Failed to start opening gripper during initialization.")
        
    def setup_ik_chain(self):
        """Set up the inverse kinematics chain for the robot arm."""
        # Define base elements
        self.base_elements = [
            "base_link", "base_link_Torso_joint", "Torso", 
            "torso_lift_joint", "torso_lift_link", 
            "torso_lift_link_TIAGo front arm_11367_joint", 
            "TIAGo front arm_11367"
        ]
        
        # Create the IK chain
        try:
            self.my_chain = Chain.from_urdf_file(
                "robot_urdf.urdf", 
                base_elements=self.base_elements
            )
            print("--- IK Chain created successfully ---")
            
            # Disable fixed links and configure active links
            self.configure_chain_links()
            
        except Exception as e:
            print(f"Error setting up IK chain: {e}")
            self.my_chain = None
        
        # Set up camera chain for coordinate transformations
        self.setup_camera_chain()
    
    def configure_chain_links(self):
        """Configure which links in the IK chain should be active."""
        # Define part names (joints) that can be controlled
        self.part_names = (
            "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
            "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint",
            "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint",
            "gripper_left_finger_joint", "gripper_right_finger_joint"
        )
        
        # Disable fixed links and any links not in part_names
        for link_id in range(len(self.my_chain.links)):
            link = self.my_chain.links[link_id]
            is_base_element = link.name in self.base_elements
            if (link.joint_type == "fixed" or 
                (link.name not in self.part_names and not is_base_element)):
                self.my_chain.active_links_mask[link_id] = False
                
        # Lock specific joints for IK
        disable_joint_names = [
            # "torso_lift_joint",  
            "gripper_right_finger_joint", 
            "gripper_left_finger_joint"
        ]
        
        for joint_name in disable_joint_names:
            try:
                link_index = [i for i, link in enumerate(self.my_chain.links) 
                             if link.name == joint_name][0]
                if self.my_chain.active_links_mask[link_index]:
                    self.my_chain.active_links_mask[link_index] = False
            except IndexError:
                print(f"  Warning: Joint '{joint_name}' not found in the IK chain.")

    def setup_camera_chain(self):
        """Set up the camera chain for coordinate transformations."""
        try:
            # Define base elements for the camera chain (up to the head)
            camera_base_elements = self.base_elements[:-2] + [
                "head_1_joint", "head_1_link", "head_2_joint", "head_2_link"
            ]
            
            # Create the chain from the URDF file
            self.camera_chain = Chain.from_urdf_file(
                "robot_urdf.urdf",
                base_elements=camera_base_elements,
            )
            print(f"  Camera IK Chain created from URDF.")
            
        except FileNotFoundError:
            print("!!! ERROR: robot_urdf.urdf not found. Cannot create camera chain.")
            self.camera_chain = None
        except Exception as e:
            print(f"!!! ERROR creating camera IK chain: {e}")
            self.camera_chain = None
    
    def setup_motors_and_sensors(self):
        """Initialize motors and sensors for the robot arm."""
        # Initialize the motors and link them to the IK chain
        for link_id in range(len(self.my_chain.links)):
            link = self.my_chain.links[link_id]
            if (self.my_chain.active_links_mask[link_id] and 
                link.joint_type != "fixed" and 
                link.name in self.part_names):
                try:
                    motor = self.robot.getDevice(link.name)
                    position_sensor = motor.getPositionSensor()
                    
                    if not position_sensor:
                        print(f"  Warning: No position sensor for motor '{link.name}', disabling link.")
                        self.my_chain.active_links_mask[link_id] = False
                        continue
                    
                    # Enable sensor
                    if position_sensor.getSamplingPeriod() <= 0:
                        position_sensor.enable(self.timestep)
                    
                    # Set appropriate velocity
                    if link.name == "torso_lift_joint":
                        motor.setVelocity(0.07)
                        # Ensure torso lift stays up at safe height
                        motor.setPosition(0.35)
                    else:
                        motor.setVelocity(motor.getMaxVelocity() * 0.8)
                    
                    self.motor_dict[link.name] = motor
                    
                except Exception as e:
                    print(f"  Error getting device/sensor for link '{link.name}': {e}. Disabling link.")
                    self.my_chain.active_links_mask[link_id] = False
        
        # Initialize and enable Position Sensors
        for part_name in self.part_names:
            # Skip wheel joints
            if "wheel" in part_name:
                continue
                
            sensor_name = part_name + "_sensor"
            sensor = self.robot.getDevice(sensor_name)
            if sensor:
                sensor.enable(self.timestep)
                self.robot_sensors[part_name] = sensor
                
        # Enable gripper encoders (position sensors)
        self.left_gripper_enc = self.robot.getDevice("gripper_left_finger_joint_sensor")
        self.right_gripper_enc = self.robot.getDevice("gripper_right_finger_joint_sensor")
        self.left_gripper_enc.enable(self.timestep)
        self.right_gripper_enc.enable(self.timestep)
    
    # Core IK helper methods
    def get_current_ik_joint_state(self, sensor_bound_tolerance=0.01):
        """
        Retrieves the current joint positions for the active links in the IK chain.
        
        Args:
            sensor_bound_tolerance: Tolerance for checking if sensor values are outside bounds.
            
        Returns:
            np.ndarray or None: Current joint angles or None if reading fails.
        """
        num_links = len(self.my_chain.links)
        current_joint_state = np.zeros(num_links)
        valid_state = True
        
        for i in range(num_links):
            if self.my_chain.active_links_mask[i]:
                link = self.my_chain.links[i]
                link_name = link.name
                if link_name in self.motor_dict:
                    try:
                        sensor = self.motor_dict[link_name].getPositionSensor()
                        if sensor is None:
                            print(f"!!! WARNING: No position sensor found for active motor '{link_name}'")
                            continue
                            
                        sensor_value = sensor.getValue()
                        
                        if not np.isfinite(sensor_value):
                            print(f"!!! ERROR: Non-finite sensor value ({sensor_value}) for {link_name}")
                            valid_state = False
                            break
                            
                        lower_bound, upper_bound = link.bounds
                        
                        if (lower_bound is not None and 
                            sensor_value < lower_bound - sensor_bound_tolerance) or \
                           (upper_bound is not None and 
                            sensor_value > upper_bound + sensor_bound_tolerance):
                            print(f"!!! WARNING: Sensor value {sensor_value:.4f} for {link_name} is outside bounds")
                            
                        if lower_bound is not None and upper_bound is not None:
                            current_joint_state[i] = np.clip(sensor_value, lower_bound, upper_bound)
                        elif lower_bound is not None:
                            current_joint_state[i] = max(sensor_value, lower_bound)
                        elif upper_bound is not None:
                            current_joint_state[i] = min(sensor_value, upper_bound)
                        else:
                            current_joint_state[i] = sensor_value
                            
                    except Exception as e:
                        print(f"Error reading sensor for {link_name}: {e}")
                        valid_state = False
                        break
        
        if not valid_state:
            return None
        else:
            return current_joint_state
            
    def rotation_matrix(self, axis, angle):
        """Creates a 3x3 rotation matrix for a given axis and angle."""
        c = np.cos(angle)
        s = np.sin(angle)
        
        if axis == 'x':  # Roll
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'y':  # Pitch
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == 'z':  # Yaw
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            print(f"Warning: Invalid rotation axis '{axis}'. Returning identity matrix.")
            return np.identity(3)
            
    def check_arm_at_position(self, ik_results, cutoff=0.01):
        """Checks if arm is close to the target position defined by ikResults."""
        arm_error = 0
        count = 0
        
        for i in range(len(ik_results)):
            # Only check active links that have a corresponding motor
            if self.my_chain.active_links_mask[i]:
                link_name = self.my_chain.links[i].name
                if link_name in self.motor_dict:
                    current_pos = self.motor_dict[link_name].getPositionSensor().getValue()
                    arm_error += (current_pos - ik_results[i])**2
                    count += 1
        
        if count > 0:
            arm_error = math.sqrt(arm_error / count)
            
        return arm_error < cutoff
        
    def move_arm_to_target(self, ik_results):
        """Commands the arm motors to the positions specified in ikResults."""
        # Set the robot motors for active links only
        for i in range(len(ik_results)):
            if self.my_chain.active_links_mask[i]:
                link_name = self.my_chain.links[i].name
                if link_name in self.motor_dict:
                    self.motor_dict[link_name].setPosition(ik_results[i])
    
    def calculate_ik(self, target_position, initial_position=None, orient=False, 
                      orientation_mode="Y", target_orientation=None):
        """Calculates IK for a target position, using provided initial joint angles."""
        print(f"Calculating IK for target: {target_position}")
        
        # If no initial position provided, get current state
        if initial_position is None:
            initial_position = self.get_current_ik_joint_state()
            if initial_position is None:
                print("Failed to get current joint state for IK calculation")
                return None
        
        try:
            # Calculate IK using the provided initial position
            ik_results = self.my_chain.inverse_kinematics(
                target_position,
                initial_position=initial_position,
                target_orientation=target_orientation if orient else None,
                orientation_mode=orientation_mode if orient else None
            )
            
            return ik_results
            
        except ValueError as e:
            print(f"IK calculation failed: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during IK calculation: {e}")
            return None
            
    def wait_for_arm_movement(self, ik_results, max_wait_steps=100, cutoff=0.02, 
                               description="position"):
        """Helper function to wait for arm to reach a target position.
        
        Args:
            ik_results: The IK solution to check against
            max_wait_steps: Maximum steps to wait
            cutoff: Position tolerance
            description: Description of the movement for logging
            
        Returns:
            True if position reached, False if timeout
        """
        print(f"  Waiting for arm to reach {description}...")
        wait_steps = 0
        
        while wait_steps < max_wait_steps:
            self.robot.step(self.timestep)
            if self.check_arm_at_position(ik_results, cutoff=cutoff):
                print(f"  Arm reached {description}.")
                return True
            wait_steps += 1
        
        print(f"  Timeout waiting for arm to reach {description}.")
        return False
        
    def move_arm_with_ik(self, target_position, initial_position=None, orientation=None, 
                          orientation_mode="all", max_wait=100, cutoff=0.02, 
                          description="position", must_succeed=True):
        """Helper function to calculate IK, move arm, and wait for completion.
        
        Args:
            target_position: Target position for end effector
            initial_position: Initial joint state (if None, uses current state)
            orientation: Target orientation matrix (if None, orientation not controlled)
            orientation_mode: Orientation control mode
            max_wait: Maximum wait steps
            cutoff: Position tolerance
            description: Description for logging
            must_succeed: If True, return False on failure
            
        Returns:
            (success, ik_results) tuple: success is True if move succeeded, 
                                        ik_results is the calculated IK solution or None
        """
        # If no initial position provided, get current state
        if initial_position is None:
            initial_position = self.get_current_ik_joint_state()
            if initial_position is None and must_succeed:
                print("Failed to get current joint state")
                return False, None
        
        # Calculate IK
        ik_results = self.calculate_ik(
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
        self.move_arm_to_target(ik_results)
        
        # Wait for completion
        success = self.wait_for_arm_movement(
            ik_results,
            max_wait_steps=max_wait,
            cutoff=cutoff,
            description=description
        )
        
        if not success and must_succeed:
            return False, ik_results
        
        return True, ik_results
    
    def convert_camera_coord_to_robot_coord(self, obj_pos):
        """Converts object coordinates from camera frame to robot base frame."""
        if self.camera_chain is None:
            print("Error: Camera chain is not initialized.")
            return None
            
        try:
            # 1. Get current head joint angles
            head_1_angle = self.robot_sensors["head_1_joint"].getValue()
            head_2_angle = self.robot_sensors["head_2_joint"].getValue()
            
            # 2. Construct the joint state for the camera chain
            num_cam_chain_joints = len(self.camera_chain.links)
            current_joint_state = np.zeros(num_cam_chain_joints)
            
            # Find indices for head joints within the camera_chain
            head_1_idx = -1
            head_2_idx = -1
            for i, link in enumerate(self.camera_chain.links):
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
            base_T_camera = self.camera_chain.forward_kinematics(current_joint_state)
            
            # 4. Convert object position to homogeneous coordinates
            obj_pos_camera_homogeneous = np.append(obj_pos, 1)
            
            # 5. Transform position to robot base frame
            obj_pos_robot_homogeneous = base_T_camera @ obj_pos_camera_homogeneous
            
            # 6. Extract Cartesian coordinates
            obj_pos_robot = obj_pos_robot_homogeneous[:3]
            
            return obj_pos_robot
            
        except KeyError as e:
            print(f"Error: Joint sensor key not found: {e}.")
            return None
        except Exception as e:
            print(f"Error during coordinate transformation: {e}")
            return None
    
    # Arm control and gripper methods
    def handle_arm_control(self, key):
        """Handles keyboard input for moving the end effector in Cartesian space."""
        key_char = chr(key).upper()  # Use upper case for matching dictionary keys

        # 1. Get Current Joint Positions using the helper function
        initial_position = self.get_current_ik_joint_state()

        if initial_position is None:
            print("Failed to get valid current joint state. Skipping arm control step.")
            return  # Exit if initial positions couldn't be read reliably

        # 2. Forward Kinematics to find current end-effector pose
        try:
            current_fk = self.my_chain.forward_kinematics(initial_position)
            if not np.all(np.isfinite(current_fk)):
                print("!!! ERROR: Non-finite values in FK result. Skipping step.")
                print(f"  Initial Position used: {initial_position}")
                return
                
            current_pos = current_fk[:3, 3]
            current_orientation_matrix = current_fk[:3, :3]
            print(f"Current Orientation Matrix:\n{current_orientation_matrix}")
            
            if not np.all(np.isfinite(current_pos)) or not np.all(np.isfinite(current_orientation_matrix)):
                print("!!! WARNING: Non-finite values in extracted pose after FK. Skipping step.")
                return

        except Exception as e:
            print(f"Forward kinematics calculation failed: {e}")
            print(f"  Initial Position used: {initial_position}")
            return

        # 3. Determine Desired Change (Position or Orientation)
        delta_pos = np.array([0.0, 0.0, 0.0])
        delta_orientation_matrix = np.identity(3)  # Start with no orientation change
        position_change_requested = False
        orientation_change_requested = False
        action_key_pressed = True  # Flag to check if any valid action key was pressed

        # Check for Position Keys (Base frame: +X Forward, +Y Left, +Z Up)
        if key_char == self.END_EFFECTOR_CONTROL_KEYS["forward"]:
            delta_pos[0] = self.CARTESIAN_STEP
            position_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["backward"]:
            delta_pos[0] = -self.CARTESIAN_STEP
            position_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["left"]:  # Move Left
            delta_pos[1] = self.CARTESIAN_STEP
            position_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["right"]:  # Move Right
            delta_pos[1] = -self.CARTESIAN_STEP
            position_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["up"]:
            delta_pos[2] = self.CARTESIAN_STEP
            position_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["down"]:
            delta_pos[2] = -self.CARTESIAN_STEP
            position_change_requested = True

        # Check for Orientation Keys (End-effector frame rotations)
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["pitch_up"]:  # Pitch Up (around local Y)
            delta_orientation_matrix = self.rotation_matrix('y', self.ORIENTATION_STEP)
            orientation_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["pitch_down"]:  # Pitch Down (around local Y)
            delta_orientation_matrix = self.rotation_matrix('y', -self.ORIENTATION_STEP)
            orientation_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["roll_left"]:  # Roll Left (around local X)
            delta_orientation_matrix = self.rotation_matrix('x', self.ORIENTATION_STEP)
            orientation_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["roll_right"]:  # Roll Right (around local X)
            delta_orientation_matrix = self.rotation_matrix('x', -self.ORIENTATION_STEP)
            orientation_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["yaw_left"]:  # Yaw Left (around local Z)
            delta_orientation_matrix = self.rotation_matrix('z', self.ORIENTATION_STEP)
            orientation_change_requested = True
        elif key_char == self.END_EFFECTOR_CONTROL_KEYS["yaw_right"]:  # Yaw Right (around local Z)
            delta_orientation_matrix = self.rotation_matrix('z', -self.ORIENTATION_STEP)
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
        # Ensure orientation control is applied
        orientation_change_requested = True
        print(f"Attempting IK: TargetPos={target_position}, OrientChange={orientation_change_requested}")
        
        # Pass the initial_position to IK
        ik_results = self.calculate_ik(
            target_position,
            initial_position=initial_position,
            orient=orientation_change_requested,
            orientation_mode="all" if orientation_change_requested else None,
            target_orientation=target_orientation if orientation_change_requested else None
        )

        # 6. Command Joints
        if ik_results is not None:
            # Optional: Add check for non-finite values in ik_results before commanding
            if np.all(np.isfinite(ik_results)):
                self.move_arm_to_target(ik_results)
            else:
                print("!!! WARNING: Non-finite values detected in IK results. Not commanding motors.")
        else:
            # Enhanced logging for IK failure
            print("IK solver failed to find a solution for the target.")
            print(f"  Failed Target Position: {target_position}")
            if orientation_change_requested:
                print(f"  Failed Target Orientation:\n{target_orientation}")
            print(f"  Initial Guess used: {initial_position}")
    
    def toggle_gripper(self):
        """Toggle the gripper between open and closed states."""
        if self.gripper_status == "closed":
            return self.open_gripper()
        elif self.gripper_status == "open":
            return self.close_gripper()
        return False
        
    def open_gripper(self):
        """Open the gripper."""
        try:
            # Get gripper motors directly
            left_gripper_motor = self.robot.getDevice("gripper_left_finger_joint")
            right_gripper_motor = self.robot.getDevice("gripper_right_finger_joint")

            if left_gripper_motor and right_gripper_motor:
                # Command gripper to open position
                left_gripper_motor.setPosition(0.045)
                right_gripper_motor.setPosition(0.045)
                self.gripper_status = "opening"
                print("Opening gripper...")
                return True
            else:
                print("Error: Could not find gripper motors.")
                return False
        except Exception as e:
            print(f"Error opening gripper: {e}")
            return False
    
    def close_gripper(self):
        """Close the gripper."""
        try:
            # Get gripper motors directly
            left_gripper_motor = self.robot.getDevice("gripper_left_finger_joint")
            right_gripper_motor = self.robot.getDevice("gripper_right_finger_joint")

            if left_gripper_motor and right_gripper_motor:
                # Command gripper to closed position
                left_gripper_motor.setPosition(0.0)
                right_gripper_motor.setPosition(0.0)
                self.gripper_status = "closing"
                print("Closing gripper...")
                return True
            else:
                print("Error: Could not find gripper motors.")
                return False
        except Exception as e:
            print(f"Error closing gripper: {e}")
            return False
    
    def update_gripper_status(self):
        """Update the gripper status based on sensor readings."""
        if self.gripper_status == "opening":
            if self.left_gripper_enc.getValue() >= 0.044:  # Check sensor
                self.gripper_status = "open"
                print("Gripper Open")
        elif self.gripper_status == "closing":
            if self.right_gripper_enc.getValue() <= 0.005:  # Check sensor
                self.gripper_status = "closed"
                print("Gripper Closed")
                
    def get_current_joint_positions(self):
        """Get a string representation of the current joint positions."""
        current_pose_list = []
        for name in self.part_names:
            if "wheel" in name:
                current_pose_list.append("'inf'")  # Add as string to match target_pos format
            elif name == "gripper_left_finger_joint":
                current_pose_list.append(f"{self.left_gripper_enc.getValue():.3f}")
            elif name == "gripper_right_finger_joint":
                current_pose_list.append(f"{self.right_gripper_enc.getValue():.3f}")
            elif name in self.robot_sensors:
                current_pose_list.append(f"{self.robot_sensors[name].getValue():.3f}")
                
        pose_str = "(" + ", ".join(current_pose_list) + ")"
        print(f"Current Pose: {pose_str}")
        return pose_str
        
    def set_arm_to_position(self, position):
        """Set the arm to a predefined position.
        
        Args:
            position: A tuple with values for each joint in part_names order
        """
        print(f"Setting arm to predefined position")
        
        # Set each joint to the corresponding position
        for i, part_name in enumerate(self.part_names):
            if part_name in self.motor_dict:
                # Skip wheels
                if "wheel" in part_name:
                    continue
                    
                # Set position for other joints
                try:
                    pos_value = float(position[i])
                    self.motor_dict[part_name].setPosition(pos_value)
                    print(f"  Setting {part_name} to {pos_value:.3f}")
                except (ValueError, IndexError) as e:
                    print(f"  Error setting {part_name}: {e}")
        
        # For the torso lift joint, ensure it's set to a safe value
        if "torso_lift_joint" in self.motor_dict:
            torso_idx = self.part_names.index("torso_lift_joint")
            if torso_idx < len(position):
                try:
                    torso_pos = float(position[torso_idx])
                    if torso_pos < 0.1:  # If torso position is too low
                        torso_pos = 0.35  # Set to a safe value
                    self.motor_dict["torso_lift_joint"].setPosition(torso_pos)
                    print(f"  Ensuring torso_lift_joint at safe position: {torso_pos:.3f}")
                except (ValueError, IndexError) as e:
                    print(f"  Error setting torso_lift_joint: {e}")
                    
        return True
        
    def approach_and_grasp_object(self, object_mask, depth_image, o3d_intrinsics, image_tools, MAX_SPEED_MS, MAX_SPEED):
        """Calculates object position, moves arm nearby, then approaches for grasp.
        
        Args:
            object_mask: Binary mask of the detected object
            depth_image: Depth image from the camera
            o3d_intrinsics: Open3D camera intrinsics
            image_tools: Image processing tools instance
            MAX_SPEED_MS: Maximum speed in m/s
            MAX_SPEED: Maximum speed in rad/s
            
        Returns:
            bool: True if grasp was successful, False otherwise
        """
        # Store original motor velocities to restore later
        original_velocities = {}
        print("Attempting to approach and grasp detected object...")
        
        #unlock base
        joint_name = "torso_lift_joint"
        link_index = [i for i, link in enumerate(self.my_chain.links) 
                             if link.name == joint_name][0]
        if self.my_chain.active_links_mask[link_index]:
            self.my_chain.active_links_mask[link_index] = True
        
        # --- Step 1: Get object position in robot frame ---
        obj_pos = image_tools.get_object_coord(object_mask, depth_image, o3d_intrinsics)
        if obj_pos is None or not np.all(np.isfinite(obj_pos)):
            print("Failed to get valid object coordinates.")
            return False
        print(f"Object Position (Camera Frame): {obj_pos}")

        robot_pos = self.convert_camera_coord_to_robot_coord(obj_pos)
        if robot_pos is None or not np.all(np.isfinite(robot_pos)):
            print("Failed to convert object coordinates to robot frame.")
            return False
        print(f"Object Position (Robot Frame): {robot_pos}")

        # Apply Z correction for head tilt
        if robot_pos is not None and np.all(np.isfinite(robot_pos)):
            try:
                current_tilt_angle = self.robot_sensors["head_2_joint"].getValue()
                Z_CORRECTION_FACTOR_PER_RADIAN = 0.12
                if current_tilt_angle != 0.0:
                    z_correction = Z_CORRECTION_FACTOR_PER_RADIAN * current_tilt_angle
                    robot_pos[2] += z_correction
                    print(f"  Head Tilt: {current_tilt_angle:.3f} rad, Z Correction: {z_correction:.4f}")
                    print(f"  Position with correction: {robot_pos}")
            except Exception as e:
                print(f"Warning: Z correction failed: {e}")
        
        # Check reachability
        max_reach_distance = 1.3  # meters
        if np.linalg.norm(robot_pos) >= max_reach_distance:
            print(f"Object is too far ({np.linalg.norm(robot_pos):.2f}m > {max_reach_distance}m). Cannot reach.")
            return False

        # --- Step 2: Prepare arm movement ---
        approach_offset = 0.2  # meters back from the object for initial alignment
        print("Object is within reach. Calculating initial arm position...")
        
        # Get current arm state
        initial_position = self.get_current_ik_joint_state()
        if initial_position is None:
            print("Failed to get current arm joint state.")
            return False

        # Get current end-effector position
        current_fk = self.my_chain.forward_kinematics(initial_position)
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
            success, ik_results_retract = self.move_arm_with_ik(
                retract_pos, 
                initial_position, 
                orientation=current_fk[:3, :3],
                description="retraction position", 
                must_succeed=False
            )
            
            if not success:
                print("  Trying alternative retraction...")
                retract_pos = np.array([0.35, current_pos_ee[1] * 0.8, current_pos_ee[2]])
                success, ik_results_retract = self.move_arm_with_ik(
                    retract_pos, 
                    initial_position, 
                    orientation=current_fk[:3, :3],
                    description="alternative retraction position"
                )
                
                if not success:
                    print("  All retraction attempts failed. Cannot proceed safely.")
                    return False
            
            # Update position after retraction
            initial_position = self.get_current_ik_joint_state()
            if initial_position is None:
                print("  Failed to get updated joint state after retraction.")
                return False
            current_fk = self.my_chain.forward_kinematics(initial_position)
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
            
            success, ik_results = self.move_arm_with_ik(
                checkpoint,
                current_checkpoint_pos,
                orientation=alignment_orientation,
                description=f"checkpoint {idx+1}",
                max_wait=80,
                cutoff=0.025,
                must_succeed=False
            )
            
            if success:
                current_checkpoint_pos = self.get_current_ik_joint_state()
                if current_checkpoint_pos is None:
                    print(f"  Failed to get updated joint state after checkpoint {idx+1}.")
                    return False
            # If checkpoint fails, continue to next one
        
        # Final alignment position
        print("  Moving to final Z-alignment position...")
        success, ik_results_z_align = self.move_arm_with_ik(
            intermediate_pos_z_aligned,
            current_checkpoint_pos,
            orientation=alignment_orientation,
            description="final Z-aligned position"
        )
        
        if not success:
            return False
        
        # --- Step 5: Orient arm for grabbing ---
        print("Orienting arm to ready for item grabbing...")
        success, ik_results_orient = self.move_arm_with_ik(
            intermediate_pos_z_aligned,
            ik_results_z_align,
            orientation=target_orientation_matrix,
            description="orientation position"
        )
        
        if not success:
            return False
        
        # --- Step 6: Move to approach position (offset from object) ---
        state_after_z_align = self.get_current_ik_joint_state()
        if state_after_z_align is None:
            print("Failed to get arm joint state after Z alignment.")
            return False

        # Get current end-effector position
        current_fk_approach = self.my_chain.forward_kinematics(state_after_z_align)
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
            
            success, ik_results = self.move_arm_with_ik(
                checkpoint,
                current_checkpoint_pos,
                orientation=target_orientation_matrix,
                description=f"approach checkpoint {idx+1}",
                max_wait=80,
                cutoff=0.025,
                must_succeed=False
            )
            
            if success:
                current_checkpoint_pos = self.get_current_ik_joint_state()
                if current_checkpoint_pos is None:
                    print(f"  Failed to get updated joint state after approach checkpoint {idx+1}.")
                    return False
        
        # Final approach position
        success, ik_results_approach = self.move_arm_with_ik(
            robot_pos_approach,
            current_checkpoint_pos,
            orientation=target_orientation_matrix,
            description="final approach position"
        )
        
        if not success:
            return False
            
        # --- Step 7: Final approach in small steps ---
        print(f"Stage 3: Performing final approach movement ({approach_offset}m)...")
        steps = 10
        step_size = approach_offset / steps
        
        for step in range(1, steps + 1):
            current_position_step = self.get_current_ik_joint_state()
            if current_position_step is None:
                print("Failed to get current joint state during final approach step.")
                return False
            
            # Calculate incremental target
            step_target = robot_pos_approach.copy()
            step_target[0] += step * step_size
            
            # Move arm incrementally
            success, _ = self.move_arm_with_ik(
                step_target,
                current_position_step,
                orientation=target_orientation_matrix,
                description=f"approach step {step}/{steps}",
                max_wait=10,
                cutoff=0.03,
                must_succeed=True
            )
            
            if not success:
                print(f"IK failed during step {step} of final approach.")
                return False
        
        # Verify final position
        final_pos_check = self.get_current_ik_joint_state()
        if final_pos_check is not None:
            final_fk = self.my_chain.forward_kinematics(final_pos_check)
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
        if not self.close_gripper():
            print("Failed to close gripper.")
            return False
            
        # Wait for gripper
        wait_duration_s = 2.0
        num_wait_steps = int(wait_duration_s * 1000 / self.timestep)
        for _ in range(num_wait_steps):
            if self.robot.step(self.timestep) == -1:
                print("Simulation stopped during gripper close wait.")
                return False
            self.update_gripper_status()
            if self.gripper_status == "closed":
                break
        
        # --- Step 9: Lift object ---
        print("Lifting object slightly...")
        current_state_before_lift = self.get_current_ik_joint_state()
        if current_state_before_lift is None:
            print("Failed to get joint state before lifting.")
            return False
        
        try:
            # Calculate lift position
            current_fk_before_lift = self.my_chain.forward_kinematics(current_state_before_lift)
            current_pos_before_lift = current_fk_before_lift[:3, 3]
            current_orientation_before_lift = current_fk_before_lift[:3, :3]
            lift_target_pos = current_pos_before_lift + np.array([0.0, 0.0, 0.15])  # 15cm lift
            
            # Slow down arm for lift
            print("  Slowing down arm motors for lifting...")
            slow_down_factor = 0.2
            for i in range(len(self.my_chain.links)):
                if self.my_chain.active_links_mask[i]:
                    link_name = self.my_chain.links[i].name
                    if link_name in self.motor_dict and "wheel" not in link_name:
                        motor = self.motor_dict[link_name]
                        original_velocities[link_name] = motor.getVelocity()
                        motor.setVelocity(motor.getMaxVelocity() * slow_down_factor)
            
            # Move arm up
            success, _ = self.move_arm_with_ik(
                lift_target_pos,
                current_state_before_lift,
                orientation=current_orientation_before_lift,
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
        
        # Get wheel motors
        try:
            wheel_left = self.robot.getDevice("wheel_left_joint")
            wheel_right = self.robot.getDevice("wheel_right_joint")
            
            reverse_distance = 0.5
            reverse_speed_factor = 0.25
            reverse_speed_ms = MAX_SPEED_MS * reverse_speed_factor
            
            if reverse_speed_ms > 0:
                duration_s = reverse_distance / reverse_speed_ms
                num_steps = int(duration_s * 1000 / self.timestep)
                reverse_velocity = -MAX_SPEED * reverse_speed_factor
                
                # Execute reverse movement
                wheel_left.setVelocity(reverse_velocity)
                wheel_right.setVelocity(reverse_velocity)
                
                for _ in range(num_steps):
                    if self.robot.step(self.timestep) == -1:
                        wheel_left.setVelocity(0)
                        wheel_right.setVelocity(0)
                        print("Simulation stopped during reverse.")
                        return False
                
                # Stop wheels
                wheel_left.setVelocity(0)
                wheel_right.setVelocity(0)
        except KeyError:
            print("Wheel motors not available, skipping reverse.")
        
        #lock base
        joint_name = "torso_lift_joint"
        link_index = [i for i, link in enumerate(self.my_chain.links) 
                             if link.name == joint_name][0]
        if self.my_chain.active_links_mask[link_index]:
            self.my_chain.active_links_mask[link_index] = False
        
        # --- Step 11: Move to final pose and open gripper ---
        print("Moving arm to final pose and releasing object...")
        final_target_pos = np.array([0.28548, 0.0, 0.33344])
        final_target_orient = np.array([
            [ 0.99527,  0.09564, -0.017219],
            [-0.01995,  0.027676, -0.99942 ],
            [-0.095108,  0.99503,  0.029453]
        ])
        
        # Get current arm state
        current_arm_pos = self.get_current_ik_joint_state()
        if current_arm_pos is not None:
            # Move to final position
            success, _ = self.move_arm_with_ik(
                final_target_pos,
                current_arm_pos,
                orientation=final_target_orient,
                description="final pose",
                max_wait=300,
                must_succeed=False
            )
        
        # Open gripper regardless of arm movement success
        self.open_gripper()
        
        # Wait for gripper to open
        open_wait_duration_s = 1.0
        num_open_wait_steps = int(open_wait_duration_s * 1000 / self.timestep)
        for _ in range(num_open_wait_steps):
            if self.robot.step(self.timestep) == -1:
                print("Simulation stopped during gripper open wait.")
                return False
            self.update_gripper_status()
            if self.gripper_status == "open":
                break
        
        # --- Restore arm speeds ---
        print("  Restoring original arm motor velocities...")
        for link_name, original_velocity in original_velocities.items():
            if link_name in self.motor_dict:
                self.motor_dict[link_name].setVelocity(original_velocity)
        
        print("Grasp sequence completed successfully.")
        return True
        