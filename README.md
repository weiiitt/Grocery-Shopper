## 2. Deliverables Overview

Below are the major modules of our final project. Each module is tied to a “tier goal” (baseline vs. enhanced). We will integrate these modules so that the robot can build a map, localize itself, plan paths, detect objects, and manipulate items in the store. The robot will keep track of a set of 20 grocery items that need to be collected. If it encounters an aisle containing any of those items, it will pick them up. There is no strict sequence—once the robot detects a target item, it attempts to collect it before moving on.

### Mapping
- **Tier Goal (Baseline):** Construct a 2D occupancy grid from LiDAR data.
- **Enhanced (Optional):** Implement or integrate a SLAM algorithm (e.g., from PythonRobotics) to update the map in real time.

### Localization
- **Tier Goal (Baseline):** Use GPS + Compass to get robot pose estimates.
- **Enhanced (Optional):** Implement Monte Carlo Localization (MCL) or a particle filter. This would let us refine our position estimate using LiDAR data against the occupancy grid.

### Path Planning
- **Tier Goal (Baseline):** Use an RRT or RRT* implementation (e.g., from prior Homework 2) to navigate from the robot’s current location to a target aisle or object.
- **Enhanced (Optional):** Add path smoothing or obstacle re-check in real time for dynamic updates.

### Object Detection
- **Tier Goal (Baseline):** Simple color blob detection for items using the Webots camera.

### Manipulation
- **Tier Goal (Baseline):** Use inverse kinematics (IK) to pick up items once the base is positioned near them. Libraries like ikpy may be used.
- **Enhanced (Optional):** Multi-waypoint or RRT-based arm motion to avoid collisions with shelves or obstacles, potentially enabling green-cube bonus pickups.

### System Integration & Teleoperation
- **Tier Goal (Baseline):** Implement a simple state machine that orchestrates mapping, localization, path planning, object detection, and pick-and-place.
- **Enhanced (Optional):** Provide a keyboard teleop mode for debugging and incorporate robust error handling so the robot can recover if it drops an item or fails to pick up an object.

---

## 3. Detailed Deliverables & Leads

Below is the final breakdown of each major deliverable, its sub-tasks, a designated lead, and the target completion date. (All dates are placeholders—adjust them to match your semester deadlines.)

### [D1] Mapping
- **Lead:** Patrick Nguyen
- **Target Completion:** April 15
  - **[D1.1] LiDAR Integration**
    - Read and store LiDAR scans in Python.
    - Construct an occupancy grid or direct obstacle map.
  - **[D1.2] Visualization**
    - Display the map in Webots (using the display device) or print to console for debugging.
    - Confirm that obstacles appear in the correct locations.
  - **Testing**
    - Drive the robot manually (or via teleop) and verify that the LiDAR data matches the expected environment layout.

### [D2] Localization
- **Lead:** Patrick Nguyen
- **Target Completion:** April 15
  - **[D2.1] GPS + Compass Baseline**
    - Fuse data to get the robot’s (x, y, θ) in the global frame.
  - **[D2.2] (Optional) MCL Implementation**
    - Create a particle filter that uses the occupancy grid and LiDAR readings to refine pose.
    - Demonstrate more accurate localization than GPS alone.
  - **Testing**
    - Compare the reported robot pose vs. known positions in Webots.
    - For MCL, place the robot in multiple known spots and verify that the filter converges near the true pose.

### [D3] Path Planning
- **Lead:** Nikko Gajowniczek
- **Target Completion:** April 21
  - **[D3.1] RRT from HW 2***
    - Adapt the existing 2D RRT* to our occupancy grid.
    - Generate collision-free paths from the robot’s current location to a desired goal.
  - **[D3.2] (Optional) Path Smoothing**
    - Post-process the waypoints to eliminate sharp corners or detours.
  - **[D3.3] Convert to Wheel Speeds**
    - Similar to Lab 5, take each waypoint and produce the appropriate forward/turn velocities.
  - **Testing**
    - Visualize or print the path to confirm it avoids shelving.
    - Let the robot navigate a short route and confirm no collisions.

### [D4] Object Detection (Vision)
- **Lead:** Wei Jiang
- **Target Completion:** April 21
  - **[D4.1] Color Thresholding for Yellow**
    - Capture frames from the Webots camera.
    - Identify and locate yellow blocks.
  - **[D4.2] Compute Object Position**
    - Estimate the object’s position in world coordinates by combining camera data with known robot pose. (LiDAR or bounding box geometry can help refine distance).
  - **[D4.3] (Optional) Green Cube Detection**
    - If going for bonus items, do color thresholding for green blocks as well.
  - **Testing**
    - Place cubes at various angles and distances; confirm that the code prints detection results.

### [D5] Manipulation
- **Lead:** Nikko Gajowniczek
- **Target Completion:** April 21
  - **[D5.1] Basic Inverse Kinematics**
    - Install and test ikpy.
    - For a known block location near the robot, solve for the end-effector joint angles.
  - **[D5.2] Gripper Control**
    - Send open/close commands to pick up and release items.
  - **[D5.3] (Optional) Collision-Free Arm Motions**
    - Add intermediate waypoints to avoid collisions with shelves or other obstacles.
  - **Testing**
    - Place a block directly in front of the robot. Confirm that it picks up the block.
    - Attempt different positions to check for reliability.

### [D6] System Integration & Teleoperation
- **Lead:** Wei Jiang
- **Target Completion:** April 30
  - **[D6.1] State Machine**
    - Create a Python-based state machine with states like:
      - “Explore & Build Map”
      - “Detect Object”
      - “Plan Path & Navigate”
      - “Pick Object”
      - “Update Set of Items” (if object is a needed item, remove from set)
    - The robot continues until it has found all items or time runs out.
  - **[D6.2] Teleop (Optional)**
    - Implement a keyboard teleop for quick tests or to override the robot if it gets stuck.
  - **[D6.3] Multiple Items**
    - Maintain a set of 20 required items.
    - If an item is detected that belongs to this set, trigger pick-up.
    - Stop the project once the set is empty or the user ends the run.
  - **Testing**
    - Run a full integrated script that moves, detects, picks up at least one object, and returns success messages.

---

## 4. Implementation Plan & Timeline

| Task                 | Target  | Notes                                   |
|----------------------|---------|-----------------------------------------|
| [D1] Mapping          | Apr 15  | Basic occupancy grid or SLAM library setup |
| [D2] Localization     | Apr 15  | GPS + Compass baseline; optional MCL    |
| [D3] Path Planning    | Apr 21  | RRT*, possibly path smoothing           |
| [D4] Object Detection | Apr 21  | Color thresholding for yellow; optionally green |
| [D5] Manipulation     | Apr 21  | IK usage, pick-and-place, possibly obstacle-free |
| [D6] Integration      | Apr 30  | Merge modules, handle set-based item pickup |
| Final Demo            | May 4   | Live test with all modules in one demonstration |

**Note:** If time permits, we will enhance mapping/localization (D1+D2) or do a collision-free manipulation pipeline for the green bonus cubes (D5.3).

---

## 5. Testing & Validation

### Mapping:
- Display the occupancy grid or save it to an image. Verify that walls and shelves align with LiDAR data.

### Localization:
- Compare reported pose vs. the ground-truth robot location in Webots.
- If implementing MCL, place the robot in multiple corners of the store and track the convergence.

### Path Planning:
- Visualize or print path nodes from RRT/RRT*. Confirm they avoid obstacles.
- Let the robot physically follow the path in Webots, ensuring no collisions.

### Object Detection:
- Place a single item at known coordinates. Check console/log outputs for correct detection.
- Repeat at multiple orientations.

### Manipulation:
- Attempt pick-and-place with the item near the front. Confirm the item ends in the basket.
- Optionally test an obstructed scenario if doing collision-free arm planning.

### Integration & Multi-Item Handling:
- The robot has a set of 20 items it needs. Manually place a few items in different aisles.
- Let the robot roam the store. Each time it detects a relevant item, it picks it up and removes it from the set.
- Observe logs to confirm that items are only collected if they are in the set.