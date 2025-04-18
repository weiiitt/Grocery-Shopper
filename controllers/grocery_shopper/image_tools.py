import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import trimesh
import open3d as o3d
from ultralytics import YOLO
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "FusionVision"))
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from utils import perform_yolo_inference

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'FoundationPose'))
# from estimater import *
# from datareader import *
# # from Utils import *


class ImageTools:
    def __init__(
        self,
        yolo_weight_path,
        fastsam_weight_path,
        yolo_conf_threshold=0.9,
        fastsam_conf=0.3,
        fastsam_iou=0.7,
    ):
        # Initialize models here
        print(f"Loading YOLO model from: {yolo_weight_path}")
        self.yolo_model = YOLO(yolo_weight_path)
        print(f"Loading FastSAM model from: {fastsam_weight_path}")
        self.fastsam_model = FastSAM(fastsam_weight_path)
        self.yolo_conf_threshold = yolo_conf_threshold
        self.fastsam_conf = fastsam_conf
        self.fastsam_iou = fastsam_iou
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print("Models loaded.")

    def get_object_pose(self, depth_data, obj_file):
        """Get the pose of an object in the depth image."""
        # Load the object mesh
        # mesh = trimesh.load(obj_file)

        # scorer = ScorePredictor()
        # refiner = PoseRefinePredictor()
        # glctx = dr.RasterizeCudaContext()
        # est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,  glctx=glctx)

        # rgb = depth_to_vis(depth_data)

        pass

    def save_depth_image(self, depth_data, directory):
        """Save the depth data as a PNG image."""
        np.save(directory + "/depth_data.npy", depth_data)
        print("Raw depth data saved as 'depth_data.npy'")

        # Convert depth data to a visualization image
        # Normalize the depth data to 0-255 range for visualization
        depth_min = np.min(depth_data[np.isfinite(depth_data)])
        depth_max = np.max(depth_data[np.isfinite(depth_data)])

        # Create a mask for invalid values (NaN or inf)
        invalid_mask = ~np.isfinite(depth_data)

        if depth_max > depth_min:  # Avoid division by zero
            # Create normalized array first as float
            normalized_depth = np.full_like(depth_data, 255, dtype=np.float32)

            # Only normalize valid values
            normalized_depth[~invalid_mask] = (
                (depth_data[~invalid_mask] - depth_min) / (depth_max - depth_min) * 255
            )

            # Convert to uint8 after normalization
            normalized_depth = normalized_depth.astype(np.uint8)

        else:
            normalized_depth = np.full_like(depth_data, 255, dtype=np.uint8)

        # Create a grayscale image from the normalized depth data
        plt.figure(figsize=(10, 8))
        plt.imshow(normalized_depth, cmap="gray")
        plt.colorbar(label="Depth")
        plt.title("Depth Image")
        plt.savefig(directory + "/depth_visualization.png")
        plt.close()
        print("Depth visualization saved as 'depth_visualization.png'")

    def save_color_image(self, color_data, directory):
        """Save the color data as a PNG image."""
        np.save(directory + "/color_data.npy", color_data)
        print("Raw color data saved as 'color_data.npy'")

        # Convert color data to a visualization image
        # Convert BGR to RGB for proper display with matplotlib
        rgb_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_data)
        plt.title("Color Image")
        plt.savefig(directory + "/color_visualization.png")
        plt.close()
        print("Color visualization saved as 'color_visualization.png'")

    def save_images(self, color_image, depth_image, directory):
        """Save the color and depth images as PNG files."""
        # Save color image
        self.save_color_image(color_image, directory)
        self.save_depth_image(depth_image, directory)

    def save_sequential_color_image(self, color_image, directory):
        """
        Saves the color image to the specified directory with a sequential filename.

        Parameters:
            color_image (np.ndarray): The color image data (expected in BGR format).
            directory (str): The directory where the image should be saved.
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

            # Find the next available sequence number
            existing_files = [
                f
                for f in os.listdir(directory)
                if f.startswith("color_image_") and f.endswith(".png")
            ]
            max_num = -1
            for f in existing_files:
                try:
                    num_str = f.replace("color_image_", "").replace(".png", "")
                    num = int(num_str)
                    if num > max_num:
                        max_num = num
                except ValueError:
                    # Ignore files that don't match the expected number format
                    continue

            next_num = max_num + 1
            filename = f"color_image_{next_num:03d}.png"
            filepath = os.path.join(directory, filename)

            # Save the color image using OpenCV
            cv2.imwrite(filepath, color_image)
            print(f"Color image saved as '{filepath}'")

        except Exception as e:
            print(f"Error saving sequential color image: {e}")

    def process_and_extract_objects(self, color_image, depth_image, intrinsics: o3d.camera.PinholeCameraIntrinsic, depth_scale=1.0, result_image: bool = True):
        """
        Processes color and depth images to detect objects, selects the most confident
        (or largest) one, segments it, and extracts its mask and point cloud.

        Args:
            color_image (np.ndarray): The BGR color image.
            depth_image (np.ndarray): The depth image (raw values).
            intrinsics (o3d.camera.PinholeCameraIntrinsic): Open3D camera intrinsic parameters.
            depth_scale (float): The factor to convert depth values to meters (depth / depth_scale).
            result_image (bool, optional): If True (default), generates and returns the overlay image.
                                           If False, returns an empty array for the overlay image.

        Returns:
            tuple: A tuple containing:
                - overlay_image (np.ndarray): Color image with the selected object's YOLO box and FastSAM mask, or an empty array if result_image is False.
                - object_masks (np.ndarray): Binary mask for the selected object (or empty array if no object).
                - object_point_clouds (o3d.geometry.PointCloud): Point cloud for the selected object.
                - detections (List[dict]): List of detection dictionaries for the selected object.
        """
        # --- 1. YOLO Detection ---
        all_detections, _ = perform_yolo_inference(color_image, self.yolo_model, confidence_threshold=self.yolo_conf_threshold)
        
        # Initialize overlay_image based on result_image flag
        overlay_image = color_image.copy() if result_image else np.array([])
        object_masks = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=bool)
        object_point_clouds = o3d.geometry.PointCloud()
        selected_detection_list = [] # Will hold the single selected detection

        if not all_detections:
            print("No objects detected by YOLO.")
            return overlay_image, object_masks, object_point_clouds, selected_detection_list

        # --- 2. Select the Best Object ---
        if len(all_detections) == 1:
            selected_detection = all_detections[0]
        else:
            # Find max confidence
            max_conf = max(det['confidence'] for det in all_detections)
            
            # Get all detections with max confidence
            top_conf_detections = [det for det in all_detections if det['confidence'] == max_conf]
            
            if len(top_conf_detections) == 1:
                selected_detection = top_conf_detections[0]
            else:
                # Tie-breaker: Largest bounding box area
                best_area = -1
                selected_detection = None
                for det in top_conf_detections:
                    x1, y1, x2, y2 = det['bounding_box']
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        selected_detection = det
            
        if selected_detection is None: # Should not happen if all_detections is not empty, but safety check
             print("Could not select a best object.")
             return overlay_image, object_masks, object_point_clouds, selected_detection_list

        selected_detection_list.append(selected_detection)
        selected_bbox = list(map(int, selected_detection['bounding_box'])) # Use integers for bbox

        # --- 3. FastSAM Segmentation for the Selected Object ---

        fastsam_results = self.fastsam_model(color_image, device=self.device, retina_masks=True, imgsz=intrinsics.width, conf=self.fastsam_conf, iou=self.fastsam_iou)
        
        if not fastsam_results:
                print("FastSAM did not return results.")
                # Draw YOLO box only if result_image is True and FastSAM fails
                if result_image:
                    x1, y1, x2, y2 = selected_bbox
                    cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(overlay_image, f"{selected_detection['class_name']}: {selected_detection['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                return overlay_image, object_masks, object_point_clouds, selected_detection_list

        prompt_process = FastSAMPrompt(color_image, fastsam_results, device=self.device)
        
        # Prompting with the single selected box
        ann = prompt_process.box_prompt(bboxes=[selected_bbox]) # Pass as list

        if len(ann) == 0:
            print("FastSAM box prompt did not return annotations for the selected box.")
            # Draw YOLO box only if result_image is True and prompting fails
            if result_image:
                x1, y1, x2, y2 = selected_bbox
                cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(overlay_image, f"{selected_detection['class_name']}: {selected_detection['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            return overlay_image, object_masks, object_point_clouds, selected_detection_list

        # --- 4. Generate Overlay Image (Conditional) ---
        if result_image:
            # ann should contain the mask(s) for the selected box
            overlay_image = prompt_process.plot_to_result(annotations=ann)
            # Optionally re-draw selected box for emphasis
            x1, y1, x2, y2 = selected_bbox
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 0), 1) # Green box

        # --- 5. Process Mask and Generate Point Cloud ---
            # ann might be a single mask or a list containing one mask
        if isinstance(ann, list):
            if len(ann) == 0:
                    print("Warning: FastSAM returned an empty list for the selected box.")
                    mask_data = None
            else:
                    if len(ann) > 1:
                        print(f"Warning: FastSAM returned {len(ann)} masks for a single box prompt. Using the first one.")
                    mask_data = ann[0] # Take the first mask if multiple are returned
        else:
                mask_data = ann # Assume it's the single mask tensor/array
        
        if mask_data is not None:
            # Create Open3D depth image
            # o3d_depth = o3d.geometry.Image(depth_image)

            # Convert mask tensor to numpy array
            if hasattr(mask_data, 'cpu'):
                mask_data = mask_data.cpu()
            if hasattr(mask_data, 'numpy'):
                mask_np = mask_data.numpy().squeeze()
            else:
                mask_np = np.array(mask_data).squeeze()

            mask_np = mask_np.astype(np.uint8) * 255

            if mask_np.shape != (intrinsics.height, intrinsics.width):
                print(f"Warning: Mask shape {mask_np.shape} mismatch with image dimensions ({intrinsics.height}, {intrinsics.width}). Resizing.")
                mask_np = cv2.resize(mask_np, (intrinsics.width, intrinsics.height), interpolation=cv2.INTER_NEAREST)

            object_masks = mask_np > 0 # Store boolean mask

            # --- 6. Generate Object Point Cloud ---
            rows, cols = np.where(mask_np > 0)
            depth_vals = depth_image[rows, cols]
            
            valid_depth_indices = depth_vals > 0
            
            rows = rows[valid_depth_indices]
            cols = cols[valid_depth_indices]
            depth_vals = depth_vals[valid_depth_indices]

            if len(depth_vals) > 0:
                fx = intrinsics.intrinsic_matrix[0, 0]
                fy = intrinsics.intrinsic_matrix[1, 1]
                cx = intrinsics.intrinsic_matrix[0, 2]
                cy = intrinsics.intrinsic_matrix[1, 2]

                z = depth_vals / depth_scale
                x = (cols - cx) * z / fx
                y = (rows - cy) * z / fy
                
                points = np.vstack((x, y, z)).T

                obj_pcd = o3d.geometry.PointCloud()
                obj_pcd.points = o3d.utility.Vector3dVector(points)
                object_point_clouds = obj_pcd
            else:
                print(f"No valid depth points found for the selected object mask.")
                object_point_clouds = o3d.geometry.PointCloud() # Add empty point cloud



        return overlay_image, object_masks, object_point_clouds, selected_detection_list
