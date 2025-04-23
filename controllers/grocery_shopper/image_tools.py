import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import trimesh
import open3d as o3d
from ultralytics import YOLO
import torch
from graspnetAPI import GraspGroup
from typing import Tuple, List, Dict, Any, Optional


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
        yolo_weight_path: str,
        fastsam_weight_path: str,
        graspnet_root: str,
        graspnet_checkpoint_path: str,
        yolo_conf_threshold: float = 0.9,
        fastsam_conf: float = 0.3,
        fastsam_iou: float = 0.7,
        num_point: int = 20000,
        num_view: int = 300,
        collision_thresh: float = 0.01,
        voxel_size: float = 0.01,
    ):
        """Initializes the ImageTools class, loading necessary models and setting up paths.

        Args:
            yolo_weight_path (str): Path to the YOLO model weights file.
            fastsam_weight_path (str): Path to the FastSAM model weights file.
            graspnet_root (str): Root directory of the GraspNet baseline.
            graspnet_checkpoint_path (str): Path to the GraspNet checkpoint file.
            yolo_conf_threshold (float, optional): Confidence threshold for YOLO detections. Defaults to 0.9.
            fastsam_conf (float, optional): Confidence threshold for FastSAM segmentation. Defaults to 0.3.
            fastsam_iou (float, optional): IoU threshold for FastSAM segmentation. Defaults to 0.7.
            num_point (int, optional): Number of points to sample for GraspNet. Defaults to 20000.
            num_view (int, optional): Number of views for GraspNet. Defaults to 300.
            collision_thresh (float, optional): Collision threshold distance for GraspNet. Defaults to 0.01.
            voxel_size (float, optional): Voxel size for collision detection downsampling. Defaults to 0.01.
        """
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
        
        print("loading graspnet")
        self.graspnet_root = graspnet_root
        self._setup_paths()
        self._import_graspnet_modules()
        self.checkpoint_path = graspnet_checkpoint_path
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.net = self._init_network()

    def _setup_paths(self):
        """Setup necessary paths for graspnet-baseline"""
        paths = [
            'models',
            'dataset',
            'utils',
            ''  # for the base directory
        ]
        
        for path in paths:
            full_path = os.path.join(self.graspnet_root, path)
            if not os.path.exists(full_path):
                raise ValueError(f"Required path does not exist: {full_path}")
            if full_path not in sys.path:
                sys.path.append(full_path)

    def _import_graspnet_modules(self):
        """Import GraspNet modules after paths are set up"""
        global GraspNet, pred_decode, GraspNetDataset, ModelFreeCollisionDetector
        global CameraInfo, create_point_cloud_from_depth_image
        
        from graspnet import GraspNet, pred_decode
        from graspnet_dataset import GraspNetDataset
        from collision_detector import ModelFreeCollisionDetector
        from data_utils import CameraInfo, create_point_cloud_from_depth_image
    
    def _init_network(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"-> loaded checkpoint {self.checkpoint_path} (epoch: {start_epoch})")
        # set model to eval mode
        net.eval()
        return net

    def get_object_pose(self, depth_data: np.ndarray, obj_file: str) -> None:
        """Get the pose of an object in the depth image (Not fully implemented).

        Args:
            depth_data (np.ndarray): The depth image data.
            obj_file (str): Path to the object's mesh file (e.g., .obj).

        Returns:
            None: Currently does not return anything.
        """
        # Load the object mesh
        # mesh = trimesh.load(obj_file)

        # scorer = ScorePredictor()
        # refiner = PoseRefinePredictor()
        # glctx = dr.RasterizeCudaContext()
        # est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,  glctx=glctx)

        # rgb = depth_to_vis(depth_data)

        pass

    def save_depth_image(self, depth_data: np.ndarray, directory: str) -> None:
        """Saves the raw depth data as .npy and a visualization as .png.

        Args:
            depth_data (np.ndarray): The depth image data.
            directory (str): The directory to save the files in.
        """
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

    def save_color_image(self, color_data: np.ndarray, directory: str) -> None:
        """Saves the raw color data (BGR) as .npy and a visualization (RGB) as .png.

        Args:
            color_data (np.ndarray): The BGR color image data.
            directory (str): The directory to save the files in.
        """
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

    def save_images(self, color_image: np.ndarray, depth_image: np.ndarray, directory: str) -> None:
        """Saves both color and depth images using their respective save functions.

        Args:
            color_image (np.ndarray): The BGR color image data.
            depth_image (np.ndarray): The depth image data.
            directory (str): The directory to save the files in.
        """
        # Save color image
        self.save_color_image(color_image, directory)
        self.save_depth_image(depth_image, directory)

    def save_sequential_color_image(self, color_image: np.ndarray, directory: str) -> None:
        """
        Saves the color image to the specified directory with a sequential filename (color_image_XXX.png).

        Args:
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

    def process_and_extract_objects(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        depth_scale: float = 1.0,
        result_image: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud, List[Dict[str, Any]]]:
        """
        Processes color/depth images: detects objects (YOLO), selects the best one,
        segments it (FastSAM), and extracts its mask and point cloud.

        Args:
            color_image (np.ndarray): The BGR color image.
            depth_image (np.ndarray): The depth image (raw values).
            intrinsics (o3d.camera.PinholeCameraIntrinsic): Open3D camera intrinsic parameters.
            depth_scale (float, optional): Factor to convert depth values to meters. Defaults to 1.0.
            result_image (bool, optional): If True, generate and return the overlay image.
                                           If False, return an empty array for overlay. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud, List[Dict[str, Any]]]: A tuple containing:
                - overlay_image (np.ndarray): Color image with YOLO box and FastSAM mask overlay (or empty array).
                - object_masks (np.ndarray): Boolean mask for the selected object.
                - object_point_clouds (o3d.geometry.PointCloud): Point cloud of the selected object.
                - selected_detection_list (List[Dict[str, Any]]): List containing the dictionary of the selected detection.
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
    def process_data(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        workspace_mask: np.ndarray,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        depth_scale: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], o3d.geometry.PointCloud]:
        """Generates a point cloud, applies workspace mask, samples points, and prepares data for GraspNet.

        Args:
            color (np.ndarray): The BGR color image.
            depth (np.ndarray): The depth image (raw values).
            workspace_mask (np.ndarray): A boolean mask defining the valid workspace area.
            intrinsics (o3d.camera.PinholeCameraIntrinsic): Open3D camera intrinsic parameters.
            depth_scale (float, optional): Factor to convert depth values to meters. Defaults to 1.0.

        Returns:
            Tuple[Dict[str, torch.Tensor], o3d.geometry.PointCloud]: A tuple containing:
                - end_points (Dict[str, torch.Tensor]): Dictionary containing 'point_clouds' and 'cloud_colors' tensors for GraspNet.
                - cloud_o3d (o3d.geometry.PointCloud): The generated Open3D point cloud (masked, but not sampled).
        """
        # Extract intrinsic parameters
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx = intrinsics.intrinsic_matrix[0, 2]
        cy = intrinsics.intrinsic_matrix[1, 2]

        # Generate point cloud
        camera = CameraInfo(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            scale=depth_scale
        )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # Get valid points
        mask = workspace_mask & (depth > 0)
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # Sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # Convert data to Open3D format
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        # Assign the colors corresponding to the masked points. 
        # Make sure colors are in the range [0, 1] and are float32.
        # OpenCV loads images as BGR, so we need to convert to RGB for Open3D.
        if color_masked.dtype == np.uint8:
            normalized_colors = color_masked.astype(np.float32) / 255.0
        else:
            # Assume colors are already float, potentially in [0,1]
            normalized_colors = color_masked.astype(np.float32) 
            # Clamp values just in case they are outside [0, 1]
            normalized_colors = np.clip(normalized_colors, 0.0, 1.0)
            
        # Convert BGR to RGB
        rgb_colors = normalized_colors[:, ::-1] 
        cloud_o3d.colors = o3d.utility.Vector3dVector(rgb_colors)

        # Prepare data for neural network
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = torch.from_numpy(color_sampled.astype(np.float32)).to(device)

        return end_points, cloud_o3d
    
    def predict_grasps(self, end_points: Dict[str, torch.Tensor]) -> GraspGroup:
        """Runs the GraspNet model to predict grasps based on the processed point cloud data.

        Args:
            end_points (Dict[str, torch.Tensor]): Dictionary containing 'point_clouds' and 'cloud_colors' tensors.

        Returns:
            GraspGroup: A GraspGroup object containing the predicted grasps.
        """
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        return GraspGroup(gg_array)
    
    def filter_collisions(self, grasp_group: GraspGroup, cloud: o3d.geometry.PointCloud) -> GraspGroup:
        """Filters predicted grasps by checking for collisions with the scene point cloud.

        Args:
            grasp_group (GraspGroup): The predicted grasps.
            cloud (o3d.geometry.PointCloud): The scene point cloud to check against.

        Returns:
            GraspGroup: The filtered grasps (those not in collision).
        """
        if self.collision_thresh <= 0:
            return grasp_group
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.05, 
                                          collision_thresh=self.collision_thresh)
        return grasp_group[~collision_mask]
    
    def visualize_grasps(self, grasp_group: GraspGroup, cloud: o3d.geometry.PointCloud, top_k: int = 50) -> None:
        """Visualizes the top K grasps along with the scene point cloud using Open3D.
        Args:
            grasp_group (GraspGroup): The grasps to visualize.
            cloud (o3d.geometry.PointCloud): The scene point cloud.
            top_k (int, optional): The number of top grasps to visualize. Defaults to 50.
        """
        grasp_group.nms()
        grasp_group.sort_by_score()
        grasp_group = grasp_group[:top_k]
        grippers = grasp_group.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    
    def detect(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        workspace_mask: np.ndarray,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        depth_scale: float = 1.0,
        visualize: bool = True,
    ) -> Tuple[GraspGroup, o3d.geometry.PointCloud]:
        """Main method to process images and detect grasps.

        Combines data processing, grasp prediction, collision filtering, and optional visualization.

        Args:
            color_image (np.ndarray): The BGR color image.
            depth_image (np.ndarray): The depth image (raw values).
            workspace_mask (np.ndarray): A boolean mask defining the valid workspace area.
            intrinsics (o3d.camera.PinholeCameraIntrinsic): Open3D camera intrinsic parameters.
            depth_scale (float, optional): Factor to convert depth values to meters. Defaults to 1.0.
            visualize (bool, optional): Whether to visualize the resulting grasps. Defaults to True.

        Returns:
            Tuple[GraspGroup, o3d.geometry.PointCloud]: A tuple containing:
                - grasp_group (GraspGroup): The final filtered grasps.
                - cloud (o3d.geometry.PointCloud): The processed point cloud used for detection.
        """
        end_points, cloud = self.process_data(color_image, depth_image, workspace_mask, intrinsics, depth_scale)
        grasp_group = self.predict_grasps(end_points)
        grasp_group = self.filter_collisions(grasp_group, cloud)
        
        if visualize:
            self.visualize_grasps(grasp_group, cloud)
            
        return grasp_group, cloud