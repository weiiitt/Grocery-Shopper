import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import trimesh
import open3d as o3d

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'FoundationPose'))
# from estimater import *
# from datareader import *
# # from Utils import *

class ImageTools:
    def __init__(self):
        pass

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

    def detect_objects(self, color_image, depth_image, meta, workspace_mask=None, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """
        Detects objects on a table using depth and color images by performing plane segmentation.

        Parameters:
            color_image (numpy.ndarray): The color image, expected in range [0,1], float32.
            depth_image (numpy.ndarray): The depth image corresponding to the color image.
            meta (dict): Dictionary containing 'intrinsic_matrix' and 'factor_depth'.
            workspace_mask (numpy.ndarray): Boolean mask of the workspace. Defaults to None (use entire frame).
            distance_threshold (float): RANSAC distance threshold for plane fitting (in meters).
            ransac_n (int): Number of initial points to estimate a plane.
            num_iterations (int): Number of RANSAC iterations.

        Returns:
            result_image (numpy.ndarray): The color image with detected objects outlined.
            object_contours (list): List of contours corresponding to detected objects.
        """

        # Unpack meta data
        factor_depth = meta.get('factor_depth', 1000.0)  # Default to 1000.0 if not provided
        intrinsic_matrix = meta['intrinsic_matrix']
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        # Convert depth image to meters
        depth_in_meters = depth_image.astype(float) / factor_depth  # Depth in meters

        # Get image dimensions
        height, width = depth_in_meters.shape

        # Generate point cloud from depth image
        x_idx = np.arange(width)
        y_idx = np.arange(height)
        x_grid, y_grid = np.meshgrid(x_idx, y_idx)

        if workspace_mask is not None:
            valid_mask = (depth_in_meters > 0) & workspace_mask
        else:
            valid_mask = depth_in_meters > 0

        z = depth_in_meters[valid_mask]
        x = (x_grid[valid_mask] - cx) * z / fx
        y = (y_grid[valid_mask] - cy) * z / fy

        points = np.vstack((x, y, z)).T

        if points.shape[0] < 3:
            print("Not enough points for plane fitting.")
            return (color_image * 255).astype(np.uint8), []

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Perform plane segmentation using RANSAC
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=ransac_n,
                                                num_iterations=num_iterations)

        # [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        # Extract inliers (table plane) and outliers (objects)
        # plane_cloud = pcd.select_by_index(inliers)
        object_cloud = pcd.select_by_index(inliers, invert=True)

        # Project object points back to image plane
        object_points = np.asarray(object_cloud.points)
        if object_points.size == 0:
            print("No objects detected.")
            return (color_image * 255).astype(np.uint8), []

        # Compute pixel coordinates
        x_img = (object_points[:, 0] * fx) / object_points[:, 2] + cx
        y_img = (object_points[:, 1] * fy) / object_points[:, 2] + cy

        x_img = np.round(x_img).astype(int)
        y_img = np.round(y_img).astype(int)

        # Create object mask
        object_mask = np.zeros((height, width), dtype=np.uint8)
        valid_pixels = (x_img >= 0) & (x_img < width) & (y_img >= 0) & (y_img < height)
        object_mask[y_img[valid_pixels], x_img[valid_pixels]] = 255

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert color image back to uint8 [0,255] for drawing
        result_image = (color_image * 255).astype(np.uint8).copy()
        object_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(result_image, [cnt], -1, (255, 0, 0), 2)
                object_contours.append(cnt)

        return result_image, object_contours

    def save_depth_image(self, depth_data, directory):
        """Save the depth data as a PNG image."""
        np.save(directory + '/depth_data.npy', depth_data)
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
            normalized_depth[~invalid_mask] = (depth_data[~invalid_mask] - depth_min) / (depth_max - depth_min) * 255

            # Convert to uint8 after normalization
            normalized_depth = normalized_depth.astype(np.uint8)
            
        else:
            normalized_depth = np.full_like(depth_data, 255, dtype=np.uint8)

        # Create a grayscale image from the normalized depth data
        plt.figure(figsize=(10, 8))
        plt.imshow(normalized_depth, cmap='gray')
        plt.colorbar(label='Depth')
        plt.title('Depth Image')
        plt.savefig(directory + '/depth_visualization.png')
        plt.close()
        print("Depth visualization saved as 'depth_visualization.png'")

    def save_color_image(self, color_data, directory):
        """Save the color data as a PNG image."""
        np.save(directory + '/color_data.npy', color_data)
        print("Raw color data saved as 'color_data.npy'")

        # Convert color data to a visualization image
        # Convert BGR to RGB for proper display with matplotlib
        rgb_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_data)
        plt.title('Color Image')
        plt.savefig(directory + '/color_visualization.png')
        plt.close()
        print("Color visualization saved as 'color_visualization.png'")

    def save_images(self, color_image, depth_image, directory):
        """Save the color and depth images as PNG files."""
        # Save color image
        self.save_color_image(color_image, directory)
        self.save_depth_image(depth_image, directory)
        
