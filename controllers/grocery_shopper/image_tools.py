import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImageTools:
    def __init__(self):
        pass

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
