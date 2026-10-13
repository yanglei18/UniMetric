import os
import torch
import numpy as np
import cv2
from PIL import Image
from metric_scale.config.path_config import ensure_dir
from metric_scale.utils.utils import file_utils
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

class MogeDepthEstimator:
    """MogeDepthEstimator - A wrapper for MOGE depth estimation models.
    
    This class provides an easy-to-use interface for loading MOGE models and running depth estimation on image sequences.
    
    Args:
        pretrained_model (str): Pretrained model name or path. Default: "Ruicheng/moge-2-vitl-normal"
        device (str, optional): Device to run the model on. Default: 'cuda' if available, otherwise 'cpu'
        dtype (torch.dtype, optional): Data type to use for inference. Default: float32
    """

    def __init__(self, pretrained_model="Ruicheng/moge-2-vitl-normal", device=None, dtype=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set data type
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        
        # Load the model
        self.model = MoGeModel.from_pretrained(pretrained_model)
        self.model = self.model.to(device=self.device)
    
    def load_images(self, path, interval=1, pixel_limit=2550000, start_frame=0, frame_count=None):
        """Load images from a directory or video file.
        
        Args:
            path (str): Path to directory containing images or a video file (.mp4)
            interval (int, optional): Interval between frames to load. Default: 1
            pixel_limit (int, optional): Maximum number of pixels per image. Images will be resized to fit this limit.
                Default: 2550000
            start_frame (int, optional): Starting frame index. Default: 0
            frame_count (int, optional): Number of frames to load. If None, load all frames from start_frame.
                Default: None
        
        Returns:
            list: List of image paths ready for inference
        """
        sources = []
        
        if os.path.isdir(path):
            print(f"Loading images from directory: {path}")
            filenames = file_utils.natural_sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Calculate end frame
            end_frame = len(filenames) if frame_count is None else start_frame + frame_count
            
            # Iterate through frames with interval, starting from start_frame
            for i in range(start_frame, end_frame, interval):
                if i < len(filenames):
                    img_path = os.path.join(path, filenames[i])
                    try:
                        sources.append(img_path)
                    except Exception as e:
                        print(f"Could not load image {filenames[i]}: {e}")
        else:
            raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")
        
        if not sources:
            print("No images found or loaded.")
            return []
        
        print(f"Found {len(sources)} images/frames.")
        return sources
    
    def estimate_depth(self, image_paths):
        """Run depth estimation on a sequence of images.
        
        Args:
            image_paths (list): List of image paths
        
        Returns:
            dict: Dictionary containing depth estimation results with keys:
                - 'depth': Depth maps
                - 'conf': Confidence maps (using mask as confidence)
                - 'point_map': Point maps
                - 'extrinsics': Extrinsic matrices
                - 'intrinsics': Intrinsic matrices
                - 'processed_images': Processed images
                - 'image_paths': Original image paths
        """
        if not image_paths:
            raise ValueError("No images provided for depth estimation.")
        
        # Process images one by one (MOGE processes single images)
        depths = []
        confs = []
        point_maps = []
        extrinsics = []
        intrinsics = []
        processed_images = []
        
        for img_path in image_paths:
            # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
            input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            processed_images.append(input_image)
            
            input_tensor = torch.tensor(input_image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
            
            # Infer
            with torch.no_grad():
                output = self.model.infer(input_tensor)
            
            # Extract results and move to CPU
            depth = output["depth"].cpu()
            mask = output["mask"].cpu()
            # 将depth中的inf替换为0
            depth = torch.where(torch.isinf(depth), torch.tensor(0.0, dtype=depth.dtype, device=depth.device), depth)
            depths.append(depth)
            confs.append(mask)
            # 将point_map中的inf替换为0
            point_map = output["points"].cpu()
            point_map = torch.where(torch.isinf(point_map), torch.tensor(0.0, dtype=point_map.dtype, device=point_map.device), point_map)
            point_maps.append(point_map)
            intrinsics.append(output["intrinsics"].cpu())
            
            # Camera to world transformation matrix
            # Camera coords: X right, Y down, Z forward
            # World coords: Z up (perpendicular to ground)
            # Correct rotation matrix:
            # Xc (right) → Xw (right)
            # Yc (down) → -Zw (since world Z is up)
            # Zc (forward) → Yw (forward)
            rotation = np.array([
                [1, 0, 0],   # Xc → Xw
                [0, 0, 1],   # Zc → Yw
                [0, -1, 0]   # Yc → -Zw
            ], dtype=np.float32)
            translation = np.array([0, 0, 0], dtype=np.float32)
            extrinsic = np.hstack((rotation, translation.reshape(3, 1)))
            extrinsics.append(extrinsic)
        
        # Convert to numpy arrays and adjust shapes
        depths = np.stack(depths)
        confs = np.stack(confs)
        point_maps = np.stack(point_maps)
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)
        processed_images = np.stack(processed_images)
        
        # Prepare results
        results = {
            'depth': depths,
            'conf': confs,
            'point_map': point_maps,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'processed_images': processed_images,
            "image_paths": image_paths
        }
        
        return results
    
    def process(self, path, interval=1, pixel_limit=2550000, start_frame=0, frame_count=None):
        """End-to-end processing: load images and run depth estimation.
        
        Args:
            path (str): Path to directory containing images or a video file (.mp4)
            interval (int, optional): Interval between frames to load. Default: 1
            pixel_limit (int, optional): Maximum number of pixels per image. Default: 2550000
            start_frame (int, optional): Starting frame index. Default: 0
            frame_count (int, optional): Number of frames to load. If None, load all frames from start_frame.
                Default: None
        
        Returns:
            dict: Dictionary containing depth estimation results
        """
        print(f"Loading images from {path}...")
        image_paths = self.load_images(path, interval, pixel_limit, start_frame, frame_count)
        if not image_paths:
            return None
        
        print(f"Loaded {len(image_paths)} images. Running depth estimation...")
        results = self.estimate_depth(image_paths)
        print("Depth estimation complete!")
        return results
    
    def visualize_depth(self, image, depth_map, colormap=cv2.COLORMAP_JET, alpha=0.5):
        """Visualize a single depth map by applying a colormap and blending with the original image.
        
        Args:
            image (np.ndarray): Original RGB image of shape (H, W, 3)
            depth_map (np.ndarray): Depth map of shape (H, W)
            colormap (int, optional): OpenCV colormap to use. Default: cv2.COLORMAP_JET
            alpha (float, optional): Blending alpha value (0.0 - original image, 1.0 - depth map). Default: 0.5
        
        Returns:
            np.ndarray: Visualized depth image of shape (H, W, 3)
        """
        # Normalize depth map to [0, 255] range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min + 1e-6) * 255
        normalized_depth = normalized_depth.astype(np.uint8)
        
        # Apply colormap to depth map
        colored_depth = cv2.applyColorMap(normalized_depth, colormap)
        colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
        
        # Resize colored depth to match original image if needed
        if colored_depth.shape[:2] != image.shape[:2]:
            colored_depth = cv2.resize(colored_depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Blend colored depth with original image
        blended = cv2.addWeighted(image, 1 - alpha, colored_depth, alpha, 0)
        
        return blended
    
    def visualize_all_depth(self, results, colormap=cv2.COLORMAP_JET, alpha=0.5):
        """Visualize all depth maps from the results.
        
        Args:
            results (dict): Depth estimation results containing processed_images and depth attributes
            colormap (int, optional): OpenCV colormap to use. Default: cv2.COLORMAP_JET
            alpha (float, optional): Blending alpha value. Default: 0.5
        
        Returns:
            list: List of visualized depth images
        """
        visualized_images = []
        
        # Get processed images and depth maps from results
        processed_images = results['processed_images']  # Shape: [N, H, W, 3] uint8
        depth_maps = results['depth']  # Shape: [N, H, W] float32
        
        # Visualize each depth map
        for i in range(processed_images.shape[0]):
            image = processed_images[i]
            depth_map = depth_maps[i]
            visualized = self.visualize_depth(image, depth_map, colormap, alpha)
            visualized_images.append(visualized)
        
        return visualized_images
    
    def save_visualizations(self, results, output_dir, start_frame=0, frame_ids=None, colormap=cv2.COLORMAP_JET, alpha=0.5):
        """Save all visualized depth images to the specified directory.
        
        Args:
            results (dict): Depth estimation results containing processed_images and depth attributes
            output_dir (str): Directory to save visualized images
            start_frame (int, optional): Starting frame index. Default: 0
            colormap (int, optional): OpenCV colormap to use. Default: cv2.COLORMAP_JET
            alpha (float, optional): Blending alpha value. Default: 0.5
        """
        # Create output directory if it doesn't exist
        ensure_dir(output_dir)
        
        # Generate visualized images
        visualized_images = self.visualize_all_depth(results, colormap, alpha)
        
        # Save each visualized image
        for i in range(len(visualized_images)):
            if frame_ids and i < len(frame_ids):
                frame_prefix = f"{frame_ids[i]}_camera1"
            else:
                actual_frame_idx = start_frame + i
                frame_prefix = f"{actual_frame_idx:05d}_camera1"
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(visualized_images[i], cv2.COLOR_RGB2BGR)
            # Save image with index
            img_path = os.path.join(output_dir, f"depth_visualization_{frame_prefix}.png")
            cv2.imwrite(img_path, img_bgr)
            print(f"Saved visualization: {img_path}")
        print(f"All {len(visualized_images)} visualizations saved to {output_dir}")
    
    def depth_to_pointmap(self, depth, intrinsics):
        """Convert depth map to point map using camera intrinsics.
        
        Args:
            depth (np.ndarray): Depth map of shape (H, W)
            intrinsics (np.ndarray): Camera intrinsics matrix of shape (3, 3)
        
        Returns:
            np.ndarray: Point map of shape (H, W, 3)
        """
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        ones = np.ones_like(depth)
        pixels = np.stack([u, v, ones], axis=-1)   # (H, W, 3)
        K_inv = np.linalg.inv(intrinsics)
        # Apply inverse intrinsics
        cam_dirs = pixels @ K_inv.T                # (H, W, 3)
        # Scale by depth
        points = cam_dirs * depth[..., None]
        return points
    
    def save_results(self, results, output_dir, start_frame=0, frame_ids=None, save_images=True, save_depth=True, save_conf=True, save_pointmap=True, save_camera_params=True):
        """Save depth estimation results frame by frame to the specified directory.
        
        Args:
            results (dict): Depth estimation results containing depth, conf, point_map, intrinsics, and extrinsics
            output_dir (str): Main output directory
            start_frame (int, optional): Starting frame index. Default: 0
            save_images (bool, optional): Whether to save processed images. Default: True
            save_depth (bool, optional): Whether to save depth maps. Default: True
            save_conf (bool, optional): Whether to save confidence maps. Default: True
            save_pointmap (bool, optional): Whether to save point maps. Default: True
            save_camera_params (bool, optional): Whether to save camera parameters. Default: True
        """
        # Create main output directory
        ensure_dir(output_dir)
        
        # Create subdirectories
        if save_images:
            images_dir = os.path.join(output_dir, "images")
            ensure_dir(images_dir)
        
        if save_depth:
            depth_dir = os.path.join(output_dir, "depth")
            ensure_dir(depth_dir)
        
        if save_conf:
            conf_dir = os.path.join(output_dir, "conf")
            ensure_dir(conf_dir)
        
        if save_camera_params:
            camera_params_dir = os.path.join(output_dir, "camera_params")
            ensure_dir(camera_params_dir)
        
        if save_pointmap:
            pointmap_dir = os.path.join(output_dir, "pointmap")
            ensure_dir(pointmap_dir)
        
        # Get number of frames
        num_frames = results['depth'].shape[0]
        # Save results frame by frame
        for frame_idx in range(num_frames):
            if frame_ids and frame_idx < len(frame_ids):
                frame_prefix = f"{frame_ids[frame_idx]}_camera1"
            else:
                actual_frame_idx = start_frame + frame_idx
                frame_prefix = f"{actual_frame_idx:05d}_camera1"
            
            # Save processed image
            if save_images and 'processed_images' in results:
                image = results['processed_images'][frame_idx]
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_path = os.path.join(images_dir, f"{frame_prefix}.png")
                cv2.imwrite(img_path, img_bgr)
            
            # Save depth map
            if save_depth:
                depth = results['depth'][frame_idx]
                depth_path = os.path.join(depth_dir, f"{frame_prefix}.npy")
                np.save(depth_path, depth)
            
            # Save confidence map
            if save_conf:
                conf = results['conf'][frame_idx]
                conf_path = os.path.join(conf_dir, f"{frame_prefix}.npy")
                np.save(conf_path, conf)
            
            if save_pointmap:
                if 'point_map' in results:
                    pointmap = results['point_map'][frame_idx]
                else:
                    pointmap = self.depth_to_pointmap(results['depth'][frame_idx], results['intrinsics'][frame_idx])
                pointmap_path = os.path.join(pointmap_dir, f"{frame_prefix}.npy")
                np.save(pointmap_path, pointmap)
            
            # Save camera parameters
            if save_camera_params:
                # Save intrinsics and extrinsics as JSON
                import json
                intrinsic = results['intrinsics'][frame_idx]
                extrinsic = results['extrinsics'][frame_idx]
                
                # Convert numpy arrays to lists for JSON serialization
                camera_params = {
                    "intrinsics": intrinsic.tolist(),
                    "extrinsics": extrinsic.tolist()
                }
                
                # Save as JSON file
                params_path = os.path.join(camera_params_dir, f"{frame_prefix}.json")
                with open(params_path, 'w') as f:
                    json.dump(camera_params, f, indent=2)
        print(f"All {num_frames} frames saved to {output_dir}")

# Example usage if run as script
if __name__ == "__main__":
    # Create estimator instance
    estimator = MogeDepthEstimator()
    
    # Process images
    example_path = "p3_images"
    results = estimator.process(example_path, interval=1)
    
    # Access outputs
    if results:
        print(f"Depth shape: {results['depth'].shape}")
        print(f"Confidence shape: {results['conf'].shape}")
        print(f"Point map shape: {results['point_map'].shape}")
        print(f"Extrinsics shape: {results['extrinsics'].shape}")
        print(f"Intrinsics shape: {results['intrinsics'].shape}")
        
        # Save results frame by frame
        estimator.save_results(
            results, 
            "moge_depth_results",
            save_images=True,
            save_depth=True,
            save_conf=True,
            save_pointmap=True,
            save_camera_params=True
        )
