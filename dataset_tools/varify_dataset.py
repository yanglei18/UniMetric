#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import glob
import numpy as np
import yaml
import cv2

from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt


def _compute_bev_bounds(
    points_xy,
    min_span=20.0,
    padding_ratio=0.08,
    max_padding=8.0,
    lower_percentile=1.0,
    upper_percentile=99.0,
    core_fraction=0.80,
    min_core_points=200,
):
    """Compute a compact BEV view window while ignoring sparse outliers."""
    if points_xy is None or len(points_xy) == 0:
        return -50.0, 50.0, -50.0, 50.0

    points_xy = np.asarray(points_xy, dtype=np.float32)
    finite_mask = np.isfinite(points_xy).all(axis=1)
    points_xy = points_xy[finite_mask]
    if len(points_xy) == 0:
        return -50.0, 50.0, -50.0, 50.0

    if len(points_xy) >= min_core_points and 0.0 < core_fraction < 1.0:
        center = np.median(points_xy, axis=0)
        scale = np.percentile(np.abs(points_xy - center), 75, axis=0)
        scale = np.maximum(scale, 1.0)
        normalized_offsets = (points_xy - center) / scale
        distances = np.linalg.norm(normalized_offsets, axis=1)
        distance_threshold = np.percentile(distances, core_fraction * 100.0)
        core_points = points_xy[distances <= distance_threshold]
        if len(core_points) >= max(20, int(0.1 * len(points_xy))):
            points_xy = core_points

    x_values = points_xy[:, 0]
    y_values = points_xy[:, 1]

    if len(points_xy) >= 20 and 0.0 <= lower_percentile < upper_percentile <= 100.0:
        x_min = float(np.percentile(x_values, lower_percentile))
        x_max = float(np.percentile(x_values, upper_percentile))
        y_min = float(np.percentile(y_values, lower_percentile))
        y_max = float(np.percentile(y_values, upper_percentile))
    else:
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    span = max(x_span, y_span, min_span)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_extent = 0.5 * span

    padding = min(max(span * padding_ratio, 2.0), max_padding)
    half_extent += padding

    return (
        x_center - half_extent,
        x_center + half_extent,
        y_center - half_extent,
        y_center + half_extent,
    )


def _compute_start_centered_bev_bounds(lidar_poses, span=3.0):
    """Return a fixed square BEV range centered at the first LiDAR pose."""
    if not lidar_poses:
        return None

    start_xy = np.asarray(lidar_poses[0][:2], dtype=np.float32)
    if not np.isfinite(start_xy).all():
        return None

    half_span = 0.5 * float(span)
    return (
        float(start_xy[0] - half_span),
        float(start_xy[0] + half_span),
        float(start_xy[1] - half_span),
        float(start_xy[1] + half_span),
    )


def _crop_bev_to_content(bev_image, margin_pixels=24):
    """Crop empty black borders so the point cloud occupies most of the output."""
    if bev_image is None or bev_image.size == 0:
        return bev_image

    non_black_mask = np.any(bev_image > 0, axis=2)
    if not np.any(non_black_mask):
        return bev_image

    rows, cols = np.where(non_black_mask)
    y_min = max(int(rows.min()) - margin_pixels, 0)
    y_max = min(int(rows.max()) + margin_pixels + 1, bev_image.shape[0])
    x_min = max(int(cols.min()) - margin_pixels, 0)
    x_max = min(int(cols.max()) + margin_pixels + 1, bev_image.shape[1])
    return bev_image[y_min:y_max, x_min:x_max]

class FrameLoader:
    def __init__(self, data_path, camera_id=0, frame_id="00000"):
        self.data_path = data_path
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.image = None
        self.image_path = os.path.join(data_path, f"{frame_id}_camera{camera_id}.jpg")
        # self.image_path = os.path.join(data_path, f"{frame_id}.jpg")
        self.img_width = None
        self.img_height = None
        self.points = None
        self.yaml_data = None
        self.lidar2camera = None
        self.camera2world = None
        self.world2camera = None
        self.intrinsic = None
        self.img_np = None
        self.object_list = []
        
    def x_to_world(self, pose):
        """
        The transformation matrix from x-coordinate system to carla world system
        Also is the pose in world coordinate: T_world_x
        
        Parameters
        ----------
        pose : list
            [x, y, z, roll, yaw, pitch], degree
        
        Returns
        -------
        matrix : np.ndarray
            The transformation matrix.
        """
        x, y, z, roll, yaw, pitch = pose[:]
        # used for rotation matrix
        c_y = np.cos(np.radians(yaw))
        s_y = np.sin(np.radians(yaw))
        c_r = np.cos(np.radians(roll))
        s_r = np.sin(np.radians(roll))
        c_p = np.cos(np.radians(pitch))
        s_p = np.sin(np.radians(pitch))
        matrix = np.identity(4)
        # translation matrix
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        # rotation matrix
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix
    
    def get_object_bbox_vertices(self, obj):
        """Generate 8 vertices of 3D bounding box for an object (vehicle, pedestrian, cyclist, etc.)"""
        location = np.array(obj['location'])
        extent = np.array(obj['extent'])
        angle = np.array(obj['angle'])
        
        # Convert angles from degrees to radians
        roll, yaw, pitch = angle
        
        # Calculate cosine and sine values for each angle
        c_y = np.cos(np.radians(yaw))
        s_y = np.sin(np.radians(yaw))
        c_r = np.cos(np.radians(roll))
        s_r = np.sin(np.radians(roll))
        c_p = np.cos(np.radians(pitch))
        s_p = np.sin(np.radians(pitch))
        
        # Create rotation matrix directly (same as x_to_world function)
        R = np.identity(3)
        R[0, 0] = c_p * c_y
        R[0, 1] = c_y * s_p * s_r - s_y * c_r
        R[0, 2] = -c_y * s_p * c_r - s_y * s_r
        R[1, 0] = s_y * c_p
        R[1, 1] = s_y * s_p * s_r + c_y * c_r
        R[1, 2] = -s_y * s_p * c_r + c_y * s_r
        R[2, 0] = s_p
        R[2, 1] = -c_p * s_r
        R[2, 2] = c_p * c_r
        
        # Generate 8 basic vertices (centered at origin)
        # extent is [half_length, half_width, half_height]
        l, w, h = extent
        vertices = np.array([
            [l, w, h],    # front-right-top
            [l, -w, h],   # front-left-top
            [-l, -w, h],  # rear-left-top
            [-l, w, h],   # rear-right-top
            [l, w, -h],   # front-right-bottom
            [l, -w, -h],  # front-left-bottom
            [-l, -w, -h], # rear-left-bottom
            [-l, w, -h]   # rear-right-bottom
        ])
        
        # Apply rotation and translation
        vertices = np.dot(vertices, R.T) + location
        
        return vertices
    
    def project_bbox_to_image(self, vertices, global2camera, intrinsic, img_width, img_height):
        """Project 3D bounding box vertices to image plane"""
        # Convert vertices to homogeneous coordinates
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        # Transform from LiDAR/World coordinates to camera coordinates
        vertices_camera = np.dot(global2camera, vertices_homogeneous.T).T
        
        # Filter vertices behind the camera
        visible_mask = vertices_camera[:, 2] > 0
        visible_vertices = vertices_camera[visible_mask]
        
        if len(visible_vertices) == 0:
            return None, None
        
        # Project vertices to image plane
        projected_vertices = np.dot(intrinsic, visible_vertices[:, :3].T).T
        projected_vertices[:, 0] /= projected_vertices[:, 2]
        projected_vertices[:, 1] /= projected_vertices[:, 2]
        
        # Check if any vertex is within image boundaries
        valid_mask = (
            (projected_vertices[:, 0] >= 0) & 
            (projected_vertices[:, 0] < img_width) &
            (projected_vertices[:, 1] >= 0) & 
            (projected_vertices[:, 1] < img_height)
        )
        
        if not np.any(valid_mask):
            return None, None
        
        # Calculate 2D bounding box based on ALL visible projected vertices
        # This ensures the bbox includes all projected vertices
        x_min = np.min(projected_vertices[:, 0])
        y_min = np.min(projected_vertices[:, 1])
        x_max = np.max(projected_vertices[:, 0])
        y_max = np.max(projected_vertices[:, 1])
        
        # Ensure 2D box is within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        
        # Create 2D box as [x_min, y_min, x_max, y_max]
        bbox_2d = np.array([x_min, y_min, x_max, y_max])
        
        return projected_vertices, bbox_2d
    
    def draw_3d_bbox(self, img_np, projected_vertices, color=(0, 255, 0), thickness=2):
        """Draw 3D bounding box on image"""
        # Skip if not enough vertices
        if len(projected_vertices) < 2:
            return img_np
        
        # Define the edges of the bounding box (indices of connected vertices)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Draw each edge only if both vertices exist
        for edge in edges:
            # Check if both vertices are within bounds
            if edge[0] < len(projected_vertices) and edge[1] < len(projected_vertices):
                # Get the two vertices of the edge
                p1 = projected_vertices[edge[0]]
                p2 = projected_vertices[edge[1]]
                
                # Convert to integer pixel coordinates
                x1, y1 = int(round(p1[0])), int(round(p1[1]))
                x2, y2 = int(round(p2[0])), int(round(p2[1]))
                
                # Draw the line
                cv2.line(img_np, (x1, y1), (x2, y2), color, thickness)
        
        return img_np
    
    def project_points_to_bev(self, points, T_tolidar=np.eye(4)):
        """Project LiDAR points to Bird's Eye View (BEV) in LiDAR coordinate system
        
        Parameters
        ----------
        points : np.ndarray
            LiDAR points with shape (N, 3) or (N, 4), containing x, y, z coordinates and optional intensity
        T_tolidar : np.ndarray, optional
            Transformation matrix from input coordinate system to LiDAR coordinate system, by default np.eye(4)
        
        Returns
        -------
        bev_image : np.ndarray
            BEV visualization image
        """
        # Convert points to homogeneous coordinates if needed
        if points.shape[1] == 3:
            points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        else:
            points_homogeneous = points[:, :4]  # Use first 4 columns if intensity is present
        
        # Transform points to LiDAR coordinate system
        points_lidar = np.dot(T_tolidar, points_homogeneous.T).T[:, :3]  # Only keep x, y, z
        
        # Build a compact BEV window from the actual point distribution so the
        # visible content occupies most of the image instead of a tiny region.
        z_min, z_max = -5.0, 5.0    # Only points within this height range

        height_mask = (
            (points_lidar[:, 2] >= z_min) &
            (points_lidar[:, 2] <= z_max)
        )
        height_filtered_points = points_lidar[height_mask]

        x_min, x_max, y_min, y_max = _compute_bev_bounds(height_filtered_points[:, :2])
        resolution = max((x_max - x_min) / 900.0, (y_max - y_min) / 900.0, 0.03)
        bev_width = max(int(np.ceil((y_max - y_min) / resolution)), 300)
        bev_height = max(int(np.ceil((x_max - x_min) / resolution)), 300)

        # Filter points within the region of interest
        mask = ((height_filtered_points[:, 0] >= x_min) & (height_filtered_points[:, 0] <= x_max) &
                (height_filtered_points[:, 1] >= y_min) & (height_filtered_points[:, 1] <= y_max))
        filtered_points = height_filtered_points[mask]
        
        # Create BEV image with black background
        bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        
        if len(filtered_points) == 0:
            return bev_image
        
        # Calculate image coordinates
        # Shift origin to center of image and scale by resolution
        # Note: In LiDAR, x is forward, y is left, z is up
        # In BEV image, we want forward to be up, right to be right
        # So we need to swap x and y, and invert y
        bev_x = ((filtered_points[:, 1] - y_min) / resolution).astype(int)  # y becomes x in BEV
        bev_y = ((x_max - filtered_points[:, 0]) / resolution).astype(int)  # x becomes y in BEV, inverted
        
        # Clip coordinates to image boundaries
        bev_x = np.clip(bev_x, 0, bev_width - 1)
        bev_y = np.clip(bev_y, 0, bev_height - 1)
        
        # Normalize z values for color mapping
        z_values = filtered_points[:, 2]
        normalized_z = (z_values - z_min) / (z_max - z_min)  # Normalize to [0, 1]
        
        # Apply color map: blue for low points, red for high points
        colors = plt.cm.coolwarm(normalized_z)[:, :3]  # Get RGB colors from colormap
        colors_bgr = (colors * 255).astype(np.uint8)[:, ::-1]  # Convert to BGR for OpenCV
        
        # Draw points on BEV image
        for i in range(len(bev_x)):
            x = bev_x[i]
            y = bev_y[i]
            color = tuple(colors_bgr[i].tolist())
            cv2.circle(bev_image, (x, y), 1, color, -1)
        
        return _crop_bev_to_content(bev_image, margin_pixels=18)
        

    def project_points_to_image(self, points, image, lidar2camera, intrinsic):
        """Project 3D points to image plane and draw them"""
        points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        points_camera = np.dot(lidar2camera, points_homogeneous.T).T
        
        # Filter points behind the camera
        visible_mask = points_camera[:, 2] > 0
        visible_points = points_camera[visible_mask]
        
        # Calculate distance to camera (in camera coordinate system)
        distances = np.linalg.norm(visible_points[:, :3], axis=1)
        
        # Project points to image plane
        projected_points = np.dot(intrinsic, visible_points[:, :3].T).T
        projected_points[:, 0] /= projected_points[:, 2]
        projected_points[:, 1] /= projected_points[:, 2]
        
        # Filter points within image boundaries
        img_width, img_height = image.size
        valid_mask = (
            (projected_points[:, 0] >= 0) & 
            (projected_points[:, 0] < img_width) &
            (projected_points[:, 1] >= 0) & 
            (projected_points[:, 1] < img_height)
        )
        valid_projected_points = projected_points[valid_mask]
        valid_distances = distances[valid_mask]
        
        # Normalize distances for color mapping
        if len(valid_distances) > 0:
            min_dist = valid_distances.min()
            max_dist = valid_distances.max()
            normalized_dist = (valid_distances - min_dist) / (max_dist - min_dist) if max_dist > min_dist else np.zeros_like(valid_distances)
        else:
            normalized_dist = np.array([])
        
        # Apply color map: coolwarm with reversed direction (blue for close, red for far)
        # Convert from matplotlib colormap to BGR for OpenCV
        colors = plt.cm.coolwarm(1 - normalized_dist)  # 1-normalized_dist反转颜色方向
        colors_bgr = (colors[:, :3] * 255).astype(np.uint8)[:, ::-1]
        
        # Draw points on image
        img_np = np.array(image)
        for point, color in zip(valid_projected_points, colors_bgr):
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(img_np, (x, y), 2, tuple(color.tolist()), -1)
        
        return img_np
    
    def load_image(self):
        """Load image from file"""
        image_path = os.path.join(self.data_path, f"{self.frame_id}_camera{self.camera_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found, skipping...")
            return False
        
        self.image = Image.open(image_path)
        self.img_width, self.img_height = self.image.size
        return True
    
    def load_lidar(self):
        """Load point cloud data"""
        lidar_path = os.path.join(self.data_path, f"{self.frame_id}.pcd")
        # Read PCD file using Open3D to support both ASCII and binary formats
        pcd = o3d.io.read_point_cloud(lidar_path)
        self.points = np.asarray(pcd.points)
        # If the PCD file doesn't have intensity information, add it as zeros
        if self.points.shape[1] == 3:
            intensity = np.zeros((self.points.shape[0], 1), dtype=np.float32)
            self.points = np.hstack((self.points, intensity))
        return True
    
    def load_yaml(self):
        """Load YAML configuration file"""
        yaml_path = os.path.join(self.data_path, f"{self.frame_id}.yaml")
        with open(yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
        return True
    
    def get_camera_parameters(self):
        """Extract camera intrinsic and extrinsic parameters"""
        if self.yaml_data is None:
            self.load_yaml()
        
        self.lidar2camera = np.array(self.yaml_data[f'camera{self.camera_id}']['extrinsic'])
        camera_pose = self.yaml_data[f'camera{self.camera_id}']['cords']
        self.camera2world = self.x_to_world(camera_pose)
        self.world2camera = np.linalg.inv(self.camera2world)
        self.intrinsic = np.array(self.yaml_data[f'camera{self.camera_id}']['intrinsic'])
    
    def get_objects(self):
        """Extract objects from YAML data"""
        self.object_list = []
        # Define object types to process
        # object_types = ['vehicles', 'pedestrians', 'cyclists']
        object_types = ['vehicles', 'cyclists']
        
        # Process each object type
        for obj_type in object_types:
            if obj_type in self.yaml_data:
                objects = self.yaml_data[obj_type]
                # Process each object
                for obj_id, obj in objects.items():
                    # Generate 3D bounding box vertices
                    vertices = self.get_object_bbox_vertices(obj)
                    # Project bounding box to image
                    projected_vertices, bbox_2d = self.project_bbox_to_image(vertices, self.world2camera, self.intrinsic, self.img_width, self.img_height)
                    
                    if projected_vertices is not None:
                        # Set color based on object type
                        if obj_type == 'vehicles':
                            if 'object_type' in obj:
                                if obj['object_type'] == 'Bus':
                                    color = (0, 0, 255)  # Red for buses
                                elif obj['object_type'] == 'Car':
                                    color = (0, 255, 0)  # Green for cars
                                else:
                                    color = (255, 0, 0)  # Blue for other vehicles
                            else:
                                color = (255, 255, 0)  # Yellow for vehicles by default
                        elif obj_type == 'pedestrians':
                            color = (128, 0, 128)  # Purple for pedestrians
                        elif obj_type == 'cyclists':
                            color = (0, 165, 255)  # Orange for cyclists
                        else:
                            color = (255, 255, 255)  # White for unknown objects
                        # Draw 3D bounding box
                        self.img_np = self.draw_3d_bbox(self.img_np, projected_vertices, color=color)
                        # Draw 2D bounding box
                        # x_min, y_min, x_max, y_max = bbox_2d
                        # cv2.rectangle(self.img_np, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                        # Calculate center of 2D bounding box
                        # center_x = int((x_min + x_max) / 2)
                        # center_y = int((y_min + y_max) / 2)
                        # Calculate actual dimension
                        dimension = 2 * np.array(obj["extent"])
                        # Format dimension as string
                        # dim_str = f"{dimension[0]:.2f}x{dimension[1]:.2f}x{dimension[2]:.2f}"
                        # Draw dimension text at center of 2D box
                        # cv2.putText(self.img_np, dim_str, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                        
                        self.object_list.append({"type": obj["object_type"], "bbox": bbox_2d, 
                                                 "relative_location": 0.3 * np.array(obj["location"]), "relative_dimension": 0.3 * 2 * np.array(obj["extent"]), 
                                                 "gt_location": np.array(obj["location"]), "gt_dimension": 2 * np.array(obj["extent"])})
                        

    def process_data(self):
        """Process data by projecting points and drawing bounding boxes"""
        # Load all necessary data
        if not self.load_image():
            return False
        self.load_lidar()
        self.load_yaml()
        self.get_camera_parameters()
        # Project LiDAR points to image
        self.img_np = self.project_points_to_image(self.points, self.image, self.lidar2camera, self.intrinsic)
        # Process objects and draw bounding boxes
        self.get_objects()
        return True
    
    def save_result(self, output_path=None):
        """Save processed image"""
        if output_path is None:
            output_path = f"projected_bboxes_camera{self.camera_id}.jpg"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            ensure_dir(output_dir)
        
        cv2.imwrite(output_path, cv2.cvtColor(self.img_np, cv2.COLOR_RGB2BGR))
        print(f"Saved result to {output_path}")
        return output_path

def ensure_dir(dir_path):
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_frame_ids(data_path):
    """Extract frame IDs from the dataset files"""
    # Get all PCD files and extract frame IDs
    pcd_files = glob.glob(os.path.join(data_path, "*.pcd"))
    frame_ids = []
    
    for pcd_file in pcd_files:
        # Skip radar PCD files
        if "_radar" in pcd_file:
            continue
        
        # Extract frame ID from filename
        frame_id = os.path.basename(pcd_file).split(".")[0]
        frame_ids.append(frame_id)
    
    # Sort frame IDs numerically
    frame_ids.sort(key=lambda x: int(x))
    return frame_ids


def get_sequence_paths(data_path):
    """Return sequence directories from either a sequence path or a dataset root."""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data path does not exist or is not a directory: {data_path}")

    if get_frame_ids(data_path):
        return [data_path]

    sequence_paths = []
    for child_name in sorted(os.listdir(data_path)):
        child_path = os.path.join(data_path, child_name)
        if os.path.isdir(child_path) and get_frame_ids(child_path):
            sequence_paths.append(child_path)

    if not sequence_paths:
        raise ValueError(f"No sequence directories with .pcd frames found under: {data_path}")

    return sequence_paths


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system
    Also is the pose in world coordinate: T_world_x
    
    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch], degree
    
    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]
    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def visualize_frame(data_path, frame_id, output_dir):
    """Visualize a single frame and save results"""
    print(f"Processing frame {frame_id}...")
    
    import cv2
    import numpy as np

    def get_available_camera_ids():
        available_camera_ids = []
        for camera_id in [0, 1, 2]:
            image_path = os.path.join(data_path, f"{frame_id}_camera{camera_id}.jpg")
            if os.path.exists(image_path):
                available_camera_ids.append(camera_id)

        if not available_camera_ids:
            print(f"No camera images found for frame {frame_id}, skipping...")
            return []

        if 1 in available_camera_ids and 0 not in available_camera_ids and 2 not in available_camera_ids:
            return [1]

        return available_camera_ids
    
    # Collect processed images from all cameras
    camera_images = []
    available_camera_ids = get_available_camera_ids()
    if not available_camera_ids:
        return
    print(f"Using camera views: {available_camera_ids}")
    save_individual_views = len(available_camera_ids) == 1
    
    # Process all camera views
    for camera_id in available_camera_ids:
        # Create FrameLoader instance
        loader = FrameLoader(data_path, camera_id=camera_id, frame_id=frame_id)
        
        # Process the frame
        if loader.process_data():
            if save_individual_views:
                output_path = os.path.join(output_dir, f"{frame_id}_camera{camera_id}_visualized.jpg")
                loader.save_result(output_path)
            # Add to camera images list for stitching
            camera_images.append(loader.img_np)
    
    # Stitch camera images horizontally if multiple cameras are available
    if len(camera_images) > 1:
        # Ensure all images have the same height
        heights = [img.shape[0] for img in camera_images]
        max_height = max(heights)
        
        # Resize images to match max height if needed
        resized_images = []
        for img in camera_images:
            if img.shape[0] != max_height:
                # Calculate aspect ratio to maintain width
                aspect_ratio = img.shape[1] / img.shape[0]
                new_width = int(aspect_ratio * max_height)
                resized = cv2.resize(img, (new_width, max_height))
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        # Stitch images horizontally
        stitched_image = cv2.hconcat(resized_images)
        
        # Save stitched image
        stitched_output_path = os.path.join(output_dir, f"{frame_id}_stitched_cameras.jpg")
        ensure_dir(os.path.dirname(stitched_output_path))
        cv2.imwrite(stitched_output_path, cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
        print(f"Saved stitched cameras visualization to {stitched_output_path}")
    
    # Also generate BEV visualization for LiDAR data
    '''
    loader = FrameLoader(data_path, frame_id=frame_id)
    if loader.load_lidar():
        bev_image = loader.project_points_to_bev(loader.points)
        bev_output_path = os.path.join(output_dir, f"{frame_id}_bev.jpg")
        ensure_dir(os.path.dirname(bev_output_path))
        cv2.imwrite(bev_output_path, bev_image)
        print(f"Saved BEV visualization to {bev_output_path}")
    '''

def collect_lidar_poses(data_path, frame_ids):
    """Collect LiDAR poses from all frames to form trajectory"""
    lidar_poses = []
    
    for frame_id in frame_ids:
        # Load YAML data to get lidar pose
        yaml_path = os.path.join(data_path, f"{frame_id}.yaml")
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Get lidar pose (x, y, z, roll, yaw, pitch)
        lidar_pose = yaml_data.get('lidar_pose', [0, 0, 0, 0, 0, 0])
        lidar_poses.append(lidar_pose)
    
    print(f"Collected {len(lidar_poses)} LiDAR poses")
    return lidar_poses


def accumulate_points(data_path, frame_ids):
    """Accumulate LiDAR points from all frames into a single point cloud"""
    accumulated_points = []
    
    for frame_id in frame_ids:
        print(f"Accumulating points from frame {frame_id}...")
        
        # Load LiDAR data
        loader = FrameLoader(data_path, frame_id=frame_id)
        if loader.load_lidar():
            # Load YAML data to get lidar pose
            yaml_path = os.path.join(data_path, f"{frame_id}.yaml")
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Get lidar pose
            lidar_pose = yaml_data.get('lidar_pose', [0, 0, 0, 0, 0, 0])
            
            # Calculate transformation matrix from LiDAR to world coordinates
            lidar_to_world = x_to_world(lidar_pose)
            
            # Transform LiDAR points to world coordinates
            points = loader.points
            # Only use x, y, z coordinates (first 3 columns), exclude intensity
            points_xyz = points[:, :3]
            # Convert to homogeneous coordinates
            points_homogeneous = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
            # Transform points
            points_world = np.dot(lidar_to_world, points_homogeneous.T).T[:, :3]
            
            # Add to accumulated points
            accumulated_points.append(points_world)
    
    # Combine all points
    if accumulated_points:
        accumulated_points = np.vstack(accumulated_points)
        accumulated_points = accumulated_points[np.isfinite(accumulated_points).all(axis=1)]
    else:
        accumulated_points = np.array([])
    
    print(f"Accumulated {len(accumulated_points)} points from {len(frame_ids)} frames")
    return accumulated_points


def visualize_accumulated_points(points, lidar_poses, output_path):
    """Visualize accumulated points in BEV with LiDAR trajectory"""
    if len(points) == 0:
        print("No points to visualize")
        return
    
    import cv2
    import numpy as np
    
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        print("No finite points to visualize")
        return

    fixed_bounds = _compute_start_centered_bev_bounds(lidar_poses, span=3.0)
    if fixed_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_bounds
    else:
        x_min, x_max, y_min, y_max = _compute_bev_bounds(
            points[:, :2],
            min_span=3.0,
            padding_ratio=0.0,
            max_padding=0.0,
        )

    bev_width = 900
    bev_height = 900
    resolution = max((x_max - x_min) / bev_height, (y_max - y_min) / bev_width, 1e-6)
    
    # Create BEV image
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    
    # Process points for BEV visualization
    if len(points) > 0:
        # Filter points within range
        mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
        filtered_points = points[mask]
        
        if len(filtered_points) > 0:
            # Calculate BEV coordinates
            bev_x = ((filtered_points[:, 1] - y_min) / resolution).astype(int)
            bev_y = ((x_max - filtered_points[:, 0]) / resolution).astype(int)
            
            # Clip coordinates
            bev_x = np.clip(bev_x, 0, bev_width - 1)
            bev_y = np.clip(bev_y, 0, bev_height - 1)
            
            # Color mapping based on z value
            z_min = filtered_points[:, 2].min()
            z_max = filtered_points[:, 2].max()
            if z_max > z_min:
                normalized_z = (filtered_points[:, 2] - z_min) / (z_max - z_min)
            else:
                normalized_z = np.zeros_like(filtered_points[:, 2])
            
            # Apply color map
            import matplotlib.pyplot as plt
            colors = plt.cm.coolwarm(normalized_z)[:, :3]
            colors_bgr = (colors * 255).astype(np.uint8)[:, ::-1]
            
            # Draw points
            for i in range(len(filtered_points)):
                cv2.circle(bev_image, (bev_x[i], bev_y[i]), 1, tuple(colors_bgr[i].tolist()), -1)
    
    # Draw LiDAR trajectory as red dots for every frame
    for i, pose in enumerate(lidar_poses):
        lidar_x, lidar_y, _ = pose[:3]
        if not (x_min <= lidar_x <= x_max and y_min <= lidar_y <= y_max):
            continue
        
        # Calculate BEV coordinates
        bev_x = int((lidar_y - y_min) / resolution)
        bev_y = int((x_max - lidar_x) / resolution)
        
        # Clip coordinates
        bev_x = max(0, min(bev_width - 1, bev_x))
        bev_y = max(0, min(bev_height - 1, bev_y))
        
        # Draw red dot
        cv2.circle(bev_image, (bev_x, bev_y), 3, (0, 0, 255), -1)
    
    # Save BEV image
    ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(output_path, bev_image)
    print(f"Saved accumulated points BEV visualization to {output_path}")
    print(f"BEV range: x [{x_min:.1f}, {x_max:.1f}], y [{y_min:.1f}, {y_max:.1f}]")
    print(f"BEV size: {bev_width}x{bev_height} pixels")


def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize dataset sequence(s) using FrameLoader")
    parser.add_argument("--data_path", type=str, 
                        default="data/official-data/Argoverse/1d676737-4110-3f7e-bec0-0c90f74c248f",
                        help="Path to a sequence directory or a dataset root containing sequence directories")
    args = parser.parse_args()
    
    data_path = args.data_path
    output_root = os.path.join("demo")

    sequence_paths = get_sequence_paths(data_path)
    print(f"Found {len(sequence_paths)} sequence(s) in {data_path}")

    for sequence_path in sequence_paths:
        sequence_name = os.path.basename(os.path.normpath(sequence_path))
        output_dir = os.path.join(output_root, sequence_name)
        ensure_dir(output_dir)

        frame_ids = get_frame_ids(sequence_path)
        if not frame_ids:
            raise ValueError(f"No frames found in sequence: {sequence_path}")

        print(f"\nProcessing sequence {sequence_name}: {len(frame_ids)} frames")

        for frame_id in frame_ids:
            visualize_frame(sequence_path, frame_id, output_dir)

        lidar_poses = collect_lidar_poses(sequence_path, frame_ids)
        accumulated_points = accumulate_points(sequence_path, frame_ids)
        accumulated_output_path = os.path.join(output_dir, "accumulated_points_bev.jpg")
        visualize_accumulated_points(accumulated_points, lidar_poses, accumulated_output_path)

    print(f"\nVisualization complete! Results saved to {output_root}")


if __name__ == "__main__":
    main()
