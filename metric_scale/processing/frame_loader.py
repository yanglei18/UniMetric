#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import os
import cv2
import numpy as np

from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

from metric_scale.config.path_config import ensure_dir

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
        
        # BEV visualization parameters
        resolution = 0.2  # meters per pixel
        bev_width = 500    # pixels (covers 20 meters left and right from center)
        bev_height = 500   # pixels (covers 20 meters front and back from center)
        
        # Define the region of interest in LiDAR coordinates
        # Assuming LiDAR is at origin, looking along positive x-axis
        x_min, x_max = -50.0, 50.0  # 20 meters in front and back
        y_min, y_max = -50.0, 50.0  # 20 meters left and right
        z_min, z_max = -5.0, 5.0    # Only points within this height range
        
        # Filter points within the region of interest
        mask = ((points_lidar[:, 0] >= x_min) & (points_lidar[:, 0] <= x_max) &
                (points_lidar[:, 1] >= y_min) & (points_lidar[:, 1] <= y_max) &
                (points_lidar[:, 2] >= z_min) & (points_lidar[:, 2] <= z_max))
        filtered_points = points_lidar[mask]
        
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
        
        return bev_image
        

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