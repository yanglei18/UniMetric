#!/usr/bin/env python3
"""
Convert nuScenes dataset to OpenV2V dataset format.

This script transforms nuScenes sensor data, calibrations, and annotations
into the OpenV2V directory structure with YAML metadata files.

Usage:
    python prepare_nuscenes_dataset.py --nuscenes_root /path/to/nuscenes \
        --output_dir ./openv2v_output --version v1.0-mini
"""

import argparse
import os
import shutil
from collections import OrderedDict
from os import path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus


# Camera mapping from nuScenes to OpenV2V (all 6 cameras)
CAMERA_MAPPING = {
    'CAM_FRONT': 'camera1',       # Frontal camera
}


def quaternion_to_euler(q: Quaternion) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, yaw, pitch) in degrees.
    
    Args:
        q: Quaternion object
        
    Returns:
        Tuple of (roll, yaw, pitch) in degrees, clamped to reasonable ranges
    """
    yaw, pitch, roll = q.yaw_pitch_roll
    
    # Convert to degrees
    roll_deg = float(np.degrees(roll))
    yaw_deg = float(np.degrees(yaw))
    pitch_deg = float(np.degrees(pitch))
    
    # Clamp angles to reasonable ranges
    # Roll: [-180, 180]
    roll_deg = ((roll_deg + 180) % 360) - 180
    # Yaw: [-180, 180]
    yaw_deg = ((yaw_deg + 180) % 360) - 180
    # Pitch: [-90, 90]
    pitch_deg = max(-90, min(90, pitch_deg))
    
    return roll_deg, yaw_deg, pitch_deg


def get_pose_6dof(translation: List[float], rotation: Quaternion) -> List[float]:
    """
    Extract 6-DOF pose (x, y, z, roll, yaw, pitch) from translation and rotation.
    
    Args:
        translation: [x, y, z] position
        rotation: Quaternion rotation
        
    Returns:
        List of [x, y, z, roll, yaw, pitch]
    """
    roll, yaw, pitch = quaternion_to_euler(rotation)
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        roll,
        yaw,
        pitch
    ]


def get_sensor_world_pose(nusc: NuScenes, sensor_token: str) -> List[float]:
    """
    Get sensor pose in world coordinate system.
    
    Args:
        nusc: NuScenes instance
        sensor_token: Token of the sensor sample_data
        
    Returns:
        List of [x, y, z, roll, yaw, pitch]
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # Get ego pose
    ego_translation = np.array(pose_record['translation'])
    ego_rotation = Quaternion(pose_record['rotation'])
    
    # Get sensor to ego transform
    sensor_translation = np.array(cs_record['translation'])
    sensor_rotation = Quaternion(cs_record['rotation'])
    
    # Transform sensor position to world coordinates
    sensor_world_pos = ego_translation + ego_rotation.rotate(sensor_translation)
    sensor_world_rot = sensor_rotation * ego_rotation
    return get_pose_6dof(sensor_world_pos.tolist(), sensor_world_rot)

def compute_lidar_to_camera_extrinsic(nusc: NuScenes, sample: Dict, 
                                       sensor_token: str) -> np.ndarray:
    """
    Compute extrinsic matrix from LiDAR to camera coordinate frame.
    This is used for projecting LiDAR points onto camera images.
    
    Args:
        nusc: NuScenes instance
        sample: Sample record
        sensor_token: Token of the camera sample_data
        
    Returns:
        4x4 extrinsic transformation matrix (LiDAR -> Camera)
    """
    # Get camera calibration
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    cam_translation = np.array(cs_record['translation'])
    cam_rotation = Quaternion(cs_record['rotation'])
    
    # Get LiDAR calibration
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd_rec = nusc.get('sample_data', lidar_token)
    lidar_cs_record = nusc.get('calibrated_sensor', lidar_sd_rec['calibrated_sensor_token'])
    lidar_translation = np.array(lidar_cs_record['translation'])
    lidar_rotation = Quaternion(lidar_cs_record['rotation'])
    
    # LiDAR to ego
    lidar_to_ego_r = lidar_rotation.rotation_matrix
    lidar_to_ego_t = lidar_translation
    
    # Camera to ego (we need inverse: ego to camera)
    cam_to_ego_r = cam_rotation.rotation_matrix
    cam_to_ego_t = cam_translation
    ego_to_cam_r = cam_to_ego_r.T
    ego_to_cam_t = -ego_to_cam_r @ cam_to_ego_t
    
    # LiDAR to Camera: lidar -> ego -> camera
    lidar_to_cam_r = ego_to_cam_r @ lidar_to_ego_r
    lidar_to_cam_t = ego_to_cam_r @ lidar_to_ego_t + ego_to_cam_t
    
    # Create 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = lidar_to_cam_r
    extrinsic[:3, 3] = lidar_to_cam_t
    
    return extrinsic





def get_camera_params(nusc: NuScenes, sample: Dict, cam_name: str) -> Optional[Dict]:
    """
    Get camera parameters including coors, extrinsic and intrinsic.
    
    Args:
        nusc: NuScenes instance
        sample: Sample record
        cam_name: Camera name (e.g., 'CAM_FRONT')
        
    Returns:
        Dict with camera parameters or None if camera not available
    """
    if cam_name not in sample['data']:
        return None
        
    cam_token = sample['data'][cam_name]
    sd_rec = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    
    # Get camera intrinsic
    intrinsic = cs_record['camera_intrinsic']
    
    # Get camera world pose (coors)
    coors = get_sensor_world_pose(nusc, cam_token)
    
    # Calculate extrinsic (camera to LiDAR transform - actually LiDAR -> Camera for projection)
    extrinsic = compute_lidar_to_camera_extrinsic(nusc, sample, cam_token)
    
    return {
        'coors': coors,
        'extrinsic': extrinsic.tolist(),
        'intrinsic': intrinsic
    }











def create_frame_yaml(nusc: NuScenes, sample: Dict) -> OrderedDict:
    """
    Create YAML data for a single frame in OpenV2V format.
    
    Args:
        nusc: NuScenes instance
        sample: Sample record
        
    Returns:
        OrderedDict containing all frame metadata
    """
    yaml_data = OrderedDict()
    
    # Camera parameters (only CAM_FRONT)
    for nus_cam, openv2v_cam in CAMERA_MAPPING.items():
        cam_params = get_camera_params(nusc, sample, nus_cam)
        if cam_params:
            yaml_data[openv2v_cam] = cam_params
    
    return yaml_data


def lidar_to_pcd(input_path: str, output_path: str) -> bool:
    """
    Convert nuScenes LiDAR bin file to PCD format.
    
    Args:
        input_path: Path to input .bin file
        output_path: Path to output .pcd file
        
    Returns:
        True if successful
    """
    try:
        # Load nuScenes LiDAR data (x, y, z, intensity, ring_index)
        points = np.fromfile(input_path, dtype=np.float32).reshape(-1, 5)
        xyz = points[:, :3]
        intensity = points[:, 3]
        
        # Write PCD file
        num_points = xyz.shape[0]
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""
        with open(output_path, 'w') as f:
            f.write(header)
            for i in range(num_points):
                f.write(f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} {intensity[i]}\n")
        
        return True
    except Exception as e:
        print(f"Error converting {input_path} to PCD: {e}")
        return False


def convert_scene(nusc: NuScenes, scene: Dict, output_dir: str) -> None:
    """
    Convert a single nuScenes scene to OpenV2V format.
    
    Args:
        nusc: NuScenes instance
        scene: Scene record
        output_dir: Base output directory
    """
    scene_name = scene['name']
    
    scene_output_dir = osp.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    print(f"Converting scene: {scene_name}")
    
    # Iterate through all samples in the scene
    sample_token = scene['first_sample_token']
    frame_idx = 0
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        frame_str = f"{frame_idx:05d}"
        
        # 1. Create YAML metadata
        yaml_data = create_frame_yaml(nusc, sample)
        yaml_path = osp.join(scene_output_dir, f"{frame_str}.yaml")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dict(yaml_data), f, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False)
        
        # 2. Copy/convert LiDAR data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path = nusc.get_sample_data_path(lidar_token)
        pcd_path = osp.join(scene_output_dir, f"{frame_str}.pcd")
        lidar_to_pcd(lidar_path, pcd_path)
        
        # 3. Copy camera images
        for nus_cam, openv2v_cam in CAMERA_MAPPING.items():
            if nus_cam in sample['data']:
                cam_token = sample['data'][nus_cam]
                cam_path = nusc.get_sample_data_path(cam_token)
                
                # Determine output filename
                cam_output = osp.join(scene_output_dir, f"{frame_str}_{openv2v_cam}.jpg")
                
                if osp.exists(cam_path):
                    shutil.copy2(cam_path, cam_output)
        
        # Move to next sample
        sample_token = sample['next']
        frame_idx += 1
    
    print(f"  Converted {frame_idx} frames")


def main():
    parser = argparse.ArgumentParser(
        description='Convert nuScenes dataset to OpenV2V format'
    )
    parser.add_argument(
        '--nuscenes_root', 
        type=str, 
        default="/home/yanglei/SAM4D/data/nuScenes",
        help='Path to nuScenes dataset root'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="/home/yanglei/SAM4D/data/nuScenes-openv2v",
        help='Output directory for OpenV2V format'
    )
    parser.add_argument(
        '--version', 
        type=str, 
        default='v1.0-mini',
        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
        help='nuScenes dataset version'
    )
    args = parser.parse_args()
    
    # Initialize nuScenes
    print(f"Loading nuScenes {args.version} from {args.nuscenes_root}")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert all scenes directly to output directory
    for scene in nusc.scene:
        convert_scene(nusc, scene, args.output_dir)
    
    print(f"\nConversion complete! Output saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
