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
    'CAM_FRONT_RIGHT': 'camera0', # Front right camera
    'CAM_FRONT': 'camera1',       # Frontal camera
    'CAM_FRONT_LEFT': 'camera2',  # Front left camera
    'CAM_BACK': 'camera3',        # Back camera
    'CAM_BACK_LEFT': 'camera4',   # Back left camera
    'CAM_BACK_RIGHT': 'camera5',  # Back right camera
}

# Radar mapping from nuScenes to OpenV2V
RADAR_MAPPING = {
    'RADAR_FRONT': 'radar',
}

# Category mappings for object types
VEHICLE_CATEGORIES = {
    'vehicle.car': 'Car',
    'vehicle.truck': 'Truck',
    'vehicle.bus.bendy': 'Bus',
    'vehicle.bus.rigid': 'Bus',
    'vehicle.trailer': 'Trailer',
    'vehicle.construction': 'Construction',
}

PEDESTRIAN_CATEGORIES = {
    'human.pedestrian.adult': 'Pedestrian',
    'human.pedestrian.child': 'Pedestrian',
    'human.pedestrian.wheelchair': 'Pedestrian',
    'human.pedestrian.stroller': 'Pedestrian',
    'human.pedestrian.personal_mobility': 'Pedestrian',
    'human.pedestrian.police_officer': 'Pedestrian',
    'human.pedestrian.construction_worker': 'Pedestrian',
}

CYCLIST_CATEGORIES = {
    'vehicle.bicycle': 'Cyclist',
    'vehicle.motorcycle': 'Cyclist',
}


def quaternion_to_euler(q: Quaternion) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, yaw, pitch) in degrees.
    
    Args:
        q: Quaternion object
        
    Returns:
        Tuple of (roll, yaw, pitch) in degrees
    """
    yaw, pitch, roll = q.yaw_pitch_roll
    return (
        float(np.degrees(roll)),
        float(np.degrees(yaw)),
        float(np.degrees(pitch))
    )


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


def get_lidar_pose(nusc: NuScenes, sample: Dict) -> List[float]:
    """
    Get LiDAR sensor pose in world coordinate system.
    
    Args:
        nusc: NuScenes instance
        sample: Sample record
        
    Returns:
        List of [x, y, z, roll, yaw, pitch]
    """
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # Get ego pose
    ego_translation = np.array(pose_record['translation'])
    ego_rotation = Quaternion(pose_record['rotation'])
    
    # Get LiDAR to ego transform
    lidar_translation = np.array(cs_record['translation'])
    lidar_rotation = Quaternion(cs_record['rotation'])
    
    # Transform LiDAR position to world coordinates
    lidar_world_pos = ego_translation + ego_rotation.rotate(lidar_translation)
    lidar_world_rot = ego_rotation * lidar_rotation
    
    return get_pose_6dof(lidar_world_pos.tolist(), lidar_world_rot)


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
    sensor_world_rot = ego_rotation * sensor_rotation
    
    return get_pose_6dof(sensor_world_pos.tolist(), sensor_world_rot)


def get_camera_params(nusc: NuScenes, sample: Dict, cam_name: str) -> Optional[Dict]:
    """
    Get camera parameters including cords, extrinsic and intrinsic.
    
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
    
    # Get camera world pose (cords)
    cords = get_sensor_world_pose(nusc, cam_token)
    
    # Calculate extrinsic (camera to LiDAR transform - actually LiDAR -> Camera for projection)
    extrinsic = compute_lidar_to_camera_extrinsic(nusc, sample, cam_token)
    
    return {
        'cords': cords,
        'extrinsic': extrinsic.tolist(),
        'intrinsic': intrinsic
    }


def get_radar_params(nusc: NuScenes, sample: Dict, radar_name: str) -> Optional[Dict]:
    """
    Get radar parameters including cords and extrinsic.
    
    Args:
        nusc: NuScenes instance
        sample: Sample record
        radar_name: Radar name (e.g., 'RADAR_FRONT')
        
    Returns:
        Dict with radar parameters or None if radar not available
    """
    if radar_name not in sample['data']:
        return None
        
    radar_token = sample['data'][radar_name]
    
    # Get radar world pose (cords)
    cords = get_sensor_world_pose(nusc, radar_token)
    
    # Calculate extrinsic (radar to LiDAR transform - actually LiDAR -> Radar)
    extrinsic = compute_lidar_to_camera_extrinsic(nusc, sample, radar_token)
    
    return {
        'cords': cords,
        'extrinsic': extrinsic.tolist()
    }


def get_ego_speed(nusc: NuScenes, nusc_can_bus: Optional[NuScenesCanBus], 
                   sample: Dict) -> int:
    """
    Get ego vehicle speed from CAN bus data.
    
    Args:
        nusc: NuScenes instance
        nusc_can_bus: NuScenesCanBus instance or None
        sample: Sample record
        
    Returns:
        Speed (integer, 0 if not available)
    """
    if nusc_can_bus is None:
        return 0
    
    try:
        scene_name = nusc.get('scene', sample['scene_token'])['name']
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
        sample_timestamp = sample['timestamp']
        
        last_pose = pose_list[0]
        for pose in pose_list:
            if pose['utime'] > sample_timestamp:
                break
            last_pose = pose
        
        # Calculate speed from velocity components
        vel = last_pose.get('vel', [0, 0, 0])
        speed_mps = np.sqrt(vel[0]**2 + vel[1]**2)
        speed_kmh = speed_mps * 3.6
        return int(round(speed_kmh))
    except Exception:
        return 0


def get_object_annotations(nusc: NuScenes, sample: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Get all object annotations separated by category (vehicles, pedestrians, cyclists).
    
    Args:
        nusc: NuScenes instance
        sample: Sample record
        
    Returns:
        Tuple of (vehicles_dict, pedestrians_dict, cyclists_dict)
    """
    vehicles = {}
    pedestrians = {}
    cyclists = {}
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        category = ann['category_name']
        
        # Skip if no LiDAR points hit
        if ann['num_lidar_pts'] == 0:
            continue
        
        # Determine object type and target dict
        obj_type = None
        target_dict = None
        
        # Check vehicle categories
        for cat_prefix, type_name in VEHICLE_CATEGORIES.items():
            if category.startswith(cat_prefix):
                obj_type = type_name
                target_dict = vehicles
                break
        
        # Check pedestrian categories
        if obj_type is None:
            for cat_prefix, type_name in PEDESTRIAN_CATEGORIES.items():
                if category.startswith(cat_prefix):
                    obj_type = type_name
                    target_dict = pedestrians
                    break
        
        # Check cyclist categories
        if obj_type is None:
            for cat_prefix, type_name in CYCLIST_CATEGORIES.items():
                if category.startswith(cat_prefix):
                    obj_type = type_name
                    target_dict = cyclists
                    break
        
        if obj_type is None:
            continue
        
        # Get object ID from instance_token (consistent across frames)
        # Use hash to get a stable numeric ID
        obj_id = abs(hash(ann['instance_token'])) % 1000
        
        # Get world position
        location = ann['translation']
        rotation = Quaternion(ann['rotation'])
        roll, yaw, pitch = quaternion_to_euler(rotation)
        
        # Get dimensions (wlh -> extent is half dimensions)
        # nuScenes size order: [width, length, height]
        size = ann['size']
        extent = [
            float(size[1] / 2),  # half_length
            float(size[0] / 2),  # half_width  
            float(size[2] / 2)   # half_height
        ]
        
        # Center offset (typically zero)
        center = [0, 0, 0]
        
        target_dict[obj_id] = {
            'angle': [roll, yaw, pitch],
            'center': center,
            'extent': extent,
            'location': [float(location[0]), float(location[1]), float(location[2])],
            'object_type': obj_type
        }
    
    return vehicles, pedestrians, cyclists


def create_frame_yaml(nusc: NuScenes, nusc_can_bus: Optional[NuScenesCanBus],
                      sample: Dict) -> OrderedDict:
    """
    Create YAML data for a single frame in OpenV2V format.
    
    Args:
        nusc: NuScenes instance
        nusc_can_bus: NuScenesCanBus instance or None
        sample: Sample record
        
    Returns:
        OrderedDict containing all frame metadata
    """
    yaml_data = OrderedDict()
    
    # RSU flag (always false for nuScenes - no roadside units)
    yaml_data['RSU'] = False
    
    # Camera parameters (all 6 cameras)
    for nus_cam, openv2v_cam in CAMERA_MAPPING.items():
        cam_params = get_camera_params(nusc, sample, nus_cam)
        if cam_params:
            yaml_data[openv2v_cam] = cam_params
    
    # Radar parameters
    for nus_radar, openv2v_radar in RADAR_MAPPING.items():
        radar_params = get_radar_params(nusc, sample, nus_radar)
        if radar_params:
            yaml_data[openv2v_radar] = radar_params
    
    # Ego speed
    yaml_data['ego_speed'] = get_ego_speed(nusc, nusc_can_bus, sample)
    
    # LiDAR pose
    yaml_data['lidar_pose'] = get_lidar_pose(nusc, sample)
    
    # True ego pose (same as LiDAR pose for consistency)
    yaml_data['true_ego_pose'] = yaml_data['lidar_pose'].copy()
    
    # Get object annotations
    vehicles, pedestrians, cyclists = get_object_annotations(nusc, sample)
    
    # Add annotations in order: vehicles, pedestrians, cyclists
    yaml_data['vehicles'] = vehicles
    yaml_data['pedestrians'] = pedestrians
    yaml_data['cyclists'] = cyclists
    
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


def convert_scene(nusc: NuScenes, nusc_can_bus: Optional[NuScenesCanBus],
                  scene: Dict, output_dir: str, split: str = 'train') -> None:
    """
    Convert a single nuScenes scene to OpenV2V format.
    
    Args:
        nusc: NuScenes instance
        nusc_can_bus: NuScenesCanBus instance or None
        scene: Scene record
        output_dir: Base output directory
        split: 'train', 'validate', or 'test'
    """
    scene_name = scene['name']
    
    scene_output_dir = osp.join(output_dir, split, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    print(f"Converting scene: {scene_name}")
    
    # Iterate through all samples in the scene
    sample_token = scene['first_sample_token']
    frame_idx = 0
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        frame_str = f"{frame_idx:05d}"
        
        # 1. Create YAML metadata
        yaml_data = create_frame_yaml(nusc, nusc_can_bus, sample)
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
                ext = osp.splitext(cam_path)[1]
                cam_output = osp.join(scene_output_dir, f"{frame_str}_{openv2v_cam}{ext}")
                
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
    parser.add_argument(
        '--can_bus_path',
        type=str,
        default=None,
        help='Path to CAN bus data (optional)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'validate', 'test'],
        help='Splits to generate'
    )
    
    args = parser.parse_args()
    
    # Initialize nuScenes
    print(f"Loading nuScenes {args.version} from {args.nuscenes_root}")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=True)
    
    # Initialize CAN bus if available
    nusc_can_bus = None
    if args.can_bus_path:
        can_bus_full_path = osp.join(args.can_bus_path, 'can_bus')
        if osp.exists(can_bus_full_path):
            nusc_can_bus = NuScenesCanBus(dataroot=args.can_bus_path)
            print("CAN bus data loaded")
        else:
            print(f"Warning: CAN bus directory not found at {can_bus_full_path}")
    
    # Create output directories
    for split in args.splits:
        os.makedirs(osp.join(args.output_dir, split), exist_ok=True)
    
    # For mini, split scenes into train/val (approximately 8/2)
    # For trainval, use official splits
    if args.version == 'v1.0-mini':
        num_scenes = len(nusc.scene)
        train_count = int(num_scenes * 0.8)
        
        for i, scene in enumerate(nusc.scene):
            if i < train_count:
                split = 'train'
            else:
                split = 'validate'
            convert_scene(nusc, nusc_can_bus, scene, args.output_dir, split)
    else:
        # Use official train/val splits
        from data_converter.splits import splits as official_splits
        
        train_scenes = set(official_splits.get('train', []))
        val_scenes = set(official_splits.get('val', []))
        test_scenes = set(official_splits.get('test', []))
        
        for scene in nusc.scene:
            scene_name = scene['name']
            if scene_name in train_scenes:
                convert_scene(nusc, nusc_can_bus, scene, args.output_dir, 'train')
            elif scene_name in val_scenes:
                convert_scene(nusc, nusc_can_bus, scene, args.output_dir, 'validate')
            elif scene_name in test_scenes:
                convert_scene(nusc, nusc_can_bus, scene, args.output_dir, 'test')
    
    print(f"\nConversion complete! Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
