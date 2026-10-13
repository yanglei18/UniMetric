#!/usr/bin/env python3
"""Visualize GT camera ego poses in the first-camera coordinate frame."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read GT camera poses from sequence YAML files and visualize the "
            "trajectory in the first camera reference frame."
        )
    )
    parser.add_argument(
        "--data-path",
        "--data_path",
        required=True,
        help="Sequence folder, e.g. data/official-data/nuScenes/scene-0004.",
    )
    parser.add_argument("--camera-id", type=int, default=1, help="Camera id to inspect. Default: 1.")
    parser.add_argument("--frame-start", type=int, default=0, help="First frame offset after sorting YAML files.")
    parser.add_argument("--frame-count", type=int, default=40, help="Number of frames to visualize. Use <=0 for all.")
    parser.add_argument(
        "--reference-mode",
        choices=("lidar_to_camera", "camera_cords"),
        default="lidar_to_camera",
        help=(
            "lidar_to_camera builds OpenCV camera poses from lidar_pose and "
            "camera extrinsic (LiDAR->Camera). camera_cords uses camera.cords directly."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="demo",
        help="Directory for output PNG/JSON files. Default: demo.",
    )
    parser.add_argument(
        "--label-step",
        type=int,
        default=1,
        help="Label every N frame id on the trajectory. Use <=0 to disable labels.",
    )
    parser.add_argument(
        "--arrow-step",
        type=int,
        default=1,
        help="Draw direction arrows every N segment. Use <=0 to disable arrows.",
    )
    parser.add_argument(
        "--disable-pointcloud",
        action="store_true",
        help="Only draw camera trajectory; skip front-view LiDAR point accumulation.",
    )
    parser.add_argument(
        "--pointcloud-mode",
        choices=("front_view", "all_lidar"),
        default="front_view",
        help=(
            "front_view keeps only LiDAR points projected into the selected camera image. "
            "all_lidar fuses full LiDAR frames without image/FOV filtering."
        ),
    )
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=100.0,
        help="Max camera-frame depth for visible LiDAR points. Use <=0 to disable.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000000000,
        help="Max accumulated visible points to draw. Use <=0 for all points.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.25,
        help="Scatter size for accumulated visible LiDAR points.",
    )
    parser.add_argument(
        "--pred-scale-dir",
        default=None,
        help=(
            "Optional prediction scale directory that contains camera_params/*.json, "
            "e.g. outputs/nuScenes/scene-0004/pi3_outputs/relative_scale."
        ),
    )
    parser.add_argument(
        "--show-pred-pose",
        action="store_true",
        help="Draw predicted camera trajectory from --pred-scale-dir/camera_params.",
    )
    parser.add_argument(
        "--show-pred-pointmap",
        action="store_true",
        help=(
            "Draw accumulated predicted pointmap samples at pixels hit by projected "
            "LiDAR points. Requires --pred-scale-dir."
        ),
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    return parser.parse_args()


def list_frame_ids(sequence_path: Path) -> list[str]:
    frame_ids = []
    for pcd_file in glob.glob(str(sequence_path / "*.pcd")):
        if "_radar" in pcd_file:
            continue
        frame_ids.append(Path(pcd_file).stem)
    return sorted(frame_ids, key=lambda frame_id: int(frame_id))


def x_to_world(pose: list[float]) -> np.ndarray:
    """Convert [x, y, z, roll, yaw, pitch] in degrees to T_world_x."""
    x, y, z, roll, yaw, pitch = pose[:]
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
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


def to_homogeneous(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape == (4, 4):
        return transform
    if transform.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3, :] = transform
        return homogeneous
    raise ValueError(f"Unsupported transform shape: {transform.shape}")


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return (transform @ homogeneous.T).T[:, :3].astype(np.float32)


def load_yaml(sequence_path: Path, frame_id: str) -> dict:
    yaml_path = sequence_path / f"{frame_id}.yaml"
    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_camera_cords_pose(payload: dict, camera_id: int, yaml_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    camera_key = f"camera{camera_id}"
    if camera_key not in payload:
        raise KeyError(f"{camera_key} not found in {yaml_path}")
    cords = payload[camera_key].get("cords")
    if cords is None:
        raise KeyError(f"{camera_key}.cords not found in {yaml_path}")
    return x_to_world(cords), {"camera_cords": list(cords)}


def load_lidar_to_camera_pose(payload: dict, camera_id: int, yaml_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    camera_key = f"camera{camera_id}"
    if camera_key not in payload:
        raise KeyError(f"{camera_key} not found in {yaml_path}")

    lidar_pose = payload.get("lidar_pose", [0, 0, 0, 0, 0, 0])
    camera_extrinsic = payload[camera_key].get("extrinsic")
    if camera_extrinsic is None:
        raise KeyError(f"{camera_key}.extrinsic not found in {yaml_path}")

    world_from_lidar = x_to_world(lidar_pose)
    camera_from_lidar = to_homogeneous(np.asarray(camera_extrinsic, dtype=np.float64))
    lidar_from_camera = np.linalg.inv(camera_from_lidar)
    world_from_camera = world_from_lidar @ lidar_from_camera
    return world_from_camera, {
        "lidar_pose": list(lidar_pose),
        "camera_extrinsic_lidar_to_camera": camera_from_lidar.tolist(),
    }


def load_camera_pose(
    sequence_path: Path,
    frame_id: str,
    camera_id: int,
    reference_mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    yaml_path = sequence_path / f"{frame_id}.yaml"
    payload = load_yaml(sequence_path, frame_id)
    if reference_mode == "camera_cords":
        pose, metadata = load_camera_cords_pose(payload, camera_id, yaml_path)
    elif reference_mode == "lidar_to_camera":
        pose, metadata = load_lidar_to_camera_pose(payload, camera_id, yaml_path)
    else:
        raise ValueError(f"Unsupported reference mode: {reference_mode}")
    metadata["reference_mode"] = reference_mode
    return pose, metadata


def select_frame_ids(frame_ids: list[str], frame_start: int, frame_count: int) -> list[str]:
    if frame_start < 0:
        raise ValueError("--frame-start must be non-negative")
    selected = frame_ids[frame_start:]
    if frame_count > 0:
        selected = selected[:frame_count]
    return selected


def relative_pose(reference_pose: np.ndarray, frame_pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(reference_pose) @ frame_pose


def collect_trajectory(
    sequence_path: Path,
    frame_ids: list[str],
    camera_id: int,
    reference_mode: str,
) -> tuple[list[dict[str, object]], np.ndarray]:
    if not frame_ids:
        raise ValueError("No frame ids selected.")

    poses = []
    metadata_by_frame = {}
    for frame_id in frame_ids:
        pose, metadata = load_camera_pose(sequence_path, frame_id, camera_id, reference_mode)
        poses.append(pose)
        metadata_by_frame[frame_id] = metadata

    reference_pose = poses[0]
    records = []
    positions = []
    for frame_id, pose in zip(frame_ids, poses):
        transform = relative_pose(reference_pose, pose)
        position = transform[:3, 3].astype(np.float64)
        positions.append(position)
        records.append(
            {
                "frame_id": frame_id,
                **metadata_by_frame[frame_id],
                "position_ref_camera": position.tolist(),
                "transform_ref_from_camera": transform.tolist(),
            }
        )
    return records, np.asarray(positions, dtype=np.float64)


def load_prediction_extrinsics(pred_scale_dir: Path, frame_id: str, camera_id: int) -> np.ndarray:
    camera_params_dir = pred_scale_dir / "camera_params"
    params_path = camera_params_dir / f"{frame_id}_camera{camera_id}.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Prediction camera params not found: {params_path}")
    with params_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if "extrinsics" not in payload:
        raise KeyError(f"'extrinsics' not found in {params_path}")
    return to_homogeneous(np.asarray(payload["extrinsics"], dtype=np.float64))


def collect_prediction_trajectory(
    pred_scale_dir: Path,
    frame_ids: list[str],
    camera_id: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    if not frame_ids:
        raise ValueError("No frame ids selected.")

    extrinsics = [
        load_prediction_extrinsics(pred_scale_dir, frame_id, camera_id)
        for frame_id in frame_ids
    ]
    reference_extrinsics = extrinsics[0]
    records = []
    positions = []
    for frame_id, frame_extrinsics in zip(frame_ids, extrinsics):
        transform_ref_from_camera = reference_extrinsics @ np.linalg.inv(frame_extrinsics)
        position = transform_ref_from_camera[:3, 3].astype(np.float64)
        positions.append(position)
        records.append(
            {
                "frame_id": frame_id,
                "position_pred_ref_camera": position.tolist(),
                "transform_pred_ref_from_camera": transform_ref_from_camera.tolist(),
            }
        )
    return records, np.asarray(positions, dtype=np.float64)


def map_pixels_to_pointmap(
    projected_points: np.ndarray,
    pointmap_shape: tuple[int, ...],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pointmap_h, pointmap_w = pointmap_shape[:2]
    image_h, image_w = image_shape
    mapped = projected_points.astype(np.float64).copy()
    mapped[:, 0] *= pointmap_w / float(image_w)
    mapped[:, 1] *= pointmap_h / float(image_h)
    pixel_coords = np.round(mapped).astype(np.int64)
    valid_mask = (
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 0] < pointmap_w)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 1] < pointmap_h)
    )
    return pixel_coords[valid_mask], valid_mask


def deduplicate_visible_pixels(
    pixel_coords: np.ndarray,
    gt_points_camera: np.ndarray,
    pointmap_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if pixel_coords.size == 0:
        return pixel_coords, gt_points_camera
    pixel_ids = pixel_coords[:, 1] * pointmap_width + pixel_coords[:, 0]
    gt_depth = gt_points_camera[:, 2]
    order = np.lexsort((gt_depth, pixel_ids))
    sorted_ids = pixel_ids[order]
    _, unique_indices = np.unique(sorted_ids, return_index=True)
    selected_indices = order[unique_indices]
    return pixel_coords[selected_indices], gt_points_camera[selected_indices]


def load_image_size(sequence_path: Path, frame_id: str, camera_id: int) -> tuple[int, int]:
    from PIL import Image

    for suffix in (".jpg", ".jpeg", ".png"):
        image_path = sequence_path / f"{frame_id}_camera{camera_id}{suffix}"
        if image_path.exists():
            with Image.open(image_path) as image:
                return image.size
    raise FileNotFoundError(f"No image found for frame {frame_id}, camera{camera_id}")


def load_lidar_points(sequence_path: Path, frame_id: str) -> np.ndarray:
    import open3d as o3d

    lidar_path = sequence_path / f"{frame_id}.pcd"
    point_cloud = o3d.io.read_point_cloud(str(lidar_path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32)
    return points[np.isfinite(points).all(axis=1)]


def load_camera_from_lidar(sequence_path: Path, frame_id: str, camera_id: int) -> np.ndarray:
    payload = load_yaml(sequence_path, frame_id)
    camera_key = f"camera{camera_id}"
    if camera_key not in payload:
        raise KeyError(f"{camera_key} not found in {sequence_path / f'{frame_id}.yaml'}")
    return to_homogeneous(np.asarray(payload[camera_key]["extrinsic"], dtype=np.float64))


def load_lidar_points_in_camera(sequence_path: Path, frame_id: str, camera_id: int) -> tuple[np.ndarray, int]:
    points_lidar = load_lidar_points(sequence_path, frame_id)
    if points_lidar.size == 0:
        return np.empty((0, 3), dtype=np.float32), 0
    camera_from_lidar = load_camera_from_lidar(sequence_path, frame_id, camera_id)
    points_camera = transform_points(points_lidar, camera_from_lidar)
    points_camera = points_camera[np.isfinite(points_camera).all(axis=1)]
    return points_camera, int(points_lidar.shape[0])


def project_front_view_lidar_points(
    sequence_path: Path,
    frame_id: str,
    camera_id: int,
    depth_threshold: float | None,
) -> tuple[np.ndarray, int]:
    payload = load_yaml(sequence_path, frame_id)
    camera_key = f"camera{camera_id}"
    if camera_key not in payload:
        raise KeyError(f"{camera_key} not found in {sequence_path / f'{frame_id}.yaml'}")

    intrinsic = np.asarray(payload[camera_key]["intrinsic"], dtype=np.float64)
    image_width, image_height = load_image_size(sequence_path, frame_id, camera_id)
    points_camera, raw_count = load_lidar_points_in_camera(sequence_path, frame_id, camera_id)
    if points_camera.size == 0:
        return np.empty((0, 3), dtype=np.float32), raw_count

    points_camera = points_camera.astype(np.float64)
    valid_mask = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-6)
    if depth_threshold is not None:
        valid_mask &= points_camera[:, 2] <= depth_threshold
    points_camera = points_camera[valid_mask]
    if points_camera.size == 0:
        return np.empty((0, 3), dtype=np.float32), raw_count

    projected = (intrinsic @ points_camera.T).T
    projected[:, 0] /= np.maximum(projected[:, 2], 1e-6)
    projected[:, 1] /= np.maximum(projected[:, 2], 1e-6)
    image_mask = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < image_width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < image_height)
    )
    return points_camera[image_mask].astype(np.float32), raw_count


def project_front_view_lidar_pixels(
    sequence_path: Path,
    frame_id: str,
    camera_id: int,
    depth_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    payload = load_yaml(sequence_path, frame_id)
    camera_key = f"camera{camera_id}"
    if camera_key not in payload:
        raise KeyError(f"{camera_key} not found in {sequence_path / f'{frame_id}.yaml'}")

    intrinsic = np.asarray(payload[camera_key]["intrinsic"], dtype=np.float64)
    image_width, image_height = load_image_size(sequence_path, frame_id, camera_id)
    points_camera, raw_count = load_lidar_points_in_camera(sequence_path, frame_id, camera_id)
    if points_camera.size == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_pixels = np.empty((0, 2), dtype=np.float32)
        return empty_pixels, empty_points, (image_height, image_width), raw_count

    points_camera = points_camera.astype(np.float64)
    valid_mask = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-6)
    if depth_threshold is not None:
        valid_mask &= points_camera[:, 2] <= depth_threshold
    points_camera = points_camera[valid_mask]
    if points_camera.size == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_pixels = np.empty((0, 2), dtype=np.float32)
        return empty_pixels, empty_points, (image_height, image_width), raw_count

    projected = (intrinsic @ points_camera.T).T
    projected[:, 0] /= np.maximum(projected[:, 2], 1e-6)
    projected[:, 1] /= np.maximum(projected[:, 2], 1e-6)
    image_mask = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < image_width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < image_height)
    )
    return (
        projected[image_mask, :2].astype(np.float32),
        points_camera[image_mask].astype(np.float32),
        (image_height, image_width),
        raw_count,
    )


def load_prediction_pointmap(pred_scale_dir: Path, frame_id: str, camera_id: int) -> np.ndarray:
    pointmap_path = pred_scale_dir / "pointmap" / f"{frame_id}_camera{camera_id}.npy"
    if not pointmap_path.exists():
        raise FileNotFoundError(f"Prediction pointmap not found: {pointmap_path}")
    pointmap = np.load(pointmap_path)
    if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
        raise ValueError(f"Expected pointmap shape [H, W, 3], got {pointmap.shape} from {pointmap_path}")
    return pointmap.astype(np.float32)


def sample_prediction_pointmap_at_lidar_pixels(
    sequence_path: Path,
    pred_scale_dir: Path,
    frame_id: str,
    camera_id: int,
    depth_threshold: float | None,
) -> tuple[np.ndarray, dict[str, float]]:
    pointmap = load_prediction_pointmap(pred_scale_dir, frame_id, camera_id)
    projected_pixels, gt_points_camera, image_shape, raw_count = project_front_view_lidar_pixels(
        sequence_path,
        frame_id,
        camera_id,
        depth_threshold,
    )
    if projected_pixels.size == 0:
        return np.empty((0, 3), dtype=np.float32), {
            "raw_lidar_points": float(raw_count),
            "projected_lidar_pixels": 0.0,
            "sampled_pred_pointmap_points": 0.0,
        }

    pixel_coords, valid_mask = map_pixels_to_pointmap(projected_pixels, pointmap.shape, image_shape)
    gt_points_camera = gt_points_camera[valid_mask]
    pixel_coords, gt_points_camera = deduplicate_visible_pixels(pixel_coords, gt_points_camera, pointmap.shape[1])
    del gt_points_camera
    if pixel_coords.size == 0:
        return np.empty((0, 3), dtype=np.float32), {
            "raw_lidar_points": float(raw_count),
            "projected_lidar_pixels": float(projected_pixels.shape[0]),
            "sampled_pred_pointmap_points": 0.0,
        }

    pred_points = pointmap[pixel_coords[:, 1], pixel_coords[:, 0]]
    valid_points = np.isfinite(pred_points).all(axis=1) & (pred_points[:, 2] > 1e-6)
    pred_points = pred_points[valid_points]
    return pred_points.astype(np.float32), {
        "raw_lidar_points": float(raw_count),
        "projected_lidar_pixels": float(projected_pixels.shape[0]),
        "sampled_pred_pointmap_points": float(pred_points.shape[0]),
    }


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(42)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def collect_visible_pointcloud(
    sequence_path: Path,
    records: list[dict[str, object]],
    camera_id: int,
    depth_threshold: float | None,
    max_points: int,
    pointcloud_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    clouds = []
    raw_points = 0
    visible_points = 0
    used_frames = 0
    for record in records:
        frame_id = str(record["frame_id"])
        if pointcloud_mode == "front_view":
            points_camera, raw_count = project_front_view_lidar_points(
                sequence_path,
                frame_id,
                camera_id,
                depth_threshold,
            )
        elif pointcloud_mode == "all_lidar":
            points_camera, raw_count = load_lidar_points_in_camera(sequence_path, frame_id, camera_id)
        else:
            raise ValueError(f"Unsupported pointcloud mode: {pointcloud_mode}")
        raw_points += raw_count
        visible_points += int(points_camera.shape[0])
        if points_camera.size == 0:
            continue
        transform_ref_from_camera = np.asarray(record["transform_ref_from_camera"], dtype=np.float64)
        clouds.append(transform_points(points_camera, transform_ref_from_camera))
        used_frames += 1

    if clouds:
        points = np.concatenate(clouds, axis=0)
        points = points[np.isfinite(points).all(axis=1)]
    else:
        points = np.empty((0, 3), dtype=np.float32)
    sampled_points = downsample_points(points, max_points)
    stats = {
        "raw_lidar_points": float(raw_points),
        "front_view_visible_points": float(visible_points if pointcloud_mode == "front_view" else 0),
        "all_lidar_points": float(visible_points if pointcloud_mode == "all_lidar" else 0),
        "pointcloud_input_points": float(visible_points),
        "fused_points_before_downsample": float(points.shape[0]),
        "fused_points_drawn": float(sampled_points.shape[0]),
        "pointcloud_frames_used": float(used_frames),
    }
    return sampled_points, stats


def collect_prediction_pointmap_cloud(
    sequence_path: Path,
    pred_scale_dir: Path,
    records: list[dict[str, object]],
    camera_id: int,
    depth_threshold: float | None,
    max_points: int,
) -> tuple[np.ndarray, dict[str, float]]:
    camera_params_dir = pred_scale_dir / "camera_params"
    if not camera_params_dir.is_dir():
        raise FileNotFoundError(f"Prediction camera_params directory does not exist: {camera_params_dir}")

    reference_frame_id = str(records[0]["frame_id"])
    reference_extrinsics = load_prediction_extrinsics(pred_scale_dir, reference_frame_id, camera_id)
    clouds = []
    raw_lidar_points = 0.0
    projected_lidar_pixels = 0.0
    sampled_pointmap_points = 0.0
    used_frames = 0
    for record in records:
        frame_id = str(record["frame_id"])
        pred_points_camera, sample_stats = sample_prediction_pointmap_at_lidar_pixels(
            sequence_path,
            pred_scale_dir,
            frame_id,
            camera_id,
            depth_threshold,
        )
        raw_lidar_points += sample_stats["raw_lidar_points"]
        projected_lidar_pixels += sample_stats["projected_lidar_pixels"]
        sampled_pointmap_points += sample_stats["sampled_pred_pointmap_points"]
        if pred_points_camera.size == 0:
            continue
        frame_extrinsics = load_prediction_extrinsics(pred_scale_dir, frame_id, camera_id)
        pred_to_reference = reference_extrinsics @ np.linalg.inv(frame_extrinsics)
        clouds.append(transform_points(pred_points_camera, pred_to_reference))
        used_frames += 1

    if clouds:
        points = np.concatenate(clouds, axis=0)
        points = points[np.isfinite(points).all(axis=1)]
    else:
        points = np.empty((0, 3), dtype=np.float32)
    sampled_points = downsample_points(points, max_points)
    stats = {
        "pred_pointmap_raw_lidar_points": raw_lidar_points,
        "pred_pointmap_projected_lidar_pixels": projected_lidar_pixels,
        "pred_pointmap_sampled_points": sampled_pointmap_points,
        "pred_pointmap_fused_points_before_downsample": float(points.shape[0]),
        "pred_pointmap_fused_points_drawn": float(sampled_points.shape[0]),
        "pred_pointmap_frames_used": float(used_frames),
    }
    return sampled_points, stats


def trajectory_length(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def step_stats(positions: np.ndarray) -> dict[str, float]:
    if positions.shape[0] < 2:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return {
        "mean": float(np.mean(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
    }


def set_equal_bev_limits(ax, trajectories: list[np.ndarray], pointcloud: np.ndarray | None = None) -> None:
    point_sets = [traj[:, [0, 2]] for traj in trajectories if traj.size]
    if pointcloud is not None and pointcloud.size:
        point_sets.append(pointcloud[:, [0, 2]])
    if not point_sets:
        return
    points = np.concatenate(point_sets, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.size == 0:
        return

    x_min, z_min = np.percentile(points, 1, axis=0)
    x_max, z_max = np.percentile(points, 99, axis=0)
    margin_x = max((x_max - x_min) * 0.12, 1.0)
    margin_z = max((z_max - z_min) * 0.12, 1.0)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(z_min - margin_z, z_max + margin_z)
    ax.set_aspect("equal", adjustable="box")


def label_frames(ax, positions: np.ndarray, frame_ids: list[str], step: int, color: str, offset: tuple[int, int]) -> None:
    if step <= 0 or positions.size == 0:
        return
    label_indices = set(range(0, positions.shape[0], step))
    label_indices.add(0)
    label_indices.add(positions.shape[0] - 1)

    for idx in sorted(label_indices):
        point = positions[idx]
        if not np.isfinite(point).all():
            continue
        ax.annotate(
            frame_ids[idx],
            (point[0], point[2]),
            textcoords="offset points",
            xytext=offset,
            fontsize=6,
            color=color,
            alpha=0.9,
        )


def draw_arrows(ax, positions: np.ndarray, step: int, color: str) -> None:
    if step <= 0 or positions.shape[0] < 2:
        return
    for idx in range(0, positions.shape[0] - 1, step):
        start = positions[idx]
        end = positions[idx + 1]
        delta = end - start
        if not np.isfinite(start).all() or not np.isfinite(delta).all():
            continue
        ax.arrow(
            start[0],
            start[2],
            delta[0],
            delta[2],
            color=color,
            alpha=0.35,
            width=0.02,
            head_width=0.6,
            length_includes_head=True,
        )


def save_plot(
    sequence_path: Path,
    camera_id: int,
    frame_ids: list[str],
    positions: np.ndarray,
    visible_points: np.ndarray,
    pred_pointmap_points: np.ndarray | None,
    pred_positions: np.ndarray | None,
    stats: dict[str, float],
    args: argparse.Namespace,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_suffix = "_with_pred" if args.pred_scale_dir else ""
    output_path = output_dir / (
        f"camera_ego_{sequence_path.name}_camera{camera_id}_"
        f"{args.reference_mode}_{args.pointcloud_mode}{pred_suffix}.png"
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    if visible_points.size:
        point_label = "Projected LiDAR points" if args.pointcloud_mode == "front_view" else "All LiDAR points"
        ax.scatter(
            visible_points[:, 0],
            visible_points[:, 2],
            s=args.point_size,
            alpha=0.35,
            color="tab:blue",
            label=point_label,
        )
    if pred_pointmap_points is not None and pred_pointmap_points.size:
        ax.scatter(
            pred_pointmap_points[:, 0],
            pred_pointmap_points[:, 2],
            s=max(args.point_size * 0.9, 0.05),
            alpha=0.45,
            color="tab:orange",
            label="Pred pointmap at LiDAR pixels",
        )

    color = "tab:green"
    ax.plot(
        positions[:, 0],
        positions[:, 2],
        "-o",
        color=color,
        markersize=3,
        linewidth=1.5,
        label="GT camera trajectory",
    )
    ax.scatter(positions[0, 0], positions[0, 2], marker="*", s=120, color=color, label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 2], marker="s", s=50, color=color, label="End")
    label_frames(ax, positions, frame_ids, args.label_step, color, (4, 4))
    draw_arrows(ax, positions, args.arrow_step, color)

    if pred_positions is not None and pred_positions.size:
        pred_color = "tab:red"
        ax.plot(
            pred_positions[:, 0],
            pred_positions[:, 2],
            "-o",
            color=pred_color,
            markersize=3,
            linewidth=1.3,
            alpha=0.9,
            label="Pred camera trajectory",
        )
        ax.scatter(
            pred_positions[0, 0],
            pred_positions[0, 2],
            marker="*",
            s=110,
            color=pred_color,
            label="Pred start",
        )
        ax.scatter(
            pred_positions[-1, 0],
            pred_positions[-1, 2],
            marker="s",
            s=45,
            color=pred_color,
            label="Pred end",
        )
        draw_arrows(ax, pred_positions, args.arrow_step, pred_color)

    title = args.title
    if title is None:
        pred_summary = ""
        if pred_positions is not None and pred_positions.size:
            pred_summary = f", pred_len={stats.get('pred_length', 0.0):.2f}m"
        pred_pointmap_summary = ""
        if pred_pointmap_points is not None and pred_pointmap_points.size:
            pred_pointmap_summary = f", pred_pointmap={int(stats.get('pred_pointmap_fused_points_drawn', 0))}"
        title = (
            f"{sequence_path.name} camera{camera_id} camera trajectories "
            f"({args.reference_mode})\n"
            f"len={stats['length']:.2f}m, step_mean={stats['step_mean']:.2f}m, "
            f"{args.pointcloud_mode}_points={int(stats.get('fused_points_drawn', 0))}"
            f"{pred_summary}"
            f"{pred_pointmap_summary}"
        )
    ax.set_title(title)
    ax.set_xlabel("Reference Camera X / right (m)")
    ax.set_ylabel("Reference Camera Z / forward (m)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="best")
    trajectories = [positions]
    if pred_positions is not None and pred_positions.size:
        trajectories.append(pred_positions)
    point_sets = [visible_points]
    if pred_pointmap_points is not None and pred_pointmap_points.size:
        point_sets.append(pred_pointmap_points)
    plot_points = np.concatenate(point_sets, axis=0) if point_sets else visible_points
    set_equal_bev_limits(ax, trajectories, plot_points)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_json(
    sequence_path: Path,
    camera_id: int,
    frame_ids: list[str],
    records: list[dict[str, object]],
    pred_records: list[dict[str, object]],
    stats: dict[str, float],
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_suffix = "_with_pred" if args.pred_scale_dir else ""
    output_path = output_dir / (
        f"camera_ego_{sequence_path.name}_camera{camera_id}_"
        f"{args.reference_mode}_{args.pointcloud_mode}{pred_suffix}.json"
    )
    payload = {
        "sequence": sequence_path.name,
        "sequence_path": str(sequence_path),
        "camera_id": camera_id,
        "reference_mode": args.reference_mode,
        "pointcloud_mode": args.pointcloud_mode,
        "pred_scale_dir": args.pred_scale_dir,
        "frame_start": args.frame_start,
        "frame_count": len(frame_ids),
        "frame_ids": frame_ids,
        "stats": stats,
        "pointcloud_enabled": not args.disable_pointcloud,
        "records": records,
        "pred_records": pred_records,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    sequence_path = Path(args.data_path)
    if not sequence_path.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {sequence_path}")

    all_frame_ids = list_frame_ids(sequence_path)
    frame_ids = select_frame_ids(all_frame_ids, args.frame_start, args.frame_count)
    if not frame_ids:
        raise RuntimeError(f"No LiDAR frames selected from {sequence_path}")

    records, positions = collect_trajectory(
        sequence_path,
        frame_ids,
        args.camera_id,
        args.reference_mode,
    )
    step_summary = step_stats(positions)
    stats = {
        "length": trajectory_length(positions),
        "step_mean": step_summary["mean"],
        "step_min": step_summary["min"],
        "step_max": step_summary["max"],
        "first_to_last_distance": float(np.linalg.norm(positions[-1] - positions[0])),
    }
    pred_records: list[dict[str, object]] = []
    pred_positions: np.ndarray | None = None
    if args.show_pred_pose or args.pred_scale_dir or args.show_pred_pointmap:
        if not args.pred_scale_dir:
            raise ValueError("--show-pred-pose/--show-pred-pointmap requires --pred-scale-dir")
        pred_scale_dir = Path(args.pred_scale_dir)
        if not pred_scale_dir.is_dir():
            raise FileNotFoundError(f"Prediction scale directory does not exist: {pred_scale_dir}")
        pred_records, pred_positions = collect_prediction_trajectory(
            pred_scale_dir,
            frame_ids,
            args.camera_id,
        )
        pred_step_summary = step_stats(pred_positions)
        stats.update(
            {
                "pred_length": trajectory_length(pred_positions),
                "pred_step_mean": pred_step_summary["mean"],
                "pred_step_min": pred_step_summary["min"],
                "pred_step_max": pred_step_summary["max"],
                "pred_first_to_last_distance": float(np.linalg.norm(pred_positions[-1] - pred_positions[0])),
            }
        )

    if args.disable_pointcloud:
        visible_points = np.empty((0, 3), dtype=np.float32)
    else:
        depth_threshold = args.depth_threshold if args.depth_threshold > 0 else None
        visible_points, pointcloud_stats = collect_visible_pointcloud(
            sequence_path,
            records,
            args.camera_id,
            depth_threshold,
            args.max_points,
            args.pointcloud_mode,
        )
        stats.update(pointcloud_stats)

    pred_pointmap_points: np.ndarray | None = None
    if args.show_pred_pointmap:
        pred_scale_dir = Path(args.pred_scale_dir)
        depth_threshold = args.depth_threshold if args.depth_threshold > 0 else None
        pred_pointmap_points, pred_pointmap_stats = collect_prediction_pointmap_cloud(
            sequence_path,
            pred_scale_dir,
            records,
            args.camera_id,
            depth_threshold,
            args.max_points,
        )
        stats.update(pred_pointmap_stats)

    plot_path = save_plot(
        sequence_path,
        args.camera_id,
        frame_ids,
        positions,
        visible_points,
        pred_pointmap_points,
        pred_positions,
        stats,
        args,
    )
    json_path = save_json(sequence_path, args.camera_id, frame_ids, records, pred_records, stats, args)

    print(f"Saved camera ego BEV: {plot_path}")
    print(f"Saved camera ego JSON: {json_path}")
    print(
        f"reference_mode={args.reference_mode}, frames={len(frame_ids)}, "
        f"first={frame_ids[0]}, last={frame_ids[-1]}"
    )
    print(
        f"length={stats['length']:.4f}m, "
        f"step_mean={stats['step_mean']:.4f}m, "
        f"step_min={stats['step_min']:.4f}m, "
        f"step_max={stats['step_max']:.4f}m, "
        f"first_to_last={stats['first_to_last_distance']:.4f}m"
    )
    if not args.disable_pointcloud:
        print(
            f"pointcloud_mode={args.pointcloud_mode}, "
            f"pointcloud_frames_used={int(stats['pointcloud_frames_used'])}, "
            f"raw_lidar_points={int(stats['raw_lidar_points'])}, "
            f"pointcloud_input_points={int(stats['pointcloud_input_points'])}, "
            f"fused_points_drawn={int(stats['fused_points_drawn'])}"
        )
    if pred_positions is not None:
        print(
            f"pred_scale_dir={args.pred_scale_dir}, "
            f"pred_length={stats['pred_length']:.4f}m, "
            f"pred_step_mean={stats['pred_step_mean']:.4f}m, "
            f"pred_first_to_last={stats['pred_first_to_last_distance']:.4f}m"
        )
    if pred_pointmap_points is not None:
        print(
            f"pred_pointmap_frames_used={int(stats['pred_pointmap_frames_used'])}, "
            f"pred_pointmap_projected_lidar_pixels={int(stats['pred_pointmap_projected_lidar_pixels'])}, "
            f"pred_pointmap_sampled_points={int(stats['pred_pointmap_sampled_points'])}, "
            f"pred_pointmap_fused_points_drawn={int(stats['pred_pointmap_fused_points_drawn'])}"
        )


if __name__ == "__main__":
    main()
