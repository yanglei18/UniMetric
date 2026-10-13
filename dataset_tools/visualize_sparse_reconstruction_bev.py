#!/usr/bin/env python3
"""Visualize sparse reconstruction proxy samples in BEV for debugging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from metric_scale.processing.frame_loader import FrameLoader

_FRAME_LOADER_CLS: type | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize sparse predicted pointmap samples and GT LiDAR samples "
            "in first-camera BEV, using the same sparse sequence reconstruction "
            "sampling/fusion convention as the metric code."
        )
    )
    parser.add_argument("--sequence-path", required=True, help="Sequence folder, e.g. data/official-data/nuScenes-New/scene-0004.")
    parser.add_argument("--depth-output", default=None, help="Depth output root, e.g. outputs/nuScenes-New/scene-0004/pi3_outputs.")
    parser.add_argument("--scale-dir", default=None, help="Direct scale directory, e.g. outputs/.../pi3_outputs/consistent_scale.")
    parser.add_argument("--scale", default="optimized_scale_original", help="Scale folder name under depth output root.")
    parser.add_argument("--output", default=None, help="Output PNG path. Default: <depth-output>/debug_sparse_bev_<scale>.png.")
    parser.add_argument("--frame-start", type=int, default=0, help="First frame index.")
    parser.add_argument("--frame-count", type=int, default=40, help="Number of frames to visualize.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Stride between frame indices (e.g. 2 for KITTI-tracking which stores even-indexed frames).",
    )
    parser.add_argument("--camera-id", type=int, default=1, help="Camera id used by Metric-Depth. Default: 1.")
    parser.add_argument("--depth-threshold", type=float, default=100.0, help="Max GT depth included in visualization.")
    parser.add_argument("--max-points", type=int, default=80000, help="Max points plotted per cloud after fusion.")
    parser.add_argument(
        "--plot-pose-trajectories",
        action="store_true",
        help="Also save a BEV comparison of GT and predicted pose trajectories.",
    )
    parser.add_argument(
        "--pose-label-step",
        type=int,
        default=1,
        help="Label every N frame index on the pose trajectory plot. Use <=0 to disable labels.",
    )
    parser.add_argument(
        "--pose-label-all-trajectories",
        action="store_true",
        help="Label all plotted pose trajectories instead of only the GT trajectory.",
    )
    parser.add_argument(
        "--transform-mode",
        choices=(
            "metric",
            "pred_current",
            "pred_current_se3",
            "pred_current_sim3",
            "pred_inverse",
            "pred_inverse_se3",
            "pred_inverse_sim3",
            "identity",
            "gt_pose_for_pred",
        ),
        default="metric",
        help=(
            "metric/pred_current: fuse frames with the current sequence reconstruction convention. "
            "pred_current_se3: additionally align predicted pose trajectory to GT with SE(3), preserving scale. "
            "pred_current_sim3: additionally align predicted pose trajectory to GT with Sim(3). "
            "pred_inverse: fuse predicted points with the inverse extrinsics convention candidate. "
            "pred_inverse_se3: additionally align the inverse-convention trajectory to GT with SE(3), preserving scale. "
            "pred_inverse_sim3: additionally align the inverse-convention trajectory to GT with Sim(3). "
            "identity: skip cross-frame predicted/GT pose transforms for debugging. "
            "gt_pose_for_pred: fuse both predicted and GT points with GT poses to isolate pointmap quality."
        ),
    )
    parser.add_argument("--title", default=None, help="Optional figure title.")
    return parser.parse_args()


def resolve_scale_dir(args: argparse.Namespace) -> Path:
    if args.scale_dir:
        return Path(args.scale_dir)
    if args.depth_output:
        return Path(args.depth_output) / args.scale
    raise ValueError("Either --scale-dir or --depth-output must be provided.")


def load_frame_loader() -> type:
    global _FRAME_LOADER_CLS
    if _FRAME_LOADER_CLS is not None:
        return _FRAME_LOADER_CLS

    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    try:
        from metric_scale.processing.frame_loader import FrameLoader
    finally:
        sys.argv = original_argv

    _FRAME_LOADER_CLS = FrameLoader
    return _FRAME_LOADER_CLS


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


def frame_prefix(frame_idx: int, camera_id: int) -> str:
    return f"{frame_idx:06d}_camera{camera_id}"


def load_prediction_extrinsics(camera_params_dir: Path, prefix: str) -> np.ndarray:
    path = camera_params_dir / f"{prefix}.json"
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return to_homogeneous(np.asarray(payload["extrinsics"], dtype=np.float64))


def pred_current_to_reference(reference_extrinsics: np.ndarray, frame_extrinsics: np.ndarray) -> np.ndarray:
    return reference_extrinsics @ np.linalg.inv(frame_extrinsics)


def pred_inverse_to_reference(reference_extrinsics: np.ndarray, frame_extrinsics: np.ndarray) -> np.ndarray:
    return np.linalg.inv(reference_extrinsics) @ frame_extrinsics


def translation_norm(transform: np.ndarray) -> float:
    return float(np.linalg.norm(transform[:3, 3]))


def trajectory_length(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def estimate_sim3(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching [N, 3] trajectories, got {source.shape} and {target.shape}")

    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if source.shape[0] < 2:
        return {"s": 1.0, "R": np.eye(3, dtype=np.float64), "t": np.zeros(3, dtype=np.float64)}

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target
    source_var = float(np.mean(np.sum(source_centered ** 2, axis=1)))
    if source_var <= 1e-12:
        return {"s": 1.0, "R": np.eye(3, dtype=np.float64), "t": mu_target - mu_source}

    covariance = source_centered.T @ target_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    scale = float(np.sum(singular_values) / source_var)
    translation = mu_target - scale * (rotation @ mu_source)
    return {"s": scale, "R": rotation, "t": translation}


def estimate_se3(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching [N, 3] trajectories, got {source.shape} and {target.shape}")

    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if source.shape[0] < 2:
        return {"s": 1.0, "R": np.eye(3, dtype=np.float64), "t": np.zeros(3, dtype=np.float64)}

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target
    covariance = source_centered.T @ target_centered / source.shape[0]
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = mu_target - rotation @ mu_source
    return {"s": 1.0, "R": rotation, "t": translation}


def apply_sim3(points: np.ndarray, sim3: dict[str, np.ndarray | float]) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    scale = float(sim3["s"])
    rotation = np.asarray(sim3["R"], dtype=np.float64)
    translation = np.asarray(sim3["t"], dtype=np.float64)
    aligned = scale * (rotation @ points.astype(np.float64).T).T + translation
    return aligned.astype(np.float32)


def sim3_residual_rmse(source: np.ndarray, target: np.ndarray, sim3: dict[str, np.ndarray | float]) -> float:
    aligned = apply_sim3(source, sim3).astype(np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(aligned).all(axis=1) & np.isfinite(target).all(axis=1)
    if not np.any(valid):
        return float("nan")
    errors = np.linalg.norm(aligned[valid] - target[valid], axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def load_gt_camera2world(sequence_path: Path, frame_idx: int, camera_id: int) -> np.ndarray:
    FrameLoader = load_frame_loader()
    frame_data = FrameLoader(str(sequence_path), camera_id=camera_id, frame_id=f"{frame_idx:06d}")
    if not frame_data.process_data():
        raise RuntimeError(f"Failed to load frame for pose diagnostics: {frame_idx:06d}_camera{camera_id}")
    return to_homogeneous(np.asarray(frame_data.camera2world, dtype=np.float64))


def compute_pose_diagnostics(
    sequence_path: Path,
    camera_params_dir: Path,
    frame_indices: list[int],
    camera_id: int,
) -> dict[str, float]:
    if len(frame_indices) < 2:
        return {}

    first_idx = frame_indices[0]
    second_idx = frame_indices[1]
    first_prefix = frame_prefix(first_idx, camera_id)
    second_prefix = frame_prefix(second_idx, camera_id)

    first_pred = load_prediction_extrinsics(camera_params_dir, first_prefix)
    second_pred = load_prediction_extrinsics(camera_params_dir, second_prefix)
    first_gt = load_gt_camera2world(sequence_path, first_idx, camera_id)
    second_gt = load_gt_camera2world(sequence_path, second_idx, camera_id)

    gt_relative = np.linalg.inv(first_gt) @ second_gt
    pred_current = pred_current_to_reference(first_pred, second_pred)
    pred_inverse = pred_inverse_to_reference(first_pred, second_pred)
    return {
        "diagnostic_frame_a": float(first_idx),
        "diagnostic_frame_b": float(second_idx),
        "gt_relative_translation_norm": translation_norm(gt_relative),
        "pred_current_translation_norm": translation_norm(pred_current),
        "pred_inverse_translation_norm": translation_norm(pred_inverse),
    }


def collect_pose_trajectories(args: argparse.Namespace) -> dict[str, np.ndarray | float]:
    sequence_path = Path(args.sequence_path)
    scale_dir = resolve_scale_dir(args)
    camera_params_dir = scale_dir / "camera_params"
    frame_indices = list(range(args.frame_start, args.frame_start + args.frame_count * args.frame_step, args.frame_step))

    reference_prefix = frame_prefix(frame_indices[0], args.camera_id)
    reference_pred = load_prediction_extrinsics(camera_params_dir, reference_prefix)
    reference_gt = load_gt_camera2world(sequence_path, frame_indices[0], args.camera_id)
    gt_reference_from_world = np.linalg.inv(reference_gt)

    gt_positions = []
    pred_current_positions = []
    pred_inverse_positions = []
    for frame_idx in frame_indices:
        prefix = frame_prefix(frame_idx, args.camera_id)
        pred_extrinsics = load_prediction_extrinsics(camera_params_dir, prefix)
        gt_camera2world = load_gt_camera2world(sequence_path, frame_idx, args.camera_id)

        gt_to_reference = gt_reference_from_world @ gt_camera2world
        current_to_reference = pred_current_to_reference(reference_pred, pred_extrinsics)
        inverse_to_reference = pred_inverse_to_reference(reference_pred, pred_extrinsics)

        gt_positions.append(gt_to_reference[:3, 3])
        pred_current_positions.append(current_to_reference[:3, 3])
        pred_inverse_positions.append(inverse_to_reference[:3, 3])

    gt_positions = np.asarray(gt_positions, dtype=np.float64)
    pred_current_positions = np.asarray(pred_current_positions, dtype=np.float64)
    pred_inverse_positions = np.asarray(pred_inverse_positions, dtype=np.float64)
    pred_current_sim3 = estimate_sim3(pred_current_positions, gt_positions)
    pred_inverse_sim3 = estimate_sim3(pred_inverse_positions, gt_positions)
    pred_current_se3 = estimate_se3(pred_current_positions, gt_positions)
    pred_inverse_se3 = estimate_se3(pred_inverse_positions, gt_positions)
    pred_current_aligned = apply_sim3(pred_current_positions, pred_current_sim3).astype(np.float64)
    pred_inverse_aligned = apply_sim3(pred_inverse_positions, pred_inverse_sim3).astype(np.float64)
    pred_current_se3_aligned = apply_sim3(pred_current_positions, pred_current_se3).astype(np.float64)
    pred_inverse_se3_aligned = apply_sim3(pred_inverse_positions, pred_inverse_se3).astype(np.float64)
    return {
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "gt": gt_positions,
        "pred_current": pred_current_positions,
        "pred_inverse": pred_inverse_positions,
        "pred_current_se3": pred_current_se3_aligned,
        "pred_inverse_se3": pred_inverse_se3_aligned,
        "pred_current_sim3": pred_current_aligned,
        "pred_inverse_sim3": pred_inverse_aligned,
        "gt_length": trajectory_length(gt_positions),
        "pred_current_length": trajectory_length(pred_current_positions),
        "pred_inverse_length": trajectory_length(pred_inverse_positions),
        "pred_current_se3_scale": float(pred_current_se3["s"]),
        "pred_inverse_se3_scale": float(pred_inverse_se3["s"]),
        "pred_current_se3_rmse": sim3_residual_rmse(pred_current_positions, gt_positions, pred_current_se3),
        "pred_inverse_se3_rmse": sim3_residual_rmse(pred_inverse_positions, gt_positions, pred_inverse_se3),
        "pred_current_se3_transform": pred_current_se3,
        "pred_inverse_se3_transform": pred_inverse_se3,
        "pred_current_sim3_scale": float(pred_current_sim3["s"]),
        "pred_inverse_sim3_scale": float(pred_inverse_sim3["s"]),
        "pred_current_sim3_rmse": sim3_residual_rmse(pred_current_positions, gt_positions, pred_current_sim3),
        "pred_inverse_sim3_rmse": sim3_residual_rmse(pred_inverse_positions, gt_positions, pred_inverse_sim3),
        "pred_current_sim3_transform": pred_current_sim3,
        "pred_inverse_sim3_transform": pred_inverse_sim3,
    }


def project_lidar_to_image(frame_data: "FrameLoader") -> tuple[np.ndarray, np.ndarray]:
    points_homogeneous = np.hstack(
        (frame_data.points[:, :3], np.ones((frame_data.points.shape[0], 1)))
    )
    points_camera = (frame_data.lidar2camera @ points_homogeneous.T).T[:, :3]
    visible_mask = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-6)
    points_camera = points_camera[visible_mask]
    if points_camera.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    projected = (frame_data.intrinsic @ points_camera.T).T
    projected[:, 0] /= np.maximum(projected[:, 2], 1e-6)
    projected[:, 1] /= np.maximum(projected[:, 2], 1e-6)
    valid_mask = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < frame_data.img_width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < frame_data.img_height)
    )
    return projected[valid_mask, :2], points_camera[valid_mask]


def map_pixels_to_pointmap(
    projected_points: np.ndarray,
    pointmap_shape: tuple[int, ...],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pointmap_h, pointmap_w = pointmap_shape[:2]
    img_h, img_w = image_shape
    scale_x = pointmap_w / float(img_w)
    scale_y = pointmap_h / float(img_h)
    mapped = projected_points.copy()
    mapped[:, 0] *= scale_x
    mapped[:, 1] *= scale_y
    coords = np.round(mapped).astype(int)
    valid_mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < pointmap_w)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < pointmap_h)
    )
    return coords[valid_mask], valid_mask


def deduplicate_visible_points(
    pixel_coords: np.ndarray,
    gt_points_camera: np.ndarray,
    pointmap_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_ids = pixel_coords[:, 1] * pointmap_width + pixel_coords[:, 0]
    gt_depth = gt_points_camera[:, 2]
    order = np.lexsort((gt_depth, pixel_ids))
    sorted_ids = pixel_ids[order]
    _, unique_indices = np.unique(sorted_ids, return_index=True)
    selected_indices = order[unique_indices]
    return pixel_coords[selected_indices], gt_points_camera[selected_indices]


def sample_frame_points(
    sequence_path: Path,
    pointmap_dir: Path,
    frame_idx: int,
    camera_id: int,
    depth_threshold: float | None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    prefix = frame_prefix(frame_idx, camera_id)
    FrameLoader = load_frame_loader()
    frame_data = FrameLoader(str(sequence_path), camera_id=camera_id, frame_id=f"{frame_idx:06d}")
    if not frame_data.process_data():
        raise RuntimeError(f"Failed to load frame: {prefix}")

    pointmap_path = pointmap_dir / f"{prefix}.npy"
    if not pointmap_path.exists():
        raise FileNotFoundError(f"Missing pointmap: {pointmap_path}")
    pointmap = np.load(pointmap_path)

    projected, gt_points = project_lidar_to_image(frame_data)
    pixel_coords, valid_mask = map_pixels_to_pointmap(
        projected,
        pointmap.shape,
        (frame_data.img_height, frame_data.img_width),
    )
    gt_points = gt_points[valid_mask]
    pixel_coords, gt_points = deduplicate_visible_points(pixel_coords, gt_points, pointmap.shape[1])

    if depth_threshold is not None:
        depth_mask = gt_points[:, 2] <= depth_threshold
        pixel_coords = pixel_coords[depth_mask]
        gt_points = gt_points[depth_mask]

    pred_points = pointmap[pixel_coords[:, 1], pixel_coords[:, 0]]
    valid_points = (
        np.isfinite(pred_points).all(axis=1)
        & np.isfinite(gt_points).all(axis=1)
        & (pred_points[:, 2] > 1e-6)
        & (gt_points[:, 2] > 1e-6)
    )
    return prefix, pred_points[valid_points], gt_points[valid_points], np.asarray(frame_data.camera2world)


def collect_fused_points(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sequence_path = Path(args.sequence_path)
    scale_dir = resolve_scale_dir(args)
    pointmap_dir = scale_dir / "pointmap"
    camera_params_dir = scale_dir / "camera_params"
    frame_indices = list(range(args.frame_start, args.frame_start + args.frame_count * args.frame_step, args.frame_step))

    reference_prefix = frame_prefix(frame_indices[0], args.camera_id)
    reference_pred_extrinsics = load_prediction_extrinsics(camera_params_dir, reference_prefix)
    FrameLoader = load_frame_loader()
    reference_frame = FrameLoader(str(sequence_path), camera_id=args.camera_id, frame_id=f"{frame_indices[0]:06d}")
    if not reference_frame.process_data():
        raise RuntimeError(f"Failed to load reference frame: {reference_prefix}")
    gt_reference_from_world = np.linalg.inv(to_homogeneous(reference_frame.camera2world))
    pose_trajectories = None
    pred_sim3 = None
    if args.transform_mode in ("pred_current_se3", "pred_current_sim3", "pred_inverse_se3", "pred_inverse_sim3"):
        pose_trajectories = collect_pose_trajectories(args)
        pred_sim3 = pose_trajectories[f"{args.transform_mode}_transform"]

    pred_clouds = []
    gt_clouds = []
    frame_count = 0
    pairwise_errors = []
    for frame_idx in frame_indices:
        prefix, pred_points, gt_points, gt_camera2world = sample_frame_points(
            sequence_path,
            pointmap_dir,
            frame_idx,
            args.camera_id,
            args.depth_threshold,
        )
        if pred_points.size == 0:
            continue

        if args.transform_mode in ("metric", "pred_current", "pred_current_se3", "pred_current_sim3"):
            pred_extrinsics = load_prediction_extrinsics(camera_params_dir, prefix)
            pred_to_reference = pred_current_to_reference(reference_pred_extrinsics, pred_extrinsics)
            gt_to_reference = gt_reference_from_world @ to_homogeneous(gt_camera2world)
        elif args.transform_mode in ("pred_inverse", "pred_inverse_se3", "pred_inverse_sim3"):
            pred_extrinsics = load_prediction_extrinsics(camera_params_dir, prefix)
            pred_to_reference = pred_inverse_to_reference(reference_pred_extrinsics, pred_extrinsics)
            gt_to_reference = gt_reference_from_world @ to_homogeneous(gt_camera2world)
        elif args.transform_mode == "gt_pose_for_pred":
            gt_to_reference = gt_reference_from_world @ to_homogeneous(gt_camera2world)
            pred_to_reference = gt_to_reference
        else:
            pred_to_reference = np.eye(4, dtype=np.float64)
            gt_to_reference = np.eye(4, dtype=np.float64)

        pred_reference = transform_points(pred_points, pred_to_reference)
        if pred_sim3 is not None:
            pred_reference = apply_sim3(pred_reference, pred_sim3)
        gt_reference = transform_points(gt_points, gt_to_reference)
        valid_mask = np.isfinite(pred_reference).all(axis=1) & np.isfinite(gt_reference).all(axis=1)
        if not np.any(valid_mask):
            continue
        pred_reference = pred_reference[valid_mask]
        gt_reference = gt_reference[valid_mask]
        pred_clouds.append(pred_reference)
        gt_clouds.append(gt_reference)
        pairwise_errors.append(np.linalg.norm(pred_reference - gt_reference, axis=1))
        frame_count += 1

    if not pred_clouds:
        raise RuntimeError("No valid sparse samples were collected.")

    pred = np.concatenate(pred_clouds, axis=0)
    gt = np.concatenate(gt_clouds, axis=0)
    errors = np.concatenate(pairwise_errors, axis=0)
    stats = {
        "frames_used": float(frame_count),
        "point_count": float(pred.shape[0]),
        "pairwise_mae": float(np.mean(errors)),
        "pairwise_rmse": float(np.sqrt(np.mean(errors ** 2))),
    }
    stats.update(compute_pose_diagnostics(sequence_path, camera_params_dir, frame_indices, args.camera_id))
    if pose_trajectories is not None:
        stats["pose_sim3_scale"] = float(pose_trajectories[f"{args.transform_mode}_scale"])
        stats["pose_sim3_rmse"] = float(pose_trajectories[f"{args.transform_mode}_rmse"])
    return downsample_points(pred, args.max_points), downsample_points(gt, args.max_points), stats


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(42)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def set_equal_bev_limits(ax, pred: np.ndarray, gt: np.ndarray) -> None:
    points = np.concatenate([pred[:, [0, 2]], gt[:, [0, 2]]], axis=0)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.size == 0:
        return
    x_min, z_min = np.percentile(points, 1, axis=0)
    x_max, z_max = np.percentile(points, 99, axis=0)
    margin_x = max((x_max - x_min) * 0.08, 1.0)
    margin_z = max((z_max - z_min) * 0.08, 1.0)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(z_min - margin_z, z_max + margin_z)
    ax.set_aspect("equal", adjustable="box")


def set_equal_trajectory_limits(ax, trajectories: list[np.ndarray]) -> None:
    points = np.concatenate([traj[:, [0, 2]] for traj in trajectories if traj.size], axis=0)
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


def label_pose_frames(
    ax,
    positions: np.ndarray,
    frame_indices: np.ndarray,
    *,
    step: int,
    color: str,
    offset: tuple[int, int],
) -> None:
    if step <= 0 or positions.size == 0:
        return

    label_indices = set(range(0, positions.shape[0], step))
    label_indices.add(0)
    label_indices.add(positions.shape[0] - 1)
    for idx in sorted(label_indices):
        if idx < 0 or idx >= positions.shape[0]:
            continue
        point = positions[idx]
        if not np.isfinite(point).all():
            continue
        ax.annotate(
            str(int(frame_indices[idx])),
            (point[0], point[2]),
            textcoords="offset points",
            xytext=offset,
            fontsize=6,
            color=color,
            alpha=0.85,
        )


def save_pose_trajectories(args: argparse.Namespace) -> tuple[Path, dict[str, np.ndarray | float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale_dir = resolve_scale_dir(args)
    trajectories = collect_pose_trajectories(args)
    output = scale_dir.parent / f"debug_pose_trajectory_{scale_dir.name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    gt = trajectories["gt"]
    pred_current = trajectories["pred_current"]
    pred_inverse = trajectories["pred_inverse"]
    pred_current_se3 = trajectories["pred_current_se3"]
    pred_inverse_se3 = trajectories["pred_inverse_se3"]
    pred_current_sim3 = trajectories["pred_current_sim3"]
    pred_inverse_sim3 = trajectories["pred_inverse_sim3"]
    frame_indices = trajectories["frame_indices"]
    ax.plot(gt[:, 0], gt[:, 2], "-o", color="tab:green", markersize=2, linewidth=1.6, label="GT pose")
    ax.plot(
        pred_current[:, 0],
        pred_current[:, 2],
        "-o",
        color="tab:red",
        markersize=2,
        linewidth=1.1,
        label="Pred current",
    )
    ax.plot(
        pred_inverse[:, 0],
        pred_inverse[:, 2],
        "-o",
        color="tab:blue",
        markersize=2,
        linewidth=1.1,
        label="Pred inverse",
    )
    ax.plot(
        pred_current_se3[:, 0],
        pred_current_se3[:, 2],
        "-.",
        color="tab:brown",
        linewidth=1.4,
        label="Pred current + SE3",
    )
    ax.plot(
        pred_inverse_se3[:, 0],
        pred_inverse_se3[:, 2],
        "-.",
        color="tab:pink",
        linewidth=1.4,
        label="Pred inverse + SE3",
    )
    ax.plot(
        pred_current_sim3[:, 0],
        pred_current_sim3[:, 2],
        "--",
        color="tab:orange",
        linewidth=1.5,
        label="Pred current + Sim3",
    )
    ax.plot(
        pred_inverse_sim3[:, 0],
        pred_inverse_sim3[:, 2],
        "--",
        color="tab:purple",
        linewidth=1.5,
        label="Pred inverse + Sim3",
    )
    ax.scatter(gt[0, 0], gt[0, 2], marker="*", s=120, color="black", label="Start")
    label_pose_frames(
        ax,
        gt,
        frame_indices,
        step=args.pose_label_step,
        color="black",
        offset=(4, 4),
    )
    if args.pose_label_all_trajectories:
        label_pose_frames(
            ax,
            pred_current,
            frame_indices,
            step=args.pose_label_step,
            color="tab:red",
            offset=(4, -8),
        )
        label_pose_frames(
            ax,
            pred_inverse,
            frame_indices,
            step=args.pose_label_step,
            color="tab:blue",
            offset=(-12, 4),
        )
        label_pose_frames(
            ax,
            pred_current_se3,
            frame_indices,
            step=args.pose_label_step,
            color="tab:brown",
            offset=(6, 10),
        )
        label_pose_frames(
            ax,
            pred_inverse_se3,
            frame_indices,
            step=args.pose_label_step,
            color="tab:pink",
            offset=(-16, -10),
        )
        label_pose_frames(
            ax,
            pred_current_sim3,
            frame_indices,
            step=args.pose_label_step,
            color="tab:orange",
            offset=(10, -2),
        )
        label_pose_frames(
            ax,
            pred_inverse_sim3,
            frame_indices,
            step=args.pose_label_step,
            color="tab:purple",
            offset=(-18, 10),
        )
    ax.set_title(
        f"{Path(args.sequence_path).name} / {scale_dir.name} pose trajectories\n"
        f"length: GT={trajectories['gt_length']:.2f}m, "
        f"current={trajectories['pred_current_length']:.2f}m, "
        f"inverse={trajectories['pred_inverse_length']:.2f}m\n"
        f"SE3 RMSE: current={trajectories['pred_current_se3_rmse']:.2f}m, "
        f"inverse={trajectories['pred_inverse_se3_rmse']:.2f}m; "
        f"Sim3 RMSE: current={trajectories['pred_current_sim3_rmse']:.2f}m, "
        f"inverse={trajectories['pred_inverse_sim3_rmse']:.2f}m"
    )
    ax.set_xlabel("Reference X (m)")
    ax.set_ylabel("Reference Z (m)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="best")
    set_equal_trajectory_limits(
        ax,
        [
            gt,
            pred_current,
            pred_inverse,
            pred_current_se3,
            pred_inverse_se3,
            pred_current_sim3,
            pred_inverse_sim3,
        ],
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output, trajectories


def save_bev(pred: np.ndarray, gt: np.ndarray, stats: dict[str, float], args: argparse.Namespace) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale_dir = resolve_scale_dir(args)
    output = Path(args.output) if args.output else scale_dir.parent / f"debug_sparse_bev_{scale_dir.name}_{args.transform_mode}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    title = args.title or f"{Path(args.sequence_path).name} / {scale_dir.name} / {args.transform_mode}"
    fig.suptitle(
        f"{title}\nframes={int(stats['frames_used'])}, points={int(stats['point_count'])}, "
        f"pairwise MAE={stats['pairwise_mae']:.3f}m, RMSE={stats['pairwise_rmse']:.3f}m",
        fontsize=12,
    )

    panels = (
        ("GT sparse samples", gt, "tab:green"),
        ("Pred sparse samples", pred, "tab:red"),
        ("Overlay", None, None),
    )
    for ax, (panel_title, points, color) in zip(axes, panels):
        if points is None:
            ax.scatter(gt[:, 0], gt[:, 2], s=1, alpha=0.35, c="tab:green", label="GT")
            ax.scatter(pred[:, 0], pred[:, 2], s=1, alpha=0.35, c="tab:red", label="Pred")
            ax.legend(loc="upper right", markerscale=4)
        else:
            ax.scatter(points[:, 0], points[:, 2], s=1, alpha=0.45, c=color)
        ax.set_title(panel_title)
        ax.set_xlabel("Camera X / right (m)")
        ax.set_ylabel("Camera Z / forward (m)")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        set_equal_bev_limits(ax, pred, gt)

    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    pred, gt, stats = collect_fused_points(args)
    output = save_bev(pred, gt, stats, args)
    print(f"Saved BEV debug visualization: {output}")
    print(
        f"frames_used={int(stats['frames_used'])}, points={int(stats['point_count'])}, "
        f"pairwise_mae={stats['pairwise_mae']:.4f}m, pairwise_rmse={stats['pairwise_rmse']:.4f}m"
    )
    if "gt_relative_translation_norm" in stats:
        print(
            "pose_diagnostic "
            f"frames={int(stats['diagnostic_frame_a']):05d}->{int(stats['diagnostic_frame_b']):05d}, "
            f"gt={stats['gt_relative_translation_norm']:.4f}m, "
            f"pred_current={stats['pred_current_translation_norm']:.4f}m, "
            f"pred_inverse={stats['pred_inverse_translation_norm']:.4f}m"
        )
    if "pose_sim3_scale" in stats:
        print(
            "pose_sim3_alignment "
            f"scale={stats['pose_sim3_scale']:.6f}, "
            f"trajectory_rmse={stats['pose_sim3_rmse']:.4f}m"
        )
    if args.plot_pose_trajectories:
        pose_output, trajectories = save_pose_trajectories(args)
        print(f"Saved pose trajectory visualization: {pose_output}")
        print(
            "pose_trajectory_length "
            f"gt={trajectories['gt_length']:.4f}m, "
            f"pred_current={trajectories['pred_current_length']:.4f}m, "
            f"pred_inverse={trajectories['pred_inverse_length']:.4f}m"
        )


if __name__ == "__main__":
    main()
