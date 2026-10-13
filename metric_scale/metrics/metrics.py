import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from metric_scale.config.config import config
from metric_scale.processing.frame_loader import FrameLoader
from metric_scale.utils.utils import data_utils, file_utils, sequence_utils


class MetricGeometryEvaluator:
    """Evaluate metric geometry accuracy for saved scale variants."""

    DEPTH_EPS = 1e-6
    POINT_REL_THRESHOLD = 0.25
    DEPTH_DELTA_THRESHOLD = 1.25
    DEFAULT_RPE_DELTA_FRAMES = 1
    SPARSE_RECONSTRUCTION_VIS_MAX_POINTS = 200000

    @staticmethod
    def _get_output_dir(depth_model, sequence_name, scale_type):
        scale_dir_names = {
            "absolute": "absolute_scale",
            "relative": "relative_scale",
            "consistent": "consistent_scale",
            "optimized": "optimized_scale",
            "optimized_legacy": "optimized_scale_legacy",
            "optimized_original": "optimized_scale_original",
            "optimized_weighted_bi": "optimized_scale_weighted_bi",
            "optimized_weighted_ifpa": "optimized_scale_weighted_ifpa",
            "selected": "selected_scale",
        }
        if scale_type not in scale_dir_names:
            raise ValueError(f"Unsupported scale_type: {scale_type}")
        scale_dir_name = scale_dir_names[scale_type]

        if depth_model == "pi3_long":
            return os.path.join(config.get_path("pi3_output", sequence_name), scale_dir_name)

        if depth_model == "dust3r_long":
            if scale_type == "absolute":
                return config.get_path("dust3r_absolute_scale", sequence_name)
            return os.path.join(config.get_path("dust3r_output", sequence_name), scale_dir_name)

        if depth_model == "must3r_long":
            if scale_type == "absolute":
                return config.get_path("must3r_absolute_scale", sequence_name)
            return os.path.join(config.get_path("must3r_output", sequence_name), scale_dir_name)

        if depth_model == "scal3r_long":
            if scale_type == "absolute":
                return config.get_path("scal3r_absolute_scale", sequence_name)
            return os.path.join(config.get_path("scal3r_output", sequence_name), scale_dir_name)

        if depth_model == "lingbot_map_long":
            if scale_type == "absolute":
                return config.get_path("lingbot_map_absolute_scale", sequence_name)
            return os.path.join(config.get_path("lingbot_map_output", sequence_name), scale_dir_name)

        if depth_model == "dvlt_long":
            if scale_type == "absolute":
                return os.path.join(config.get_path("dvlt_output", sequence_name), "absolute_scale")
            return os.path.join(config.get_path("dvlt_output", sequence_name), scale_dir_name)

        if depth_model == "4rc_long":
            if scale_type == "absolute":
                return os.path.join(config.get_path("fourrc_output", sequence_name), "absolute_scale")
            return os.path.join(config.get_path("fourrc_output", sequence_name), scale_dir_name)

        if depth_model == "stream3r_long":
            if scale_type == "absolute":
                return os.path.join(config.get_path("stream3r_output", sequence_name), "absolute_scale")
            return os.path.join(config.get_path("stream3r_output", sequence_name), scale_dir_name)

        if depth_model == "vggt_omega_long":
            if scale_type == "absolute":
                return config.get_path("vggt_omega_absolute_scale", sequence_name)
            return os.path.join(config.get_path("vggt_omega_output", sequence_name), scale_dir_name)

        if depth_model == "vggt_long":
            if scale_type == "absolute":
                return config.get_path("vggt_absolute_scale", sequence_name)
            return os.path.join(config.get_path("vggt_output", sequence_name), scale_dir_name)

        if depth_model == "moge":
            return os.path.join(
                os.path.dirname(config.get_path("da3_output", sequence_name)),
                "moge_output",
                scale_dir_name,
            )

        if depth_model == "da3_long":
            if scale_type == "absolute":
                return config.get_path("da3_absolute_scale", sequence_name)
            return os.path.join(config.get_path("da3_output", sequence_name), scale_dir_name)

        raise ValueError(f"Unsupported depth model: {depth_model}")

    @staticmethod
    def _get_frame_name(frame_data):
        return os.path.splitext(os.path.basename(frame_data.image_path))[0]

    @staticmethod
    def _project_lidar_to_image(frame_data):
        points_homogeneous = np.hstack((frame_data.points[:, :3], np.ones((frame_data.points.shape[0], 1))))
        points_camera = np.dot(frame_data.lidar2camera, points_homogeneous.T).T[:, :3]

        visible_mask = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0)
        points_camera = points_camera[visible_mask]
        if points_camera.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        projected_points = np.dot(frame_data.intrinsic, points_camera.T).T
        projected_points[:, 0] /= np.maximum(projected_points[:, 2], MetricGeometryEvaluator.DEPTH_EPS)
        projected_points[:, 1] /= np.maximum(projected_points[:, 2], MetricGeometryEvaluator.DEPTH_EPS)

        valid_mask = (
            np.isfinite(projected_points).all(axis=1)
            & (projected_points[:, 0] >= 0)
            & (projected_points[:, 0] < frame_data.img_width)
            & (projected_points[:, 1] >= 0)
            & (projected_points[:, 1] < frame_data.img_height)
        )
        return projected_points[valid_mask, :2], points_camera[valid_mask]

    @staticmethod
    def _map_pixels_to_pointmap(projected_points, pointmap_shape, image_shape):
        pointmap_height, pointmap_width = pointmap_shape[:2]
        img_height, img_width = image_shape
        scale_factors = data_utils.calculate_scale_factors(img_width, img_height, pointmap_width, pointmap_height)

        mapped_points = projected_points.copy()
        mapped_points[:, 0] *= scale_factors["scale_x"]
        mapped_points[:, 1] *= scale_factors["scale_y"]

        pixel_coords = np.round(mapped_points).astype(int)
        pixel_x = pixel_coords[:, 0]
        pixel_y = pixel_coords[:, 1]

        valid_mask = (
            (pixel_x >= 0)
            & (pixel_x < pointmap_width)
            & (pixel_y >= 0)
            & (pixel_y < pointmap_height)
        )
        return pixel_coords[valid_mask], valid_mask

    @staticmethod
    def _map_pixels_to_pointmap_float(projected_points, pointmap_shape, image_shape):
        pointmap_height, pointmap_width = pointmap_shape[:2]
        img_height, img_width = image_shape
        scale_factors = data_utils.calculate_scale_factors(img_width, img_height, pointmap_width, pointmap_height)

        mapped_points = projected_points.copy()
        mapped_points[:, 0] *= scale_factors["scale_x"]
        mapped_points[:, 1] *= scale_factors["scale_y"]

        valid_mask = (
            np.isfinite(mapped_points).all(axis=1)
            & (mapped_points[:, 0] >= 0)
            & (mapped_points[:, 0] <= pointmap_width - 1)
            & (mapped_points[:, 1] >= 0)
            & (mapped_points[:, 1] <= pointmap_height - 1)
        )
        return mapped_points[valid_mask], valid_mask

    @staticmethod
    def _sample_pointmap(pointmap, projected_points, image_shape):
        if config.depth_native_sampling_mode == "bilinear":
            mapped_points, valid_mask = MetricGeometryEvaluator._map_pixels_to_pointmap_float(
                projected_points,
                pointmap.shape,
                image_shape,
            )
            if mapped_points.size == 0:
                return np.empty((0, 3), dtype=np.float32), valid_mask

            x = mapped_points[:, 0]
            y = mapped_points[:, 1]
            x0 = np.floor(x).astype(np.int64)
            y0 = np.floor(y).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, pointmap.shape[1] - 1)
            y1 = np.clip(y0 + 1, 0, pointmap.shape[0] - 1)
            wx = (x - x0)[:, None]
            wy = (y - y0)[:, None]

            top = pointmap[y0, x0] * (1.0 - wx) + pointmap[y0, x1] * wx
            bottom = pointmap[y1, x0] * (1.0 - wx) + pointmap[y1, x1] * wx
            return (top * (1.0 - wy) + bottom * wy).astype(np.float32), valid_mask

        pixel_coords, valid_mask = MetricGeometryEvaluator._map_pixels_to_pointmap(
            projected_points,
            pointmap.shape,
            image_shape,
        )
        if pixel_coords.size == 0:
            return np.empty((0, 3), dtype=np.float32), valid_mask
        return pointmap[pixel_coords[:, 1], pixel_coords[:, 0]], valid_mask

    @staticmethod
    def _deduplicate_visible_points(pixel_coords, gt_points_camera, pointmap_width, return_indices=False):
        pixel_ids = pixel_coords[:, 1] * pointmap_width + pixel_coords[:, 0]
        gt_depth = gt_points_camera[:, 2]

        order = np.lexsort((gt_depth, pixel_ids))
        sorted_ids = pixel_ids[order]
        _, unique_indices = np.unique(sorted_ids, return_index=True)
        selected_indices = order[unique_indices]

        if return_indices:
            return pixel_coords[selected_indices], gt_points_camera[selected_indices], selected_indices
        return pixel_coords[selected_indices], gt_points_camera[selected_indices]

    @staticmethod
    def _filter_valid_predictions(pred_points, gt_points_camera):
        valid_mask = (
            np.isfinite(pred_points).all(axis=1)
            & np.isfinite(gt_points_camera).all(axis=1)
            & (pred_points[:, 2] > MetricGeometryEvaluator.DEPTH_EPS)
            & (gt_points_camera[:, 2] > MetricGeometryEvaluator.DEPTH_EPS)
        )
        return pred_points[valid_mask], gt_points_camera[valid_mask]

    @staticmethod
    def _to_homogeneous_transform(transform):
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape == (4, 4):
            return transform
        if transform.shape == (3, 4):
            homogeneous = np.eye(4, dtype=np.float64)
            homogeneous[:3, :] = transform
            return homogeneous
        raise ValueError(f"Unsupported transform shape: {transform.shape}")

    @staticmethod
    def _normalize_frame_id(frame_id):
        return str(frame_id).split("_camera")[0]

    @staticmethod
    def _frame_name(frame_id, camera_id=1):
        raw_frame_id = MetricGeometryEvaluator._normalize_frame_id(frame_id)
        return f"{raw_frame_id}_camera{camera_id}"

    @staticmethod
    def _x_to_world(pose):
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

    @staticmethod
    def _load_gt_camera_pose_from_lidar(sequence_path, frame_id, camera_id=1):
        raw_frame_id = MetricGeometryEvaluator._normalize_frame_id(frame_id)
        yaml_path = os.path.join(sequence_path, f"{raw_frame_id}.yaml")
        with open(yaml_path, "r", encoding="utf-8") as yaml_file:
            payload = yaml.safe_load(yaml_file) or {}

        camera_key = f"camera{camera_id}"
        if camera_key not in payload:
            raise KeyError(f"{camera_key} not found in {yaml_path}")

        lidar_pose = payload.get("lidar_pose", [0, 0, 0, 0, 0, 0])
        camera_extrinsic = payload[camera_key].get("extrinsic")
        if camera_extrinsic is None:
            raise KeyError(f"{camera_key}.extrinsic not found in {yaml_path}")

        world_from_lidar = MetricGeometryEvaluator._x_to_world(lidar_pose)
        camera_from_lidar = MetricGeometryEvaluator._to_homogeneous_transform(camera_extrinsic)
        lidar_from_camera = np.linalg.inv(camera_from_lidar)
        return world_from_lidar @ lidar_from_camera

    @staticmethod
    def _transform_points(points, transform):
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        homogeneous_points = np.concatenate(
            [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        transformed = (transform @ homogeneous_points.T).T[:, :3]
        return transformed.astype(np.float32)

    @staticmethod
    def _load_prediction_extrinsics(camera_params_dir, frame_name):
        camera_params_path = os.path.join(camera_params_dir, frame_name + ".json")
        if not os.path.exists(camera_params_path):
            return None, {
                "frame_id": frame_name,
                "status": "missing_prediction_camera_params",
                "camera_params_path": camera_params_path,
            }

        try:
            camera_params = file_utils.read_json(camera_params_path)
            extrinsics = MetricGeometryEvaluator._to_homogeneous_transform(camera_params["extrinsics"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return None, {
                "frame_id": frame_name,
                "status": "invalid_prediction_camera_params",
                "camera_params_path": camera_params_path,
                "error": str(exc),
            }
        return extrinsics, None

    @staticmethod
    def _sample_sparse_prediction_points(prepared_frame, pointmap_dir, depth_threshold=None):
        frame_name = prepared_frame["frame_id"]
        pointmap_path = os.path.join(pointmap_dir, frame_name + ".npy")
        if not os.path.exists(pointmap_path):
            return {
                "frame_id": frame_name,
                "status": "missing_prediction",
                "pointmap_path": pointmap_path,
            }, None, None

        try:
            pointmap = np.load(pointmap_path)
        except (OSError, ValueError) as exc:
            return {
                "frame_id": frame_name,
                "status": "invalid_prediction_pointmap",
                "pointmap_path": pointmap_path,
                "error": str(exc),
            }, None, None
        projected_points = prepared_frame["projected_points"]
        gt_points_camera = prepared_frame["gt_points_camera"]

        pred_points, pointmap_valid_mask = MetricGeometryEvaluator._sample_pointmap(
            pointmap,
            projected_points,
            prepared_frame["image_shape"],
        )
        gt_points_camera = gt_points_camera[pointmap_valid_mask]

        if pred_points.size == 0:
            return {"frame_id": frame_name, "status": "no_valid_projected_pixels"}, None, None

        pixel_coords, pointmap_valid_mask = MetricGeometryEvaluator._map_pixels_to_pointmap(
            projected_points[pointmap_valid_mask],
            pointmap.shape,
            prepared_frame["image_shape"],
        )
        gt_points_camera = gt_points_camera[pointmap_valid_mask]
        pred_points = pred_points[pointmap_valid_mask]

        pixel_coords, gt_points_camera, selected_indices = MetricGeometryEvaluator._deduplicate_visible_points(
            pixel_coords,
            gt_points_camera,
            pointmap.shape[1],
            return_indices=True,
        )
        del pixel_coords
        pred_points = pred_points[selected_indices]

        if depth_threshold is not None:
            depth_mask = gt_points_camera[:, 2] <= depth_threshold
            pred_points = pred_points[depth_mask]
            gt_points_camera = gt_points_camera[depth_mask]

        if pred_points.size == 0:
            return {"frame_id": frame_name, "status": "no_samples_within_depth_threshold"}, None, None
        pred_points, gt_points_camera = MetricGeometryEvaluator._filter_valid_predictions(
            pred_points,
            gt_points_camera,
        )

        if pred_points.size == 0:
            return {"frame_id": frame_name, "status": "no_valid_predicted_points"}, None, None

        return {"frame_id": frame_name, "status": "ok"}, pred_points, gt_points_camera

    @staticmethod
    def _summarize_depth_metrics(pred_depth, gt_depth):
        rel = np.abs(pred_depth - gt_depth) / np.maximum(gt_depth, MetricGeometryEvaluator.DEPTH_EPS)
        delta_ratio = np.maximum(
            pred_depth / np.maximum(gt_depth, MetricGeometryEvaluator.DEPTH_EPS),
            gt_depth / np.maximum(pred_depth, MetricGeometryEvaluator.DEPTH_EPS),
        )

        return {
            "sample_count": int(rel.size),
            "rel_sum": float(np.sum(rel)),
            "delta1_hits": int(np.sum(delta_ratio < MetricGeometryEvaluator.DEPTH_DELTA_THRESHOLD)),
            "mae_sum": float(np.sum(np.abs(pred_depth - gt_depth))),
            "rmse_sum": float(np.sum((pred_depth - gt_depth) ** 2)),
        }

    @staticmethod
    def _summarize_point_metrics(pred_points, gt_points_camera):
        point_error = np.linalg.norm(pred_points - gt_points_camera, axis=1)
        gt_norm = np.linalg.norm(gt_points_camera, axis=1)
        rel = point_error / np.maximum(gt_norm, MetricGeometryEvaluator.DEPTH_EPS)

        return {
            "sample_count": int(rel.size),
            "rel_sum": float(np.sum(rel)),
            "delta1_hits": int(np.sum(rel < MetricGeometryEvaluator.POINT_REL_THRESHOLD)),
            "mae_sum": float(np.sum(point_error)),
            "rmse_sum": float(np.sum(point_error ** 2)),
        }

    @staticmethod
    def _nearest_distances(query_points, reference_points, chunk_size=4096):
        if query_points.size == 0 or reference_points.size == 0:
            return np.empty((0,), dtype=np.float32)

        try:
            from scipy.spatial import cKDTree

            distances, _ = cKDTree(reference_points).query(query_points, k=1)
            return distances.astype(np.float32)
        except ImportError:
            distances = []
            reference = reference_points.astype(np.float32)
            for start in range(0, len(query_points), chunk_size):
                query = query_points[start:start + chunk_size].astype(np.float32)
                diff = query[:, None, :] - reference[None, :, :]
                chunk_dist = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
                distances.append(chunk_dist.astype(np.float32))
            return np.concatenate(distances, axis=0)

    @staticmethod
    def _summarize_reconstruction_metrics(pred_points, gt_points_camera):
        pairwise_error = np.linalg.norm(pred_points - gt_points_camera, axis=1)
        pred_to_gt = MetricGeometryEvaluator._nearest_distances(pred_points, gt_points_camera)
        gt_to_pred = MetricGeometryEvaluator._nearest_distances(gt_points_camera, pred_points)

        return {
            "pair_count": int(pairwise_error.size),
            "pred_point_count": int(pred_to_gt.size),
            "gt_point_count": int(gt_to_pred.size),
            "acc_sum": float(np.sum(pred_to_gt)),
            "comp_sum": float(np.sum(gt_to_pred)),
            "pairwise_mae_sum": float(np.sum(pairwise_error)),
            "pairwise_rmse_sum": float(np.sum(pairwise_error ** 2)),
        }

    @staticmethod
    def _finalize_metrics(raw_metrics):
        sample_count = raw_metrics["sample_count"]
        if sample_count == 0:
            return {
                "sample_count": 0,
                "rel": None,
                "delta1": None,
                "mae": None,
                "rmse": None,
            }

        return {
            "sample_count": sample_count,
            "rel": raw_metrics["rel_sum"] / sample_count,
            "delta1": raw_metrics["delta1_hits"] / sample_count,
            "mae": raw_metrics["mae_sum"] / sample_count,
            "rmse": np.sqrt(raw_metrics["rmse_sum"] / sample_count),
        }

    @staticmethod
    def _finalize_reconstruction_metrics(raw_metrics):
        pair_count = raw_metrics["pair_count"]
        pred_point_count = raw_metrics["pred_point_count"]
        gt_point_count = raw_metrics["gt_point_count"]
        if pair_count == 0 or pred_point_count == 0 or gt_point_count == 0:
            return {
                "sample_count": 0,
                "pair_count": 0,
                "pred_point_count": pred_point_count,
                "gt_point_count": gt_point_count,
                "acc": None,
                "comp": None,
                "chamfer": None,
                "pairwise_mae": None,
                "pairwise_rmse": None,
            }

        acc = raw_metrics["acc_sum"] / pred_point_count
        comp = raw_metrics["comp_sum"] / gt_point_count
        return {
            "sample_count": pair_count,
            "pair_count": pair_count,
            "pred_point_count": pred_point_count,
            "gt_point_count": gt_point_count,
            "acc": acc,
            "comp": comp,
            "chamfer": 0.5 * (acc + comp),
            "pairwise_mae": raw_metrics["pairwise_mae_sum"] / pair_count,
            "pairwise_rmse": np.sqrt(raw_metrics["pairwise_rmse_sum"] / pair_count),
        }

    @staticmethod
    def _average_frame_metrics(per_frame_results, metric_name):
        metric_keys = ("rel", "delta1", "mae", "rmse")
        valid_frame_metrics = []

        for frame_result in per_frame_results:
            if frame_result.get("status") != "ok":
                continue
            frame_metrics = frame_result.get(metric_name)
            if not frame_metrics or frame_metrics.get("sample_count", 0) == 0:
                continue
            valid_frame_metrics.append(frame_metrics)

        if not valid_frame_metrics:
            return {
                "frame_count": 0,
                "rel": None,
                "delta1": None,
                "mae": None,
                "rmse": None,
            }

        averaged_metrics = {"frame_count": len(valid_frame_metrics)}
        for key in metric_keys:
            averaged_metrics[key] = float(np.mean([metrics[key] for metrics in valid_frame_metrics]))
        return averaged_metrics

    @staticmethod
    def _average_frame_reconstruction_metrics(per_frame_results):
        metric_keys = ("acc", "comp", "chamfer", "pairwise_mae", "pairwise_rmse")
        valid_frame_metrics = []

        for frame_result in per_frame_results:
            if frame_result.get("status") != "ok":
                continue
            frame_metrics = frame_result.get("reconstruction")
            if not frame_metrics or frame_metrics.get("sample_count", 0) == 0:
                continue
            valid_frame_metrics.append(frame_metrics)

        if not valid_frame_metrics:
            return {
                "frame_count": 0,
                "acc": None,
                "comp": None,
                "chamfer": None,
                "pairwise_mae": None,
                "pairwise_rmse": None,
            }

        averaged_metrics = {"frame_count": len(valid_frame_metrics)}
        for key in metric_keys:
            averaged_metrics[key] = float(np.mean([metrics[key] for metrics in valid_frame_metrics]))
        return averaged_metrics

    @staticmethod
    def _average_sequence_reconstruction_metrics(group_results):
        metric_keys = ("acc", "comp", "chamfer", "pairwise_mae", "pairwise_rmse")
        valid_group_metrics = []

        for group_result in group_results:
            if group_result.get("status") != "ok":
                continue
            group_metrics = group_result.get("reconstruction_se3")
            if not group_metrics or group_metrics.get("sample_count", 0) == 0:
                continue
            valid_group_metrics.append(group_metrics)

        if not valid_group_metrics:
            return {
                "group_count": 0,
                "acc": None,
                "comp": None,
                "chamfer": None,
                "pairwise_mae": None,
                "pairwise_rmse": None,
            }

        averaged_metrics = {"group_count": len(valid_group_metrics)}
        for key in metric_keys:
            averaged_metrics[key] = float(np.mean([metrics[key] for metrics in valid_group_metrics]))
        return averaged_metrics

    def _empty_metric_accumulator():
        return {
            "sample_count": 0,
            "rel_sum": 0.0,
            "delta1_hits": 0,
            "mae_sum": 0.0,
            "rmse_sum": 0.0,
        }

    @staticmethod
    def _empty_reconstruction_accumulator():
        return {
            "pair_count": 0,
            "pred_point_count": 0,
            "gt_point_count": 0,
            "acc_sum": 0.0,
            "comp_sum": 0.0,
            "pairwise_mae_sum": 0.0,
            "pairwise_rmse_sum": 0.0,
        }

    @staticmethod
    def _empty_error_accumulator():
        return {
            "sample_count": 0,
            "sum": 0.0,
            "sq_sum": 0.0,
            "min": None,
            "max": None,
        }

    @staticmethod
    def _empty_pose_accumulator():
        return {
            "mode": config.pose_evaluation_mode,
            "alignment": "global_se3",
            "scale": config.pose_eval_scale,
            "groups_requested": 0,
            "groups_evaluated": 0,
            "frames_requested": 0,
            "frames_evaluated": 0,
            "rpe_pairs": 0,
            "ate": MetricGeometryEvaluator._empty_error_accumulator(),
            "rpe_trans": MetricGeometryEvaluator._empty_error_accumulator(),
            "rpe_rot": MetricGeometryEvaluator._empty_error_accumulator(),
        }

    @staticmethod
    def _accumulate_error_values(error_accumulator, errors):
        errors = np.asarray(errors, dtype=np.float64).reshape(-1)
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            return

        error_accumulator["sample_count"] += int(errors.size)
        error_accumulator["sum"] += float(np.sum(errors))
        error_accumulator["sq_sum"] += float(np.sum(errors ** 2))

        min_value = float(np.min(errors))
        max_value = float(np.max(errors))
        if error_accumulator["min"] is None:
            error_accumulator["min"] = min_value
        else:
            error_accumulator["min"] = min(error_accumulator["min"], min_value)
        if error_accumulator["max"] is None:
            error_accumulator["max"] = max_value
        else:
            error_accumulator["max"] = max(error_accumulator["max"], max_value)

    @staticmethod
    def _accumulate_error_accumulator(overall_accumulator, error_accumulator):
        sample_count = int(error_accumulator.get("sample_count", 0) or 0)
        if sample_count == 0:
            return

        overall_accumulator["sample_count"] += sample_count
        overall_accumulator["sum"] += float(error_accumulator.get("sum", 0.0) or 0.0)
        overall_accumulator["sq_sum"] += float(error_accumulator.get("sq_sum", 0.0) or 0.0)

        min_value = error_accumulator.get("min")
        max_value = error_accumulator.get("max")
        if min_value is not None:
            overall_accumulator["min"] = (
                float(min_value)
                if overall_accumulator["min"] is None
                else min(overall_accumulator["min"], float(min_value))
            )
        if max_value is not None:
            overall_accumulator["max"] = (
                float(max_value)
                if overall_accumulator["max"] is None
                else max(overall_accumulator["max"], float(max_value))
            )

    @staticmethod
    def _accumulate_pose_accumulator(overall_accumulator, pose_accumulator):
        overall_accumulator["groups_requested"] += int(pose_accumulator.get("groups_requested", 0) or 0)
        overall_accumulator["groups_evaluated"] += int(pose_accumulator.get("groups_evaluated", 0) or 0)
        overall_accumulator["frames_requested"] += int(pose_accumulator.get("frames_requested", 0) or 0)
        overall_accumulator["frames_evaluated"] += int(pose_accumulator.get("frames_evaluated", 0) or 0)
        overall_accumulator["rpe_pairs"] += int(pose_accumulator.get("rpe_pairs", 0) or 0)
        MetricGeometryEvaluator._accumulate_error_accumulator(
            overall_accumulator["ate"],
            pose_accumulator.get("ate", {}),
        )
        MetricGeometryEvaluator._accumulate_error_accumulator(
            overall_accumulator["rpe_trans"],
            pose_accumulator.get("rpe_trans", {}),
        )
        MetricGeometryEvaluator._accumulate_error_accumulator(
            overall_accumulator["rpe_rot"],
            pose_accumulator.get("rpe_rot", {}),
        )

    @staticmethod
    def _finalize_error_accumulator(error_accumulator):
        sample_count = int(error_accumulator.get("sample_count", 0) or 0)
        if sample_count == 0:
            return {
                "sample_count": 0,
                "rmse": None,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

        error_sum = float(error_accumulator.get("sum", 0.0) or 0.0)
        sq_sum = float(error_accumulator.get("sq_sum", 0.0) or 0.0)
        mean = error_sum / sample_count
        mean_sq = sq_sum / sample_count
        variance = max(mean_sq - mean ** 2, 0.0)
        return {
            "sample_count": sample_count,
            "rmse": float(np.sqrt(mean_sq)),
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "min": error_accumulator.get("min"),
            "max": error_accumulator.get("max"),
        }

    @staticmethod
    def _finalize_pose_metrics(pose_accumulator):
        return {
            "mode": pose_accumulator.get("mode", config.pose_evaluation_mode),
            "alignment": pose_accumulator.get("alignment", "global_se3"),
            "scale": float(pose_accumulator.get("scale", config.pose_eval_scale)),
            "groups_requested": int(pose_accumulator.get("groups_requested", 0) or 0),
            "groups_evaluated": int(pose_accumulator.get("groups_evaluated", 0) or 0),
            "frames_requested": int(pose_accumulator.get("frames_requested", 0) or 0),
            "frames_evaluated": int(pose_accumulator.get("frames_evaluated", 0) or 0),
            "rpe_pairs": int(pose_accumulator.get("rpe_pairs", 0) or 0),
            "ate": MetricGeometryEvaluator._finalize_error_accumulator(
                pose_accumulator.get("ate", {})
            ),
            "rpe_trans": MetricGeometryEvaluator._finalize_error_accumulator(
                pose_accumulator.get("rpe_trans", {})
            ),
            "rpe_rot": MetricGeometryEvaluator._finalize_error_accumulator(
                pose_accumulator.get("rpe_rot", {})
            ),
        }

    @staticmethod
    def _accumulate(overall_metrics, frame_metrics):
        for key in overall_metrics:
            overall_metrics[key] += frame_metrics[key]

    @staticmethod
    def _get_selected_frame_ids(sequence_path, frame_count):
        frame_ids = sequence_utils.get_sequence_frame_ids(sequence_path, camera_id=1)
        if frame_count is not None and frame_count > 0:
            return frame_ids[:frame_count]
        return frame_ids

    @staticmethod
    def _empty_sequence_metrics(scale_type, output_dir, frames_requested, distance_bin_specs=None):
        empty_metrics = {
            "scale_type": scale_type,
            "output_dir": output_dir,
            "frames_evaluated": 0,
            "frames_skipped": frames_requested,
            "frames_requested": frames_requested,
            "depth": MetricGeometryEvaluator._finalize_metrics(
                MetricGeometryEvaluator._empty_metric_accumulator()
            ),
            "point": MetricGeometryEvaluator._finalize_metrics(
                MetricGeometryEvaluator._empty_metric_accumulator()
            ),
            "reconstruction": MetricGeometryEvaluator._finalize_reconstruction_metrics(
                MetricGeometryEvaluator._empty_reconstruction_accumulator()
            ),
            "reconstruction_se3": MetricGeometryEvaluator._finalize_reconstruction_metrics(
                MetricGeometryEvaluator._empty_reconstruction_accumulator()
            ),
            "depth_frame_mean": {
                "frame_count": 0,
                "rel": None,
                "delta1": None,
                "mae": None,
                "rmse": None,
            },
            "point_frame_mean": {
                "frame_count": 0,
                "rel": None,
                "delta1": None,
                "mae": None,
                "rmse": None,
            },
            "reconstruction_frame_mean": {
                "frame_count": 0,
                "acc": None,
                "comp": None,
                "chamfer": None,
                "pairwise_mae": None,
                "pairwise_rmse": None,
            },
            "reconstruction_sequence_se3_mean": {
                "group_count": 0,
                "acc": None,
                "comp": None,
                "chamfer": None,
                "pairwise_mae": None,
                "pairwise_rmse": None,
            },
            "reconstruction_mode": config.sparse_reconstruction_mode,
            "reconstruction_groups": [],
            "pose": MetricGeometryEvaluator._finalize_pose_metrics(
                MetricGeometryEvaluator._empty_pose_accumulator()
            ),
            "pose_groups": [],
            "per_frame": [],
            "distance_bins": {
                spec["key"]: MetricCalculator._empty_distance_bin_metrics(
                    scale_type,
                    output_dir,
                    frames_requested,
                    spec,
                )
                for spec in (distance_bin_specs or [])
            },
        }
        if config.sparse_reconstruction_mode == "sequence":
            empty_metrics.pop("reconstruction", None)
            empty_metrics.pop("reconstruction_frame_mean", None)
        else:
            empty_metrics.pop("reconstruction_se3", None)
            empty_metrics.pop("reconstruction_sequence_se3_mean", None)
        return empty_metrics

    @staticmethod
    def _prepare_gt_frame(sequence_path, frame_id, frame_cache=None):
        frame_data = None if frame_cache is None else frame_cache.get(frame_id)
        if frame_data is None:
            frame_data = FrameLoader(sequence_path, camera_id=1, frame_id=frame_id)
            if not frame_data.process_data():
                return {"frame_id": frame_id, "status": "failed_to_load_ground_truth"}

        if frame_data.points is None or frame_data.image_path is None:
            return {"frame_id": frame_id, "status": "failed_to_load_ground_truth"}

        cached_prepared = getattr(frame_data, "_metric_geometry_prepared", None)
        if cached_prepared is not None:
            return cached_prepared

        frame_name = MetricGeometryEvaluator._get_frame_name(frame_data)
        projected_points, gt_points_camera = MetricGeometryEvaluator._project_lidar_to_image(frame_data)
        gt_camera2world = np.asarray(frame_data.camera2world, dtype=np.float32)
        if projected_points.size == 0:
            prepared = {
                "frame_id": frame_name,
                "status": "no_visible_lidar_points",
                "gt_camera2world": gt_camera2world,
            }
            setattr(frame_data, "_metric_geometry_prepared", prepared)
            return prepared

        prepared = {
            "frame_id": frame_name,
            "status": "ok",
            "projected_points": projected_points,
            "gt_points_camera": gt_points_camera,
            "image_shape": (frame_data.img_height, frame_data.img_width),
            "gt_camera2world": gt_camera2world,
        }
        setattr(frame_data, "_metric_geometry_prepared", prepared)
        return prepared

    @staticmethod
    def _evaluate_variant_for_prepared_frame(prepared_frame, pointmap_dir, depth_threshold=None, distance_bin_specs=None):
        frame_name = prepared_frame["frame_id"]
        sample_result, pred_points, gt_points_camera = (
            MetricGeometryEvaluator._sample_sparse_prediction_points(
                prepared_frame,
                pointmap_dir,
                depth_threshold=depth_threshold,
            )
        )
        if sample_result.get("status") != "ok":
            return sample_result, None, None, None, {}

        def summarize_subset(pred_subset, gt_subset, empty_status):
            if pred_subset.size == 0:
                return {"frame_id": frame_name, "status": empty_status}, None, None

            pred_depth = pred_subset[:, 2]
            gt_depth = gt_subset[:, 2]

            subset_depth_metrics = MetricGeometryEvaluator._summarize_depth_metrics(pred_depth, gt_depth)
            subset_point_metrics = MetricGeometryEvaluator._summarize_point_metrics(pred_subset, gt_subset)

            return {
                "frame_id": frame_name,
                "status": "ok",
                "depth": MetricGeometryEvaluator._finalize_metrics(subset_depth_metrics),
                "point": MetricGeometryEvaluator._finalize_metrics(subset_point_metrics),
            }, subset_depth_metrics, subset_point_metrics

        frame_result, frame_depth_metrics, frame_point_metrics = summarize_subset(
            pred_points,
            gt_points_camera,
            "no_valid_predicted_points",
        )
        frame_reconstruction_metrics = None
        if config.enable_sparse_reconstruction and config.sparse_reconstruction_mode == "frame":
            min_points = config.sparse_reconstruction_min_points
            if pred_points.shape[0] >= min_points:
                frame_reconstruction_metrics = (
                    MetricGeometryEvaluator._summarize_reconstruction_metrics(
                        pred_points,
                        gt_points_camera,
                    )
                )
                frame_result["reconstruction"] = (
                    MetricGeometryEvaluator._finalize_reconstruction_metrics(
                        frame_reconstruction_metrics
                    )
                )
                frame_result["reconstruction_status"] = "ok"
            else:
                frame_result["reconstruction"] = (
                    MetricGeometryEvaluator._finalize_reconstruction_metrics(
                        MetricGeometryEvaluator._empty_reconstruction_accumulator()
                    )
                )
                frame_result["reconstruction_status"] = "too_few_reconstruction_points"
                frame_result["reconstruction_min_points"] = min_points
                frame_result["reconstruction_point_count"] = int(pred_points.shape[0])

        distance_bin_results = {}
        for spec in distance_bin_specs or []:
            gt_depth = gt_points_camera[:, 2]
            depth_mask = gt_depth > spec["min_depth"]
            if spec["max_depth"] is not None:
                depth_mask &= gt_depth <= spec["max_depth"]
            bin_pred_points = pred_points[depth_mask]
            bin_gt_points = gt_points_camera[depth_mask]
            bin_result, bin_depth_metrics, bin_point_metrics = summarize_subset(
                bin_pred_points,
                bin_gt_points,
                "no_samples_in_distance_bin",
            )
            distance_bin_results[spec["key"]] = {
                "frame_result": {
                    **bin_result,
                    "label": spec["label"],
                    "min_depth": spec["min_depth"],
                    "max_depth": spec["max_depth"],
                },
                "depth_metrics": bin_depth_metrics,
                "point_metrics": bin_point_metrics,
            }

        frame_result["distance_bins"] = {
            key: value["frame_result"] for key, value in distance_bin_results.items()
        }
        return (
            frame_result,
            frame_depth_metrics,
            frame_point_metrics,
            frame_reconstruction_metrics,
            distance_bin_results,
        )

    @staticmethod
    def _evaluate_frame_variants(sequence_path, pointmap_dirs_by_key, frame_id, depth_threshold=None, distance_bin_specs=None, frame_cache=None):
        prepared_frame = MetricGeometryEvaluator._prepare_gt_frame(sequence_path, frame_id, frame_cache=frame_cache)
        results_by_key = {}

        if prepared_frame.get("status") != "ok":
            for scale_key in pointmap_dirs_by_key:
                frame_result = dict(prepared_frame)
                frame_result.pop("gt_camera2world", None)
                results_by_key[scale_key] = (frame_result, None, None, None, {})
            return results_by_key

        variant_items = list(pointmap_dirs_by_key.items())
        if config.metric_workers <= 1 and len(variant_items) > 1:
            with ThreadPoolExecutor(max_workers=len(variant_items)) as executor:
                variant_outputs = list(
                    executor.map(
                        lambda item: (
                            item[0],
                            MetricGeometryEvaluator._evaluate_variant_for_prepared_frame(
                                prepared_frame,
                                item[1],
                                depth_threshold=depth_threshold,
                                distance_bin_specs=distance_bin_specs,
                            ),
                        ),
                        variant_items,
                    )
                )
            for scale_key, variant_result in variant_outputs:
                results_by_key[scale_key] = variant_result
        else:
            for scale_key, pointmap_dir in variant_items:
                results_by_key[scale_key] = MetricGeometryEvaluator._evaluate_variant_for_prepared_frame(
                    prepared_frame,
                    pointmap_dir,
                    depth_threshold=depth_threshold,
                    distance_bin_specs=distance_bin_specs,
                )

        return results_by_key

    @staticmethod
    def _evaluate_sequence_reconstruction_group(
        sequence_path,
        pointmap_dir,
        frame_ids,
        depth_threshold=None,
        frame_cache=None,
        group_count=1,
    ):
        group_id = f"{frame_ids[0]}_{frame_ids[-1]}" if frame_ids else "empty"
        pose_source = config.sparse_reconstruction_pose_source
        pred_pose_source = (
            "camera_params_extrinsics_w2c"
            if pose_source == "pred"
            else "gt_lidar_pose_camera_extrinsic"
        )
        base_result = {
            "group_id": group_id,
            "frame_start": frame_ids[0] if frame_ids else None,
            "frame_end": frame_ids[-1] if frame_ids else None,
            "frames_requested": len(frame_ids),
            "frames_used": 0,
            "reference_frame": frame_ids[0] if frame_ids else None,
            "gt_pose_source": "lidar_pose_camera_extrinsic",
            "pred_pose_source": pred_pose_source,
            "sparse_reconstruction_pose_source": pose_source,
        }
        if not frame_ids:
            return {**base_result, "status": "empty_group"}, None

        camera_params_dir = os.path.join(os.path.dirname(pointmap_dir), "camera_params")
        reference_name = MetricGeometryEvaluator._frame_name(frame_ids[0], camera_id=1)
        reference_extrinsics = None
        if pose_source == "pred":
            reference_extrinsics, reference_error = MetricGeometryEvaluator._load_prediction_extrinsics(
                camera_params_dir,
                reference_name,
            )
            if reference_error is not None:
                return {
                    **base_result,
                    "status": "missing_reference_prediction_pose",
                    "reference_frame": reference_name,
                    "reference_error": reference_error,
                }, None

        try:
            reference_gt_camera2world = MetricGeometryEvaluator._load_gt_camera_pose_from_lidar(
                sequence_path,
                frame_ids[0],
                camera_id=1,
            )
        except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
            return {
                **base_result,
                "status": "missing_reference_gt_pose",
                "reference_frame": reference_name,
                "error": str(exc),
            }, None

        try:
            gt_reference_from_world = np.linalg.inv(reference_gt_camera2world)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {
                **base_result,
                "status": "invalid_reference_gt_pose",
                "reference_frame": reference_name,
                "error": str(exc),
            }, None

        pred_group_points = []
        gt_group_points = []
        used_frames = []
        skipped_frames = []

        for frame_id in frame_ids:
            prepared_frame = MetricGeometryEvaluator._prepare_gt_frame(
                sequence_path,
                frame_id,
                frame_cache=frame_cache,
            )
            frame_name = prepared_frame.get("frame_id", frame_id)
            if prepared_frame.get("status") != "ok":
                skipped_frames.append({"frame_id": frame_name, "status": prepared_frame.get("status")})
                continue

            sample_result, pred_points, gt_points_camera = (
                MetricGeometryEvaluator._sample_sparse_prediction_points(
                    prepared_frame,
                    pointmap_dir,
                    depth_threshold=depth_threshold,
                )
            )
            if sample_result.get("status") != "ok":
                skipped_frames.append({"frame_id": frame_name, "status": sample_result.get("status")})
                continue

            pred_extrinsics = None
            if pose_source == "pred":
                pred_extrinsics, pred_error = MetricGeometryEvaluator._load_prediction_extrinsics(
                    camera_params_dir,
                    frame_name,
                )
                if pred_error is not None:
                    skipped_frames.append({"frame_id": frame_name, "status": pred_error.get("status")})
                    continue

            try:
                frame_gt_camera2world = MetricGeometryEvaluator._load_gt_camera_pose_from_lidar(
                    sequence_path,
                    frame_id,
                    camera_id=1,
                )
                gt_frame_to_reference = gt_reference_from_world @ frame_gt_camera2world
                if pose_source == "pred":
                    pred_frame_to_reference = reference_extrinsics @ np.linalg.inv(pred_extrinsics)
                else:
                    pred_frame_to_reference = gt_frame_to_reference
            except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
                skipped_frames.append(
                    {
                        "frame_id": frame_name,
                        "status": "invalid_frame_pose",
                        "error": str(exc),
                    }
                )
                continue

            pred_points_reference = MetricGeometryEvaluator._transform_points(
                pred_points,
                pred_frame_to_reference,
            )
            gt_points_reference = MetricGeometryEvaluator._transform_points(
                gt_points_camera,
                gt_frame_to_reference,
            )
            pred_camera_position_reference = pred_frame_to_reference[:3, 3]
            gt_camera_position_reference = gt_frame_to_reference[:3, 3]
            valid_mask = np.isfinite(pred_points_reference).all(axis=1) & np.isfinite(gt_points_reference).all(axis=1)
            if not np.any(valid_mask):
                skipped_frames.append({"frame_id": frame_name, "status": "no_finite_reference_points"})
                continue

            pred_group_points.append(pred_points_reference[valid_mask])
            gt_group_points.append(gt_points_reference[valid_mask])
            used_frames.append(
                {
                    "frame_name": frame_name,
                    "pred_camera_position_reference": pred_camera_position_reference.astype(float).tolist(),
                    "gt_camera_position_reference": gt_camera_position_reference.astype(float).tolist(),
                }
            )

        if not pred_group_points:
            return {
                **base_result,
                "status": "no_valid_group_frames",
                "reference_frame": reference_name,
                "skipped_frame_count": len(skipped_frames),
            }, None

        pred_points = np.concatenate(pred_group_points, axis=0)
        gt_points = np.concatenate(gt_group_points, axis=0)
        pred_positions = np.asarray(
            [frame["pred_camera_position_reference"] for frame in used_frames],
            dtype=np.float64,
        )
        gt_positions = np.asarray(
            [frame["gt_camera_position_reference"] for frame in used_frames],
            dtype=np.float64,
        )
        min_points = config.sparse_reconstruction_min_points
        if pred_points.shape[0] < min_points:
            return {
                **base_result,
                "status": "too_few_reconstruction_points",
                "reference_frame": reference_name,
                "frames_used": len(used_frames),
                "used_frames": [frame["frame_name"] for frame in used_frames],
                "skipped_frame_count": len(skipped_frames),
                "reconstruction_min_points": min_points,
                "reconstruction_point_count": int(pred_points.shape[0]),
            }, None

        aligned_pred_points, aligned_gt_points, reconstruction_metrics, reconstruction_alignment = (
            MetricGeometryEvaluator._compute_global_se3_reconstruction_metrics(
                pred_points,
                gt_points,
                pred_positions,
                gt_positions,
            )
        )
        if reconstruction_metrics is None:
            return {
                **base_result,
                "status": "failed_reconstruction_global_se3_alignment",
                "reference_frame": reference_name,
                "frames_used": len(used_frames),
                "used_frames": [frame["frame_name"] for frame in used_frames],
                "skipped_frame_count": len(skipped_frames),
                "pred_point_count": int(pred_points.shape[0]),
                "gt_point_count": int(gt_points.shape[0]),
                "reconstruction_se3_alignment": reconstruction_alignment,
            }, None

        visualization_path = MetricGeometryEvaluator._save_sparse_reconstruction_bev_visualization(
            aligned_pred_points,
            aligned_gt_points,
            pointmap_dir,
            group_id,
            group_count=group_count,
        )
        return {
            **base_result,
            "status": "ok",
            "reference_frame": reference_name,
            "frames_used": len(used_frames),
            "used_frames": [frame["frame_name"] for frame in used_frames],
            "skipped_frame_count": len(skipped_frames),
            "pred_point_count": int(pred_points.shape[0]),
            "gt_point_count": int(gt_points.shape[0]),
            "reconstruction_se3": MetricGeometryEvaluator._finalize_reconstruction_metrics(
                reconstruction_metrics
            ),
            "reconstruction_se3_alignment": reconstruction_alignment,
            "sparse_reconstruction_visualization_path": visualization_path,
        }, reconstruction_metrics

    @staticmethod
    def _downsample_visualization_points(points):
        points = np.asarray(points, dtype=np.float32)
        if points.shape[0] <= MetricGeometryEvaluator.SPARSE_RECONSTRUCTION_VIS_MAX_POINTS:
            return points
        rng = np.random.default_rng(42)
        indices = rng.choice(
            points.shape[0],
            size=MetricGeometryEvaluator.SPARSE_RECONSTRUCTION_VIS_MAX_POINTS,
            replace=False,
        )
        return points[indices]

    @staticmethod
    def _compute_global_se3_reconstruction_metrics(pred_points, gt_points, pred_positions, gt_positions):
        pred_points = np.asarray(pred_points, dtype=np.float32)
        gt_points = np.asarray(gt_points, dtype=np.float32)
        alignment_source = config.sparse_reconstruction_alignment
        valid_mask = np.isfinite(pred_points).all(axis=1) & np.isfinite(gt_points).all(axis=1)
        if not np.any(valid_mask):
            return None, None, None, {
                "type": "global_se3",
                "source": alignment_source,
                "pose_source": config.sparse_reconstruction_pose_source,
                "status": "no_finite_points_for_global_se3",
            }

        pred_points = pred_points[valid_mask]
        gt_points = gt_points[valid_mask]
        if alignment_source == "point_cloud":
            rotation, translation = MetricGeometryEvaluator._fit_global_se3(
                pred_points,
                gt_points,
            )
        else:
            rotation, translation = MetricGeometryEvaluator._fit_global_se3(
                pred_positions,
                gt_positions,
            )
        if rotation is None or translation is None:
            return None, None, None, {
                "type": "global_se3",
                "source": alignment_source,
                "pose_source": config.sparse_reconstruction_pose_source,
                "status": "failed_global_se3_alignment",
            }
        aligned_pred_points = MetricGeometryEvaluator._apply_global_se3(
            pred_points,
            rotation,
            translation,
        ).astype(np.float32)
        metrics = MetricGeometryEvaluator._summarize_reconstruction_metrics(
            aligned_pred_points,
            gt_points,
        )
        return aligned_pred_points, gt_points, metrics, {
            "type": "global_se3",
            "source": alignment_source,
            "pose_source": config.sparse_reconstruction_pose_source,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "status": "ok",
        }

    @staticmethod
    def _set_sparse_reconstruction_bev_limits(axes, pred_points, gt_points):
        point_sets = []
        if pred_points.size:
            point_sets.append(pred_points[:, [0, 2]])
        if gt_points.size:
            point_sets.append(gt_points[:, [0, 2]])
        if not point_sets:
            return

        points = np.concatenate(point_sets, axis=0)
        points = points[np.isfinite(points).all(axis=1)]
        if points.size == 0:
            return

        min_x, min_z = np.percentile(points, 1, axis=0)
        max_x, max_z = np.percentile(points, 99, axis=0)
        margin = max(max_x - min_x, max_z - min_z, 1.0) * 0.08
        for ax in axes:
            ax.set_xlim(float(min_x - margin), float(max_x + margin))
            ax.set_ylim(float(min_z - margin), float(max_z + margin))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Reference Camera X / right (m)")
            ax.set_ylabel("Reference Camera Z / forward (m)")

    @staticmethod
    def _save_sparse_reconstruction_bev_visualization(
        pred_points,
        gt_points,
        pointmap_dir,
        group_id,
        group_count=1,
    ):
        if not config.enable_sparse_reconstruction_visualization:
            return None
        if pred_points.size == 0 or gt_points.size == 0:
            return None

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        gt_plot = MetricGeometryEvaluator._downsample_visualization_points(gt_points)
        pred_plot = MetricGeometryEvaluator._downsample_visualization_points(pred_points)

        scale_dir = os.path.dirname(pointmap_dir)
        scale_name = os.path.basename(scale_dir)
        model_output_dir = os.path.dirname(scale_dir)
        visualization_dir = os.path.join(model_output_dir, "sparse_reconstruction_visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
        safe_group_id = str(group_id).replace(os.sep, "_")
        output_filename = f"{scale_name}.png" if group_count <= 1 else f"{scale_name}_{safe_group_id}.png"
        output_path = os.path.join(visualization_dir, output_filename)

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), dpi=180)
        axes[0].scatter(gt_plot[:, 0], gt_plot[:, 2], s=0.25, alpha=0.45, color="#2ca02c")
        axes[0].set_title(f"GT sparse samples\nN={gt_points.shape[0]}")

        axes[1].scatter(pred_plot[:, 0], pred_plot[:, 2], s=0.25, alpha=0.45, color="#1f77b4")
        axes[1].set_title(f"Pred sparse samples\nN={pred_points.shape[0]}")

        axes[2].scatter(gt_plot[:, 0], gt_plot[:, 2], s=0.25, alpha=0.35, color="#2ca02c", label="GT")
        axes[2].scatter(pred_plot[:, 0], pred_plot[:, 2], s=0.25, alpha=0.35, color="#1f77b4", label="Pred")
        axes[2].set_title("Overlay")
        axes[2].legend(loc="best", fontsize=8)

        MetricGeometryEvaluator._set_sparse_reconstruction_bev_limits(axes, pred_plot, gt_plot)
        alignment_label = (
            "point-cloud global SE3"
            if config.sparse_reconstruction_alignment == "point_cloud"
            else "trajectory global SE3"
        )
        fig.suptitle(f"{scale_name} sparse reconstruction BEV ({alignment_label}): {group_id}", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    @staticmethod
    def _evaluate_sequence_reconstruction_groups(
        sequence_path,
        pointmap_dir,
        frame_ids,
        depth_threshold=None,
        frame_cache=None,
    ):
        subsequence_size = max(1, int(config.subsequence_size))
        groups = data_utils.generate_frame_id_subsequences(frame_ids, subsequence_size)
        group_results = []
        group_metrics = []
        group_count = len(groups)
        for group_frame_ids in groups:
            group_result, reconstruction_metrics = (
                MetricGeometryEvaluator._evaluate_sequence_reconstruction_group(
                    sequence_path,
                    pointmap_dir,
                    group_frame_ids,
                    depth_threshold=depth_threshold,
                    frame_cache=frame_cache,
                    group_count=group_count,
                )
            )
            group_results.append(group_result)
            if reconstruction_metrics is not None:
                group_metrics.append(reconstruction_metrics)
        return group_results, group_metrics

    @staticmethod
    def _pose_group_frame_ids(frame_ids):
        if config.pose_evaluation_mode == "sequence":
            return [frame_ids]
        subsequence_size = max(1, int(config.subsequence_size))
        return data_utils.generate_frame_id_subsequences(frame_ids, subsequence_size)

    @staticmethod
    def _scale_pose_translation(pose_c2w):
        pose_c2w = np.asarray(pose_c2w, dtype=np.float64).copy()
        pose_c2w[:3, 3] *= float(config.pose_eval_scale)
        return pose_c2w

    @staticmethod
    def _load_pose_records(sequence_path, camera_params_dir, frame_ids, frame_cache=None):
        del frame_cache
        pose_entries = []
        records = []
        skipped_frames = []

        for frame_id in frame_ids:
            frame_name = MetricGeometryEvaluator._frame_name(frame_id, camera_id=1)

            try:
                gt_c2w = MetricGeometryEvaluator._load_gt_camera_pose_from_lidar(
                    sequence_path,
                    frame_id,
                    camera_id=1,
                )
            except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
                skipped_frames.append(
                    {
                        "frame_id": frame_name,
                        "status": "missing_gt_pose",
                        "error": str(exc),
                    }
                )
                continue

            pred_extrinsics, pred_error = MetricGeometryEvaluator._load_prediction_extrinsics(
                camera_params_dir,
                frame_name,
            )
            if pred_error is not None:
                skipped_frames.append(
                    {
                        "frame_id": frame_name,
                        "status": pred_error.get("status", "missing_prediction_pose"),
                    }
                )
                continue

            try:
                pred_c2w = np.linalg.inv(pred_extrinsics)
            except (ValueError, np.linalg.LinAlgError) as exc:
                skipped_frames.append(
                    {
                        "frame_id": frame_name,
                        "status": "invalid_pose",
                        "error": str(exc),
                    }
                )
                continue

            pose_entries.append(
                {
                    "frame_id": frame_name,
                    "pred_w2c": pred_extrinsics,
                    "pred_c2w_raw": pred_c2w,
                    "gt_c2w": gt_c2w,
                }
            )

        if not pose_entries:
            return records, skipped_frames

        try:
            gt_reference_inv = np.linalg.inv(pose_entries[0]["gt_c2w"])
            pred_reference_w2c = pose_entries[0]["pred_w2c"]
        except np.linalg.LinAlgError as exc:
            skipped_frames.append(
                {
                    "frame_id": pose_entries[0]["frame_id"],
                    "status": "invalid_reference_pose",
                    "error": str(exc),
                }
            )
            return records, skipped_frames

        for pose_entry in pose_entries:
            gt_ref_from_camera = gt_reference_inv @ pose_entry["gt_c2w"]
            pred_ref_from_camera = pred_reference_w2c @ pose_entry["pred_c2w_raw"]
            records.append(
                {
                    "frame_id": pose_entry["frame_id"],
                    "pred_c2w": MetricGeometryEvaluator._scale_pose_translation(pred_ref_from_camera),
                    "gt_c2w": gt_ref_from_camera,
                }
            )

        return records, skipped_frames

    @staticmethod
    def _trajectory_length(positions):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

    @staticmethod
    def _save_pose_bev_visualization(records, camera_params_dir, group_id, group_count=1):
        if not config.enable_pose_visualization:
            return None
        if not records:
            return None

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        pose_alignment = MetricGeometryEvaluator._compute_global_se3_pose_alignment(records)
        if pose_alignment is None:
            return None

        gt_positions = pose_alignment["gt_positions"]
        pred_positions = pose_alignment["pred_positions"]
        aligned_positions = pose_alignment["aligned_positions"]

        scale_dir = os.path.dirname(camera_params_dir)
        scale_name = os.path.basename(scale_dir)
        model_output_dir = os.path.dirname(scale_dir)
        visualization_dir = os.path.join(model_output_dir, "pose_visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
        safe_group_id = str(group_id).replace(os.sep, "_")
        output_filename = f"{scale_name}.png" if group_count <= 1 else f"{scale_name}_{safe_group_id}.png"
        output_path = os.path.join(visualization_dir, output_filename)

        plot_positions = [gt_positions, pred_positions]
        if aligned_positions.size > 0:
            plot_positions.append(aligned_positions)

        all_x = np.concatenate([positions[:, 0] for positions in plot_positions])
        all_z = np.concatenate([positions[:, 2] for positions in plot_positions])
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        z_min, z_max = float(np.min(all_z)), float(np.max(all_z))
        padding = max(x_max - x_min, z_max - z_min, 1.0) * 0.08

        fig, ax = plt.subplots(figsize=(7.0, 8.0), dpi=180)
        ax.plot(
            gt_positions[:, 0],
            gt_positions[:, 2],
            "-o",
            color="#2ca02c",
            markersize=2.5,
            linewidth=1.5,
            label="GT pose",
        )
        ax.plot(
            pred_positions[:, 0],
            pred_positions[:, 2],
            "-o",
            color="#d62728",
            markersize=2.5,
            linewidth=1.3,
            label="Pred pose raw",
        )
        if aligned_positions.size > 0:
            ax.plot(
                aligned_positions[:, 0],
                aligned_positions[:, 2],
                "--o",
                color="#1f77b4",
                markersize=2.5,
                linewidth=1.3,
                label="Pred pose + global SE3",
            )
        ax.scatter(gt_positions[0, 0], gt_positions[0, 2], marker="*", s=110, color="#2ca02c", label="GT start")
        ax.scatter(pred_positions[0, 0], pred_positions[0, 2], marker="*", s=90, color="#d62728", label="Pred start")
        ax.scatter(gt_positions[-1, 0], gt_positions[-1, 2], marker="s", s=45, color="#2ca02c", label="GT end")
        ax.scatter(pred_positions[-1, 0], pred_positions[-1, 2], marker="s", s=40, color="#d62728", label="Pred end")
        if aligned_positions.size > 0:
            ax.scatter(
                aligned_positions[0, 0],
                aligned_positions[0, 2],
                marker="*",
                s=80,
                color="#1f77b4",
                label="Global SE3 start",
            )
            ax.scatter(
                aligned_positions[-1, 0],
                aligned_positions[-1, 2],
                marker="s",
                s=36,
                color="#1f77b4",
                label="Global SE3 end",
            )

        label_step = max(1, len(gt_positions) // 10)
        for idx in range(0, len(gt_positions), label_step):
            ax.text(gt_positions[idx, 0], gt_positions[idx, 2], str(idx), fontsize=6, color="#2ca02c")
            ax.text(pred_positions[idx, 0], pred_positions[idx, 2], str(idx), fontsize=6, color="#d62728")
            if aligned_positions.size > 0:
                ax.text(
                    aligned_positions[idx, 0],
                    aligned_positions[idx, 2],
                    str(idx),
                    fontsize=6,
                    color="#1f77b4",
                )

        gt_length = MetricGeometryEvaluator._trajectory_length(gt_positions)
        pred_length = MetricGeometryEvaluator._trajectory_length(pred_positions)
        raw_rmse = float(np.sqrt(np.mean(np.sum((pred_positions - gt_positions) ** 2, axis=1))))
        aligned_rmse = pose_alignment["rmse"]
        rmse_text = f"raw RMSE={raw_rmse:.2f}m"
        rmse_text += f", global SE3 RMSE={aligned_rmse:.2f}m"
        ax.set_title(
            f"{scale_name} pose BEV: {group_id}\n"
            f"frames={len(gt_positions)}, GT length={gt_length:.2f}m, Pred length={pred_length:.2f}m\n"
            f"{rmse_text}"
        )
        ax.set_xlabel("Reference Camera X / right (m)")
        ax.set_ylabel("Reference Camera Z / forward (m)")
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(z_min - padding, z_max + padding)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    @staticmethod
    def _fit_global_se3(pred_points, gt_points):
        pred_points = np.asarray(pred_points, dtype=np.float64)
        gt_points = np.asarray(gt_points, dtype=np.float64)
        if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
            return None, None

        finite_mask = np.isfinite(pred_points).all(axis=1) & np.isfinite(gt_points).all(axis=1)
        pred_points = pred_points[finite_mask]
        gt_points = gt_points[finite_mask]
        if pred_points.shape[0] == 0:
            return None, None

        pred_mean = np.mean(pred_points, axis=0)
        gt_mean = np.mean(gt_points, axis=0)
        pred_centered = pred_points - pred_mean
        gt_centered = gt_points - gt_mean
        covariance = pred_centered.T @ gt_centered
        try:
            u_matrix, _, vt_matrix = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            return None, None

        rotation = vt_matrix.T @ u_matrix.T
        if np.linalg.det(rotation) < 0.0:
            vt_matrix[-1, :] *= -1.0
            rotation = vt_matrix.T @ u_matrix.T
        translation = gt_mean - rotation @ pred_mean
        return rotation, translation

    @staticmethod
    def _apply_global_se3(points, rotation, translation):
        points = np.asarray(points, dtype=np.float64)
        return (rotation @ points.T).T + translation

    @staticmethod
    def _align_positions_global_se3(pred_positions, gt_positions):
        pred_positions = np.asarray(pred_positions, dtype=np.float64)
        gt_positions = np.asarray(gt_positions, dtype=np.float64)
        rotation, translation = MetricGeometryEvaluator._fit_global_se3(
            pred_positions,
            gt_positions,
        )
        if rotation is None or translation is None:
            return None, None, None
        aligned_positions = MetricGeometryEvaluator._apply_global_se3(
            pred_positions,
            rotation,
            translation,
        )
        return aligned_positions, rotation, translation

    @staticmethod
    def _rotation_angle_degrees(rotation):
        trace_value = float(np.trace(rotation))
        cosine = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _compute_global_se3_pose_alignment(records):
        pred_positions = np.asarray(
            [record["pred_c2w"][:3, 3] for record in records],
            dtype=np.float64,
        )
        gt_positions = np.asarray(
            [record["gt_c2w"][:3, 3] for record in records],
            dtype=np.float64,
        )
        finite_mask = np.isfinite(pred_positions).all(axis=1) & np.isfinite(gt_positions).all(axis=1)
        pred_positions = pred_positions[finite_mask]
        gt_positions = gt_positions[finite_mask]
        if pred_positions.shape[0] < config.pose_eval_min_frames:
            return None

        aligned_positions, rotation, translation = MetricGeometryEvaluator._align_positions_global_se3(
            pred_positions,
            gt_positions,
        )
        if aligned_positions is None:
            return None

        errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)
        return {
            "gt_positions": gt_positions,
            "pred_positions": pred_positions,
            "aligned_positions": aligned_positions,
            "errors": errors,
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "alignment": {
                "type": "global_se3",
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            },
        }

    @staticmethod
    def _compute_tum_ate_se3(records):
        pose_alignment = MetricGeometryEvaluator._compute_global_se3_pose_alignment(records)
        if pose_alignment is None:
            return None, None
        return pose_alignment["errors"], pose_alignment["alignment"]

    @staticmethod
    def _compute_tum_rpe(records):
        trans_errors = []
        rot_errors = []
        delta = MetricGeometryEvaluator.DEFAULT_RPE_DELTA_FRAMES

        for idx in range(0, len(records) - delta):
            pred_i = records[idx]["pred_c2w"]
            pred_j = records[idx + delta]["pred_c2w"]
            gt_i = records[idx]["gt_c2w"]
            gt_j = records[idx + delta]["gt_c2w"]
            try:
                pred_relative = np.linalg.inv(pred_i) @ pred_j
                gt_relative = np.linalg.inv(gt_i) @ gt_j
                error_transform = np.linalg.inv(gt_relative) @ pred_relative
            except np.linalg.LinAlgError:
                continue

            translation_error = error_transform[:3, 3]
            rotation_error = error_transform[:3, :3]
            if not np.isfinite(translation_error).all() or not np.isfinite(rotation_error).all():
                continue

            trans_errors.append(float(np.linalg.norm(translation_error)))
            rot_errors.append(MetricGeometryEvaluator._rotation_angle_degrees(rotation_error))

        return np.asarray(trans_errors, dtype=np.float64), np.asarray(rot_errors, dtype=np.float64)

    @staticmethod
    def _evaluate_pose_group(sequence_path, camera_params_dir, frame_ids, frame_cache=None, group_count=1):
        group_id = f"{frame_ids[0]}_{frame_ids[-1]}" if frame_ids else "empty"
        base_result = {
            "group_id": group_id,
            "frame_start": frame_ids[0] if frame_ids else None,
            "frame_end": frame_ids[-1] if frame_ids else None,
            "frames_requested": len(frame_ids),
            "frames_used": 0,
            "rpe_delta_frames": MetricGeometryEvaluator.DEFAULT_RPE_DELTA_FRAMES,
        }
        if len(frame_ids) < config.pose_eval_min_frames:
            return {**base_result, "status": "too_few_requested_frames"}, None

        records, skipped_frames = MetricGeometryEvaluator._load_pose_records(
            sequence_path,
            camera_params_dir,
            frame_ids,
            frame_cache=frame_cache,
        )
        if len(records) < config.pose_eval_min_frames:
            return {
                **base_result,
                "status": "too_few_valid_poses",
                "frames_used": len(records),
                "skipped_frame_count": len(skipped_frames),
            }, None

        ate_errors, alignment = MetricGeometryEvaluator._compute_tum_ate_se3(records)
        if ate_errors is None:
            return {
                **base_result,
                "status": "failed_ate_alignment",
                "frames_used": len(records),
                "skipped_frame_count": len(skipped_frames),
            }, None

        rpe_trans_errors, rpe_rot_errors = MetricGeometryEvaluator._compute_tum_rpe(records)
        pose_visualization_path = MetricGeometryEvaluator._save_pose_bev_visualization(
            records,
            camera_params_dir,
            group_id,
            group_count=group_count,
        )
        pose_accumulator = MetricGeometryEvaluator._empty_pose_accumulator()
        pose_accumulator["groups_requested"] = 1
        pose_accumulator["groups_evaluated"] = 1
        pose_accumulator["frames_requested"] = len(frame_ids)
        pose_accumulator["frames_evaluated"] = int(ate_errors.size)
        pose_accumulator["rpe_pairs"] = int(rpe_trans_errors.size)
        MetricGeometryEvaluator._accumulate_error_values(pose_accumulator["ate"], ate_errors)
        MetricGeometryEvaluator._accumulate_error_values(
            pose_accumulator["rpe_trans"],
            rpe_trans_errors,
        )
        MetricGeometryEvaluator._accumulate_error_values(
            pose_accumulator["rpe_rot"],
            rpe_rot_errors,
        )

        finalized_pose = MetricGeometryEvaluator._finalize_pose_metrics(pose_accumulator)
        return {
            **base_result,
            "status": "ok",
            "frames_used": len(records),
            "used_frames": [record["frame_id"] for record in records],
            "skipped_frame_count": len(skipped_frames),
            "ate": finalized_pose["ate"],
            "rpe_trans": finalized_pose["rpe_trans"],
            "rpe_rot": finalized_pose["rpe_rot"],
            "alignment": alignment,
            "pose_visualization_path": pose_visualization_path,
        }, pose_accumulator

    @staticmethod
    def _evaluate_pose_groups(sequence_path, camera_params_dir, frame_ids, frame_cache=None):
        group_frame_ids = MetricGeometryEvaluator._pose_group_frame_ids(frame_ids)
        pose_accumulator = MetricGeometryEvaluator._empty_pose_accumulator()
        group_results = []

        group_count = len(group_frame_ids)
        for group_ids in group_frame_ids:
            pose_accumulator["groups_requested"] += 1
            pose_accumulator["frames_requested"] += len(group_ids)
            group_result, group_accumulator = MetricGeometryEvaluator._evaluate_pose_group(
                sequence_path,
                camera_params_dir,
                group_ids,
                frame_cache=frame_cache,
                group_count=group_count,
            )
            group_results.append(group_result)
            if group_accumulator is None:
                continue
            MetricGeometryEvaluator._accumulate_pose_accumulator(
                pose_accumulator,
                {
                    **group_accumulator,
                    "groups_requested": 0,
                    "frames_requested": 0,
                },
            )

        return group_results, pose_accumulator

    @staticmethod
    def evaluate_sequence(sequence_path, frame_count, depth_model, sequence_name, scale_type="absolute", depth_threshold=None, frame_cache=None):
        sequence_metrics = MetricCalculator._evaluate_variants(
            sequence_path=sequence_path,
            frame_count=frame_count,
            depth_model=depth_model,
            sequence_name=sequence_name,
            distance_threshold=depth_threshold,
            frame_cache=frame_cache,
        )
        return sequence_metrics[scale_type]


class ScaleAccuracyEvaluator:
    """尺度准确性评估器"""

    @staticmethod
    def evaluate_scale_accuracy(predicted_scales, ground_truth_scales):
        if len(predicted_scales) != len(ground_truth_scales):
            raise ValueError("Predicted and ground truth scales must have the same length")

        scale_errors = np.abs(np.array(predicted_scales) - np.array(ground_truth_scales))
        thresholds = [0.05, 0.1, 0.15, 0.2]

        accuracies = {}
        for threshold in thresholds:
            accuracies[f"acc_{threshold}"] = np.mean(scale_errors < threshold)

        return {
            "mae": np.mean(scale_errors),
            "rmse": np.sqrt(np.mean(scale_errors ** 2)),
            "median_error": np.median(scale_errors),
            "max_error": np.max(scale_errors),
            "min_error": np.min(scale_errors),
            "accuracies": accuracies,
            "errors": scale_errors.tolist(),
        }


class MetricCalculator:
    """Metric geometry evaluator for multiple scale variants."""

    _RESULTS_RUN_ID = None

    VARIANT_ORDER_DEFINITIONS = (
        ("relative_scale", "relative", None),
        ("consistent_scale", "consistent", None),
        ("optimized_scale_legacy", "optimized_legacy", "optimized_scale_legacy"),
        ("optimized_scale_original", "optimized_original", "optimized_scale_original"),
        ("optimized_scale_weighted_bi", "optimized_weighted_bi", "optimized_scale_weighted_bi"),
        ("optimized_scale_weighted_ifpa", "optimized_weighted_ifpa", "optimized_scale_weighted_ifpa"),
        ("selected_scale", "selected", None),
    )
    DIFFICULTY_GROUP_ORDER = ("easy", "mid", "hard")
    PRIMARY_METRIC_NOTE = "* marks primary metrics."

    @staticmethod
    def _variant_order():
        enabled_optimized_variants = set(config.enabled_optimized_variants)
        enabled_metric_variants = set(config.enabled_metric_variants)
        return tuple(
            (label, scale_key)
            for label, scale_key, optimized_variant_key in MetricCalculator.VARIANT_ORDER_DEFINITIONS
            if scale_key in enabled_metric_variants
            and (optimized_variant_key is None or optimized_variant_key in enabled_optimized_variants)
        )

    @staticmethod
    def _primary_scale_key():
        return MetricCalculator._variant_order()[0][1]

    @staticmethod
    def _distance_bin_specs(distance_threshold):
        if not config.enable_metric_depth_bins:
            return []

        bin_edges = list(config.metric_depth_bins)
        if not bin_edges:
            return []

        specs = []
        lower = 0.0
        for upper in bin_edges:
            if distance_threshold is not None and lower >= distance_threshold:
                break
            specs.append(
                {
                    "key": f"{int(lower)}_{int(upper)}m",
                    "label": f"{int(lower)}-{int(upper)}m",
                    "min_depth": lower,
                    "max_depth": float(upper),
                }
            )
            lower = float(upper)

        if distance_threshold is None or lower < distance_threshold:
            specs.append(
                {
                    "key": f"{int(lower)}m_plus",
                    "label": f"{int(lower)}m+",
                    "min_depth": lower,
                    "max_depth": None if distance_threshold is None else float(distance_threshold),
                }
            )
        return specs

    @staticmethod
    def _results_run_id():
        if MetricCalculator._RESULTS_RUN_ID is None:
            MetricCalculator._RESULTS_RUN_ID = time.strftime("%Y%m%d_%H%M%S")
        return MetricCalculator._RESULTS_RUN_ID

    @staticmethod
    def _results_root(depth_model):
        metric_dir_name = f"{depth_model}_{MetricCalculator._results_run_id()}"
        return os.path.join(config.get_path("output_root"), config.dataset_name, "metric_results", metric_dir_name)

    @staticmethod
    def _sequence_results_dir(depth_model):
        return os.path.join(MetricCalculator._results_root(depth_model), "sequences")

    @staticmethod
    def _dataset_results_path(depth_model):
        return os.path.join(MetricCalculator._results_root(depth_model), "dataset_summary.json")

    @staticmethod
    def _dataset_report_path(depth_model):
        return os.path.join(MetricCalculator._results_root(depth_model), "dataset_summary.txt")

    @staticmethod
    def _sequence_result_path(depth_model, sequence_name):
        return os.path.join(MetricCalculator._sequence_results_dir(depth_model), f"{sequence_name}.json")

    @staticmethod
    def _sequence_report_path(depth_model, sequence_name):
        return os.path.join(MetricCalculator._sequence_results_dir(depth_model), f"{sequence_name}.txt")

    @staticmethod
    def _format_value(value, precision=4, suffix=""):
        if value is None:
            return "-"
        return f"{value:.{precision}f}{suffix}"

    @staticmethod
    def _scale_debug_dir(depth_model, sequence_name):
        relative_dir = MetricGeometryEvaluator._get_output_dir(
            depth_model,
            sequence_name,
            "relative",
        )
        return os.path.join(os.path.dirname(relative_dir), "semantic_scale_debug")

    @staticmethod
    def _load_scale_estimation_summaries(depth_model, sequence_name):
        if not config.save_scale_intermediate_results:
            return []

        debug_dir = MetricCalculator._scale_debug_dir(depth_model, sequence_name)
        if not os.path.isdir(debug_dir):
            return []

        summaries = []
        debug_files = [
            filename
            for filename in file_utils.natural_sorted(os.listdir(debug_dir))
            if filename.startswith("shared_scale_") and filename.endswith(".json")
        ]
        for filename in debug_files:
            payload = file_utils.read_json(os.path.join(debug_dir, filename))
            rejection_summary = payload.get("scale_interval_rejection_summary") or {}
            frame_ids = payload.get("frame_ids") or []
            summaries.append(
                {
                    "filename": filename,
                    "frame_start": frame_ids[0] if frame_ids else None,
                    "frame_end": frame_ids[-1] if frame_ids else None,
                    "use_depth_geometry_filter": payload.get("use_depth_geometry_filter"),
                    "total_candidates": payload.get("candidate_anchor_count", 0),
                    "valid_intervals": payload.get(
                        "valid_scale_interval_count",
                        payload.get("scale_interval_count", 0),
                    ),
                    "solver_intervals": payload.get("scale_interval_count", 0),
                    "rejected_candidates": rejection_summary.get("invalid_count", 0),
                    "rejection_reasons": rejection_summary.get("reason_counts", {}),
                    "consistent_scale": payload.get("consistent_scale"),
                    "consistent_count": payload.get("consistent_count"),
                    "optimized_scale": payload.get("optimized_scale"),
                    "consistent_scale_fallback_applied": payload.get(
                        "consistent_scale_fallback_applied",
                        False,
                    ),
                    "consistent_scale_fallback_source": payload.get(
                        "consistent_scale_fallback_source"
                    ),
                    "consistent_scale_history_mean": payload.get(
                        "consistent_scale_history_mean"
                    ),
                    "consistent_scale_history_count": payload.get(
                        "consistent_scale_history_count"
                    ),
                    "consistent_scale_before_fallback": payload.get(
                        "consistent_scale_before_fallback"
                    ),
                    "optimized_scale_before_fallback": payload.get(
                        "optimized_scale_before_fallback"
                    ),
                    "enable_selected_scale": payload.get("enable_selected_scale"),
                    "selected_scale": payload.get("selected_scale"),
                    "selected_scale_variant": payload.get("selected_scale_variant"),
                    "selected_scale_reason": payload.get("selected_scale_reason"),
                    "selected_scale_default": payload.get("selected_scale_default"),
                    "selected_scale_gate_passed": payload.get("selected_scale_gate_passed"),
                    "selected_scale_valid_intervals": payload.get("selected_scale_valid_intervals"),
                    "min_valid_intervals_for_optimized": payload.get("min_valid_intervals_for_optimized"),
                    "selected_scale_consistent_count": payload.get("selected_scale_consistent_count"),
                    "min_consistent_count_for_optimized": payload.get("min_consistent_count_for_optimized"),
                    "selected_scale_valid_interval_gate_passed": payload.get("selected_scale_valid_interval_gate_passed"),
                    "selected_scale_consistent_count_gate_passed": payload.get("selected_scale_consistent_count_gate_passed"),
                    "selected_scale_optimized_shift_gate_passed": payload.get("selected_scale_optimized_shift_gate_passed"),
                    "selected_scale_optimized_shift_ratio": payload.get("selected_scale_optimized_shift_ratio"),
                    "max_optimized_shift_ratio": payload.get("max_optimized_shift_ratio"),
                }
            )
        return summaries

    @staticmethod
    def _render_scale_estimation_summary_lines(scale_summaries):
        lines = []
        for summary in scale_summaries:
            lines.append("=== Shared Scale Estimation Results ===")
            if summary["frame_start"] is not None and summary["frame_end"] is not None:
                lines.append(f"Scale subsequence: {summary['frame_start']} -> {summary['frame_end']}")
            lines.append(f"Total scale intervals collected: {summary['total_candidates']}")
            if summary["use_depth_geometry_filter"]:
                lines.append(
                    f"Valid scale intervals after depth geometry: {summary['valid_intervals']}"
                )
                if summary["rejected_candidates"] > 0:
                    lines.append(
                        "Rejected scale interval candidates: "
                        f"{summary['rejected_candidates']} "
                        f"({summary['rejection_reasons']})"
                    )
            else:
                lines.append("Depth geometry filtering: disabled")
                lines.append(f"Scale intervals used by solver: {summary['solver_intervals']}")
            lines.append(
                "Consistent scale: "
                f"{MetricCalculator._format_value(summary['consistent_scale'])}"
            )
            lines.append(f"Consistent count: {summary['consistent_count']}")
            if summary.get("consistent_scale_fallback_applied"):
                lines.append(
                    "Consistent fallback: "
                    f"{summary.get('consistent_scale_fallback_source')}, "
                    f"history_mean={MetricCalculator._format_value(summary.get('consistent_scale_history_mean'))}, "
                    f"history_count={summary.get('consistent_scale_history_count')}, "
                    f"before={MetricCalculator._format_value(summary.get('consistent_scale_before_fallback'))}"
                )
            lines.append(
                "Optimized scale: "
                f"{MetricCalculator._format_value(summary['optimized_scale'])}"
            )
            if summary.get("enable_selected_scale"):
                lines.append(
                    "Selected scale: "
                    f"{MetricCalculator._format_value(summary.get('selected_scale'))} "
                    f"({summary.get('selected_scale_variant')}, {summary.get('selected_scale_reason')})"
                )
                lines.append(
                    "Selected scale gate: "
                    f"shift={MetricCalculator._format_value(summary.get('selected_scale_optimized_shift_ratio'))}, "
                    f"max_shift={MetricCalculator._format_value(summary.get('max_optimized_shift_ratio'))}, "
                    f"passed={summary.get('selected_scale_gate_passed')}, "
                    f"default={summary.get('selected_scale_default')}"
                )
        return lines

    @staticmethod
    def _print_scale_estimation_summaries(scale_summaries):
        for line in MetricCalculator._render_scale_estimation_summary_lines(scale_summaries):
            print(line)

    @staticmethod
    def _format_count(value):
        if value is None:
            return "-"
        return str(value)

    @staticmethod
    def _empty_distance_bin_metrics(scale_type, output_dir, frames_requested, spec):
        return {
            "label": spec["label"],
            "min_depth": spec["min_depth"],
            "max_depth": spec["max_depth"],
            "scale_type": scale_type,
            "output_dir": output_dir,
            "frames_evaluated": 0,
            "frames_skipped": frames_requested,
            "frames_requested": frames_requested,
            "depth": MetricGeometryEvaluator._finalize_metrics(
                MetricGeometryEvaluator._empty_metric_accumulator()
            ),
            "point": MetricGeometryEvaluator._finalize_metrics(
                MetricGeometryEvaluator._empty_metric_accumulator()
            ),
            "depth_frame_mean": {
                "frame_count": 0,
                "rel": None,
                "delta1": None,
                "mae": None,
                "rmse": None,
            },
            "point_frame_mean": {
                "frame_count": 0,
                "rel": None,
                "delta1": None,
                "mae": None,
                "rmse": None,
            },
            "per_frame": [],
        }

    @staticmethod
    def _distance_bin_sections(variant_metrics):
        if not variant_metrics:
            return []
        first_distance_bins = variant_metrics[0][1].get("distance_bins", {})
        sections = []
        for bin_key, first_bin_metrics in first_distance_bins.items():
            sections.append(
                (
                    first_bin_metrics["label"],
                    [(label, metrics["distance_bins"][bin_key]) for label, metrics in variant_metrics],
                )
            )
        return sections

    @staticmethod
    def _build_table(headers, rows):
        widths = [len(str(header)) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(str(cell)))

        def make_separator():
            return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

        def make_row(row):
            return "| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

        lines = [make_separator(), make_row(headers), make_separator()]
        lines.extend(make_row(row) for row in rows)
        lines.append(make_separator())
        return "\n".join(lines)

    @staticmethod
    def _make_sample_weighted_rows(variant_metrics):
        rows = []
        for label, metrics in variant_metrics:
            rows.append(
                (
                    label,
                    MetricCalculator._format_count(metrics["depth"]["sample_count"]),
                    MetricCalculator._format_value(metrics["depth"]["rel"]),
                    MetricCalculator._format_value(metrics["depth"]["delta1"]),
                    MetricCalculator._format_value(metrics["depth"]["mae"], suffix=" m"),
                    MetricCalculator._format_value(metrics["depth"]["rmse"], suffix=" m"),
                    MetricCalculator._format_count(metrics["point"]["sample_count"]),
                    MetricCalculator._format_value(metrics["point"]["rel"]),
                    MetricCalculator._format_value(metrics["point"]["delta1"]),
                    MetricCalculator._format_value(metrics["point"]["mae"], suffix=" m"),
                    MetricCalculator._format_value(metrics["point"]["rmse"], suffix=" m"),
                )
            )
        return rows

    @staticmethod
    def _make_frame_mean_rows(variant_metrics):
        rows = []
        for label, metrics in variant_metrics:
            rows.append(
                (
                    label,
                    MetricCalculator._format_count(metrics["depth_frame_mean"]["frame_count"]),
                    MetricCalculator._format_value(metrics["depth_frame_mean"]["rel"]),
                    MetricCalculator._format_value(metrics["depth_frame_mean"]["delta1"]),
                    MetricCalculator._format_value(metrics["depth_frame_mean"]["mae"], suffix=" m"),
                    MetricCalculator._format_value(metrics["depth_frame_mean"]["rmse"], suffix=" m"),
                    MetricCalculator._format_count(metrics["point_frame_mean"]["frame_count"]),
                    MetricCalculator._format_value(metrics["point_frame_mean"]["rel"]),
                    MetricCalculator._format_value(metrics["point_frame_mean"]["delta1"]),
                    MetricCalculator._format_value(metrics["point_frame_mean"]["mae"], suffix=" m"),
                    MetricCalculator._format_value(metrics["point_frame_mean"]["rmse"], suffix=" m"),
                )
            )
        return rows

    @staticmethod
    def _make_reconstruction_rows(variant_metrics):
        rows = []
        mode = config.sparse_reconstruction_mode
        metric_key = "reconstruction_sequence_se3_mean" if mode == "sequence" else "reconstruction_frame_mean"
        count_key = "group_count" if mode == "sequence" else "frame_count"
        for label, metrics in variant_metrics:
            reconstruction = metrics.get(metric_key, {})
            rows.append(
                (
                    label,
                    MetricCalculator._format_count(reconstruction.get(count_key)),
                    MetricCalculator._format_value(reconstruction.get("acc"), suffix=" m"),
                    MetricCalculator._format_value(reconstruction.get("comp"), suffix=" m"),
                    MetricCalculator._format_value(reconstruction.get("chamfer"), suffix=" m"),
                    MetricCalculator._format_value(reconstruction.get("pairwise_mae"), suffix=" m"),
                    MetricCalculator._format_value(reconstruction.get("pairwise_rmse"), suffix=" m"),
                )
            )
        return rows

    @staticmethod
    def _make_pose_rows(variant_metrics):
        rows = []
        for label, metrics in variant_metrics:
            pose = metrics.get("pose", {})
            ate = pose.get("ate", {})
            rpe_trans = pose.get("rpe_trans", {})
            rpe_rot = pose.get("rpe_rot", {})
            rows.append(
                (
                    label,
                    MetricCalculator._format_count(pose.get("groups_evaluated")),
                    MetricCalculator._format_count(pose.get("frames_evaluated")),
                    MetricCalculator._format_value(ate.get("rmse"), suffix=" m"),
                    MetricCalculator._format_value(ate.get("mean"), suffix=" m"),
                    MetricCalculator._format_count(pose.get("rpe_pairs")),
                    MetricCalculator._format_value(rpe_trans.get("rmse"), suffix=" m"),
                    MetricCalculator._format_value(rpe_rot.get("rmse"), suffix=" deg"),
                )
            )
        return rows

    @staticmethod
    def _sparse_reconstruction_title():
        if config.sparse_reconstruction_mode != "sequence":
            return "Sparse 3D Reconstruction Proxy (camera-frame LiDAR samples, frame-mean)"

        pose_label = "pred-pose" if config.sparse_reconstruction_pose_source == "pred" else "gt-pose"
        alignment_label = (
            "point-cloud global SE3"
            if config.sparse_reconstruction_alignment == "point_cloud"
            else "trajectory global SE3"
        )
        return f"Sparse 3D Reconstruction Proxy ({pose_label}, {alignment_label} aligned, subsequence-mean)"

    @staticmethod
    def _print_metric_tables(variant_metrics, include_distance_bins=True):
        frame_mean_headers = (
            "Scale",
            "Depth Frames",
            "Depth Rel↓*",
            "Depth d1↑*",
            "Depth MAE↓",
            "Depth RMSE↓",
            "Point Frames",
            "Point Rel↓",
            "Point d1↑",
            "Point MAE↓",
            "Point RMSE↓",
        )

        print("Metric geometry (frame-mean)")
        print(
            MetricCalculator._build_table(
                frame_mean_headers,
                MetricCalculator._make_frame_mean_rows(variant_metrics),
            )
        )

        if config.enable_sparse_reconstruction:
            reconstruction_mode = config.sparse_reconstruction_mode
            reconstruction_headers = (
                "Scale",
                "Groups" if reconstruction_mode == "sequence" else "Frames",
                "Acc↓*",
                "Comp↓*",
                "Chamfer↓",
                "Pairwise MAE↓",
                "Pairwise RMSE↓",
            )
            reconstruction_title = MetricCalculator._sparse_reconstruction_title()
            print(reconstruction_title)
            print(
                MetricCalculator._build_table(
                    reconstruction_headers,
                    MetricCalculator._make_reconstruction_rows(variant_metrics),
                )
            )

        if config.enable_pose_evaluation:
            pose_headers = (
                "Scale",
                "Groups",
                "Frames",
                "ATE RMSE↓*",
                "ATE Mean↓",
                "RPE Pairs",
                "RPE Trans RMSE↓*",
                "RPE Rot RMSE↓",
            )
            print(
                "TUM Pose "
                f"({config.pose_evaluation_mode}, global SE3 alignment, scale={config.pose_eval_scale:.4f})"
            )
            print(
                MetricCalculator._build_table(
                    pose_headers,
                    MetricCalculator._make_pose_rows(variant_metrics),
                )
            )

        print(MetricCalculator.PRIMARY_METRIC_NOTE)

        if include_distance_bins:
            for bin_label, bin_variant_metrics in MetricCalculator._distance_bin_sections(variant_metrics):
                print(f"Metric geometry ({bin_label}, frame-mean)")
                print(
                    MetricCalculator._build_table(
                        frame_mean_headers,
                        MetricCalculator._make_frame_mean_rows(bin_variant_metrics),
                    )
                )

    @staticmethod
    def _render_metric_tables_text(variant_metrics, include_distance_bins=True):
        frame_mean_headers = (
            "Scale",
            "Depth Frames",
            "Depth Rel↓*",
            "Depth d1↑*",
            "Depth MAE↓",
            "Depth RMSE↓",
            "Point Frames",
            "Point Rel↓",
            "Point d1↑",
            "Point MAE↓",
            "Point RMSE↓",
        )

        sections = [
            "Metric geometry (frame-mean)",
            MetricCalculator._build_table(
                frame_mean_headers,
                MetricCalculator._make_frame_mean_rows(variant_metrics),
            ),
        ]
        if config.enable_sparse_reconstruction:
            reconstruction_mode = config.sparse_reconstruction_mode
            reconstruction_headers = (
                "Scale",
                "Groups" if reconstruction_mode == "sequence" else "Frames",
                "Acc↓*",
                "Comp↓*",
                "Chamfer↓",
                "Pairwise MAE↓",
                "Pairwise RMSE↓",
            )
            reconstruction_title = MetricCalculator._sparse_reconstruction_title()
            sections.extend(
                [
                    reconstruction_title,
                    MetricCalculator._build_table(
                        reconstruction_headers,
                        MetricCalculator._make_reconstruction_rows(variant_metrics),
                    ),
                ]
            )

        if config.enable_pose_evaluation:
            pose_headers = (
                "Scale",
                "Groups",
                "Frames",
                "ATE RMSE↓*",
                "ATE Mean↓",
                "RPE Pairs",
                "RPE Trans RMSE↓*",
                "RPE Rot RMSE↓",
            )
            sections.extend(
                [
                    "TUM Pose "
                    f"({config.pose_evaluation_mode}, global SE3 alignment, scale={config.pose_eval_scale:.4f})",
                    MetricCalculator._build_table(
                        pose_headers,
                        MetricCalculator._make_pose_rows(variant_metrics),
                    ),
                ]
            )

        sections.append(MetricCalculator.PRIMARY_METRIC_NOTE)

        if include_distance_bins:
            for bin_label, bin_variant_metrics in MetricCalculator._distance_bin_sections(variant_metrics):
                sections.extend(
                    [
                        f"Metric geometry ({bin_label}, frame-mean)",
                        MetricCalculator._build_table(
                            frame_mean_headers,
                            MetricCalculator._make_frame_mean_rows(bin_variant_metrics),
                        ),
                    ]
                )

        return "\n".join(sections)

    @staticmethod
    def _build_sequence_difficulty_summary(sequence_results):
        if not config.sequence_difficulty_enabled:
            return None

        groups = {
            group_name: {
                "sequence_results": [],
                "sequence_names": [],
            }
            for group_name in MetricCalculator.DIFFICULTY_GROUP_ORDER
        }
        sequence_profiles = []
        for sequence_result in sequence_results:
            profile = config.classify_sequence_difficulty_result(
                sequence_result,
                include_overrides=False,
            )
            group_name = profile["profile_name"]
            groups[group_name]["sequence_results"].append(sequence_result)
            groups[group_name]["sequence_names"].append(profile["sequence_name"])
            sequence_profiles.append(profile)

        for group_name in MetricCalculator.DIFFICULTY_GROUP_ORDER:
            groups[group_name]["sequence_names"].sort()
        sequence_profiles.sort(key=lambda item: item["sequence_name"])

        summary = {
            "mode": config.sequence_difficulty_mode,
            "baseline_variant": config.sequence_difficulty_baseline_variant,
            "candidate_variants": list(config.sequence_difficulty_candidate_variants),
            "groups": groups,
            "sequence_profiles": sequence_profiles,
            "group_counts": {
                group_name: len(groups[group_name]["sequence_results"])
                for group_name in MetricCalculator.DIFFICULTY_GROUP_ORDER
            },
        }
        if config.sequence_difficulty_mode == "best_variant_vs_baseline":
            summary["easy_d1_gain_threshold"] = config.sequence_difficulty_easy_d1_gain_threshold
            summary["easy_rule"] = (
                "any valid candidate has delta1 gain vs baseline >= "
                f"{config.sequence_difficulty_easy_d1_gain_threshold:.4f}"
            )
            summary["hard_rule"] = (
                "after excluding easy, any valid candidate has lower delta1 than baseline"
            )
            summary["mid_rule"] = "all other sequences"
        return summary

    @staticmethod
    def _render_sequence_difficulty_summary(difficulty_summary):
        if not difficulty_summary:
            return ""

        lines = [
            "Sequence Difficulty Summary",
            f"Mode: {difficulty_summary['mode']}",
            f"Baseline variant: {difficulty_summary['baseline_variant']}",
            "Candidate variants: " + ", ".join(difficulty_summary["candidate_variants"]),
        ]
        if difficulty_summary["mode"] == "best_variant_vs_baseline":
            lines.extend(
                [
                    f"Easy d1 gain threshold: {difficulty_summary['easy_d1_gain_threshold']:.4f}",
                    f"Easy rule: {difficulty_summary['easy_rule']}",
                    f"Hard rule: {difficulty_summary['hard_rule']}",
                    f"Mid rule: {difficulty_summary['mid_rule']}",
                ]
            )
        lines.append(
            "Group counts: "
            + ", ".join(
                f"{group_name}={difficulty_summary['group_counts'][group_name]}"
                for group_name in MetricCalculator.DIFFICULTY_GROUP_ORDER
            )
        )

        for group_name in MetricCalculator.DIFFICULTY_GROUP_ORDER:
            group_sequence_results = difficulty_summary["groups"][group_name]["sequence_results"]
            lines.append("-" * 80)
            lines.append(f"Metric geometry ({group_name})")
            lines.append(f"Sequences: {len(group_sequence_results)}")
            if not group_sequence_results:
                lines.append("No sequences in this group.")
                continue
            group_variant_metrics = [
                (
                    label,
                    MetricCalculator._aggregate_sequence_metrics(group_sequence_results, scale_key),
                )
                for label, scale_key in MetricCalculator._variant_order()
            ]
            lines.append(
                MetricCalculator._build_table(
                    (
                        "Scale",
                        "Depth Frames",
                        "Depth Rel↓",
                        "Depth d1↑",
                        "Depth MAE↓",
                        "Depth RMSE↓",
                        "Point Frames",
                        "Point Rel↓",
                        "Point d1↑",
                        "Point MAE↓",
                        "Point RMSE↓",
                    ),
                    MetricCalculator._make_frame_mean_rows(group_variant_metrics),
                )
            )

        return "\n".join(lines)

    @staticmethod
    def _sequence_group_map_path(depth_model):
        return os.path.join(MetricCalculator._results_root(depth_model), "sequence_group_map.txt")

    @staticmethod
    def _render_sequence_group_map(difficulty_summary, depth_model):
        if not difficulty_summary:
            return ""

        lines = [
            "=" * 80,
            "Sequence Group Map",
            "=" * 80,
            f"summary_path: {MetricCalculator._dataset_results_path(depth_model)}",
            f"mode: {difficulty_summary['mode']}",
            f"baseline_variant: {difficulty_summary['baseline_variant']}",
            "candidate_variants: " + ", ".join(difficulty_summary["candidate_variants"]),
        ]
        if difficulty_summary["mode"] == "best_variant_vs_baseline":
            lines.extend(
                [
                    f"easy_d1_gain_threshold: {difficulty_summary['easy_d1_gain_threshold']:.4f}",
                    f"easy_rule: {difficulty_summary['easy_rule']}",
                    f"hard_rule: {difficulty_summary['hard_rule']}",
                    f"mid_rule: {difficulty_summary['mid_rule']}",
                ]
            )
        lines.append(
            "group_counts: "
            + ", ".join(
                f"{group_name}={difficulty_summary['group_counts'][group_name]}"
                for group_name in MetricCalculator.DIFFICULTY_GROUP_ORDER
            )
        )
        lines.append("-" * 80)

        for profile in difficulty_summary["sequence_profiles"]:
            baseline_rel = profile.get("baseline_rel")
            baseline_delta1 = profile.get("baseline_delta1")
            lines.append(
                f"{profile['sequence_name']}: group={profile['profile_name']}, "
                f"baseline_rel={MetricCalculator._format_value(baseline_rel)}, "
                f"baseline_d1={MetricCalculator._format_value(baseline_delta1)}"
            )
            lines.append(f"  reason: {profile.get('classification_reason')}")
            if profile.get("per_variant_metrics"):
                lines.append(
                    "  candidates: "
                    + ", ".join(
                        f"{name}(rel={MetricCalculator._format_value(value['rel'])}, "
                        f"d1={MetricCalculator._format_value(value['delta1'])}, "
                        f"rel_delta={value['rel_delta_vs_baseline']:+.4f}, "
                        f"d1_gain={value['d1_gain_vs_baseline']:+.4f})"
                        for name, value in profile["per_variant_metrics"].items()
                    )
                )
            if profile.get("missing_variant_metrics"):
                lines.append(
                    "  missing: "
                    + ", ".join(
                        f"{name}={reason}"
                        for name, reason in profile["missing_variant_metrics"].items()
                    )
                )

        return "\n".join(lines) + "\n"

    @staticmethod
    def _evaluate_variants(sequence_path, frame_count, depth_model, sequence_name, distance_threshold, frame_cache=None):
        selected_frame_ids = MetricGeometryEvaluator._get_selected_frame_ids(sequence_path, frame_count)
        distance_bin_specs = MetricCalculator._distance_bin_specs(distance_threshold)
        output_dirs = {
            scale_key: MetricGeometryEvaluator._get_output_dir(depth_model, sequence_name, scale_key)
            for _, scale_key in MetricCalculator._variant_order()
        }
        pointmap_dirs_by_key = {
            scale_key: os.path.join(output_dir, "pointmap")
            for scale_key, output_dir in output_dirs.items()
        }

        metrics_by_key = {
            scale_key: MetricGeometryEvaluator._empty_sequence_metrics(
                scale_key,
                output_dirs[scale_key],
                len(selected_frame_ids),
                distance_bin_specs=distance_bin_specs,
            )
            for _, scale_key in MetricCalculator._variant_order()
        }

        missing_scale_keys = {
            scale_key
            for scale_key, pointmap_dir in pointmap_dirs_by_key.items()
            if not os.path.isdir(pointmap_dir)
        }

        available_pointmap_dirs = {
            scale_key: pointmap_dir
            for scale_key, pointmap_dir in pointmap_dirs_by_key.items()
            if scale_key not in missing_scale_keys
        }
        if not available_pointmap_dirs:
            return metrics_by_key

        workers = config.metric_workers
        if workers > 1 and len(selected_frame_ids) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                frame_outputs = list(
                    executor.map(
                        lambda frame_id: MetricGeometryEvaluator._evaluate_frame_variants(
                            sequence_path,
                            available_pointmap_dirs,
                            frame_id,
                            distance_threshold,
                            distance_bin_specs=distance_bin_specs,
                            frame_cache=frame_cache,
                        ),
                        selected_frame_ids,
                    )
                )
        else:
            frame_outputs = [
                MetricGeometryEvaluator._evaluate_frame_variants(
                    sequence_path,
                    available_pointmap_dirs,
                    frame_id,
                    distance_threshold,
                    distance_bin_specs=distance_bin_specs,
                    frame_cache=frame_cache,
                )
                for frame_id in selected_frame_ids
            ]

        accumulators_by_key = {
            scale_key: {
                "depth": MetricGeometryEvaluator._empty_metric_accumulator(),
                "point": MetricGeometryEvaluator._empty_metric_accumulator(),
                "reconstruction": MetricGeometryEvaluator._empty_reconstruction_accumulator(),
                "frames_evaluated": 0,
                "reconstruction_frames_evaluated": 0,
                "reconstruction_groups_evaluated": 0,
                "reconstruction_groups": [],
                "pose": MetricGeometryEvaluator._empty_pose_accumulator(),
                "pose_groups": [],
                "per_frame": [],
                "distance_bins": {
                    spec["key"]: {
                        "label": spec["label"],
                        "min_depth": spec["min_depth"],
                        "max_depth": spec["max_depth"],
                        "depth": MetricGeometryEvaluator._empty_metric_accumulator(),
                        "point": MetricGeometryEvaluator._empty_metric_accumulator(),
                        "frames_evaluated": 0,
                        "per_frame": [],
                    }
                    for spec in distance_bin_specs
                },
            }
            for scale_key in available_pointmap_dirs
        }

        for frame_variant_results in frame_outputs:
            for scale_key, (
                frame_result,
                frame_depth_metrics,
                frame_point_metrics,
                frame_reconstruction_metrics,
                distance_bin_results,
            ) in frame_variant_results.items():
                accumulators_by_key[scale_key]["per_frame"].append(frame_result)
                if frame_result.get("status") != "ok":
                    continue

                MetricGeometryEvaluator._accumulate(
                    accumulators_by_key[scale_key]["depth"],
                    frame_depth_metrics,
                )
                MetricGeometryEvaluator._accumulate(
                    accumulators_by_key[scale_key]["point"],
                    frame_point_metrics,
                )
                accumulators_by_key[scale_key]["frames_evaluated"] += 1
                if frame_reconstruction_metrics is not None:
                    MetricGeometryEvaluator._accumulate(
                        accumulators_by_key[scale_key]["reconstruction"],
                        frame_reconstruction_metrics,
                    )
                    accumulators_by_key[scale_key]["reconstruction_frames_evaluated"] += 1

                for bin_key, bin_payload in distance_bin_results.items():
                    bin_accumulator = accumulators_by_key[scale_key]["distance_bins"][bin_key]
                    bin_frame_result = bin_payload["frame_result"]
                    bin_accumulator["per_frame"].append(bin_frame_result)
                    if bin_frame_result.get("status") != "ok":
                        continue
                    MetricGeometryEvaluator._accumulate(
                        bin_accumulator["depth"],
                        bin_payload["depth_metrics"],
                    )
                    MetricGeometryEvaluator._accumulate(
                        bin_accumulator["point"],
                        bin_payload["point_metrics"],
                    )
                    bin_accumulator["frames_evaluated"] += 1

        if config.enable_sparse_reconstruction and config.sparse_reconstruction_mode == "sequence":
            for scale_key, pointmap_dir in available_pointmap_dirs.items():
                group_results, group_metrics = (
                    MetricGeometryEvaluator._evaluate_sequence_reconstruction_groups(
                        sequence_path,
                        pointmap_dir,
                        selected_frame_ids,
                        depth_threshold=distance_threshold,
                        frame_cache=frame_cache,
                    )
                )
                accumulators_by_key[scale_key]["reconstruction_groups"] = group_results
                for reconstruction_metrics in group_metrics:
                    MetricGeometryEvaluator._accumulate(
                        accumulators_by_key[scale_key]["reconstruction"],
                        reconstruction_metrics,
                    )
                    accumulators_by_key[scale_key]["reconstruction_groups_evaluated"] += 1

        if config.enable_pose_evaluation:
            for scale_key, output_dir in output_dirs.items():
                if scale_key not in accumulators_by_key:
                    continue
                camera_params_dir = os.path.join(output_dir, "camera_params")
                group_results, pose_accumulator = MetricGeometryEvaluator._evaluate_pose_groups(
                    sequence_path,
                    camera_params_dir,
                    selected_frame_ids,
                    frame_cache=frame_cache,
                )
                accumulators_by_key[scale_key]["pose_groups"] = group_results
                MetricGeometryEvaluator._accumulate_pose_accumulator(
                    accumulators_by_key[scale_key]["pose"],
                    pose_accumulator,
                )

        for scale_key in available_pointmap_dirs:
            accumulator = accumulators_by_key[scale_key]
            metrics_by_key[scale_key] = {
                "scale_type": scale_key,
                "output_dir": output_dirs[scale_key],
                "frames_evaluated": accumulator["frames_evaluated"],
                "frames_skipped": len(selected_frame_ids) - accumulator["frames_evaluated"],
                "frames_requested": len(selected_frame_ids),
                "depth": MetricGeometryEvaluator._finalize_metrics(accumulator["depth"]),
                "point": MetricGeometryEvaluator._finalize_metrics(accumulator["point"]),
                "reconstruction": MetricGeometryEvaluator._finalize_reconstruction_metrics(
                    accumulator["reconstruction"]
                ),
                "reconstruction_se3": MetricGeometryEvaluator._finalize_reconstruction_metrics(
                    accumulator["reconstruction"]
                ),
                "reconstruction_mode": config.sparse_reconstruction_mode,
                "depth_frame_mean": MetricGeometryEvaluator._average_frame_metrics(accumulator["per_frame"], "depth"),
                "point_frame_mean": MetricGeometryEvaluator._average_frame_metrics(accumulator["per_frame"], "point"),
                "reconstruction_frame_mean": MetricGeometryEvaluator._average_frame_reconstruction_metrics(
                    accumulator["per_frame"]
                ),
                "reconstruction_sequence_se3_mean": MetricGeometryEvaluator._average_sequence_reconstruction_metrics(
                    accumulator["reconstruction_groups"]
                ),
                "reconstruction_groups_evaluated": accumulator["reconstruction_groups_evaluated"],
                "reconstruction_groups": accumulator["reconstruction_groups"],
                "pose": MetricGeometryEvaluator._finalize_pose_metrics(accumulator["pose"]),
                "pose_groups": accumulator["pose_groups"],
                "per_frame": accumulator["per_frame"],
                "distance_bins": {
                    bin_key: {
                        "label": bin_accumulator["label"],
                        "min_depth": bin_accumulator["min_depth"],
                        "max_depth": bin_accumulator["max_depth"],
                        "scale_type": scale_key,
                        "output_dir": output_dirs[scale_key],
                        "frames_evaluated": bin_accumulator["frames_evaluated"],
                        "frames_skipped": len(selected_frame_ids) - bin_accumulator["frames_evaluated"],
                        "frames_requested": len(selected_frame_ids),
                        "depth": MetricGeometryEvaluator._finalize_metrics(bin_accumulator["depth"]),
                        "point": MetricGeometryEvaluator._finalize_metrics(bin_accumulator["point"]),
                        "depth_frame_mean": MetricGeometryEvaluator._average_frame_metrics(
                            bin_accumulator["per_frame"], "depth"
                        ),
                        "point_frame_mean": MetricGeometryEvaluator._average_frame_metrics(
                            bin_accumulator["per_frame"], "point"
                        ),
                        "per_frame": bin_accumulator["per_frame"],
                    }
                    for bin_key, bin_accumulator in accumulator["distance_bins"].items()
                },
            }
            if config.sparse_reconstruction_mode == "sequence":
                metrics_by_key[scale_key].pop("reconstruction", None)
                metrics_by_key[scale_key].pop("reconstruction_frame_mean", None)
            else:
                metrics_by_key[scale_key].pop("reconstruction_se3", None)
                metrics_by_key[scale_key].pop("reconstruction_sequence_se3_mean", None)

        return metrics_by_key

    @staticmethod
    def _variant_metrics_to_rows(metrics_by_key):
        return [(label, metrics_by_key[key]) for label, key in MetricCalculator._variant_order()]

    @staticmethod
    def _to_raw_accumulator(metric_summary):
        sample_count = metric_summary.get("sample_count", 0) or 0
        if sample_count == 0:
            return MetricGeometryEvaluator._empty_metric_accumulator()

        rel = metric_summary.get("rel")
        delta1 = metric_summary.get("delta1")
        mae = metric_summary.get("mae")
        rmse = metric_summary.get("rmse")

        return {
            "sample_count": int(sample_count),
            "rel_sum": float((rel or 0.0) * sample_count),
            "delta1_hits": int(round((delta1 or 0.0) * sample_count)),
            "mae_sum": float((mae or 0.0) * sample_count),
            "rmse_sum": float(((rmse or 0.0) ** 2) * sample_count),
        }

    @staticmethod
    def _to_raw_reconstruction_accumulator(metric_summary):
        pair_count = metric_summary.get("pair_count", metric_summary.get("sample_count", 0)) or 0
        if pair_count == 0:
            return MetricGeometryEvaluator._empty_reconstruction_accumulator()

        pred_point_count = metric_summary.get("pred_point_count", pair_count) or 0
        gt_point_count = metric_summary.get("gt_point_count", pair_count) or 0
        acc = metric_summary.get("acc")
        comp = metric_summary.get("comp")
        pairwise_mae = metric_summary.get("pairwise_mae")
        pairwise_rmse = metric_summary.get("pairwise_rmse")

        return {
            "pair_count": int(pair_count),
            "pred_point_count": int(pred_point_count),
            "gt_point_count": int(gt_point_count),
            "acc_sum": float((acc or 0.0) * pred_point_count),
            "comp_sum": float((comp or 0.0) * gt_point_count),
            "pairwise_mae_sum": float((pairwise_mae or 0.0) * pair_count),
            "pairwise_rmse_sum": float(((pairwise_rmse or 0.0) ** 2) * pair_count),
        }

    @staticmethod
    def _to_raw_error_accumulator(metric_summary):
        sample_count = metric_summary.get("sample_count", 0) or 0
        if sample_count == 0:
            return MetricGeometryEvaluator._empty_error_accumulator()

        mean = metric_summary.get("mean")
        rmse = metric_summary.get("rmse")
        return {
            "sample_count": int(sample_count),
            "sum": float((mean or 0.0) * sample_count),
            "sq_sum": float(((rmse or 0.0) ** 2) * sample_count),
            "min": metric_summary.get("min"),
            "max": metric_summary.get("max"),
        }

    @staticmethod
    def _to_raw_pose_accumulator(metric_summary):
        accumulator = MetricGeometryEvaluator._empty_pose_accumulator()
        accumulator["mode"] = metric_summary.get("mode", config.pose_evaluation_mode)
        accumulator["alignment"] = metric_summary.get("alignment", "global_se3")
        accumulator["scale"] = metric_summary.get("scale", config.pose_eval_scale)
        accumulator["groups_requested"] = int(metric_summary.get("groups_requested", 0) or 0)
        accumulator["groups_evaluated"] = int(metric_summary.get("groups_evaluated", 0) or 0)
        accumulator["frames_requested"] = int(metric_summary.get("frames_requested", 0) or 0)
        accumulator["frames_evaluated"] = int(metric_summary.get("frames_evaluated", 0) or 0)
        accumulator["rpe_pairs"] = int(metric_summary.get("rpe_pairs", 0) or 0)
        accumulator["ate"] = MetricCalculator._to_raw_error_accumulator(
            metric_summary.get("ate", {})
        )
        accumulator["rpe_trans"] = MetricCalculator._to_raw_error_accumulator(
            metric_summary.get("rpe_trans", {})
        )
        accumulator["rpe_rot"] = MetricCalculator._to_raw_error_accumulator(
            metric_summary.get("rpe_rot", {})
        )
        return accumulator

    @staticmethod
    def _aggregate_sequence_metrics(sequence_results, scale_key):
        overall_depth = MetricGeometryEvaluator._empty_metric_accumulator()
        overall_point = MetricGeometryEvaluator._empty_metric_accumulator()
        overall_reconstruction = MetricGeometryEvaluator._empty_reconstruction_accumulator()
        overall_pose = MetricGeometryEvaluator._empty_pose_accumulator()
        combined_per_frame = []
        combined_reconstruction_groups = []
        combined_pose_groups = []
        frames_requested = 0
        frames_evaluated = 0
        frames_skipped = 0
        distance_bin_accumulators = {}

        for sequence_result in sequence_results:
            metrics = sequence_result[scale_key]
            MetricGeometryEvaluator._accumulate(
                overall_depth,
                MetricCalculator._to_raw_accumulator(metrics["depth"]),
            )
            MetricGeometryEvaluator._accumulate(
                overall_point,
                MetricCalculator._to_raw_accumulator(metrics["point"]),
            )
            MetricGeometryEvaluator._accumulate(
                overall_reconstruction,
                MetricCalculator._to_raw_reconstruction_accumulator(
                    metrics.get("reconstruction_se3", metrics.get("reconstruction", {}))
                ),
            )
            combined_per_frame.extend(metrics.get("per_frame", []))
            combined_reconstruction_groups.extend(metrics.get("reconstruction_groups", []))
            combined_pose_groups.extend(metrics.get("pose_groups", []))
            MetricGeometryEvaluator._accumulate_pose_accumulator(
                overall_pose,
                MetricCalculator._to_raw_pose_accumulator(metrics.get("pose", {})),
            )
            frames_requested += metrics.get("frames_requested", 0)
            frames_evaluated += metrics.get("frames_evaluated", 0)
            frames_skipped += metrics.get("frames_skipped", 0)
            for bin_key, bin_metrics in metrics.get("distance_bins", {}).items():
                if bin_key not in distance_bin_accumulators:
                    distance_bin_accumulators[bin_key] = {
                        "label": bin_metrics["label"],
                        "min_depth": bin_metrics["min_depth"],
                        "max_depth": bin_metrics["max_depth"],
                        "depth": MetricGeometryEvaluator._empty_metric_accumulator(),
                        "point": MetricGeometryEvaluator._empty_metric_accumulator(),
                        "frames_requested": 0,
                        "frames_evaluated": 0,
                        "frames_skipped": 0,
                        "per_frame": [],
                    }
                accumulator = distance_bin_accumulators[bin_key]
                MetricGeometryEvaluator._accumulate(
                    accumulator["depth"],
                    MetricCalculator._to_raw_accumulator(bin_metrics["depth"]),
                )
                MetricGeometryEvaluator._accumulate(
                    accumulator["point"],
                    MetricCalculator._to_raw_accumulator(bin_metrics["point"]),
                )
                accumulator["frames_requested"] += bin_metrics.get("frames_requested", 0)
                accumulator["frames_evaluated"] += bin_metrics.get("frames_evaluated", 0)
                accumulator["frames_skipped"] += bin_metrics.get("frames_skipped", 0)
                accumulator["per_frame"].extend(bin_metrics.get("per_frame", []))

        output_dirs = [sequence_result[scale_key]["output_dir"] for sequence_result in sequence_results]
        aggregated_metrics = {
            "scale_type": scale_key,
            "output_dir": output_dirs,
            "frames_requested": frames_requested,
            "frames_evaluated": frames_evaluated,
            "frames_skipped": frames_skipped,
            "depth": MetricGeometryEvaluator._finalize_metrics(overall_depth),
            "point": MetricGeometryEvaluator._finalize_metrics(overall_point),
            "reconstruction": MetricGeometryEvaluator._finalize_reconstruction_metrics(
                overall_reconstruction
            ),
            "reconstruction_se3": MetricGeometryEvaluator._finalize_reconstruction_metrics(
                overall_reconstruction
            ),
            "reconstruction_mode": config.sparse_reconstruction_mode,
            "depth_frame_mean": MetricGeometryEvaluator._average_frame_metrics(combined_per_frame, "depth"),
            "point_frame_mean": MetricGeometryEvaluator._average_frame_metrics(combined_per_frame, "point"),
            "reconstruction_frame_mean": MetricGeometryEvaluator._average_frame_reconstruction_metrics(
                combined_per_frame
            ),
            "reconstruction_sequence_se3_mean": MetricGeometryEvaluator._average_sequence_reconstruction_metrics(
                combined_reconstruction_groups
            ),
            "reconstruction_groups_evaluated": sum(
                1 for group in combined_reconstruction_groups if group.get("status") == "ok"
            ),
            "reconstruction_groups": combined_reconstruction_groups,
            "pose": MetricGeometryEvaluator._finalize_pose_metrics(overall_pose),
            "pose_groups": combined_pose_groups,
            "per_frame": combined_per_frame,
            "distance_bins": {
                bin_key: {
                    "label": accumulator["label"],
                    "min_depth": accumulator["min_depth"],
                    "max_depth": accumulator["max_depth"],
                    "scale_type": scale_key,
                    "output_dir": output_dirs,
                    "frames_requested": accumulator["frames_requested"],
                    "frames_evaluated": accumulator["frames_evaluated"],
                    "frames_skipped": accumulator["frames_skipped"],
                    "depth": MetricGeometryEvaluator._finalize_metrics(accumulator["depth"]),
                    "point": MetricGeometryEvaluator._finalize_metrics(accumulator["point"]),
                    "depth_frame_mean": MetricGeometryEvaluator._average_frame_metrics(
                        accumulator["per_frame"], "depth"
                    ),
                    "point_frame_mean": MetricGeometryEvaluator._average_frame_metrics(
                        accumulator["per_frame"], "point"
                    ),
                    "per_frame": accumulator["per_frame"],
                }
                for bin_key, accumulator in distance_bin_accumulators.items()
            },
        }
        if config.sparse_reconstruction_mode == "sequence":
            aggregated_metrics.pop("reconstruction", None)
            aggregated_metrics.pop("reconstruction_frame_mean", None)
        else:
            aggregated_metrics.pop("reconstruction_se3", None)
            aggregated_metrics.pop("reconstruction_sequence_se3_mean", None)
        return aggregated_metrics

    @staticmethod
    def _build_dataset_summary(sequence_results, elapsed):
        dataset_metrics = {}
        for _, scale_key in MetricCalculator._variant_order():
            dataset_metrics[scale_key] = MetricCalculator._aggregate_sequence_metrics(sequence_results, scale_key)

        return {
            "dataset": dataset_metrics,
            "sequence_results": sequence_results,
            "elapsed_time": elapsed,
        }

    @staticmethod
    def _drop_per_frame_details(value):
        if isinstance(value, dict):
            return {
                key: MetricCalculator._drop_per_frame_details(item)
                for key, item in value.items()
                if key not in {"per_frame", "reconstruction_groups", "pose_groups"}
            }
        if isinstance(value, list):
            return [MetricCalculator._drop_per_frame_details(item) for item in value]
        return value

    @staticmethod
    def _print_sequence_report(sequence_result, depth_model):
        sequence_name = sequence_result["sequence_name"]
        variant_metrics = MetricCalculator._variant_metrics_to_rows(sequence_result)
        primary_scale_key = MetricCalculator._primary_scale_key()
        scale_summaries = sequence_result.get("scale_estimation_summaries", [])

        print(f"Sequence: {sequence_name}")
        print(f"Model: {depth_model}")
        print(f"Requested frames: {sequence_result[primary_scale_key]['frames_requested']}")
        MetricCalculator._print_scale_estimation_summaries(scale_summaries)
        for label, metrics in variant_metrics:
            print(f"{label} dir: {metrics['output_dir']}")
        print(
            "Frames evaluated: "
            + ", ".join(f"{label}={metrics['frames_evaluated']}" for label, metrics in variant_metrics)
        )
        MetricCalculator._print_metric_tables(variant_metrics)

    @staticmethod
    def _print_dataset_report(sequence_results, dataset_metrics, depth_model, elapsed):
        print("=" * 80)
        print("Calculating Metric Geometry...")
        print("=" * 80)

        for sequence_result in sequence_results:
            MetricCalculator._print_sequence_report(sequence_result, depth_model)
            print("-" * 80)

        variant_metrics = MetricCalculator._variant_metrics_to_rows(dataset_metrics)
        primary_scale_key = MetricCalculator._primary_scale_key()
        print("Dataset Summary")
        print(f"Model: {depth_model}")
        print(f"Sequences: {len(sequence_results)}")
        print(f"Requested frames: {dataset_metrics[primary_scale_key]['frames_requested']}")
        print(
            "Requested frames by scale: "
            + ", ".join(f"{label}={metrics['frames_requested']}" for label, metrics in variant_metrics)
        )
        print(
            "Frames evaluated: "
            + ", ".join(f"{label}={metrics['frames_evaluated']}" for label, metrics in variant_metrics)
        )
        MetricCalculator._print_metric_tables(variant_metrics)
        difficulty_summary = MetricCalculator._build_sequence_difficulty_summary(sequence_results)
        if difficulty_summary:
            print("-" * 80)
            print(MetricCalculator._render_sequence_difficulty_summary(difficulty_summary))
        print(f"Metric geometry calculation time: {elapsed:.4f}s")
        print("=" * 80)

    @staticmethod
    def _render_sequence_report(sequence_result, depth_model):
        sequence_name = sequence_result["sequence_name"]
        variant_metrics = MetricCalculator._variant_metrics_to_rows(sequence_result)
        primary_scale_key = MetricCalculator._primary_scale_key()
        scale_summaries = sequence_result.get("scale_estimation_summaries", [])

        lines = [
            "=" * 80,
            "Calculating Metric Geometry...",
            "=" * 80,
            f"Sequence: {sequence_name}",
            f"Model: {depth_model}",
            f"Requested frames: {sequence_result[primary_scale_key]['frames_requested']}",
        ]
        lines.extend(MetricCalculator._render_scale_estimation_summary_lines(scale_summaries))
        for label, metrics in variant_metrics:
            lines.append(f"{label} dir: {metrics['output_dir']}")
        lines.append(
            "Frames evaluated: "
            + ", ".join(f"{label}={metrics['frames_evaluated']}" for label, metrics in variant_metrics)
        )
        lines.append(MetricCalculator._render_metric_tables_text(variant_metrics))
        lines.append(f"Metric geometry calculation time: {sequence_result.get('elapsed_time', 0.0):.4f}s")
        lines.append("=" * 80)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_dataset_report(sequence_results, dataset_metrics, depth_model, elapsed, difficulty_summary=None):
        variant_metrics = MetricCalculator._variant_metrics_to_rows(dataset_metrics)
        primary_scale_key = MetricCalculator._primary_scale_key()
        lines = [
            "=" * 80,
            "Calculating Metric Geometry...",
            "=" * 80,
        ]

        for sequence_result in sequence_results:
            lines.append(MetricCalculator._render_sequence_report(sequence_result, depth_model).rstrip())
            lines.append("-" * 80)

        lines.extend(
            [
                "Dataset Summary",
                f"Model: {depth_model}",
                f"Sequences: {len(sequence_results)}",
                f"Requested frames: {dataset_metrics[primary_scale_key]['frames_requested']}",
                "Requested frames by scale: "
                + ", ".join(f"{label}={metrics['frames_requested']}" for label, metrics in variant_metrics),
                "Frames evaluated: "
                + ", ".join(f"{label}={metrics['frames_evaluated']}" for label, metrics in variant_metrics),
                MetricCalculator._render_metric_tables_text(variant_metrics),
            ]
        )
        if difficulty_summary is None:
            difficulty_summary = MetricCalculator._build_sequence_difficulty_summary(sequence_results)
        if difficulty_summary:
            lines.extend(
                [
                    "-" * 80,
                    MetricCalculator._render_sequence_difficulty_summary(difficulty_summary),
                ]
            )
        lines.extend(
            [
                f"Metric geometry calculation time: {elapsed:.4f}s",
                "=" * 80,
            ]
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_dataset_summary_report(sequence_results, dataset_metrics, depth_model, elapsed, difficulty_summary=None):
        variant_metrics = MetricCalculator._variant_metrics_to_rows(dataset_metrics)
        primary_scale_key = MetricCalculator._primary_scale_key()
        lines = [
            "Dataset Summary",
            f"Model: {depth_model}",
            f"Sequences: {len(sequence_results)}",
            f"Requested frames: {dataset_metrics[primary_scale_key]['frames_requested']}",
            "Requested frames by scale: "
            + ", ".join(f"{label}={metrics['frames_requested']}" for label, metrics in variant_metrics),
            "Frames evaluated: "
            + ", ".join(f"{label}={metrics['frames_evaluated']}" for label, metrics in variant_metrics),
            MetricCalculator._render_metric_tables_text(variant_metrics),
        ]
        if difficulty_summary is None:
            difficulty_summary = MetricCalculator._build_sequence_difficulty_summary(sequence_results)
        if difficulty_summary:
            lines.extend(
                [
                    "-" * 80,
                    MetricCalculator._render_sequence_difficulty_summary(difficulty_summary),
                ]
            )
        lines.extend(
            [
                f"Metric geometry calculation time: {elapsed:.4f}s",
                "=" * 80,
            ]
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def calculate_all_metrics(sequence_path, frame_count, depth_model, sequence_name, distance_threshold=100, frame_cache=None):
        print("=" * 80)
        print("Calculating Metric Geometry...")
        print("=" * 80)

        start_time = time.time()
        sequence_metrics = MetricCalculator._evaluate_variants(
            sequence_path=sequence_path,
            frame_count=frame_count,
            depth_model=depth_model,
            sequence_name=sequence_name,
            distance_threshold=distance_threshold,
            frame_cache=frame_cache,
        )
        elapsed = time.time() - start_time

        variant_metrics = MetricCalculator._variant_metrics_to_rows(sequence_metrics)
        primary_scale_key = MetricCalculator._primary_scale_key()
        scale_summaries = MetricCalculator._load_scale_estimation_summaries(
            depth_model,
            sequence_name,
        )

        print(f"Sequence: {sequence_name}")
        print(f"Model: {depth_model}")
        print(f"Requested frames: {sequence_metrics[primary_scale_key]['frames_requested']}")
        MetricCalculator._print_scale_estimation_summaries(scale_summaries)
        for label, metrics in variant_metrics:
            print(f"{label} dir: {metrics['output_dir']}")
        print(
            "Frames evaluated: "
            + ", ".join(f"{label}={metrics['frames_evaluated']}" for label, metrics in variant_metrics)
        )
        MetricCalculator._print_metric_tables(variant_metrics)

        print(f"Metric geometry calculation time: {elapsed:.4f}s")
        print("=" * 80)

        result = {
            "sequence_name": sequence_name,
            "elapsed_time": elapsed,
            "scale_estimation_summaries": scale_summaries,
        }
        for _, scale_key in MetricCalculator._variant_order():
            result[scale_key] = sequence_metrics[scale_key]
        return result

    @staticmethod
    def calculate_dataset_metrics(sequences, frame_count, depth_model, distance_threshold=100):
        start_time = time.time()
        sequence_results = []

        for sequence_path, sequence_name in sequences:
            sequence_metrics = MetricCalculator._evaluate_variants(
                sequence_path=sequence_path,
                frame_count=frame_count,
                depth_model=depth_model,
                sequence_name=sequence_name,
                distance_threshold=distance_threshold,
            )
            sequence_results.append(
                {
                    "sequence_name": sequence_name,
                    **{
                        scale_key: sequence_metrics[scale_key]
                        for _, scale_key in MetricCalculator._variant_order()
                    },
                }
            )

        elapsed = time.time() - start_time
        summary = MetricCalculator._build_dataset_summary(sequence_results, elapsed)
        MetricCalculator._print_dataset_report(
            sequence_results=sequence_results,
            dataset_metrics=summary["dataset"],
            depth_model=depth_model,
            elapsed=elapsed,
        )
        return summary

    @staticmethod
    def save_sequence_result(sequence_result, depth_model):
        file_utils.ensure_directory(MetricCalculator._sequence_results_dir(depth_model))
        payload = {
            "dataset": config.dataset,
            "depth_model": depth_model,
            "metric_results_run_id": MetricCalculator._results_run_id(),
            "metric_results_dir": MetricCalculator._results_root(depth_model),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **sequence_result,
        }
        file_utils.write_json(
            payload,
            MetricCalculator._sequence_result_path(depth_model, sequence_result["sequence_name"]),
        )
        with open(
            MetricCalculator._sequence_report_path(depth_model, sequence_result["sequence_name"]),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(MetricCalculator._render_sequence_report(sequence_result, depth_model))

    @staticmethod
    def save_dataset_summary(sequence_results, depth_model, total_sequences=None, elapsed=None):
        file_utils.ensure_directory(MetricCalculator._results_root(depth_model))
        summary = MetricCalculator._build_dataset_summary(sequence_results, elapsed or 0.0)
        difficulty_summary = MetricCalculator._build_sequence_difficulty_summary(sequence_results)
        report_text = MetricCalculator._render_dataset_report(
            sequence_results=sequence_results,
            dataset_metrics=summary["dataset"],
            depth_model=depth_model,
            elapsed=summary["elapsed_time"],
            difficulty_summary=difficulty_summary,
        )
        payload = {
            "dataset": config.dataset,
            "depth_model": depth_model,
            "metric_results_run_id": MetricCalculator._results_run_id(),
            "metric_results_dir": MetricCalculator._results_root(depth_model),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_sequences": total_sequences if total_sequences is not None else len(sequence_results),
            "processed_sequences": len(sequence_results),
            **summary,
        }
        file_utils.write_json(
            MetricCalculator._drop_per_frame_details(payload),
            MetricCalculator._dataset_results_path(depth_model),
        )
        with open(MetricCalculator._dataset_report_path(depth_model), "w", encoding="utf-8") as f:
            f.write(report_text)
        if total_sequences is None or len(sequence_results) >= total_sequences:
            print(
                MetricCalculator._render_dataset_summary_report(
                    sequence_results=sequence_results,
                    dataset_metrics=summary["dataset"],
                    depth_model=depth_model,
                    elapsed=payload["elapsed_time"],
                    difficulty_summary=difficulty_summary,
                ).rstrip()
            )
        if difficulty_summary:
            with open(
                MetricCalculator._sequence_group_map_path(depth_model),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    MetricCalculator._render_sequence_group_map(
                        difficulty_summary,
                        depth_model,
                    )
                )
        return payload


metric_geometry_evaluator = MetricGeometryEvaluator()
scale_accuracy_evaluator = ScaleAccuracyEvaluator()
metric_calculator = MetricCalculator()
