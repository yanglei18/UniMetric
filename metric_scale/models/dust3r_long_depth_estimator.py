import gc
import importlib
import json
import math
import os
import os.path as osp
import shutil
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as tvf

from metric_scale.config.config import config
from metric_scale.config.path_config import ensure_dir
from metric_scale.models.output_metadata import build_camera_params
from metric_scale.utils.utils import file_utils


class Dust3rLongDepthEstimator:
    """Chunked long-sequence DUST3R estimator with overlap-based Sim3 alignment."""

    def __init__(
        self,
        model_name: str = "dust3r",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        resolution_set: int = 512,
        norm_type: str = "dust3r",
        preprocess_mode: str = "resize_only",
        patch_size: int = 16,
        torch_hub_force_reload: bool = False,
        quat_order: str = "xyzw",
        chunk_size: int = 60,
        overlap: int = 30,
        delete_temp_files: bool = True,
        temp_root: str | None = None,
        max_points_per_frame: int = 50000,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        if dtype is None:
            if self.device_type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            elif self.device_type == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        self.model_name = model_name
        self.resolution_set = int(resolution_set)
        self.norm_type = str(norm_type)
        self.preprocess_mode = str(preprocess_mode or "resize_only").strip().lower()
        if self.preprocess_mode != "resize_only":
            raise ValueError(
                f"Unsupported DUST3R preprocess mode: {self.preprocess_mode}. "
                "Expected 'resize_only' so DUST3R uses the same Metric-Depth "
                "input resizing path as Pi3."
            )
        self.patch_size = int(patch_size)
        self.torch_hub_force_reload = bool(torch_hub_force_reload)
        self.quat_order = str(quat_order).strip().lower()
        if self.quat_order not in {"xyzw", "wxyz"}:
            raise ValueError(f"Unsupported quaternion order: {quat_order}")

        self._import_external_factory()
        self.model = self.model_factory(
            "dust3r",
            name="dust3r",
            ckpt_path=self.model_name,
            torch_hub_force_reload=self.torch_hub_force_reload,
        ).to(self.device).eval()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.delete_temp_files = delete_temp_files
        self.temp_root = temp_root
        self.max_points_per_frame = max_points_per_frame

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than chunk size ({self.chunk_size})."
            )

    def _import_external_factory(self) -> None:
        try:
            models_module = importlib.import_module("mapanything.models")
            image_module = importlib.import_module("mapanything.utils.image")
        except ImportError as exc:
            raise ImportError(
                "DUST3R external wrapper dependencies are not installed. Install the required "
                "external DUST3R wrapper dependencies before using dust3r_long."
            ) from exc

        self.model_factory = getattr(models_module, "model_factory")
        self.image_normalization_dict = getattr(image_module, "IMAGE_NORMALIZATION_DICT")

    @staticmethod
    def _frame_prefix(frame_idx: int, frame_ids: list[str] | None = None, start_frame: int = 0) -> str:
        if frame_ids and frame_idx < len(frame_ids):
            return f"{frame_ids[frame_idx]}_camera1"
        return f"{start_frame + frame_idx:05d}_camera1"

    @staticmethod
    def _tensor_to_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _squeeze_prediction_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = Dust3rLongDepthEstimator._tensor_to_numpy(value)
        while array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        return array.astype(np.float32)

    @staticmethod
    def _resize_frame_output(
        array: np.ndarray,
        target_size: tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        target_h, target_w = target_size
        if array.shape[:2] == (target_h, target_w):
            return array.astype(np.float32)
        resized = cv2.resize(array, (target_w, target_h), interpolation=interpolation)
        return resized.astype(np.float32)

    @staticmethod
    def _rescale_intrinsics(
        intrinsic: np.ndarray,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> np.ndarray:
        source_h, source_w = source_size
        target_h, target_w = target_size
        if (source_h, source_w) == (target_h, target_w):
            return intrinsic.astype(np.float32)

        scaled = intrinsic.astype(np.float32).copy()
        scale_x = float(target_w) / float(source_w)
        scale_y = float(target_h) / float(source_h)
        scaled[0, 0] *= scale_x
        scaled[1, 1] *= scale_y
        scaled[0, 2] *= scale_x
        scaled[1, 2] *= scale_y
        return scaled

    @staticmethod
    def _default_intrinsics(height: int, width: int) -> np.ndarray:
        focal = float(max(height, width))
        return np.array(
            [
                [focal, 0.0, (width - 1.0) * 0.5],
                [0.0, focal, (height - 1.0) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _estimate_intrinsics_from_rays(ray_directions: np.ndarray) -> np.ndarray | None:
        if ray_directions is None or ray_directions.ndim != 3 or ray_directions.shape[-1] != 3:
            return None

        height, width, _ = ray_directions.shape
        ray_z = ray_directions[..., 2]
        valid_mask = np.isfinite(ray_directions).all(axis=-1) & (np.abs(ray_z) > 1e-6)
        if np.count_nonzero(valid_mask) < 64:
            return None

        u_coords, v_coords = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        norm_x = ray_directions[..., 0] / ray_z
        norm_y = ray_directions[..., 1] / ray_z

        try:
            fx_cx, *_ = np.linalg.lstsq(
                np.stack([norm_x[valid_mask], np.ones(np.count_nonzero(valid_mask), dtype=np.float32)], axis=1),
                u_coords[valid_mask],
                rcond=None,
            )
            fy_cy, *_ = np.linalg.lstsq(
                np.stack([norm_y[valid_mask], np.ones(np.count_nonzero(valid_mask), dtype=np.float32)], axis=1),
                v_coords[valid_mask],
                rcond=None,
            )
        except np.linalg.LinAlgError:
            return None

        fx, cx = float(fx_cx[0]), float(fx_cx[1])
        fy, cy = float(fy_cy[0]), float(fy_cy[1])
        if not all(np.isfinite(value) for value in (fx, fy, cx, cy)):
            return None
        if abs(fx) < 1e-6 or abs(fy) < 1e-6:
            return None

        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
        if q.size != 4:
            raise ValueError(f"Expected quaternion with 4 elements, got shape {quaternion.shape}")

        if self.quat_order == "wxyz":
            qw, qx, qy, qz = q
        else:
            qx, qy, qz, qw = q

        norm = math.sqrt(float(qw * qw + qx * qx + qy * qy + qz * qz))
        if norm <= 1e-8:
            return np.eye(3, dtype=np.float32)
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float32,
        )

    def _build_camera_pose_c2w(
        self,
        cam_trans: np.ndarray | None,
        cam_quats: np.ndarray | None,
    ) -> np.ndarray:
        if cam_trans is None or cam_quats is None:
            return np.eye(4, dtype=np.float32)

        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = self._quaternion_to_rotation_matrix(cam_quats)
        camera_pose[:3, 3] = np.asarray(cam_trans, dtype=np.float32).reshape(3)
        return camera_pose

    @staticmethod
    def _depth_to_pointmap(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        height, width = depth.shape
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones_like(depth)
        pixels = np.stack([u_coords, v_coords, ones], axis=-1)
        intrinsic_inv = np.linalg.inv(intrinsics)
        camera_dirs = pixels @ intrinsic_inv.T
        return (camera_dirs * depth[..., None]).astype(np.float32)

    @staticmethod
    def _move_to_device(value: Any, device: str) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: Dust3rLongDepthEstimator._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [Dust3rLongDepthEstimator._move_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(Dust3rLongDepthEstimator._move_to_device(item, device) for item in value)
        return value

    @staticmethod
    def _build_view_identifier(image_path: str) -> tuple[str, str]:
        parent_name = os.path.basename(os.path.dirname(image_path)) or "sequence"
        image_name = os.path.basename(image_path)
        return parent_name, image_name

    def _attach_view_metadata(
        self,
        views: list[dict[str, Any]],
        image_paths: list[str],
    ) -> list[dict[str, Any]]:
        for view, image_path in zip(views, image_paths):
            label, instance = self._build_view_identifier(image_path)
            if "label" not in view or not view["label"]:
                view["label"] = [label]
            if "instance" not in view or not view["instance"]:
                view["instance"] = [instance]
        return views

    def _collect_image_paths(
        self,
        path: str,
        interval: int = 1,
        start_frame: int = 0,
        frame_count: int | None = None,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        if not osp.isdir(path):
            raise ValueError(f"Unsupported path. Must be a directory: {path}")

        filenames = file_utils.natural_sorted(
            [name for name in os.listdir(path) if name.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        end_frame = len(filenames) if frame_count is None else start_frame + frame_count

        image_paths: list[str] = []
        orig_sizes: list[tuple[int, int]] = []
        for index in range(start_frame, end_frame, interval):
            if index >= len(filenames):
                break
            image_path = osp.join(path, filenames[index])
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                orig_sizes.append((rgb.height, rgb.width))
            image_paths.append(image_path)
        return image_paths, orig_sizes

    def _compute_target_size(
        self,
        orig_height: int,
        orig_width: int,
        pixel_limit: int | None,
        max_long_edge: int | None = None,
    ) -> tuple[int, int]:
        divisor = int(self.patch_size)
        if max_long_edge is not None and max_long_edge > 0:
            max_side = max(orig_width, orig_height)
            scale = min(1.0, float(max_long_edge) / float(max_side)) if max_side > 0 else 1.0
            target_w = max(1, int(math.floor((orig_width * scale) / divisor))) * divisor
            target_h = max(1, int(math.floor((orig_height * scale) / divisor))) * divisor
            return target_h, target_w

        if pixel_limit is None or pixel_limit <= 0:
            target_h = max(1, int(math.ceil(float(orig_height) / divisor))) * divisor
            target_w = max(1, int(math.ceil(float(orig_width) / divisor))) * divisor
            return target_h, target_w

        scale = math.sqrt(float(pixel_limit) / float(orig_width * orig_height)) if orig_width * orig_height > 0 else 1.0
        target_w = orig_width * scale
        target_h = orig_height * scale
        k = round(target_w / divisor)
        m = round(target_h / divisor)
        while (k * divisor) * (m * divisor) > pixel_limit:
            if k / max(m, 1) > target_w / max(target_h, 1.0):
                k -= 1
            else:
                m -= 1
        return max(1, m) * divisor, max(1, k) * divisor

    def _load_chunk_images_resized(
        self,
        image_paths: list[str],
        target_size: tuple[int, int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.norm_type not in self.image_normalization_dict:
            raise ValueError(
                f"Unknown image normalization type: {self.norm_type}. "
                f"Available options: {list(self.image_normalization_dict.keys())}"
            )

        target_h, target_w = target_size
        img_norm = self.image_normalization_dict[self.norm_type]
        image_transform = tvf.Compose(
            [
                tvf.ToTensor(),
                tvf.Normalize(mean=img_norm.mean, std=img_norm.std),
            ]
        )

        views: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for local_idx, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                orig_w, orig_h = rgb.size
                resized = rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)

            label, instance = self._build_view_identifier(image_path)
            views.append(
                {
                    "img": image_transform(resized)[None],
                    "true_shape": np.int32([[target_h, target_w]]),
                    "idx": local_idx,
                    "instance": [instance],
                    "label": [label],
                    "data_norm_type": [self.norm_type],
                    "preprocess_mode": [self.preprocess_mode],
                }
            )
            metas.append(
                {
                    "image_path": image_path,
                    "orig_h": orig_h,
                    "orig_w": orig_w,
                    "infer_h": target_h,
                    "infer_w": target_w,
                }
            )
        return views, metas

    def _log_prefix(self) -> str:
        return f"[{self.device}]"

    def _build_chunk_indices(self, num_frames: int) -> list[tuple[int, int]]:
        if num_frames <= self.chunk_size:
            return [(0, num_frames)]

        step = self.chunk_size - self.overlap
        chunk_indices = []
        start = 0
        while start < num_frames:
            end = min(start + self.chunk_size, num_frames)
            chunk_indices.append((start, end))
            if end >= num_frames:
                break
            start += step
        return chunk_indices

    def _run_chunk_inference(
        self,
        chunk_idx: int,
        image_paths: list[str],
        global_start_idx: int,
        temp_dir: str,
        target_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        views, metas = self._load_chunk_images_resized(image_paths, target_size)
        views = self._attach_view_metadata(views, image_paths)
        views = self._move_to_device(views, self.device)

        autocast_enabled = self.device_type == "cuda"
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device_type, dtype=self.dtype, enabled=autocast_enabled):
                predictions = self.model(views)

        chunk_dir = os.path.join(temp_dir, "chunks", f"chunk_{chunk_idx:04d}")
        ensure_dir(chunk_dir)

        records = []
        for local_idx, (meta, prediction) in enumerate(zip(metas, predictions)):
            image_path = meta["image_path"]
            frame_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(chunk_dir, f"{frame_name}.npz")

            world_points = self._squeeze_prediction_array(prediction.get("pts3d"))
            local_points = self._squeeze_prediction_array(prediction.get("pts3d_cam"))
            if local_points is None:
                local_points = world_points
            if world_points is None:
                world_points = local_points
            if local_points is None or world_points is None:
                raise ValueError("DUST3R long prediction is missing both 'pts3d' and 'pts3d_cam'.")

            conf = self._squeeze_prediction_array(prediction.get("conf"))
            if conf is None:
                conf = np.ones(local_points.shape[:2], dtype=np.float32)
            if conf.ndim == 3 and conf.shape[-1] == 1:
                conf = conf[..., 0]

            ray_directions = self._squeeze_prediction_array(prediction.get("ray_directions"))
            intrinsic = self._estimate_intrinsics_from_rays(ray_directions)
            if intrinsic is None:
                intrinsic = self._default_intrinsics(local_points.shape[0], local_points.shape[1])

            cam_trans = self._squeeze_prediction_array(prediction.get("cam_trans"))
            cam_quats = self._squeeze_prediction_array(prediction.get("cam_quats"))
            camera_pose_c2w = self._build_camera_pose_c2w(cam_trans, cam_quats)

            np.savez_compressed(
                save_path,
                world_points=world_points.astype(np.float32),
                local_points=local_points.astype(np.float32),
                conf=conf.astype(np.float32),
                intrinsics=intrinsic.astype(np.float32),
                camera_pose_c2w=camera_pose_c2w.astype(np.float32),
                image_path=np.array(image_path),
                orig_h=np.array(meta["orig_h"], dtype=np.int32),
                orig_w=np.array(meta["orig_w"], dtype=np.int32),
                infer_h=np.array(meta["infer_h"], dtype=np.int32),
                infer_w=np.array(meta["infer_w"], dtype=np.int32),
                preprocess_mode=np.array(self.preprocess_mode),
            )

            records.append(
                {
                    "frame_idx": global_start_idx + local_idx,
                    "frame_name": frame_name,
                    "chunk_idx": chunk_idx,
                    "chunk_local_idx": local_idx,
                    "pred_path": save_path,
                    "orig_h": meta["orig_h"],
                    "orig_w": meta["orig_w"],
                    "infer_h": meta["infer_h"],
                    "infer_w": meta["infer_w"],
                    "image_path": image_path,
                }
            )

        del predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return records

    @staticmethod
    def _scalar_int(value: np.ndarray | int) -> int:
        return int(np.asarray(value).item())

    def _load_frame_prediction(self, pred_path: str) -> dict[str, Any]:
        with np.load(pred_path) as prediction:
            keys = set(prediction.files)
            return {
                "world_points": prediction["world_points"],
                "local_points": prediction["local_points"],
                "conf": prediction["conf"],
                "intrinsics": prediction["intrinsics"],
                "camera_pose_c2w": prediction["camera_pose_c2w"],
                "image_path": str(prediction["image_path"]),
                "orig_h": self._scalar_int(prediction["orig_h"]) if "orig_h" in keys else None,
                "orig_w": self._scalar_int(prediction["orig_w"]) if "orig_w" in keys else None,
                "infer_h": self._scalar_int(prediction["infer_h"]) if "infer_h" in keys else None,
                "infer_w": self._scalar_int(prediction["infer_w"]) if "infer_w" in keys else None,
                "preprocess_mode": str(prediction["preprocess_mode"]) if "preprocess_mode" in keys else "resize_only",
            }

    def _sample_valid_points(
        self,
        points_a: np.ndarray,
        conf_a: np.ndarray,
        points_b: np.ndarray,
        conf_b: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        common_height = min(points_a.shape[0], points_b.shape[0])
        common_width = min(points_a.shape[1], points_b.shape[1])
        points_a = points_a[:common_height, :common_width]
        points_b = points_b[:common_height, :common_width]
        conf_a = conf_a[:common_height, :common_width]
        conf_b = conf_b[:common_height, :common_width]

        conf_threshold = min(float(np.median(conf_a)), float(np.median(conf_b))) * 0.1
        valid_mask = (conf_a > conf_threshold) & (conf_b > conf_threshold)
        valid_indices = np.flatnonzero(valid_mask.reshape(-1))
        if valid_indices.size == 0:
            return None

        if valid_indices.size > self.max_points_per_frame:
            rng = np.random.default_rng(seed)
            valid_indices = rng.choice(valid_indices, size=self.max_points_per_frame, replace=False)

        flat_points_a = points_a.reshape(-1, 3)[valid_indices]
        flat_points_b = points_b.reshape(-1, 3)[valid_indices]
        weights = np.sqrt(conf_a.reshape(-1)[valid_indices] * conf_b.reshape(-1)[valid_indices])
        return flat_points_a, flat_points_b, weights

    @staticmethod
    def _estimate_weighted_sim3(
        source_points: np.ndarray,
        target_points: np.ndarray,
        weights: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        eps = 1e-8
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.maximum(weights, eps)
        weights /= np.sum(weights)

        source_points = np.asarray(source_points, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)

        mu_source = np.sum(weights[:, None] * source_points, axis=0)
        mu_target = np.sum(weights[:, None] * target_points, axis=0)

        source_centered = source_points - mu_source
        target_centered = target_points - mu_target
        covariance = (weights[:, None] * source_centered).T @ target_centered
        u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
        rotation = vt_matrix.T @ u_matrix.T
        if np.linalg.det(rotation) < 0:
            vt_matrix[-1, :] *= -1
            rotation = vt_matrix.T @ u_matrix.T

        source_variance = np.sum(weights * np.sum(source_centered ** 2, axis=1))
        scale = 1.0 if source_variance <= eps else float(np.sum(singular_values) / source_variance)
        translation = mu_target - scale * (rotation @ mu_source)
        return {"s": float(scale), "R": rotation.astype(np.float32), "t": translation.astype(np.float32)}

    def _estimate_adjacent_chunk_sim3(
        self,
        left_records: list[dict[str, Any]],
        right_records: list[dict[str, Any]],
        overlap_indices: list[int],
    ) -> dict[str, np.ndarray | float]:
        left_map = {record["frame_idx"]: record for record in left_records}
        right_map = {record["frame_idx"]: record for record in right_records}

        matched_target = []
        matched_source = []
        matched_weights = []
        for frame_idx in overlap_indices:
            left_record = left_map.get(frame_idx)
            right_record = right_map.get(frame_idx)
            if left_record is None or right_record is None:
                continue

            left_prediction = self._load_frame_prediction(left_record["pred_path"])
            right_prediction = self._load_frame_prediction(right_record["pred_path"])
            sampled = self._sample_valid_points(
                left_prediction["world_points"],
                left_prediction["conf"],
                right_prediction["world_points"],
                right_prediction["conf"],
                seed=frame_idx,
            )
            if sampled is None:
                continue
            points_left, points_right, weights = sampled
            matched_target.append(points_left)
            matched_source.append(points_right)
            matched_weights.append(weights)

        if not matched_target:
            return {"s": 1.0, "R": np.eye(3, dtype=np.float32), "t": np.zeros(3, dtype=np.float32)}

        return self._estimate_weighted_sim3(
            np.concatenate(matched_source, axis=0),
            np.concatenate(matched_target, axis=0),
            np.concatenate(matched_weights, axis=0),
        )

    @staticmethod
    def _accumulate_chunk_sim3(relative_transforms: list[dict[str, np.ndarray | float]]) -> list[dict[str, np.ndarray | float]]:
        cumulative = [{"s": 1.0, "R": np.eye(3, dtype=np.float32), "t": np.zeros(3, dtype=np.float32)}]
        for transform in relative_transforms:
            previous = cumulative[-1]
            next_scale = previous["s"] * transform["s"]
            next_rotation = previous["R"] @ transform["R"]
            next_translation = previous["s"] * (previous["R"] @ transform["t"]) + previous["t"]
            cumulative.append(
                {
                    "s": float(next_scale),
                    "R": next_rotation.astype(np.float32),
                    "t": next_translation.astype(np.float32),
                }
            )
        return cumulative

    @staticmethod
    def _select_best_records(
        chunk_records: list[list[dict[str, Any]]],
        chunk_indices: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        candidates_by_frame: dict[int, list[dict[str, Any]]] = {}
        for records in chunk_records:
            for record in records:
                candidates_by_frame.setdefault(record["frame_idx"], []).append(record)

        selected_records = []
        for frame_idx in sorted(candidates_by_frame):
            candidates = candidates_by_frame[frame_idx]

            def sort_key(record: dict[str, Any]) -> tuple[float, int]:
                start, end = chunk_indices[record["chunk_idx"]]
                center = 0.5 * (start + end - 1)
                return (abs(frame_idx - center), record["chunk_idx"])

            selected_records.append(min(candidates, key=sort_key))
        return selected_records

    def _apply_sim3_to_frame(
        self,
        local_points: np.ndarray,
        camera_pose_c2w: np.ndarray,
        sim3: dict[str, np.ndarray | float],
        intrinsics: np.ndarray | None = None,
        conf: np.ndarray | None = None,
        output_size: tuple[int, int] | None = None,
        source_size: tuple[int, int] | None = None,
    ) -> dict[str, np.ndarray]:
        scale = float(sim3["s"])
        rotation = sim3["R"]
        translation = sim3["t"]

        aligned_pointmap = (scale * local_points).astype(np.float32)
        depth = aligned_pointmap[..., 2].astype(np.float32)
        aligned_intrinsics = None if intrinsics is None else intrinsics.astype(np.float32)
        aligned_conf = None if conf is None else conf.astype(np.float32)

        if output_size is not None:
            if source_size is None:
                source_size = depth.shape[:2]
            if depth.shape[:2] != output_size:
                depth = self._resize_frame_output(depth, output_size, interpolation=cv2.INTER_LINEAR)
            if aligned_conf is not None and aligned_conf.shape[:2] != output_size:
                aligned_conf = self._resize_frame_output(aligned_conf, output_size, interpolation=cv2.INTER_LINEAR)
            if aligned_intrinsics is not None:
                aligned_intrinsics = self._rescale_intrinsics(aligned_intrinsics, source_size, output_size)
                aligned_pointmap = self._depth_to_pointmap(depth, aligned_intrinsics)
            elif aligned_pointmap.shape[:2] != output_size:
                aligned_pointmap = self._resize_frame_output(aligned_pointmap, output_size, interpolation=cv2.INTER_LINEAR)

        sim3_matrix = np.eye(4, dtype=np.float32)
        sim3_matrix[:3, :3] = scale * rotation
        sim3_matrix[:3, 3] = translation

        aligned_c2w = (sim3_matrix @ camera_pose_c2w).astype(np.float32)
        aligned_c2w[:3, :3] /= scale
        aligned_w2c = np.linalg.inv(aligned_c2w)[:3, :].astype(np.float32)

        return {
            "pointmap": aligned_pointmap,
            "depth": depth,
            "conf": aligned_conf,
            "intrinsics": aligned_intrinsics,
            "extrinsics": aligned_w2c,
        }

    def process(
        self,
        path: str,
        interval: int = 1,
        pixel_limit: int | None = None,
        max_long_edge: int | None = None,
        mode: str = "batch",
        start_frame: int = 0,
        frame_count: int | None = None,
    ) -> dict[str, Any] | None:
        if mode != "batch":
            raise ValueError("Dust3rLongDepthEstimator only supports batch mode.")

        image_paths, orig_sizes = self._collect_image_paths(
            path,
            interval=interval,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        if not image_paths:
            print("No images loaded. Cannot run depth estimation.")
            return None

        target_h, target_w = self._compute_target_size(
            orig_sizes[0][0],
            orig_sizes[0][1],
            pixel_limit,
            max_long_edge,
        )

        frame_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]
        print(
            f"{self._log_prefix()} Processing {len(image_paths)} images "
            f"with chunk size {self.chunk_size}, overlap {self.overlap}, "
            f"preprocess={self.preprocess_mode}, and inference size ({target_w}, {target_h})."
        )

        if self.temp_root:
            ensure_dir(self.temp_root)
        temp_dir = tempfile.mkdtemp(prefix="dust3r_long_", dir=self.temp_root)
        try:
            chunk_indices = self._build_chunk_indices(len(image_paths))
            chunk_records = []
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_indices):
                print(
                    f"{self._log_prefix()} Running chunk {chunk_idx + 1}/{len(chunk_indices)}: "
                    f"frames {chunk_start} -> {chunk_end - 1}"
                )
                records = self._run_chunk_inference(
                    chunk_idx=chunk_idx,
                    image_paths=image_paths[chunk_start:chunk_end],
                    global_start_idx=chunk_start,
                    temp_dir=temp_dir,
                    target_size=(target_h, target_w),
                )
                chunk_records.append(records)

            relative_transforms = []
            for chunk_idx in range(len(chunk_indices) - 1):
                overlap_start = chunk_indices[chunk_idx + 1][0]
                overlap_end = min(chunk_indices[chunk_idx][1], chunk_indices[chunk_idx + 1][1])
                overlap_indices = list(range(overlap_start, overlap_end))
                transform = self._estimate_adjacent_chunk_sim3(
                    chunk_records[chunk_idx],
                    chunk_records[chunk_idx + 1],
                    overlap_indices,
                )
                relative_transforms.append(transform)

            selected_records = self._select_best_records(chunk_records, chunk_indices)
            return {
                "type": "dust3r_long_manifest",
                "image_paths": image_paths,
                "frame_names": frame_names,
                "orig_sizes": orig_sizes,
                "chunk_indices": chunk_indices,
                "chunk_transforms": self._accumulate_chunk_sim3(relative_transforms),
                "frame_records": selected_records,
                "infer_size": (target_h, target_w),
                "temp_dir": temp_dir,
            }
        except Exception:
            if self.delete_temp_files:
                self._cleanup_temp(temp_dir)
            raise

    def save_visualizations(
        self,
        results: dict[str, Any],
        output_dir: str,
        start_frame: int = 0,
        frame_ids: list[str] | None = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
    ) -> None:
        ensure_dir(output_dir)
        frame_records = sorted(results["frame_records"], key=lambda record: record["frame_idx"])
        infer_size = results.get("infer_size")
        restore_to_original_resolution = config.depth_output_resolution != "native"
        for output_idx, record in enumerate(frame_records):
            prediction = self._load_frame_prediction(record["pred_path"])
            output_size = (
                (record["orig_h"], record["orig_w"])
                if restore_to_original_resolution
                else (record["infer_h"], record["infer_w"])
            )
            original_size = (int(record["orig_h"]), int(record["orig_w"]))
            inference_size = (
                (int(infer_size[0]), int(infer_size[1]))
                if infer_size is not None
                else (int(record["infer_h"]), int(record["infer_w"]))
            )
            aligned = self._apply_sim3_to_frame(
                prediction["local_points"],
                prediction["camera_pose_c2w"],
                results["chunk_transforms"][record["chunk_idx"]],
                intrinsics=prediction["intrinsics"],
                conf=prediction["conf"],
                output_size=output_size,
                source_size=inference_size,
            )
            image = cv2.cvtColor(cv2.imread(record["image_path"]), cv2.COLOR_BGR2RGB)
            depth_map = aligned["depth"]
            depth_min, depth_max = depth_map.min(), depth_map.max()
            normalized_depth = ((depth_map - depth_min) / (depth_max - depth_min + 1e-6) * 255).astype(np.uint8)
            colored_depth = cv2.cvtColor(cv2.applyColorMap(normalized_depth, colormap), cv2.COLOR_BGR2RGB)
            if colored_depth.shape[:2] != image.shape[:2]:
                colored_depth = cv2.resize(colored_depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            blended = cv2.addWeighted(image, 1 - alpha, colored_depth, alpha, 0)
            frame_prefix = self._frame_prefix(output_idx, frame_ids=frame_ids, start_frame=start_frame)
            cv2.imwrite(
                os.path.join(output_dir, f"depth_visualization_{frame_prefix}.png"),
                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
            )

    def save_results(
        self,
        results: dict[str, Any],
        output_dir: str,
        start_frame: int = 0,
        frame_ids: list[str] | None = None,
        frame_count: int | None = None,
        save_pointmap: bool = True,
        save_camera_params: bool = True,
        save_conf: bool = True,
        save_depth: bool = True,
    ) -> None:
        del frame_count
        ensure_dir(output_dir)
        if save_camera_params:
            camera_params_dir = ensure_dir(os.path.join(output_dir, "camera_params"))
        if save_pointmap:
            pointmap_dir = ensure_dir(os.path.join(output_dir, "pointmap"))
        if save_conf:
            conf_dir = ensure_dir(os.path.join(output_dir, "conf"))
        if save_depth:
            depth_dir = ensure_dir(os.path.join(output_dir, "depth"))

        frame_records = sorted(results["frame_records"], key=lambda record: record["frame_idx"])
        infer_size = results.get("infer_size")
        restore_to_original_resolution = config.depth_output_resolution != "native"
        for output_idx, record in enumerate(frame_records):
            prediction = self._load_frame_prediction(record["pred_path"])
            output_size = (
                (record["orig_h"], record["orig_w"])
                if restore_to_original_resolution
                else (record["infer_h"], record["infer_w"])
            )
            original_size = (int(record["orig_h"]), int(record["orig_w"]))
            inference_size = (
                (int(infer_size[0]), int(infer_size[1]))
                if infer_size is not None
                else (int(record["infer_h"]), int(record["infer_w"]))
            )
            aligned = self._apply_sim3_to_frame(
                prediction["local_points"],
                prediction["camera_pose_c2w"],
                results["chunk_transforms"][record["chunk_idx"]],
                intrinsics=prediction["intrinsics"],
                conf=prediction["conf"],
                output_size=output_size,
                source_size=inference_size,
            )
            frame_prefix = self._frame_prefix(output_idx, frame_ids=frame_ids, start_frame=start_frame)

            if save_pointmap:
                np.save(os.path.join(pointmap_dir, f"{frame_prefix}.npy"), aligned["pointmap"])
            if save_depth:
                np.save(os.path.join(depth_dir, f"{frame_prefix}.npy"), aligned["depth"])
            if save_conf:
                np.save(os.path.join(conf_dir, f"{frame_prefix}.npy"), aligned["conf"])
            if save_camera_params:
                with open(os.path.join(camera_params_dir, f"{frame_prefix}.json"), "w", encoding="utf-8") as file:
                    json.dump(
                        build_camera_params(
                            aligned["intrinsics"],
                            aligned["extrinsics"],
                            output_size,
                            inference_size=inference_size,
                            original_size=original_size,
                            output_resolution_mode=config.depth_output_resolution,
                            preprocess_mode=self.preprocess_mode,
                        ),
                        file,
                        indent=2,
                    )

        print(f"All {len(frame_records)} frames saved to {output_dir}")
        if self.delete_temp_files:
            self._cleanup_temp(results.get("temp_dir"))

    @staticmethod
    def _cleanup_temp(temp_dir: str | None) -> None:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
