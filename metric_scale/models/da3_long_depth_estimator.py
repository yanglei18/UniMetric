import gc
import json
import math
import os
import shutil
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.io.output_processor import OutputProcessor
from metric_scale.config.config import config
from metric_scale.config.path_config import ensure_dir
from metric_scale.models.output_metadata import build_camera_params
from metric_scale.utils.utils import file_utils


class Da3LongDepthEstimator:
    """Chunked long-sequence Depth Anything 3 estimator with Pi3-long-style resize preprocessing."""

    def __init__(
        self,
        model_name="depth-anything/DA3METRIC-LARGE",
        device=None,
        dtype=None,
        chunk_size=60,
        overlap=30,
        delete_temp_files=True,
        temp_root=None,
        ref_view_strategy="saddle_balanced",
        max_points_per_frame=50000,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.delete_temp_files = delete_temp_files
        self.temp_root = temp_root
        self.ref_view_strategy = ref_view_strategy
        self.max_points_per_frame = max_points_per_frame

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.output_processor = OutputProcessor()

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than chunk size ({self.chunk_size})."
            )

        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(device=self.device).eval()

        if dtype is None:
            if self.device_type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            elif self.device_type == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

    def _log_prefix(self):
        return f"[{self.device}]"

    @staticmethod
    def _frame_prefix(frame_idx, frame_ids=None, start_frame=0):
        if frame_ids and frame_idx < len(frame_ids):
            return f"{frame_ids[frame_idx]}_camera1"
        return f"{start_frame + frame_idx:05d}_camera1"

    @staticmethod
    def _next_multiple(value, divisor):
        return int(math.ceil(float(value) / divisor) * divisor)

    @staticmethod
    def _compute_target_size(orig_height, orig_width, pixel_limit, max_long_edge=None):
        if max_long_edge is not None and max_long_edge > 0:
            max_side = max(orig_width, orig_height)
            scale = min(1.0, float(max_long_edge) / float(max_side)) if max_side > 0 else 1.0
            target_w = max(1, int(math.floor((orig_width * scale) / 14.0))) * 14
            target_h = max(1, int(math.floor((orig_height * scale) / 14.0))) * 14
            return target_h, target_w

        if pixel_limit is None or pixel_limit <= 0:
            target_h = Da3LongDepthEstimator._next_multiple(orig_height, 14)
            target_w = Da3LongDepthEstimator._next_multiple(orig_width, 14)
            return target_h, target_w

        scale = math.sqrt(float(pixel_limit) / float(orig_width * orig_height)) if orig_width * orig_height > 0 else 1.0
        target_w = orig_width * scale
        target_h = orig_height * scale
        k = round(target_w / 14)
        m = round(target_h / 14)
        while (k * 14) * (m * 14) > pixel_limit:
            if k / max(m, 1) > target_w / max(target_h, 1.0):
                k -= 1
            else:
                m -= 1
        return max(1, m) * 14, max(1, k) * 14

    @staticmethod
    def _resize_frame_output(array, target_size, interpolation=cv2.INTER_LINEAR):
        target_h, target_w = target_size
        if array.shape[:2] == (target_h, target_w):
            return array.astype(np.float32)
        resized = cv2.resize(array, (target_w, target_h), interpolation=interpolation)
        return resized.astype(np.float32)

    @staticmethod
    def _rescale_intrinsics(intrinsic, source_size, target_size):
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

    def _collect_image_paths(self, path, interval=1, start_frame=0, frame_count=None):
        if not os.path.isdir(path):
            raise ValueError(f"Unsupported path. Must be a directory: {path}")

        filenames = file_utils.natural_sorted(
            [name for name in os.listdir(path) if name.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        end_frame = len(filenames) if frame_count is None else start_frame + frame_count

        image_paths = []
        frame_names = []
        orig_sizes = []

        for index in range(start_frame, end_frame, interval):
            if index >= len(filenames):
                break
            image_path = os.path.join(path, filenames[index])
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
            image_paths.append(image_path)
            frame_names.append(os.path.splitext(os.path.basename(image_path))[0])
            orig_sizes.append((height, width))

        return image_paths, frame_names, orig_sizes

    def _build_chunk_indices(self, num_frames):
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

    def _load_chunk_images_resized(self, image_paths, target_size):
        tensors = []
        metas = []
        target_h, target_w = target_size

        for image_path in image_paths:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                orig_width, orig_height = rgb.size
                resized = rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
                tensor = self.normalize(self.to_tensor(resized))

            tensors.append(tensor)
            metas.append(
                {
                    "image_path": image_path,
                    "orig_h": orig_height,
                    "orig_w": orig_width,
                    "infer_h": target_h,
                    "infer_w": target_w,
                }
            )

        return torch.stack(tensors, dim=0), metas

    @staticmethod
    def _depth_to_pointmap(depth, intrinsics):
        height, width = depth.shape
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones_like(depth)
        pixels = np.stack([u_coords, v_coords, ones], axis=-1)
        intrinsic_inv = np.linalg.inv(intrinsics)
        camera_dirs = pixels @ intrinsic_inv.T
        return (camera_dirs * depth[..., None]).astype(np.float32)

    @staticmethod
    def _to_homogeneous_extrinsic(extrinsic_w2c):
        extrinsic_w2c = np.asarray(extrinsic_w2c, dtype=np.float32)
        if extrinsic_w2c.shape == (3, 4):
            extrinsic_h = np.eye(4, dtype=np.float32)
            extrinsic_h[:3, :] = extrinsic_w2c
            return extrinsic_h
        if extrinsic_w2c.shape == (4, 4):
            return extrinsic_w2c
        raise ValueError(f"Unsupported extrinsic shape: {extrinsic_w2c.shape}")

    def _depth_to_world_points(self, depth, intrinsics, extrinsic_w2c):
        camera_points = self._depth_to_pointmap(depth, intrinsics)
        return self._local_to_world_points(camera_points, extrinsic_w2c)

    def _local_to_world_points(self, local_points, extrinsic_w2c):
        local_points = np.asarray(local_points, dtype=np.float32)
        height, width = local_points.shape[:2]
        local_points_h = np.concatenate(
            [local_points, np.ones((height, width, 1), dtype=np.float32)],
            axis=-1,
        )
        extrinsic_h = self._to_homogeneous_extrinsic(extrinsic_w2c)
        c2w = np.linalg.inv(extrinsic_h).astype(np.float32)
        world_points = local_points_h @ c2w.T
        return world_points[..., :3].astype(np.float32)

    def _run_chunk_inference(
        self,
        chunk_idx,
        image_paths,
        global_start_idx,
        temp_dir,
        target_size,
    ):
        images, metas = self._load_chunk_images_resized(image_paths, target_size)
        batched_images = images.to(self.device, non_blocking=True)[None].float()
        autocast_enabled = self.device_type == "cuda"

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device_type,
                dtype=self.dtype,
                enabled=autocast_enabled,
            ):
                raw_output = self.model.forward(
                    batched_images,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False,
                    use_ray_pose=False,
                    ref_view_strategy=self.ref_view_strategy,
                )

        prediction = self.output_processor(raw_output)

        depth = prediction.depth
        conf = prediction.conf
        extrinsics = prediction.extrinsics
        intrinsics = prediction.intrinsics

        if conf is None:
            raise ValueError("Depth Anything 3 did not return confidence maps.")
        if conf.ndim == 4 and conf.shape[1] == 1:
            conf = conf[:, 0]
        elif conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if extrinsics is None or intrinsics is None:
            raise ValueError("Depth Anything 3 did not return camera parameters.")

        chunk_dir = os.path.join(temp_dir, "chunks", f"chunk_{chunk_idx:04d}")
        ensure_dir(chunk_dir)

        records = []
        for local_idx, meta in enumerate(metas):
            frame_name = os.path.splitext(os.path.basename(meta["image_path"]))[0]
            save_path = os.path.join(chunk_dir, f"{frame_name}.npz")
            depth_frame = depth[local_idx].astype(np.float32)
            conf_frame = conf[local_idx].astype(np.float32)
            intrinsic_frame = intrinsics[local_idx].astype(np.float32)
            extrinsic_w2c_frame = extrinsics[local_idx].astype(np.float32)
            local_points = self._depth_to_pointmap(depth_frame, intrinsic_frame)
            world_points = self._local_to_world_points(local_points, extrinsic_w2c_frame)
            camera_pose_c2w = np.linalg.inv(
                self._to_homogeneous_extrinsic(extrinsic_w2c_frame)
            ).astype(np.float32)

            np.savez_compressed(
                save_path,
                world_points=world_points.astype(np.float32),
                local_points=local_points.astype(np.float32),
                depth=depth_frame,
                conf=conf_frame,
                intrinsic=intrinsic_frame,
                camera_pose_c2w=camera_pose_c2w,
                image_path=np.array(meta["image_path"]),
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
                }
            )

        del raw_output
        del prediction
        del batched_images
        del images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return records

    @staticmethod
    def _load_frame_prediction(pred_path):
        with np.load(pred_path) as prediction:
            return {
                "world_points": prediction["world_points"],
                "local_points": prediction["local_points"],
                "depth": prediction["depth"],
                "conf": prediction["conf"],
                "intrinsic": prediction["intrinsic"],
                "camera_pose_c2w": prediction["camera_pose_c2w"],
                "image_path": str(prediction["image_path"]),
            }

    def _sample_valid_world_points(self, points_a, conf_a, points_b, conf_b, seed):
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
    def _estimate_weighted_sim3(source_points, target_points, weights):
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
        u, singular_values, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T

        source_variance = np.sum(weights * np.sum(source_centered ** 2, axis=1))
        if source_variance <= eps:
            scale = 1.0
        else:
            scale = float(np.sum(singular_values) / source_variance)

        translation = mu_target - scale * (rotation @ mu_source)
        return {
            "s": float(scale),
            "R": rotation.astype(np.float32),
            "t": translation.astype(np.float32),
        }

    def _estimate_adjacent_chunk_sim3(self, left_records, right_records, overlap_indices):
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

            sampled = self._sample_valid_world_points(
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
            return {
                "s": 1.0,
                "R": np.eye(3, dtype=np.float32),
                "t": np.zeros(3, dtype=np.float32),
            }

        return self._estimate_weighted_sim3(
            np.concatenate(matched_source, axis=0),
            np.concatenate(matched_target, axis=0),
            np.concatenate(matched_weights, axis=0),
        )

    @staticmethod
    def _accumulate_chunk_sim3(relative_transforms):
        cumulative = [
            {
                "s": 1.0,
                "R": np.eye(3, dtype=np.float32),
                "t": np.zeros(3, dtype=np.float32),
            }
        ]

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
    def _select_best_records(chunk_records, chunk_indices):
        candidates_by_frame = {}
        for records in chunk_records:
            for record in records:
                candidates_by_frame.setdefault(record["frame_idx"], []).append(record)

        selected_records = []
        for frame_idx in sorted(candidates_by_frame):
            candidates = candidates_by_frame[frame_idx]

            def sort_key(record):
                start, end = chunk_indices[record["chunk_idx"]]
                center = 0.5 * (start + end - 1)
                return (abs(frame_idx - center), record["chunk_idx"])

            selected_records.append(min(candidates, key=sort_key))

        return selected_records

    def _apply_sim3_and_export_frame(self, prediction, sim3, output_size=None, source_size=None):
        scale = float(sim3["s"])
        rotation = sim3["R"]
        translation = sim3["t"]

        depth = (prediction["depth"] * scale).astype(np.float32)
        conf = prediction["conf"].astype(np.float32)
        intrinsic = prediction["intrinsic"].astype(np.float32)

        if output_size is not None:
            if source_size is None:
                source_size = depth.shape[:2]
            if depth.shape[:2] != output_size:
                depth = self._resize_frame_output(depth, output_size, interpolation=cv2.INTER_LINEAR)
            if conf.shape[:2] != output_size:
                conf = self._resize_frame_output(conf, output_size, interpolation=cv2.INTER_LINEAR)
            intrinsic = self._rescale_intrinsics(intrinsic, source_size, output_size)

        pointmap = self._depth_to_pointmap(depth, intrinsic)

        sim3_matrix = np.eye(4, dtype=np.float32)
        sim3_matrix[:3, :3] = scale * rotation
        sim3_matrix[:3, 3] = translation

        aligned_c2w = sim3_matrix @ prediction["camera_pose_c2w"]
        aligned_c2w = aligned_c2w.astype(np.float32)
        aligned_c2w[:3, :3] /= scale
        aligned_w2c = np.linalg.inv(aligned_c2w)[:3, :].astype(np.float32)

        return {
            "depth": depth,
            "conf": conf,
            "pointmap": pointmap,
            "intrinsics": intrinsic,
            "extrinsics": aligned_w2c,
            "image_path": prediction["image_path"],
        }

    def process(
        self,
        path,
        interval=1,
        process_res=None,
        conf_thresh_percentile=None,
        start_frame=0,
        frame_count=None,
        pixel_limit=255000,
        max_long_edge=None,
    ):
        del process_res
        del conf_thresh_percentile
        image_paths, frame_names, orig_sizes = self._collect_image_paths(
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

        print(
            f"{self._log_prefix()} Processing {len(image_paths)} images "
            f"with chunk size {self.chunk_size}, overlap {self.overlap}, "
            f"and inference size ({target_w}, {target_h})."
        )
        if self.temp_root:
            ensure_dir(self.temp_root)
        temp_dir = tempfile.mkdtemp(prefix="da3_long_", dir=self.temp_root)
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
                "type": "da3_long_manifest",
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

    def visualize_depth(self, image_path, depth_map, colormap=cv2.COLORMAP_JET, alpha=0.5):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min + 1e-6) * 255
        normalized_depth = normalized_depth.astype(np.uint8)

        colored_depth = cv2.applyColorMap(normalized_depth, colormap)
        colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)

        if colored_depth.shape[:2] != image.shape[:2]:
            colored_depth = cv2.resize(
                colored_depth,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return cv2.addWeighted(image, 1 - alpha, colored_depth, alpha, 0)

    def save_visualizations(self, results, output_dir, start_frame=0, frame_ids=None, colormap=cv2.COLORMAP_JET, alpha=0.5):
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
            exported = self._apply_sim3_and_export_frame(
                prediction,
                results["chunk_transforms"][record["chunk_idx"]],
                output_size=output_size,
                source_size=inference_size,
            )
            frame_prefix = self._frame_prefix(output_idx, frame_ids=frame_ids, start_frame=start_frame)
            visualized = self.visualize_depth(exported["image_path"], exported["depth"], colormap, alpha)
            image_bgr = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"depth_visualization_{frame_prefix}.png"), image_bgr)

        print(f"All {len(frame_records)} visualizations saved to {output_dir}")

    def save_results(
        self,
        results,
        output_dir,
        start_frame=0,
        frame_ids=None,
        save_images=True,
        save_depth=True,
        save_conf=True,
        save_pointmap=True,
        save_camera_params=True,
    ):
        del save_images
        ensure_dir(output_dir)

        if save_depth:
            depth_dir = os.path.join(output_dir, "depth")
            ensure_dir(depth_dir)
        if save_conf:
            conf_dir = os.path.join(output_dir, "conf")
            ensure_dir(conf_dir)
        if save_pointmap:
            pointmap_dir = os.path.join(output_dir, "pointmap")
            ensure_dir(pointmap_dir)
        if save_camera_params:
            camera_params_dir = os.path.join(output_dir, "camera_params")
            ensure_dir(camera_params_dir)

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
            exported = self._apply_sim3_and_export_frame(
                prediction,
                results["chunk_transforms"][record["chunk_idx"]],
                output_size=output_size,
                source_size=inference_size,
            )
            frame_prefix = self._frame_prefix(output_idx, frame_ids=frame_ids, start_frame=start_frame)

            if save_depth:
                np.save(os.path.join(depth_dir, f"{frame_prefix}.npy"), exported["depth"])
            if save_conf:
                np.save(os.path.join(conf_dir, f"{frame_prefix}.npy"), exported["conf"])
            if save_pointmap:
                np.save(os.path.join(pointmap_dir, f"{frame_prefix}.npy"), exported["pointmap"])
            if save_camera_params:
                with open(os.path.join(camera_params_dir, f"{frame_prefix}.json"), "w") as file:
                    json.dump(
                        build_camera_params(
                            exported["intrinsics"],
                            exported["extrinsics"],
                            output_size,
                            inference_size=inference_size,
                            original_size=original_size,
                            output_resolution_mode=config.depth_output_resolution,
                        ),
                        file,
                        indent=2,
                    )

        print(f"All {len(frame_records)} frames saved to {output_dir}")
        if self.delete_temp_files:
            self._cleanup_temp(results.get("temp_dir"))

    @staticmethod
    def _cleanup_temp(temp_dir):
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
