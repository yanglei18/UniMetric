import gc
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from metric_scale.config.config import config
from metric_scale.config.path_config import ensure_dir
from metric_scale.utils.utils import file_utils

_pi3_vendor_root = Path(__file__).resolve().parents[2] / "vendor" / "Pi3"
if (_pi3_vendor_root / "pi3" / "models" / "pi3x.py").is_file():
    sys.path.insert(0, str(_pi3_vendor_root))
from pi3.models.pi3 import Pi3
from pi3.models.pi3x import Pi3X


class Pi3LongDepthEstimator:
    """Chunked long-sequence Pi3/Pi3X estimator without loop closure."""

    DEFAULT_INTRINSICS = np.array(
        [[1000, 0, 512], [0, 1000, 512], [0, 0, 1]],
        dtype=np.float32,
    )

    def __init__(
        self,
        model_type="Pi3X",
        device=None,
        dtype=None,
        pretrained_model=None,
        chunk_size=60,
        overlap=30,
        delete_temp_files=True,
        temp_root=None,
        max_points_per_frame=50000,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.model_type = model_type
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.delete_temp_files = delete_temp_files
        self.temp_root = temp_root
        self.max_points_per_frame = max_points_per_frame
        self.to_tensor = transforms.ToTensor()

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than chunk size ({self.chunk_size})."
            )

        if pretrained_model is None:
            pretrained_model = "yyfz233/Pi3X" if model_type == "Pi3X" else "yyfz233/Pi3"

        if model_type == "Pi3X":
            self.model = Pi3X.from_pretrained(pretrained_model).to(self.device).eval()
        elif model_type == "Pi3":
            self.model = Pi3.from_pretrained(pretrained_model).to(self.device).eval()
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Options are 'Pi3' or 'Pi3X'.")

        if dtype is None:
            if self.device_type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            elif self.device_type == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

    @staticmethod
    def _frame_prefix(frame_idx, frame_ids=None, start_frame=0):
        if frame_ids and frame_idx < len(frame_ids):
            return f"{frame_ids[frame_idx]}_camera1"
        return f"{start_frame + frame_idx:05d}_camera1"

    @staticmethod
    def _dummy_intrinsics():
        return Pi3LongDepthEstimator.DEFAULT_INTRINSICS.copy()

    def _log_prefix(self):
        return f"[{self.device}]"

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
            target_h = Pi3LongDepthEstimator._next_multiple(orig_height, 14)
            target_w = Pi3LongDepthEstimator._next_multiple(orig_width, 14)
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
    def _dummy_intrinsics_for_size(height, width):
        scale_x = float(width) / 1024.0
        scale_y = float(height) / 1024.0
        return np.array(
            [
                [1000.0 * scale_x, 0.0, width / 2.0],
                [0.0, 1000.0 * scale_y, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _scale_intrinsics(intrinsics, source_size, target_size):
        source_h, source_w = source_size
        target_h, target_w = target_size
        scale_x = float(target_w) / max(float(source_w), 1.0)
        scale_y = float(target_h) / max(float(source_h), 1.0)
        scaled = intrinsics.astype(np.float32).copy()
        scaled[0, 0] *= scale_x
        scaled[0, 2] *= scale_x
        scaled[1, 1] *= scale_y
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
                tensor = self.to_tensor(resized)

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

    def _run_chunk_inference(self, chunk_idx, image_paths, global_start_idx, temp_dir, target_size):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images, metas = self._load_chunk_images_resized(image_paths, target_size)
        images = images.to(self.device)
        autocast_enabled = self.device_type == "cuda"

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device_type, dtype=self.dtype, enabled=autocast_enabled):
                predictions = self.model(images[None])

        world_points = predictions["points"].detach().float().cpu().squeeze(0).numpy()
        local_points = predictions["local_points"].detach().float().cpu().squeeze(0).numpy()
        conf = torch.sigmoid(predictions["conf"]).detach().float().cpu().squeeze(0).numpy()
        camera_poses = predictions["camera_poses"].detach().float().cpu().squeeze(0).numpy()

        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]

        chunk_dir = os.path.join(temp_dir, "chunks", f"chunk_{chunk_idx:04d}")
        ensure_dir(chunk_dir)

        records = []
        for local_idx, meta in enumerate(metas):
            frame_name = os.path.splitext(os.path.basename(meta["image_path"]))[0]
            save_path = os.path.join(chunk_dir, f"{frame_name}.npz")

            np.savez_compressed(
                save_path,
                world_points=world_points[local_idx].astype(np.float32),
                local_points=local_points[local_idx].astype(np.float32),
                conf=conf[local_idx].astype(np.float32),
                camera_pose_c2w=camera_poses[local_idx].astype(np.float32),
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

        del predictions
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
                "conf": prediction["conf"],
                "camera_pose_c2w": prediction["camera_pose_c2w"],
            }

    def _sample_valid_points(self, points_a, conf_a, points_b, conf_b, seed):
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

    @staticmethod
    def _apply_sim3_to_frame(local_points, camera_pose_c2w, sim3):
        scale = float(sim3["s"])
        rotation = sim3["R"]
        translation = sim3["t"]

        aligned_pointmap = (scale * local_points).astype(np.float32)

        sim3_matrix = np.eye(4, dtype=np.float32)
        sim3_matrix[:3, :3] = scale * rotation
        sim3_matrix[:3, 3] = translation

        aligned_c2w = sim3_matrix @ camera_pose_c2w
        aligned_c2w = aligned_c2w.astype(np.float32)
        aligned_c2w[:3, :3] /= scale
        aligned_w2c = np.linalg.inv(aligned_c2w)[:3, :].astype(np.float32)

        return {
            "pointmap": aligned_pointmap,
            "depth": aligned_pointmap[..., 2].astype(np.float32),
            "extrinsics": aligned_w2c,
        }

    def process(self, path, interval=1, pixel_limit=None, max_long_edge=None, mode="batch", start_frame=0, frame_count=None):
        if mode != "batch":
            raise ValueError("Pi3LongDepthEstimator only supports batch mode.")

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
        temp_dir = tempfile.mkdtemp(prefix="pi3_long_", dir=self.temp_root)
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
                "type": "pi3_long_manifest",
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

    def save_visualizations(self, results, output_dir, colormap=cv2.COLORMAP_JET, alpha=0.5):
        del results
        del output_dir
        del colormap
        del alpha
        print("Visualization not implemented for Pi3LongDepthEstimator. Skipping...")

    def save_results(
        self,
        results,
        output_dir,
        start_frame=0,
        frame_ids=None,
        frame_count=None,
        save_pointmap=True,
        save_camera_params=True,
        save_conf=True,
        save_depth=True,
    ):
        del frame_count
        ensure_dir(output_dir)

        if save_camera_params:
            camera_params_dir = os.path.join(output_dir, "camera_params")
            ensure_dir(camera_params_dir)
        if save_pointmap:
            pointmap_dir = os.path.join(output_dir, "pointmap")
            ensure_dir(pointmap_dir)
        if save_conf:
            conf_dir = os.path.join(output_dir, "conf")
            ensure_dir(conf_dir)
        if save_depth:
            depth_dir = os.path.join(output_dir, "depth")
            ensure_dir(depth_dir)

        frame_records = sorted(results["frame_records"], key=lambda record: record["frame_idx"])
        infer_size = results.get("infer_size")
        restore_to_original_resolution = config.depth_output_resolution != "native"
        output_resolution_mode = str(config.depth_output_resolution)
        for output_idx, record in enumerate(frame_records):
            prediction = self._load_frame_prediction(record["pred_path"])
            aligned = self._apply_sim3_to_frame(
                prediction["local_points"],
                prediction["camera_pose_c2w"],
                results["chunk_transforms"][record["chunk_idx"]],
            )
            frame_prefix = self._frame_prefix(output_idx, frame_ids=frame_ids, start_frame=start_frame)
            output_size = (record["orig_h"], record["orig_w"]) if restore_to_original_resolution else (record["infer_h"], record["infer_w"])

            pointmap_to_save = aligned["pointmap"]
            depth_to_save = aligned["depth"]
            conf_to_save = prediction["conf"].astype(np.float32)

            if conf_to_save.ndim == 3 and conf_to_save.shape[-1] == 1:
                conf_to_save = conf_to_save[..., 0]

            if pointmap_to_save.shape[:2] != output_size:
                pointmap_to_save = self._resize_frame_output(pointmap_to_save, output_size, interpolation=cv2.INTER_LINEAR)
            if depth_to_save.shape[:2] != output_size:
                depth_to_save = self._resize_frame_output(depth_to_save, output_size, interpolation=cv2.INTER_LINEAR)
            if conf_to_save.shape[:2] != output_size:
                conf_to_save = self._resize_frame_output(conf_to_save, output_size, interpolation=cv2.INTER_LINEAR)

            if save_pointmap:
                np.save(os.path.join(pointmap_dir, f"{frame_prefix}.npy"), pointmap_to_save)
            if save_depth:
                np.save(os.path.join(depth_dir, f"{frame_prefix}.npy"), depth_to_save)
            if save_conf:
                np.save(os.path.join(conf_dir, f"{frame_prefix}.npy"), conf_to_save)
            if save_camera_params:
                original_size = (int(record["orig_h"]), int(record["orig_w"]))
                inference_size = (
                    (int(infer_size[0]), int(infer_size[1]))
                    if infer_size is not None
                    else (int(record["infer_h"]), int(record["infer_w"]))
                )
                intrinsics_original = self._dummy_intrinsics_for_size(*original_size)
                intrinsics_prediction = self._scale_intrinsics(
                    intrinsics_original,
                    original_size,
                    output_size,
                )
                scale_x = float(output_size[1]) / max(float(original_size[1]), 1.0)
                scale_y = float(output_size[0]) / max(float(original_size[0]), 1.0)
                with open(os.path.join(camera_params_dir, f"{frame_prefix}.json"), "w") as file:
                    json.dump(
                        {
                            "intrinsics": intrinsics_prediction.tolist(),
                            "intrinsics_original": intrinsics_original.tolist(),
                            "intrinsics_prediction": intrinsics_prediction.tolist(),
                            "extrinsics": aligned["extrinsics"].tolist(),
                            "original_resolution": [int(original_size[0]), int(original_size[1])],
                            "inference_resolution": [int(inference_size[0]), int(inference_size[1])],
                            "output_resolution": [int(output_size[0]), int(output_size[1])],
                            "prediction_resolution": [int(output_size[0]), int(output_size[1])],
                            "output_resolution_mode": output_resolution_mode,
                            "preprocess": {
                                "mode": "resize_only",
                                "scale_x": scale_x,
                                "scale_y": scale_y,
                                "crop_x": 0.0,
                                "crop_y": 0.0,
                                "pad_x": 0.0,
                                "pad_y": 0.0,
                            },
                        },
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
