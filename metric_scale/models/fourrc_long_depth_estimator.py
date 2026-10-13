import gc
import importlib
import json
import math
import os
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from metric_scale.config.config import config
from metric_scale.config.path_config import ensure_dir
from metric_scale.models.output_metadata import build_camera_params
from metric_scale.models.vggt_long_depth_estimator import VGGTLongDepthEstimator


class FourRCLongDepthEstimator(VGGTLongDepthEstimator):
    """Long-sequence 4RC wrapper for dense point/depth prediction."""

    def __init__(
        self,
        project_root: str | None = None,
        checkpoint: str = "Luo-Yihang/4RC",
        device: str | torch.device | None = None,
        image_size: int = 512,
        patch_size: int = 14,
        dtype: str = "bf16-mixed",
        use_center_as_anchor: bool = False,
        preprocess_mode: str = "resize_only",
        chunk_size: int = 30,
        overlap: int = 10,
        delete_temp_files: bool = True,
        temp_root: str | None = None,
        max_points_per_frame: int = 50000,
    ) -> None:
        self.project_root = project_root
        self.checkpoint = checkpoint or "Luo-Yihang/4RC"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.dtype = str(dtype or "bf16-mixed").strip()
        self.use_center_as_anchor = bool(use_center_as_anchor)
        self.preprocess_mode = str(preprocess_mode or "resize_only").strip().lower()
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.delete_temp_files = bool(delete_temp_files)
        self.temp_root = temp_root
        self.max_points_per_frame = int(max_points_per_frame)

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than chunk size ({self.chunk_size})."
            )
        if self.preprocess_mode not in {"resize_only", "official"}:
            raise ValueError(
                f"Unsupported 4RC preprocess mode: {self.preprocess_mode}. "
                "Expected 'resize_only' or 'official'."
            )

        self._import_fourrc()
        self.model = self.arc_class.from_pretrained(self.checkpoint).to(self.device).eval()

    def _import_fourrc(self) -> None:
        if self.project_root and os.path.isdir(self.project_root):
            abs_project_root = os.path.abspath(self.project_root)
            if abs_project_root not in sys.path:
                sys.path.insert(0, abs_project_root)

        try:
            model_module = importlib.import_module("arc.models.arc")
            inference_module = importlib.import_module("arc.dust3r.inference_multiview")
            image_module = importlib.import_module("arc.dust3r.utils.image")
        except ImportError:
            try:
                model_module = importlib.import_module("arc.models.arc.arc")
                inference_module = importlib.import_module("arc.dust3r.inference_multiview")
                image_module = importlib.import_module("arc.dust3r.utils.image")
            except ImportError as exc:
                raise ImportError(
                    "Failed to import 4RC. Set models.fourrc_project_root to the local "
                    "4RC repository or install Luo-Yihang/4RC in the active environment."
                ) from exc

        self.arc_class = model_module.Arc
        self.inference_fn = inference_module.inference
        self.load_images_fn = image_module.load_images

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _squeeze_array(value: Any) -> np.ndarray:
        array = FourRCLongDepthEstimator._as_numpy(value)
        while array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        return array.astype(np.float32)

    @staticmethod
    def _to_4x4_pose(pose: Any) -> np.ndarray:
        pose = FourRCLongDepthEstimator._squeeze_array(pose)
        if pose.shape == (4, 4):
            return pose.astype(np.float32)
        if pose.shape == (3, 4):
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :4] = pose
            return pose_4x4
        raise ValueError(f"Unsupported 4RC camera pose shape: {pose.shape}")

    @staticmethod
    def _to_intrinsics(value: Any) -> np.ndarray:
        intrinsic = FourRCLongDepthEstimator._squeeze_array(value)
        if intrinsic.shape == (3, 3):
            return intrinsic.astype(np.float32)
        if intrinsic.size >= 9:
            return intrinsic.reshape(3, 3).astype(np.float32)
        raise ValueError(f"Unsupported 4RC intrinsic shape: {intrinsic.shape}")

    @staticmethod
    def _to_points(value: Any) -> np.ndarray:
        points = FourRCLongDepthEstimator._squeeze_array(value)
        if points.ndim == 3 and points.shape[-1] == 3:
            return points.astype(np.float32)
        if points.ndim == 2 and points.shape[-1] == 3:
            side = int(round(np.sqrt(points.shape[0])))
            if side * side == points.shape[0]:
                return points.reshape(side, side, 3).astype(np.float32)
        raise ValueError(f"Unsupported 4RC point shape: {points.shape}")

    @staticmethod
    def _to_conf(value: Any, target_shape: tuple[int, int]) -> np.ndarray:
        if value is None:
            return np.ones(target_shape, dtype=np.float32)
        conf = FourRCLongDepthEstimator._squeeze_array(value)
        if conf.ndim == 3 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if conf.ndim == 1 and conf.size == target_shape[0] * target_shape[1]:
            conf = conf.reshape(target_shape)
        if conf.shape != target_shape:
            return np.ones(target_shape, dtype=np.float32)
        return conf.astype(np.float32)

    @staticmethod
    def _world_to_camera_points(world_points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
        w2c = np.linalg.inv(c2w).astype(np.float32)
        flat_points = world_points.reshape(-1, 3)
        rotation = w2c[:3, :3]
        translation = w2c[:3, 3]
        camera_points = flat_points @ rotation.T + translation[None]
        return camera_points.reshape(world_points.shape).astype(np.float32)

    @staticmethod
    def _resize_pointmap(pointmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_size
        if pointmap.shape[:2] == (target_h, target_w):
            return pointmap.astype(np.float32)
        return cv2.resize(pointmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    @staticmethod
    def _next_multiple(value: int, divisor: int) -> int:
        return int(math.ceil(float(value) / divisor) * divisor)

    @staticmethod
    def _compute_target_size(
        orig_height: int,
        orig_width: int,
        pixel_limit: int | None,
        max_long_edge: int | None = None,
        patch_size: int = 14,
    ) -> tuple[int, int]:
        patch_size = max(1, int(patch_size))
        if max_long_edge is not None and max_long_edge > 0:
            max_side = max(orig_width, orig_height)
            scale = min(1.0, float(max_long_edge) / float(max_side)) if max_side > 0 else 1.0
            target_w = max(1, int(math.floor((orig_width * scale) / patch_size))) * patch_size
            target_h = max(1, int(math.floor((orig_height * scale) / patch_size))) * patch_size
            return int(target_h), int(target_w)

        if pixel_limit is not None and pixel_limit > 0:
            scale = math.sqrt(float(pixel_limit) / float(orig_width * orig_height)) if orig_width * orig_height > 0 else 1.0
            target_w = orig_width * scale
            target_h = orig_height * scale
            k = round(target_w / patch_size)
            m = round(target_h / patch_size)
            while (k * patch_size) * (m * patch_size) > pixel_limit:
                if k / max(m, 1) > target_w / max(target_h, 1.0):
                    k -= 1
                else:
                    m -= 1
            return max(1, m) * patch_size, max(1, k) * patch_size

        target_h = FourRCLongDepthEstimator._next_multiple(orig_height, patch_size)
        target_w = FourRCLongDepthEstimator._next_multiple(orig_width, patch_size)
        return int(target_h), int(target_w)

    @staticmethod
    def _normalize_image_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor * 2.0 - 1.0

    def _load_images_resize_only(
        self,
        image_paths: list[str],
        target_size: tuple[int, int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        target_h, target_w = target_size
        views: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []

        print(f">> Loading a list of {len(image_paths)} images with 4RC resize-only preprocessing")
        for image_path in image_paths:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                orig_w, orig_h = rgb.size
                resized = rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
                tensor = self._normalize_image_tensor(resized)

            print(
                f" - adding {image_path} with resolution {orig_w}x{orig_h} "
                f"--> {target_w}x{target_h}"
            )
            views.append(
                {
                    "img": tensor[None],
                    "true_shape": np.int32([[target_h, target_w]]),
                    "idx": len(views),
                    "instance": str(len(views)),
                }
            )
            metas.append(
                {
                    "image_path": image_path,
                    "orig_h": int(orig_h),
                    "orig_w": int(orig_w),
                    "infer_h": int(target_h),
                    "infer_w": int(target_w),
                }
            )

        if not views:
            raise ValueError("No images loaded for 4RC resize-only preprocessing.")
        print(f" (Found {len(views)} images)")
        return views, metas

    def _run_chunk_inference(
        self,
        chunk_idx: int,
        image_paths: list[str],
        global_start_idx: int,
        temp_dir: str,
        target_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        if self.preprocess_mode == "official":
            views = self.load_images_fn(
                image_paths,
                size=self.image_size,
                patch_size=self.patch_size,
                verbose=True,
            )
            metas = None
        else:
            views, metas = self._load_images_resize_only(image_paths, target_size)

        for view in views:
            view["track_query_idx"] = torch.tensor([len(views) // 2])

        with torch.no_grad():
            output = self.inference_fn(
                views,
                self.model,
                torch.device(self.device),
                dtype=self.dtype,
                verbose=True,
                profiling=False,
                use_center_as_anchor=self.use_center_as_anchor,
            )

        preds = output["preds"]
        chunk_dir = os.path.join(temp_dir, "chunks", f"chunk_{chunk_idx:04d}")
        ensure_dir(chunk_dir)
        records = []

        for local_idx, pred in enumerate(preds):
            image_path = image_paths[local_idx]
            frame_name = os.path.splitext(os.path.basename(image_path))[0]
            world_points = self._to_points(pred["pts"])
            conf = self._to_conf(pred.get("conf"), world_points.shape[:2])
            intrinsic = self._to_intrinsics(pred["intrinsic"])
            camera_pose_c2w = self._to_4x4_pose(pred["extrinsic"])
            camera_points = self._world_to_camera_points(world_points, camera_pose_c2w)
            depth = camera_points[..., 2].astype(np.float32)

            save_path = os.path.join(chunk_dir, f"{frame_name}.npz")
            np.savez_compressed(
                save_path,
                depth=depth,
                depth_conf=conf,
                world_points=world_points.astype(np.float32),
                world_points_conf=conf,
                intrinsic=intrinsic,
                camera_pose_c2w=camera_pose_c2w,
                image_path=np.array(image_path),
            )

            if metas is None:
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Failed to read image: {image_path}")
                orig_h, orig_w = image.shape[:2]
                infer_h, infer_w = depth.shape[:2]
            else:
                meta = metas[local_idx]
                orig_h, orig_w = meta["orig_h"], meta["orig_w"]
                infer_h, infer_w = meta["infer_h"], meta["infer_w"]
            records.append(
                {
                    "frame_idx": global_start_idx + local_idx,
                    "frame_name": frame_name,
                    "chunk_idx": chunk_idx,
                    "chunk_local_idx": local_idx,
                    "pred_path": save_path,
                    "orig_h": int(orig_h),
                    "orig_w": int(orig_w),
                    "infer_h": int(infer_h),
                    "infer_w": int(infer_w),
                }
            )

        del output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return records

    @staticmethod
    def _load_frame_prediction(pred_path: str) -> dict[str, Any]:
        with np.load(pred_path) as prediction:
            return {
                "depth": prediction["depth"],
                "depth_conf": prediction["depth_conf"],
                "world_points": prediction["world_points"],
                "world_points_conf": prediction["world_points_conf"],
                "intrinsic": prediction["intrinsic"],
                "camera_pose_c2w": prediction["camera_pose_c2w"],
                "image_path": str(prediction["image_path"]),
            }

    def _apply_sim3_and_export_frame(
        self,
        prediction: dict[str, Any],
        sim3: dict[str, Any],
        output_size: tuple[int, int] | None = None,
        source_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        del source_size
        scale = float(sim3["s"])
        rotation = sim3["R"].astype(np.float32)
        translation = sim3["t"].astype(np.float32)

        world_points = prediction["world_points"].astype(np.float32)
        aligned_world_points = scale * (world_points @ rotation.T) + translation[None, None, :]

        sim3_matrix = np.eye(4, dtype=np.float32)
        sim3_matrix[:3, :3] = scale * rotation
        sim3_matrix[:3, 3] = translation
        aligned_c2w = sim3_matrix @ prediction["camera_pose_c2w"].astype(np.float32)
        aligned_c2w[:3, :3] /= scale
        aligned_w2c = np.linalg.inv(aligned_c2w)[:3, :].astype(np.float32)

        pointmap = self._world_to_camera_points(aligned_world_points, aligned_c2w)
        depth = pointmap[..., 2].astype(np.float32)
        conf = prediction["depth_conf"].astype(np.float32)
        intrinsic = prediction["intrinsic"].astype(np.float32)
        source_size_local = depth.shape[:2]

        if output_size is not None and tuple(output_size) != source_size_local:
            depth = self._resize_frame_output(depth, output_size, interpolation=cv2.INTER_LINEAR)
            conf = self._resize_frame_output(conf, output_size, interpolation=cv2.INTER_LINEAR)
            intrinsic = self._rescale_intrinsics(intrinsic, source_size_local, output_size)
            pointmap = self._depth_to_pointmap(depth, intrinsic)

        return {
            "depth": depth.astype(np.float32),
            "conf": conf.astype(np.float32),
            "pointmap": pointmap.astype(np.float32),
            "intrinsics": intrinsic.astype(np.float32),
            "extrinsics": aligned_w2c,
            "image_path": prediction["image_path"],
            "preprocess": {
                "mode": self.preprocess_mode,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "crop_left": 0,
                "crop_top": 0,
                "target_h": int(source_size_local[0]),
                "target_w": int(source_size_local[1]),
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 0,
                "pad_bottom": 0,
            },
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
            raise ValueError("FourRCLongDepthEstimator only supports batch mode.")

        image_paths, frame_names, orig_sizes = self._collect_image_paths(
            path,
            interval=interval,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        if not image_paths:
            print("No images loaded. Cannot run 4RC depth estimation.")
            return None

        target_size = self._compute_target_size(
            orig_sizes[0][0],
            orig_sizes[0][1],
            pixel_limit=pixel_limit,
            max_long_edge=max_long_edge,
            patch_size=self.patch_size,
        )
        resize_policy = (
            f"max_long_edge={max_long_edge}"
            if max_long_edge is not None and max_long_edge > 0
            else f"pixel_limit={pixel_limit}"
            if pixel_limit is not None and pixel_limit > 0
            else "original_patch_multiple"
        )
        print(
            f"{self._log_prefix()} Processing {len(image_paths)} images with 4RC "
            f"preprocess={self.preprocess_mode}, resize_policy={resize_policy}, "
            f"patch_size={self.patch_size}, inference size ({target_size[1]}, {target_size[0]}), "
            f"chunk size {self.chunk_size}, and overlap {self.overlap}."
        )

        if self.temp_root:
            ensure_dir(self.temp_root)
        temp_dir = tempfile.mkdtemp(prefix="4rc_long_", dir=self.temp_root)
        try:
            chunk_indices = self._build_chunk_indices(len(image_paths))
            chunk_records = []
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_indices):
                print(
                    f"{self._log_prefix()} Running 4RC chunk {chunk_idx + 1}/{len(chunk_indices)}: "
                    f"frames {chunk_start} -> {chunk_end - 1}"
                )
                chunk_records.append(
                    self._run_chunk_inference(
                        chunk_idx=chunk_idx,
                        image_paths=image_paths[chunk_start:chunk_end],
                        global_start_idx=chunk_start,
                        temp_dir=temp_dir,
                        target_size=target_size,
                    )
                )

            relative_transforms = []
            for chunk_idx in range(len(chunk_indices) - 1):
                overlap_start = chunk_indices[chunk_idx + 1][0]
                overlap_end = min(chunk_indices[chunk_idx][1], chunk_indices[chunk_idx + 1][1])
                relative_transforms.append(
                    self._estimate_adjacent_chunk_sim3(
                        chunk_records[chunk_idx],
                        chunk_records[chunk_idx + 1],
                        list(range(overlap_start, overlap_end)),
                    )
                )

            return {
                "type": "4rc_long_manifest",
                "image_paths": image_paths,
                "frame_names": frame_names,
                "orig_sizes": orig_sizes,
                "chunk_indices": chunk_indices,
                "chunk_transforms": self._accumulate_chunk_sim3(relative_transforms),
                "frame_records": self._select_best_records(chunk_records, chunk_indices),
                "infer_size": target_size,
                "preprocess_mode": self.preprocess_mode,
                "temp_dir": temp_dir,
            }
        except Exception:
            if self.delete_temp_files:
                self._cleanup_temp(temp_dir)
            raise

    def visualize_depth(
        self,
        image_path: str,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
    ) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        valid_depth = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
        if valid_depth.size == 0:
            normalized_depth = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_min = float(valid_depth.min())
            depth_max = float(valid_depth.max())
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min + 1e-6) * 255
            normalized_depth = np.nan_to_num(normalized_depth, nan=0.0).astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth, colormap)
        colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
        if colored_depth.shape[:2] != image.shape[:2]:
            colored_depth = cv2.resize(
                colored_depth,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return cv2.addWeighted(image, 1 - alpha, colored_depth, alpha, 0)

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
        restore_to_original_resolution = config.depth_output_resolution != "native"
        frame_records = sorted(results["frame_records"], key=lambda item: item["frame_idx"])
        for output_idx, record in enumerate(frame_records):
            prediction = self._load_frame_prediction(record["pred_path"])
            output_size = (
                (record["orig_h"], record["orig_w"])
                if restore_to_original_resolution
                else (record["infer_h"], record["infer_w"])
            )
            exported = self._apply_sim3_and_export_frame(
                prediction,
                results["chunk_transforms"][record["chunk_idx"]],
                output_size=output_size,
            )
            frame_prefix = self._frame_prefix(output_idx, frame_ids=frame_ids, start_frame=start_frame)
            visualized = self.visualize_depth(exported["image_path"], exported["depth"], colormap, alpha)
            cv2.imwrite(
                os.path.join(output_dir, f"depth_visualization_{frame_prefix}.png"),
                cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR),
            )
        print(f"All {len(frame_records)} 4RC visualizations saved to {output_dir}")

    def save_results(
        self,
        results: dict[str, Any],
        output_dir: str,
        start_frame: int = 0,
        frame_ids: list[str] | None = None,
        save_images: bool = True,
        save_depth: bool = True,
        save_conf: bool = True,
        save_pointmap: bool = True,
        save_camera_params: bool = True,
    ) -> None:
        del save_images
        ensure_dir(output_dir)

        if save_depth:
            depth_dir = ensure_dir(os.path.join(output_dir, "depth"))
        if save_conf:
            conf_dir = ensure_dir(os.path.join(output_dir, "conf"))
        if save_pointmap:
            pointmap_dir = ensure_dir(os.path.join(output_dir, "pointmap"))
        if save_camera_params:
            camera_params_dir = ensure_dir(os.path.join(output_dir, "camera_params"))

        restore_to_original_resolution = config.depth_output_resolution != "native"
        frame_records = sorted(results["frame_records"], key=lambda item: item["frame_idx"])
        for output_idx, record in enumerate(frame_records):
            prediction = self._load_frame_prediction(record["pred_path"])
            output_size = (
                (record["orig_h"], record["orig_w"])
                if restore_to_original_resolution
                else (record["infer_h"], record["infer_w"])
            )
            output_size = (int(output_size[0]), int(output_size[1]))
            inference_size = (int(record["infer_h"]), int(record["infer_w"]))
            original_size = (int(record["orig_h"]), int(record["orig_w"]))
            exported = self._apply_sim3_and_export_frame(
                prediction,
                results["chunk_transforms"][record["chunk_idx"]],
                output_size=output_size,
            )
            frame_prefix = self._frame_prefix(output_idx, frame_ids=frame_ids, start_frame=start_frame)

            if save_depth:
                np.save(os.path.join(depth_dir, f"{frame_prefix}.npy"), exported["depth"])
            if save_conf:
                np.save(os.path.join(conf_dir, f"{frame_prefix}.npy"), exported["conf"])
            if save_pointmap:
                np.save(os.path.join(pointmap_dir, f"{frame_prefix}.npy"), exported["pointmap"])
            if save_camera_params:
                preprocess = dict(exported.get("preprocess", {}))
                preprocess.update(
                    {
                        "crop_left": 0,
                        "crop_top": 0,
                        "crop_h": int(original_size[0]),
                        "crop_w": int(original_size[1]),
                    }
                )
                camera_params = build_camera_params(
                    exported["intrinsics"],
                    exported["extrinsics"],
                    output_size,
                    inference_size=inference_size,
                    original_size=original_size,
                    output_resolution_mode=config.depth_output_resolution,
                    preprocess_mode=self.preprocess_mode,
                )
                camera_params["preprocess"].update(
                    {
                        "mode": self.preprocess_mode,
                        "image_size": self.image_size,
                        "patch_size": self.patch_size,
                        **preprocess,
                    }
                )
                with open(os.path.join(camera_params_dir, f"{frame_prefix}.json"), "w") as file:
                    json.dump(camera_params, file, indent=2)

        print(f"All {len(frame_records)} 4RC frames saved to {output_dir}")
        if self.delete_temp_files:
            self._cleanup_temp(results.get("temp_dir"))
