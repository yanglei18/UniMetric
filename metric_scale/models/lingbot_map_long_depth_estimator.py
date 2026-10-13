import gc
import contextlib
import io
import importlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from metric_scale.config.config import config
from metric_scale.config.path_config import ensure_dir
from metric_scale.models.output_metadata import build_camera_params
from metric_scale.utils.utils import file_utils


class LingBotMapLongDepthEstimator:
    """LingBot-Map long-sequence wrapper with resize-only preprocessing."""

    def __init__(
        self,
        project_root: str | None = None,
        checkpoint: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        mode: str = "streaming",
        image_size: int = 518,
        patch_size: int = 14,
        num_scale_frames: int = 8,
        keyframe_interval: int | None = None,
        max_frame_num: int = 1024,
        kv_cache_sliding_window: int = 64,
        camera_num_iterations: int = 4,
        use_sdpa: bool = True,
        offload_to_cpu: bool = True,
        window_size: int = 128,
        overlap_size: int | None = None,
        overlap_keyframes: int | None = None,
        delete_temp_files: bool = True,
        temp_root: str | None = None,
    ) -> None:
        self.project_root = project_root
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.mode = str(mode or "streaming").strip().lower()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_scale_frames = int(num_scale_frames)
        self.keyframe_interval = keyframe_interval
        self.max_frame_num = int(max_frame_num)
        self.kv_cache_sliding_window = int(kv_cache_sliding_window)
        self.camera_num_iterations = int(camera_num_iterations)
        self.use_sdpa = bool(use_sdpa)
        self.offload_to_cpu = bool(offload_to_cpu)
        self.window_size = int(window_size)
        self.overlap_size = overlap_size
        self.overlap_keyframes = overlap_keyframes
        self.delete_temp_files = bool(delete_temp_files)
        self.temp_root = temp_root
        self.to_tensor = transforms.ToTensor()

        if self.mode not in {"streaming", "windowed"}:
            raise ValueError(
                f"Unsupported LingBot-Map mode: {self.mode}. "
                "Expected 'streaming' or 'windowed'."
            )

        if dtype is None:
            if self.device_type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            elif self.device_type == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        self._import_lingbot_map()
        self.model = self._build_model()
        self._load_checkpoint()
        self.model = self.model.to(self.device).eval()

        if self.dtype != torch.float32 and getattr(self.model, "aggregator", None) is not None:
            print(f"Casting LingBot-Map aggregator to {self.dtype} (heads kept in fp32)")
            self.model.aggregator = self.model.aggregator.to(dtype=self.dtype)

    def _log_prefix(self) -> str:
        return f"[{self.device}]"

    @staticmethod
    def _frame_prefix(frame_idx: int, frame_ids: list[str] | None = None, start_frame: int = 0) -> str:
        if frame_ids and frame_idx < len(frame_ids):
            return f"{frame_ids[frame_idx]}_camera1"
        return f"{start_frame + frame_idx:05d}_camera1"

    def _import_lingbot_map(self) -> None:
        if self.project_root and os.path.isdir(self.project_root):
            abs_project_root = os.path.abspath(self.project_root)
            if abs_project_root not in sys.path:
                sys.path.insert(0, abs_project_root)

        module_name = (
            "lingbot_map.models.gct_stream_window"
            if self.mode == "windowed"
            else "lingbot_map.models.gct_stream"
        )
        try:
            model_module = importlib.import_module(module_name)
            pose_module = importlib.import_module("lingbot_map.utils.pose_enc")
        except ImportError as exc:
            raise ImportError(
                "Failed to import LingBot-Map. Set models.lingbot_map_project_root "
                "to the local lingbot-map repository or install it in the environment."
            ) from exc

        self.gct_stream_class = model_module.GCTStream
        self.pose_encoding_to_extri_intri = pose_module.pose_encoding_to_extri_intri

    def _build_model(self) -> torch.nn.Module:
        aggregator_logger = logging.getLogger("lingbot_map.aggregator.base")
        previous_level = aggregator_logger.level
        aggregator_logger.setLevel(logging.ERROR)
        try:
            # LingBot-Map's aggregator attempts torch.load("") for its optional
            # DINO pretrained path before the full LingBot checkpoint is loaded.
            # The full checkpoint below provides the actual weights, so suppress
            # this harmless construction-time warning/noise at the wrapper level.
            with contextlib.redirect_stdout(io.StringIO()):
                return self.gct_stream_class(
                    img_size=self.image_size,
                    patch_size=self.patch_size,
                    enable_3d_rope=True,
                    max_frame_num=self.max_frame_num,
                    kv_cache_sliding_window=self.kv_cache_sliding_window,
                    kv_cache_scale_frames=self.num_scale_frames,
                    kv_cache_cross_frame_special=True,
                    kv_cache_include_scale_frames=True,
                    use_sdpa=self.use_sdpa,
                )
        finally:
            aggregator_logger.setLevel(previous_level)

    def _load_checkpoint(self) -> None:
        if not self.checkpoint:
            return
        checkpoint_path = os.path.abspath(self.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"LingBot-Map checkpoint does not exist: {checkpoint_path}")

        print(f"Loading LingBot-Map checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  LingBot-Map checkpoint loaded.")

    @staticmethod
    def _next_multiple(value: int, divisor: int) -> int:
        return int(math.ceil(float(value) / float(divisor)) * divisor)

    def _compute_target_size(
        self,
        orig_height: int,
        orig_width: int,
        pixel_limit: int | None,
        max_long_edge: int | None = None,
    ) -> tuple[int, int]:
        divisor = self.patch_size
        if max_long_edge is not None and max_long_edge > 0:
            max_side = max(orig_width, orig_height)
            scale = min(1.0, float(max_long_edge) / float(max_side)) if max_side > 0 else 1.0
            target_w = max(1, int(math.floor((orig_width * scale) / divisor))) * divisor
            target_h = max(1, int(math.floor((orig_height * scale) / divisor))) * divisor
            return target_h, target_w

        if pixel_limit is None or pixel_limit <= 0:
            target_h = self._next_multiple(orig_height, divisor)
            target_w = self._next_multiple(orig_width, divisor)
            return target_h, target_w

        pixel_count = max(orig_width * orig_height, 1)
        scale = math.sqrt(float(pixel_limit) / float(pixel_count))
        target_w = orig_width * scale
        target_h = orig_height * scale
        k = max(1, round(target_w / divisor))
        m = max(1, round(target_h / divisor))
        while (k * divisor) * (m * divisor) > pixel_limit:
            if k / max(m, 1) > target_w / max(target_h, 1.0):
                k = max(1, k - 1)
            else:
                m = max(1, m - 1)
        return max(1, m) * divisor, max(1, k) * divisor

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

    def _collect_image_paths(
        self,
        path: str,
        interval: int = 1,
        start_frame: int = 0,
        frame_count: int | None = None,
    ) -> tuple[list[str], list[str], list[tuple[int, int]]]:
        if not os.path.isdir(path):
            raise ValueError(f"Unsupported path. Must be a directory: {path}")

        filenames = file_utils.natural_sorted(
            [
                name
                for name in os.listdir(path)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
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

    def _load_images_resized(
        self,
        image_paths: list[str],
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        target_h, target_w = target_size
        tensors = []
        metas = []
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
    def _tensor_to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().float().cpu()

    @staticmethod
    def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
        if depth.ndim == 5 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return depth.astype(np.float32)

    @staticmethod
    def _squeeze_conf(conf: np.ndarray | None, depth_shape: tuple[int, int, int]) -> np.ndarray:
        if conf is None:
            return np.ones(depth_shape, dtype=np.float32)
        if conf.ndim == 5 and conf.shape[0] == 1:
            conf = conf[0]
        if conf.ndim == 4 and conf.shape[0] == 1:
            conf = conf[0]
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        return conf.astype(np.float32)

    @staticmethod
    def _squeeze_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
        if intrinsics.ndim == 4 and intrinsics.shape[0] == 1:
            intrinsics = intrinsics[0]
        return intrinsics.astype(np.float32)

    @staticmethod
    def _squeeze_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
        if extrinsics.ndim == 4 and extrinsics.shape[0] == 1:
            extrinsics = extrinsics[0]
        return extrinsics.astype(np.float32)

    @staticmethod
    def _to_4x4_pose(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape == (4, 4):
            return pose
        if pose.shape == (3, 4):
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :] = pose
            return pose_4x4
        raise ValueError(f"Unsupported LingBot-Map camera pose shape: {pose.shape}")

    def _resolve_keyframe_interval(self, num_frames: int) -> int:
        if self.keyframe_interval is not None:
            return max(1, int(self.keyframe_interval))
        if self.mode == "streaming" and num_frames > 320:
            return max(1, int(math.ceil(float(num_frames) / 320.0)))
        return 1

    def _run_model_inference(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        keyframe_interval = self._resolve_keyframe_interval(int(images.shape[0]))
        output_device = torch.device("cpu") if self.offload_to_cpu else None
        autocast_enabled = self.device_type == "cuda"

        print(
            f"{self._log_prefix()} Running LingBot-Map {self.mode} inference "
            f"(dtype={self.dtype}, keyframe_interval={keyframe_interval})."
        )

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device_type,
                dtype=self.dtype,
                enabled=autocast_enabled,
            ):
                if self.mode == "streaming":
                    return self.model.inference_streaming(
                        images,
                        num_scale_frames=self.num_scale_frames,
                        keyframe_interval=keyframe_interval,
                        output_device=output_device,
                    )
                return self.model.inference_windowed(
                    images,
                    window_size=self.window_size,
                    overlap_size=self.overlap_size,
                    overlap_keyframes=self.overlap_keyframes,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=output_device,
                )

    def _export_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        images: torch.Tensor,
        metas: list[dict[str, Any]],
        frame_names: list[str],
        temp_dir: str,
    ) -> list[dict[str, Any]]:
        pose_enc = predictions["pose_enc"]
        pose_enc = pose_enc.to(self.device) if pose_enc.device.type == "cpu" else pose_enc
        camera_poses_c2w, intrinsics = self.pose_encoding_to_extri_intri(
            pose_enc,
            images.shape[-2:],
        )

        depth = self._squeeze_depth(self._tensor_to_cpu(predictions["depth"]).numpy())
        depth_conf_tensor = self._tensor_to_cpu(predictions.get("depth_conf"))
        depth_conf = self._squeeze_conf(
            None if depth_conf_tensor is None else depth_conf_tensor.numpy(),
            depth.shape,
        )
        intrinsics_np = self._squeeze_intrinsics(self._tensor_to_cpu(intrinsics).numpy())
        poses_c2w_np = self._squeeze_extrinsics(self._tensor_to_cpu(camera_poses_c2w).numpy())

        prediction_dir = os.path.join(temp_dir, "predictions")
        ensure_dir(prediction_dir)

        records = []
        for frame_idx, meta in enumerate(metas):
            frame_name = frame_names[frame_idx]
            save_path = os.path.join(prediction_dir, f"{frame_name}.npz")
            np.savez_compressed(
                save_path,
                depth=depth[frame_idx].astype(np.float32),
                depth_conf=depth_conf[frame_idx].astype(np.float32),
                intrinsic=intrinsics_np[frame_idx].astype(np.float32),
                camera_pose_c2w=poses_c2w_np[frame_idx].astype(np.float32),
                image_path=np.array(meta["image_path"]),
            )
            records.append(
                {
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "pred_path": save_path,
                    "orig_h": meta["orig_h"],
                    "orig_w": meta["orig_w"],
                    "infer_h": meta["infer_h"],
                    "infer_w": meta["infer_w"],
                }
            )

        return records

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
            raise ValueError("LingBotMapLongDepthEstimator only supports batch mode.")

        image_paths, frame_names, orig_sizes = self._collect_image_paths(
            path,
            interval=interval,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        if not image_paths:
            print("No images loaded. Cannot run LingBot-Map depth estimation.")
            return None

        target_h, target_w = self._compute_target_size(
            orig_sizes[0][0],
            orig_sizes[0][1],
            pixel_limit,
            max_long_edge,
        )
        print(
            f"{self._log_prefix()} Processing {len(image_paths)} images with "
            f"LingBot-Map resize-only input size ({target_w}, {target_h})."
        )

        if self.temp_root:
            ensure_dir(self.temp_root)
        temp_dir = tempfile.mkdtemp(prefix="lingbot_map_long_", dir=self.temp_root)
        try:
            images, metas = self._load_images_resized(image_paths, (target_h, target_w))
            images = images.to(self.device)
            predictions = self._run_model_inference(images)
            records = self._export_predictions(predictions, images, metas, frame_names, temp_dir)

            del images
            del predictions
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "type": "lingbot_map_long_manifest",
                "image_paths": image_paths,
                "frame_names": frame_names,
                "orig_sizes": orig_sizes,
                "frame_records": records,
                "infer_size": (target_h, target_w),
                "temp_dir": temp_dir,
                "preprocess": "resize_only",
            }
        except Exception:
            if self.delete_temp_files:
                self._cleanup_temp(temp_dir)
            raise

    @staticmethod
    def _load_frame_prediction(pred_path: str) -> dict[str, Any]:
        with np.load(pred_path) as prediction:
            return {
                "depth": prediction["depth"],
                "depth_conf": prediction["depth_conf"],
                "intrinsic": prediction["intrinsic"],
                "camera_pose_c2w": prediction["camera_pose_c2w"],
                "image_path": str(prediction["image_path"]),
            }

    def _export_frame(
        self,
        prediction: dict[str, Any],
        output_size: tuple[int, int],
        source_size: tuple[int, int],
    ) -> dict[str, Any]:
        depth = self._resize_frame_output(
            prediction["depth"].astype(np.float32),
            output_size,
            interpolation=cv2.INTER_LINEAR,
        )
        conf = self._resize_frame_output(
            prediction["depth_conf"].astype(np.float32),
            output_size,
            interpolation=cv2.INTER_LINEAR,
        )
        intrinsics = self._rescale_intrinsics(
            prediction["intrinsic"].astype(np.float32),
            source_size,
            output_size,
        )
        pointmap = self._depth_to_pointmap(depth, intrinsics)
        camera_pose_c2w = self._to_4x4_pose(prediction["camera_pose_c2w"])
        extrinsics = np.linalg.inv(camera_pose_c2w)[:3, :].astype(np.float32)

        return {
            "depth": depth,
            "conf": conf,
            "pointmap": pointmap,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "image_path": prediction["image_path"],
        }

    def visualize_depth(
        self,
        image_path: str,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
    ) -> np.ndarray:
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
            source_size = infer_size or (record["infer_h"], record["infer_w"])
            original_size = (int(record["orig_h"]), int(record["orig_w"]))
            output_size = (int(output_size[0]), int(output_size[1]))
            source_size = (int(source_size[0]), int(source_size[1]))
            exported = self._export_frame(prediction, output_size, source_size)
            frame_prefix = self._frame_prefix(output_idx, frame_ids=frame_ids, start_frame=start_frame)
            visualized = self.visualize_depth(exported["image_path"], exported["depth"], colormap, alpha)
            image_bgr = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"depth_visualization_{frame_prefix}.png"), image_bgr)

        print(f"All {len(frame_records)} LingBot-Map visualizations saved to {output_dir}")

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
            source_size = infer_size or (record["infer_h"], record["infer_w"])
            original_size = (int(record["orig_h"]), int(record["orig_w"]))
            output_size = (int(output_size[0]), int(output_size[1]))
            source_size = (int(source_size[0]), int(source_size[1]))
            exported = self._export_frame(prediction, output_size, source_size)
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
                            inference_size=source_size,
                            original_size=original_size,
                            output_resolution_mode=config.depth_output_resolution,
                        ),
                        file,
                        indent=2,
                    )

        print(f"All {len(frame_records)} LingBot-Map frames saved to {output_dir}")
        if self.delete_temp_files:
            self._cleanup_temp(results.get("temp_dir"))

    @staticmethod
    def _cleanup_temp(temp_dir: str | None) -> None:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
