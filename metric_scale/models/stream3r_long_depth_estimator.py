import gc
import importlib
import json
import math
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

from metric_scale.config.config import config
from metric_scale.config.path_config import ensure_dir
from metric_scale.models.output_metadata import build_camera_params
from metric_scale.models.vggt_long_depth_estimator import VGGTLongDepthEstimator


class STream3RLongDepthEstimator(VGGTLongDepthEstimator):
    """Long-sequence STream3R wrapper using resize-only preprocessing."""

    def __init__(
        self,
        project_root: str | None = None,
        checkpoint: str = "yslan/STream3R",
        device: str | torch.device | None = None,
        mode: str = "causal",
        use_stream_session: bool = True,
        preprocess_mode: str = "resize_only",
        image_size: int = 518,
        patch_size: int = 14,
        dtype: torch.dtype | None = None,
        chunk_size: int = 60,
        overlap: int = 20,
        delete_temp_files: bool = True,
        temp_root: str | None = None,
        max_points_per_frame: int = 50000,
    ) -> None:
        self.project_root = project_root
        self.checkpoint = checkpoint or "yslan/STream3R"
        self.mode = str(mode or "causal").strip().lower()
        self.use_stream_session = bool(use_stream_session)
        self.preprocess_mode = str(preprocess_mode or "resize_only").strip().lower()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.delete_temp_files = bool(delete_temp_files)
        self.temp_root = temp_root
        self.max_points_per_frame = int(max_points_per_frame)
        self.to_tensor = TF.ToTensor()

        if self.mode not in {"causal", "window", "full"}:
            raise ValueError("STream3R mode must be one of: causal, window, full.")
        if self.use_stream_session and self.mode == "full":
            raise ValueError("STream3R StreamSession supports only causal/window modes.")
        if self.preprocess_mode not in {"resize_only", "crop", "pad"}:
            raise ValueError("STream3R preprocess mode must be one of: resize_only, crop, pad.")
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than chunk size ({self.chunk_size})."
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

        self._import_stream3r()
        self.model = self._build_model()

    def _import_stream3r(self) -> None:
        if self.project_root and os.path.isdir(self.project_root):
            abs_project_root = os.path.abspath(self.project_root)
            if abs_project_root not in sys.path:
                sys.path.insert(0, abs_project_root)

        try:
            model_module = importlib.import_module("stream3r.models.stream3r")
            session_module = importlib.import_module("stream3r.stream_session")
            pose_module = importlib.import_module("stream3r.models.components.utils.pose_enc")
        except ImportError as exc:
            raise ImportError(
                "Failed to import STream3R. Set models.stream3r_project_root "
                "to the local STream3R repository or install NIRVANALAN/STream3R."
            ) from exc

        self.stream3r_class = model_module.STream3R
        self.stream_session_class = session_module.StreamSession
        self.pose_encoding_to_extri_intri = pose_module.pose_encoding_to_extri_intri

    def _build_model(self) -> torch.nn.Module:
        print(f"Loading STream3R checkpoint: {self.checkpoint}")
        model = self.stream3r_class.from_pretrained(self.checkpoint).to(self.device).eval()
        print("  STream3R checkpoint loaded.")
        return model

    @staticmethod
    def _round_to_patch(value: float, patch_size: int) -> int:
        return max(patch_size, int(round(float(value) / patch_size)) * patch_size)

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
            target_h = STream3RLongDepthEstimator._next_multiple(orig_height, 14)
            target_w = STream3RLongDepthEstimator._next_multiple(orig_width, 14)
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

    def _load_official_image(self, image_path: str, target_size: tuple[int, int] | None = None) -> tuple[torch.Tensor, dict]:
        with Image.open(image_path) as image:
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            rgb = image.convert("RGB")

        orig_w, orig_h = rgb.size
        if self.preprocess_mode == "resize_only":
            if target_size is None:
                raise ValueError("resize_only preprocessing requires an explicit target size.")
            target_h, target_w = target_size
            resized = rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
            tensor = self.to_tensor(resized)
            crop_left = 0
            crop_top = 0
            crop_w = orig_w
            crop_h = orig_h
            pad_top = pad_bottom = pad_left = pad_right = 0
        elif self.preprocess_mode == "pad":
            if orig_w >= orig_h:
                resized_w = self.image_size
                resized_h = self._round_to_patch(orig_h * (resized_w / max(orig_w, 1)), self.patch_size)
            else:
                resized_h = self.image_size
                resized_w = self._round_to_patch(orig_w * (resized_h / max(orig_h, 1)), self.patch_size)

            resized = rgb.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
            tensor = self.to_tensor(resized)
            pad_h = self.image_size - tensor.shape[1]
            pad_w = self.image_size - tensor.shape[2]
            pad_top = max(pad_h // 2, 0)
            pad_bottom = max(pad_h - pad_top, 0)
            pad_left = max(pad_w // 2, 0)
            pad_right = max(pad_w - pad_left, 0)
            if pad_h > 0 or pad_w > 0:
                tensor = torch.nn.functional.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            crop_left = 0
            crop_top = 0
            crop_w = orig_w
            crop_h = orig_h
            target_h = resized_h
            target_w = resized_w
        else:
            resized_w = self.image_size
            resized_h = self._round_to_patch(orig_h * (resized_w / max(orig_w, 1)), self.patch_size)
            resized = rgb.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
            tensor = self.to_tensor(resized)

            crop_top_resized = 0
            if resized_h > self.image_size:
                crop_top_resized = (resized_h - self.image_size) // 2
                tensor = tensor[:, crop_top_resized : crop_top_resized + self.image_size, :]
            scale_y = float(resized_h) / float(max(orig_h, 1))
            crop_top = int(round(crop_top_resized / max(scale_y, 1e-8)))
            crop_h = int(round(tensor.shape[1] / max(scale_y, 1e-8)))
            crop_h = max(1, min(orig_h - crop_top, crop_h))
            crop_left = 0
            crop_w = orig_w
            target_h = int(tensor.shape[1])
            target_w = int(tensor.shape[2])
            pad_top = pad_bottom = pad_left = pad_right = 0

        meta = {
            "image_path": image_path,
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
            "crop_left": int(crop_left),
            "crop_top": int(crop_top),
            "crop_h": int(crop_h),
            "crop_w": int(crop_w),
            "target_h": int(target_h),
            "target_w": int(target_w),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
            "pad_right": int(pad_right),
            "pad_bottom": int(pad_bottom),
            "infer_h": int(tensor.shape[1]),
            "infer_w": int(tensor.shape[2]),
        }
        return tensor, meta

    def _load_chunk_images_official(
        self,
        image_paths: list[str],
        target_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        tensors = []
        metas = []
        shapes = set()
        for image_path in image_paths:
            tensor, meta = self._load_official_image(image_path, target_size=target_size)
            tensors.append(tensor)
            metas.append(meta)
            shapes.add((tensor.shape[1], tensor.shape[2]))

        if len(shapes) <= 1:
            return torch.stack(tensors, dim=0), metas

        max_h = max(shape[0] for shape in shapes)
        max_w = max(shape[1] for shape in shapes)
        padded_tensors = []
        for tensor, meta in zip(tensors, metas):
            extra_h = max_h - tensor.shape[1]
            extra_w = max_w - tensor.shape[2]
            extra_top = extra_h // 2
            extra_bottom = extra_h - extra_top
            extra_left = extra_w // 2
            extra_right = extra_w - extra_left
            if extra_h > 0 or extra_w > 0:
                tensor = torch.nn.functional.pad(
                    tensor,
                    (extra_left, extra_right, extra_top, extra_bottom),
                    mode="constant",
                    value=1.0,
                )
            meta.update(
                {
                    "pad_left": int(meta["pad_left"] + extra_left),
                    "pad_top": int(meta["pad_top"] + extra_top),
                    "pad_right": int(meta["pad_right"] + extra_right),
                    "pad_bottom": int(meta["pad_bottom"] + extra_bottom),
                    "infer_h": int(max_h),
                    "infer_w": int(max_w),
                }
            )
            padded_tensors.append(tensor)

        return torch.stack(padded_tensors, dim=0), metas

    @staticmethod
    def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _squeeze_batch(value: torch.Tensor | np.ndarray) -> np.ndarray:
        array = STream3RLongDepthEstimator._to_numpy(value)
        if array.ndim >= 1 and array.shape[0] == 1:
            array = array[0]
        return array.astype(np.float32)

    @staticmethod
    def _w2c_to_c2w(extrinsic_w2c: np.ndarray) -> np.ndarray:
        poses = np.tile(np.eye(4, dtype=np.float32)[None], (extrinsic_w2c.shape[0], 1, 1))
        poses[:, :3, :4] = extrinsic_w2c.astype(np.float32)
        return np.linalg.inv(poses).astype(np.float32)

    def _run_model(self, images: torch.Tensor) -> dict:
        autocast_enabled = self.device_type == "cuda"
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device_type,
                dtype=self.dtype,
                enabled=autocast_enabled,
            ):
                if self.use_stream_session:
                    session = self.stream_session_class(self.model, mode=self.mode)
                    predictions = None
                    for index in range(images.shape[0]):
                        predictions = session.forward_stream(images[index : index + 1])
                    if predictions is None:
                        raise ValueError("STream3R StreamSession produced no predictions.")
                else:
                    predictions = self.model(images, mode=self.mode)
        return predictions

    def _run_chunk_inference(
        self,
        chunk_idx: int,
        image_paths: list[str],
        global_start_idx: int,
        temp_dir: str,
        target_size: tuple[int, int] | None,
    ) -> list[dict]:
        images, metas = self._load_chunk_images_official(image_paths, target_size=target_size)
        images = images.to(self.device)
        predictions = self._run_model(images)

        extrinsic_w2c, intrinsic = self.pose_encoding_to_extri_intri(
            predictions["pose_enc"].float(),
            images.shape[-2:],
        )
        depth = self._squeeze_batch(predictions["depth"])
        depth_conf = self._squeeze_batch(predictions["depth_conf"])
        world_points = self._squeeze_batch(predictions["world_points"])
        world_points_conf = self._squeeze_batch(predictions["world_points_conf"])
        extrinsic_w2c = self._squeeze_batch(extrinsic_w2c)
        intrinsic = self._squeeze_batch(intrinsic)

        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]
        if world_points_conf.ndim == 4 and world_points_conf.shape[-1] == 1:
            world_points_conf = world_points_conf[..., 0]

        camera_poses = self._w2c_to_c2w(extrinsic_w2c)
        chunk_dir = os.path.join(temp_dir, "chunks", f"chunk_{chunk_idx:04d}")
        ensure_dir(chunk_dir)

        records = []
        for local_idx, meta in enumerate(metas):
            frame_name = os.path.splitext(os.path.basename(meta["image_path"]))[0]
            save_path = os.path.join(chunk_dir, f"{frame_name}.npz")
            np.savez_compressed(
                save_path,
                depth=depth[local_idx].astype(np.float32),
                depth_conf=depth_conf[local_idx].astype(np.float32),
                world_points=world_points[local_idx].astype(np.float32),
                world_points_conf=world_points_conf[local_idx].astype(np.float32),
                intrinsic=intrinsic[local_idx].astype(np.float32),
                camera_pose_c2w=camera_poses[local_idx].astype(np.float32),
                image_path=np.array(meta["image_path"]),
                orig_h=np.array(meta["orig_h"], dtype=np.int32),
                orig_w=np.array(meta["orig_w"], dtype=np.int32),
                crop_left=np.array(meta["crop_left"], dtype=np.int32),
                crop_top=np.array(meta["crop_top"], dtype=np.int32),
                crop_h=np.array(meta["crop_h"], dtype=np.int32),
                crop_w=np.array(meta["crop_w"], dtype=np.int32),
                target_h=np.array(meta["target_h"], dtype=np.int32),
                target_w=np.array(meta["target_w"], dtype=np.int32),
                pad_left=np.array(meta["pad_left"], dtype=np.int32),
                pad_top=np.array(meta["pad_top"], dtype=np.int32),
                pad_right=np.array(meta["pad_right"], dtype=np.int32),
                pad_bottom=np.array(meta["pad_bottom"], dtype=np.int32),
                preprocess_mode=np.array(self.preprocess_mode),
                image_size=np.array(self.image_size, dtype=np.int32),
                patch_size=np.array(self.patch_size, dtype=np.int32),
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
                    "crop_left": meta["crop_left"],
                    "crop_top": meta["crop_top"],
                    "crop_h": meta["crop_h"],
                    "crop_w": meta["crop_w"],
                    "target_h": meta["target_h"],
                    "target_w": meta["target_w"],
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
    def _scalar_int(value: np.ndarray | int) -> int:
        return int(np.asarray(value).item())

    def _load_frame_prediction(self, pred_path: str) -> dict:
        with np.load(pred_path) as prediction:
            result = {
                "depth": prediction["depth"],
                "depth_conf": prediction["depth_conf"],
                "world_points": prediction["world_points"],
                "world_points_conf": prediction["world_points_conf"],
                "intrinsic": prediction["intrinsic"],
                "camera_pose_c2w": prediction["camera_pose_c2w"],
                "image_path": str(prediction["image_path"]),
                "orig_h": self._scalar_int(prediction["orig_h"]),
                "orig_w": self._scalar_int(prediction["orig_w"]),
                "crop_left": self._scalar_int(prediction["crop_left"]),
                "crop_top": self._scalar_int(prediction["crop_top"]),
                "crop_h": self._scalar_int(prediction["crop_h"]),
                "crop_w": self._scalar_int(prediction["crop_w"]),
                "target_h": self._scalar_int(prediction["target_h"]),
                "target_w": self._scalar_int(prediction["target_w"]),
                "pad_left": self._scalar_int(prediction["pad_left"]),
                "pad_top": self._scalar_int(prediction["pad_top"]),
                "pad_right": self._scalar_int(prediction["pad_right"]),
                "pad_bottom": self._scalar_int(prediction["pad_bottom"]),
                "preprocess_mode": str(prediction["preprocess_mode"]),
                "image_size": self._scalar_int(prediction["image_size"]),
                "patch_size": self._scalar_int(prediction["patch_size"]),
            }
        return self._remove_padding(result)

    @staticmethod
    def _remove_padding(prediction: dict) -> dict:
        row_slice = slice(prediction["pad_top"], prediction["pad_top"] + prediction["target_h"])
        col_slice = slice(prediction["pad_left"], prediction["pad_left"] + prediction["target_w"])

        unpadded = dict(prediction)
        for key in ("depth", "depth_conf", "world_points", "world_points_conf"):
            unpadded[key] = prediction[key][row_slice, col_slice].astype(np.float32)

        intrinsic = prediction["intrinsic"].astype(np.float32).copy()
        intrinsic[0, 2] -= float(prediction["pad_left"])
        intrinsic[1, 2] -= float(prediction["pad_top"])
        unpadded["intrinsic"] = intrinsic
        return unpadded

    def _apply_sim3_and_export_frame(self, prediction, sim3, output_size=None, source_size=None):
        del source_size
        scale = float(sim3["s"])
        rotation = sim3["R"]
        translation = sim3["t"]

        depth = (prediction["depth"] * scale).astype(np.float32)
        conf = prediction["depth_conf"].astype(np.float32)
        intrinsic = prediction["intrinsic"].astype(np.float32)
        target_size = (int(prediction["target_h"]), int(prediction["target_w"]))

        if output_size is None or tuple(output_size) == target_size:
            output_depth = depth
            output_conf = conf
            output_intrinsic = intrinsic
            valid_mask = np.ones(output_depth.shape, dtype=bool)
        elif tuple(output_size) == (int(prediction["orig_h"]), int(prediction["orig_w"])):
            crop_size = (int(prediction["crop_h"]), int(prediction["crop_w"]))
            crop_depth = self._resize_frame_output(depth, crop_size, interpolation=cv2.INTER_LINEAR)
            crop_conf = self._resize_frame_output(conf, crop_size, interpolation=cv2.INTER_LINEAR)
            crop_intrinsic = self._rescale_intrinsics(intrinsic, target_size, crop_size)

            output_depth = np.zeros(output_size, dtype=np.float32)
            output_conf = np.zeros(output_size, dtype=np.float32)
            valid_mask = np.zeros(output_size, dtype=bool)

            top = int(prediction["crop_top"])
            left = int(prediction["crop_left"])
            bottom = min(top + crop_size[0], output_size[0])
            right = min(left + crop_size[1], output_size[1])
            valid_h = max(bottom - top, 0)
            valid_w = max(right - left, 0)
            output_depth[top:bottom, left:right] = crop_depth[:valid_h, :valid_w]
            output_conf[top:bottom, left:right] = crop_conf[:valid_h, :valid_w]
            valid_mask[top:bottom, left:right] = True

            output_intrinsic = crop_intrinsic.astype(np.float32)
            output_intrinsic[0, 2] += float(left)
            output_intrinsic[1, 2] += float(top)
        else:
            output_depth = self._resize_frame_output(depth, output_size, interpolation=cv2.INTER_LINEAR)
            output_conf = self._resize_frame_output(conf, output_size, interpolation=cv2.INTER_LINEAR)
            output_intrinsic = self._rescale_intrinsics(intrinsic, target_size, output_size)
            valid_mask = np.ones(output_depth.shape, dtype=bool)

        pointmap = self._depth_to_pointmap(output_depth, output_intrinsic)
        pointmap = pointmap.astype(np.float32)
        pointmap[~valid_mask] = np.nan

        sim3_matrix = np.eye(4, dtype=np.float32)
        sim3_matrix[:3, :3] = scale * rotation
        sim3_matrix[:3, 3] = translation

        aligned_c2w = sim3_matrix @ prediction["camera_pose_c2w"]
        aligned_c2w = aligned_c2w.astype(np.float32)
        aligned_c2w[:3, :3] /= scale
        aligned_w2c = np.linalg.inv(aligned_c2w)[:3, :].astype(np.float32)

        return {
            "depth": output_depth,
            "conf": output_conf,
            "pointmap": pointmap,
            "intrinsics": output_intrinsic,
            "extrinsics": aligned_w2c,
            "image_path": prediction["image_path"],
            "preprocess": {
                "mode": prediction["preprocess_mode"],
                "image_size": int(prediction["image_size"]),
                "patch_size": int(prediction["patch_size"]),
                "crop_left": int(prediction["crop_left"]),
                "crop_top": int(prediction["crop_top"]),
                "crop_h": int(prediction["crop_h"]),
                "crop_w": int(prediction["crop_w"]),
                "target_h": int(prediction["target_h"]),
                "target_w": int(prediction["target_w"]),
                "pad_left": int(prediction["pad_left"]),
                "pad_top": int(prediction["pad_top"]),
                "pad_right": int(prediction["pad_right"]),
                "pad_bottom": int(prediction["pad_bottom"]),
            },
        }

    def process(self, path, interval=1, pixel_limit=None, max_long_edge=None, mode="batch", start_frame=0, frame_count=None):
        if mode != "batch":
            raise ValueError("STream3RLongDepthEstimator only supports batch mode.")

        image_paths, frame_names, orig_sizes = self._collect_image_paths(
            path,
            interval=interval,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        if not image_paths:
            print("No images loaded. Cannot run depth estimation.")
            return None

        target_size = None
        size_message = f"image_size={self.image_size}"
        if self.preprocess_mode == "resize_only":
            target_size = self._compute_target_size(
                orig_sizes[0][0],
                orig_sizes[0][1],
                pixel_limit,
                max_long_edge,
            )
            size_message = f"inference size ({target_size[1]}, {target_size[0]})"

        print(
            f"{self._log_prefix()} Processing {len(image_paths)} images with STream3R "
            f"mode={self.mode}, streaming={self.use_stream_session}, "
            f"preprocess={self.preprocess_mode}, {size_message}, "
            f"chunk size {self.chunk_size}, and overlap {self.overlap}."
        )
        if self.temp_root:
            ensure_dir(self.temp_root)
        temp_dir = tempfile.mkdtemp(prefix="stream3r_long_", dir=self.temp_root)
        try:
            chunk_indices = self._build_chunk_indices(len(image_paths))
            chunk_records = []

            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_indices):
                print(
                    f"{self._log_prefix()} Running STream3R chunk {chunk_idx + 1}/{len(chunk_indices)}: "
                    f"frames {chunk_start} -> {chunk_end - 1}"
                )
                records = self._run_chunk_inference(
                    chunk_idx=chunk_idx,
                    image_paths=image_paths[chunk_start:chunk_end],
                    global_start_idx=chunk_start,
                    temp_dir=temp_dir,
                    target_size=target_size,
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
                "type": "stream3r_long_manifest",
                "image_paths": image_paths,
                "frame_names": frame_names,
                "orig_sizes": orig_sizes,
                "chunk_indices": chunk_indices,
                "chunk_transforms": self._accumulate_chunk_sim3(relative_transforms),
                "frame_records": selected_records,
                "infer_size": target_size,
                "temp_dir": temp_dir,
            }
        except Exception:
            if self.delete_temp_files:
                self._cleanup_temp(temp_dir)
            raise

    def save_visualizations(
        self,
        results,
        output_dir,
        start_frame=0,
        frame_ids=None,
        colormap=cv2.COLORMAP_JET,
        alpha=0.5,
    ):
        ensure_dir(output_dir)
        frame_records = sorted(results["frame_records"], key=lambda record: record["frame_idx"])
        restore_to_original_resolution = config.depth_output_resolution != "native"

        for output_idx, record in enumerate(frame_records):
            prediction = self._load_frame_prediction(record["pred_path"])
            output_size = (
                (record["orig_h"], record["orig_w"])
                if restore_to_original_resolution
                else (prediction["target_h"], prediction["target_w"])
            )
            exported = self._apply_sim3_and_export_frame(
                prediction,
                results["chunk_transforms"][record["chunk_idx"]],
                output_size=output_size,
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
        restore_to_original_resolution = config.depth_output_resolution != "native"
        output_resolution_mode = str(config.depth_output_resolution)

        for output_idx, record in enumerate(frame_records):
            prediction = self._load_frame_prediction(record["pred_path"])
            output_size = (
                (record["orig_h"], record["orig_w"])
                if restore_to_original_resolution
                else (prediction["target_h"], prediction["target_w"])
            )
            original_size = (int(record["orig_h"]), int(record["orig_w"]))
            inference_size = (int(record["infer_h"]), int(record["infer_w"]))
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
                camera_params = build_camera_params(
                    exported["intrinsics"],
                    exported["extrinsics"],
                    output_size,
                    inference_size=inference_size,
                    original_size=original_size,
                    output_resolution_mode=output_resolution_mode,
                    preprocess_mode=exported["preprocess"]["mode"],
                )
                camera_params["preprocess"].update(exported["preprocess"])
                with open(os.path.join(camera_params_dir, f"{frame_prefix}.json"), "w") as file:
                    json.dump(camera_params, file, indent=2)

        print(f"All {len(frame_records)} frames saved to {output_dir}")
        if self.delete_temp_files:
            self._cleanup_temp(results.get("temp_dir"))

    @staticmethod
    def _cleanup_temp(temp_dir: str | None) -> None:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
