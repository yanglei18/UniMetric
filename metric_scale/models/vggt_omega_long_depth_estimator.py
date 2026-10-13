import gc
import filecmp
import importlib
import json
import math
import os
import re
import sys
import tempfile

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as TF

from metric_scale.config.config import config
from metric_scale.config.path_config import ensure_dir
from metric_scale.models.output_metadata import build_camera_params
from metric_scale.models.vggt_long_depth_estimator import VGGTLongDepthEstimator


class VGGTOmegaLongDepthEstimator(VGGTLongDepthEstimator):
    """Omega wrapper with legacy resizing or official balanced, uncropped inputs."""

    def __init__(
        self,
        project_root: str | None = None,
        checkpoint: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        image_resolution: int = 512,
        preprocess_mode: str = "resize_only",
        enable_alignment: bool = False,
        chunk_size: int = 60,
        overlap: int = 30,
        delete_temp_files: bool = True,
        temp_root: str | None = None,
        max_points_per_frame: int = 50000,
        geometry_intrinsics_source: str = "predicted",
    ) -> None:
        self.project_root = project_root
        self.checkpoint = checkpoint
        self.image_resolution = int(image_resolution)
        self.preprocess_mode = str(preprocess_mode or "resize_only").strip().lower()
        self.enable_alignment = bool(enable_alignment)
        self.geometry_intrinsics_source = str(geometry_intrinsics_source).strip().lower()
        if self.geometry_intrinsics_source not in {"predicted", "dataset"}:
            raise ValueError("Omega geometry_intrinsics_source must be 'predicted' or 'dataset'.")
        self._dataset_intrinsics = {}

        if self.preprocess_mode not in {"resize_only", "balanced"}:
            raise ValueError(
                f"Unsupported VGGT-Omega preprocess mode: {self.preprocess_mode}. "
                "Expected 'resize_only' or 'balanced'."
            )
        if self.preprocess_mode == "balanced" and (
            self.image_resolution <= 0 or self.image_resolution % 16 != 0
        ):
            raise ValueError("Balanced image_resolution must be positive and divisible by 16.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.delete_temp_files = bool(delete_temp_files)
        self.temp_root = temp_root
        self.max_points_per_frame = int(max_points_per_frame)
        self.patch_size = 16
        self.to_tensor = TF.ToTensor()

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

        self._import_vggt_omega()
        self.model = self._build_model()

    def _import_vggt_omega(self) -> None:
        if self.project_root and os.path.isdir(self.project_root):
            abs_project_root = os.path.abspath(self.project_root)
            if abs_project_root not in sys.path:
                sys.path.insert(0, abs_project_root)

        try:
            model_module = importlib.import_module("vggt_omega.models")
            pose_module = importlib.import_module("vggt_omega.utils.pose_enc")
        except ImportError as exc:
            raise ImportError(
                "Failed to import VGGT-Omega. Set models.vggt_omega_project_root "
                "to the local vggt-omega repository or install it in the environment."
            ) from exc

        self.vggt_omega_class = model_module.VGGTOmega
        self.encoding_to_camera = pose_module.encoding_to_camera

    def _build_model(self) -> torch.nn.Module:
        model = self.vggt_omega_class(enable_alignment=self.enable_alignment)
        checkpoint_path = self._resolve_checkpoint_path()
        print(f"Loading VGGT-Omega checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  VGGT-Omega checkpoint loaded.")
        return model.to(self.device).eval()

    def _resolve_checkpoint_path(self) -> str:
        if not self.checkpoint:
            raise FileNotFoundError("VGGT-Omega checkpoint path is empty.")
        checkpoint_path = os.path.abspath(self.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VGGT-Omega checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> dict:
        if not isinstance(checkpoint, dict):
            return checkpoint
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint

    def _compute_target_size(self, orig_height, orig_width, pixel_limit, max_long_edge=None):
        divisor = int(self.patch_size)
        if self.preprocess_mode == "balanced":
            if orig_height <= 0 or orig_width <= 0:
                raise ValueError("Balanced preprocessing requires positive image dimensions.")
            aspect = orig_height / orig_width
            if not 0.5 <= aspect <= 2.0:
                raise ValueError(
                    "Balanced preprocessing supports H/W in [0.5, 2.0] without cropping; "
                    "use resize_only for extreme aspect ratios to preserve SAM coordinates."
                )
            # Match Omega's official token budget and numpy rounding exactly.
            # Legacy pixel_limit/max_long_edge intentionally do not cap this mode.
            tokens = (self.image_resolution // divisor) ** 2
            width_patches = np.sqrt(tokens / aspect)
            height_patches = tokens / width_patches
            return (max(1, int(np.round(height_patches))) * divisor,
                    max(1, int(np.round(width_patches))) * divisor)
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

    def _load_chunk_images_resized(self, image_paths, target_size):
        if self.preprocess_mode == "resize_only":
            return super()._load_chunk_images_resized(image_paths, target_size)
        tensors, metas = [], []
        target_h, target_w = target_size
        for image_path in image_paths:
            with Image.open(image_path) as image:
                orig_w, orig_h = image.size
                if self._compute_target_size(orig_h, orig_w, None) != tuple(target_size):
                    raise ValueError("Balanced inputs must share a target shape; padding is not supported.")
                if image.mode == "RGBA":
                    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, image)
                rgb = image.convert("RGB")
                tensors.append(self.to_tensor(rgb.resize((target_w, target_h), Image.Resampling.BICUBIC)))
            metas.append({"image_path": image_path, "orig_h": orig_h, "orig_w": orig_w,
                          "infer_h": target_h, "infer_w": target_w})
        return torch.stack(tensors, dim=0), metas

    def _prepare_dataset_intrinsics(self, image_paths, orig_sizes, target_size,
                                    calibration_dir, expected_frame_ids=None):
        """Validate copied RGB/frame correspondence; read ONLY K, never GT depth/pose.

        Both supported preprocessors resize without crop/pad. Reject unsupported
        calibration rather than silently falling back to predicted K.
        """
        self._dataset_intrinsics = {}
        if self.geometry_intrinsics_source == "predicted":
            return
        if not calibration_dir or not os.path.isdir(calibration_dir):
            raise ValueError("Dataset intrinsics require the original sequence calibration_dir.")
        if len(image_paths) != len(orig_sizes):
            raise ValueError("Image and original-size counts differ.")
        if expected_frame_ids is not None and len(expected_frame_ids) != len(image_paths):
            raise ValueError("Prepared RGB count does not match requested frame IDs.")
        for index, (image_path, original_size) in enumerate(zip(image_paths, orig_sizes)):
            name = os.path.splitext(os.path.basename(image_path))[0]
            match = re.fullmatch(r"(.+)_camera(\d+)", name)
            if match is None:
                raise ValueError(f"Expected <frame>_camera<id> image name: {image_path}")
            frame_id, camera_id = match.groups()
            if expected_frame_ids is not None and str(expected_frame_ids[index]) not in {frame_id, name}:
                raise ValueError(f"Prepared RGB frame mismatch: {name} != {expected_frame_ids[index]}")
            source_image = os.path.join(calibration_dir, os.path.basename(image_path))
            if not os.path.isfile(source_image) or not filecmp.cmp(image_path, source_image, shallow=False):
                raise ValueError(f"Prepared RGB is not an unchanged dataset image: {image_path}")
            yaml_path = os.path.abspath(os.path.join(calibration_dir, frame_id + ".yaml"))
            with open(yaml_path, encoding="utf-8") as stream:
                calibration = yaml.safe_load(stream)
            try:
                intrinsic = np.asarray(calibration[f"camera{camera_id}"]["intrinsic"], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing or invalid camera{camera_id}.intrinsic in {yaml_path}") from exc
            if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                    or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                    or not np.allclose(intrinsic[2], [0, 0, 1])
                    or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], 0)):
                raise ValueError(f"Expected finite positive-focal zero-skew pinhole K in {yaml_path}")
            self._dataset_intrinsics[image_path] = {
                "intrinsic": self._rescale_intrinsics(intrinsic, original_size, target_size),
                "calibration_path": yaml_path,
            }

    def _run_chunk_inference(
        self,
        chunk_idx: int,
        image_paths: list[str],
        global_start_idx: int,
        temp_dir: str,
        target_size: tuple[int, int],
    ) -> list[dict]:
        images, metas = self._load_chunk_images_resized(image_paths, target_size)
        images = images.to(self.device)

        with torch.inference_mode():
            predictions = self.model(images)
            extrinsic_w2c, intrinsic = self.encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )

        depth_map = predictions["depth"].detach().float().cpu().numpy()
        depth_conf = predictions["depth_conf"].detach().float().cpu().numpy()
        extrinsic_w2c = extrinsic_w2c.detach().float().cpu().numpy()
        intrinsic = intrinsic.detach().float().cpu().numpy()

        if depth_map.shape[0] == 1:
            depth_map = depth_map[0]
        if depth_conf.shape[0] == 1:
            depth_conf = depth_conf[0]
        if extrinsic_w2c.shape[0] == 1:
            extrinsic_w2c = extrinsic_w2c[0]
        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic[0]
        if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
            depth_map = depth_map[..., 0]
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]

        network_intrinsic = intrinsic.copy()
        if self.geometry_intrinsics_source == "dataset":
            intrinsic = np.stack([self._dataset_intrinsics[m["image_path"]]["intrinsic"] for m in metas])
        # Rebuild world points BEFORE overlap Sim3 estimation; no GT poses are used.
        camera_poses = self._w2c_to_c2w(extrinsic_w2c)
        world_points = self._depth_to_world_points(depth_map, intrinsic, extrinsic_w2c)
        world_points_conf = depth_conf.astype(np.float32)

        chunk_dir = os.path.join(temp_dir, "chunks", f"chunk_{chunk_idx:04d}")
        os.makedirs(chunk_dir, exist_ok=True)

        records = []
        for local_idx, meta in enumerate(metas):
            frame_name = os.path.splitext(os.path.basename(meta["image_path"]))[0]
            save_path = os.path.join(chunk_dir, f"{frame_name}.npz")
            np.savez_compressed(
                save_path,
                depth=depth_map[local_idx].astype(np.float32),
                depth_conf=depth_conf[local_idx].astype(np.float32),
                world_points=world_points[local_idx].astype(np.float32),
                world_points_conf=world_points_conf[local_idx].astype(np.float32),
                intrinsic=intrinsic[local_idx].astype(np.float32),
                network_intrinsic=network_intrinsic[local_idx].astype(np.float32),
                geometry_intrinsics_source=np.array(self.geometry_intrinsics_source),
                calibration_path=np.array(self._dataset_intrinsics.get(meta["image_path"], {}).get("calibration_path", "")),
                camera_pose_c2w=camera_poses[local_idx].astype(np.float32),
                image_path=np.array(meta["image_path"]),
                orig_h=np.array(meta["orig_h"], dtype=np.int32),
                orig_w=np.array(meta["orig_w"], dtype=np.int32),
                crop_left=np.array(0, dtype=np.int32),
                crop_top=np.array(0, dtype=np.int32),
                crop_h=np.array(meta["orig_h"], dtype=np.int32),
                crop_w=np.array(meta["orig_w"], dtype=np.int32),
                target_h=np.array(meta["infer_h"], dtype=np.int32),
                target_w=np.array(meta["infer_w"], dtype=np.int32),
                pad_left=np.array(0, dtype=np.int32),
                pad_top=np.array(0, dtype=np.int32),
                pad_right=np.array(0, dtype=np.int32),
                pad_bottom=np.array(0, dtype=np.int32),
                preprocess_mode=np.array(self.preprocess_mode),
                image_resolution=np.array(self.image_resolution, dtype=np.int32),
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
                    "crop_left": 0,
                    "crop_top": 0,
                    "crop_h": meta["orig_h"],
                    "crop_w": meta["orig_w"],
                    "target_h": meta["infer_h"],
                    "target_w": meta["infer_w"],
                    "pad_left": 0,
                    "pad_top": 0,
                    "pad_right": 0,
                    "pad_bottom": 0,
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
                "network_intrinsic": prediction["network_intrinsic"] if "network_intrinsic" in prediction else prediction["intrinsic"],
                "geometry_intrinsics_source": str(prediction["geometry_intrinsics_source"]) if "geometry_intrinsics_source" in prediction else "predicted",
                "calibration_path": str(prediction["calibration_path"]) if "calibration_path" in prediction else "",
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
                "image_resolution": self._scalar_int(prediction["image_resolution"]),
            }
        return result

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
        else:
            output_depth = self._resize_frame_output(depth, output_size, interpolation=cv2.INTER_LINEAR)
            output_conf = self._resize_frame_output(conf, output_size, interpolation=cv2.INTER_LINEAR)
            output_intrinsic = self._rescale_intrinsics(intrinsic, target_size, output_size)

        pointmap = self._depth_to_pointmap(output_depth, output_intrinsic)
        pointmap = pointmap.astype(np.float32)

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
                "image_resolution": int(prediction["image_resolution"]),
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

    def process(self, path, interval=1, pixel_limit=None, max_long_edge=None, mode="batch", start_frame=0, frame_count=None,
                calibration_dir=None, expected_frame_ids=None):
        if mode != "batch":
            raise ValueError("VGGTOmegaLongDepthEstimator only supports batch mode.")

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
        if self.preprocess_mode == "balanced":
            for height, width in orig_sizes:
                if self._compute_target_size(height, width, None) != (target_h, target_w):
                    raise ValueError("Balanced sequence requires a common target shape; padding is not supported.")

        self._prepare_dataset_intrinsics(image_paths, orig_sizes, (target_h, target_w),
                                         calibration_dir, expected_frame_ids)
        print(
            f"{self._log_prefix()} Processing {len(image_paths)} images with VGGT-Omega "
            f"geometry_intrinsics={self.geometry_intrinsics_source}, "
            f"preprocess={self.preprocess_mode}, inference size ({target_w}, {target_h}), "
            f"chunk size {self.chunk_size}, "
            f"and overlap {self.overlap}."
        )
        if self.temp_root:
            ensure_dir(self.temp_root)
        temp_dir = tempfile.mkdtemp(prefix="vggt_omega_long_", dir=self.temp_root)
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
                "type": "vggt_omega_long_manifest",
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
                # 'intrinsics' always describes the exported geometry. Preserve the
                # network's original K separately, at an explicitly named resolution.
                camera_params["geometry_intrinsics_source"] = prediction["geometry_intrinsics_source"]
                camera_params["calibration_path"] = prediction["calibration_path"]
                camera_params["network_intrinsics_inference"] = prediction["network_intrinsic"].tolist()
                camera_params["network_intrinsics_original"] = self._rescale_intrinsics(
                    prediction["network_intrinsic"], inference_size, original_size).tolist()
                camera_params["geometry_experiment"] = "dataset_k_reprojection" if prediction["geometry_intrinsics_source"] == "dataset" else "predicted_k"
                with open(os.path.join(camera_params_dir, f"{frame_prefix}.json"), "w") as file:
                    json.dump(camera_params, file, indent=2)

        print(f"All {len(frame_records)} frames saved to {output_dir}")
        if self.delete_temp_files:
            self._cleanup_temp(results.get("temp_dir"))

    @staticmethod
    def _w2c_to_c2w(extrinsic_w2c: np.ndarray) -> np.ndarray:
        num_frames = extrinsic_w2c.shape[0]
        extrinsic_homo = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))
        extrinsic_homo[:, :3, :4] = extrinsic_w2c.astype(np.float32)
        return np.linalg.inv(extrinsic_homo).astype(np.float32)

    @staticmethod
    def _depth_to_world_points(
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics_w2c: np.ndarray,
    ) -> np.ndarray:
        num_frames, height, width = depth.shape
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x_coords = np.broadcast_to(x_coords[None], (num_frames, height, width))
        y_coords = np.broadcast_to(y_coords[None], (num_frames, height, width))

        fx = intrinsics[:, 0, 0][:, None, None]
        fy = intrinsics[:, 1, 1][:, None, None]
        cx = intrinsics[:, 0, 2][:, None, None]
        cy = intrinsics[:, 1, 2][:, None, None]
        camera_points = np.stack(
            [
                (x_coords - cx) / fx * depth,
                (y_coords - cy) / fy * depth,
                depth,
            ],
            axis=-1,
        )

        rotation = extrinsics_w2c[:, :3, :3]
        translation = extrinsics_w2c[:, :3, 3]
        world_points = np.einsum(
            "sij,shwj->shwi",
            np.transpose(rotation, (0, 2, 1)),
            camera_points - translation[:, None, None, :],
        )
        return world_points.astype(np.float32)
