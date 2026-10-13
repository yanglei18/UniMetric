import gc
import importlib
import json
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


class DVLTLongDepthEstimator(VGGTLongDepthEstimator):
    """Long-sequence DVLT wrapper using Metric-Depth resize-only preprocessing."""

    def __init__(
        self,
        project_root: str | None = None,
        checkpoint: str = "nvidia/dvlt",
        device: str | torch.device | None = None,
        image_size: int = 504,
        patch_size: int = 14,
        mixed_precision: str = "bf16",
        chunk_size: int = 60,
        overlap: int = 20,
        delete_temp_files: bool = True,
        temp_root: str | None = None,
        max_points_per_frame: int = 50000,
    ) -> None:
        self.project_root = project_root
        self.checkpoint = checkpoint or "nvidia/dvlt"
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.mixed_precision = str(mixed_precision or "bf16").strip().lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.delete_temp_files = bool(delete_temp_files)
        self.temp_root = temp_root
        self.max_points_per_frame = int(max_points_per_frame)

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than chunk size ({self.chunk_size})."
            )
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported DVLT mixed precision: {self.mixed_precision}. "
                "Expected one of: no, fp16, bf16."
            )
        if self.device_type != "cuda":
            self.mixed_precision = "no"

        self._import_dvlt()
        self.accelerator = self.accelerator_class(mixed_precision=self.mixed_precision)
        self.model = self._build_model()

    def _import_dvlt(self) -> None:
        if self.project_root and os.path.isdir(self.project_root):
            abs_project_root = os.path.abspath(self.project_root)
            if abs_project_root not in sys.path:
                sys.path.insert(0, abs_project_root)

        try:
            accelerator_module = importlib.import_module("accelerate")
            constants_module = importlib.import_module("dvlt.common.constants")
            model_module = importlib.import_module("dvlt.model.dvlt.model")
        except ImportError as exc:
            raise ImportError(
                "Failed to import DVLT. Set models.dvlt_project_root to the local "
                "dvlt repository or install nv-tlabs/dvlt in the active environment."
            ) from exc

        self.accelerator_class = accelerator_module.Accelerator
        self.image_field = constants_module.DataField.IMAGES
        self.dvlt_class = getattr(model_module, "DVLT", None) or getattr(model_module, "DVLTModel")

    def _build_model(self) -> Any:
        model = self.dvlt_class(img_size=self.image_size, patch_size=self.patch_size, load_patch_embed_weights=False)
        print(f"Loading DVLT checkpoint: {self.checkpoint}")
        model.load_pretrained(self.checkpoint, strict=True)
        model.setup_test(self.accelerator)
        print("  DVLT checkpoint loaded.")
        return model

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _to_4x4_c2w(camera_to_worlds: Any) -> np.ndarray:
        c2w = DVLTLongDepthEstimator._as_numpy(camera_to_worlds)
        if c2w.ndim == 4 and c2w.shape[0] == 1:
            c2w = c2w[0]
        if c2w.ndim != 3:
            raise ValueError(f"Unsupported DVLT camera_to_worlds shape: {c2w.shape}")
        if c2w.shape[-2:] == (4, 4):
            return c2w.astype(np.float32)
        if c2w.shape[-2:] == (3, 4):
            poses = np.tile(np.eye(4, dtype=np.float32)[None], (c2w.shape[0], 1, 1))
            poses[:, :3, :4] = c2w
            return poses
        raise ValueError(f"Unsupported DVLT camera_to_worlds shape: {c2w.shape}")

    @staticmethod
    def _squeeze_depth(value: Any) -> np.ndarray:
        array = DVLTLongDepthEstimator._as_numpy(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 3:
            raise ValueError(f"Unsupported DVLT depth shape: {array.shape}")
        return array.astype(np.float32)

    @staticmethod
    def _squeeze_points(value: Any) -> np.ndarray:
        array = DVLTLongDepthEstimator._as_numpy(value)
        if array.ndim == 5 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"Unsupported DVLT world_points shape: {array.shape}")
        return array.astype(np.float32)

    @staticmethod
    def _squeeze_conf(value: Any, depth_shape: tuple[int, int, int]) -> np.ndarray:
        if value is None:
            return np.ones(depth_shape, dtype=np.float32)
        array = DVLTLongDepthEstimator._as_numpy(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.shape != depth_shape:
            return np.ones(depth_shape, dtype=np.float32)
        return array.astype(np.float32)

    @staticmethod
    def _squeeze_intrinsics(value: Any) -> np.ndarray:
        array = DVLTLongDepthEstimator._as_numpy(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[-2:] != (3, 3):
            raise ValueError(f"Unsupported DVLT intrinsics shape: {array.shape}")
        return array.astype(np.float32)

    @staticmethod
    def _make_divisible(value: int, divisor: int) -> int:
        return max(divisor, (int(value) // divisor) * divisor)

    def _build_resize_only_metas(
        self,
        image_paths: list[str],
        target_size: tuple[int, int],
    ) -> list[dict[str, int | str]]:
        target_h, target_w = target_size
        metas = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                orig_w, orig_h = image.convert("RGB").size

            metas.append(
                {
                    "image_path": image_path,
                    "orig_h": orig_h,
                    "orig_w": orig_w,
                    "resized_h": target_h,
                    "resized_w": target_w,
                    "target_h": target_h,
                    "target_w": target_w,
                    "crop_top": 0,
                    "crop_left": 0,
                    "pad_top": 0,
                    "pad_left": 0,
                    "pad_bottom": 0,
                    "pad_right": 0,
                    "infer_h": target_h,
                    "infer_w": target_w,
                }
            )
        return metas

    def _load_chunk_batch_resized(
        self,
        image_paths: list[str],
        target_size: tuple[int, int],
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        target_h, target_w = target_size
        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                array = np.asarray(rgb, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append(tensor)

        images = torch.stack(tensors, dim=0).unsqueeze(0).to(self.accelerator.device)
        valid_pixels = torch.ones(
            (1, len(tensors), target_h, target_w),
            dtype=torch.bool,
            device=self.accelerator.device,
        )
        batch = {self.image_field: images, "gradio_valid_pixels": valid_pixels}
        metas = self._build_resize_only_metas(image_paths, target_size)
        return batch, metas

    @staticmethod
    def _extract_depth_conf(predictions: dict, depth_shape: tuple[int, int, int]) -> np.ndarray:
        for key in ("depth_conf", "depth_confidence", "conf", "confidence", "confidences"):
            if key in predictions:
                return DVLTLongDepthEstimator._squeeze_conf(predictions[key], depth_shape)
        return np.ones(depth_shape, dtype=np.float32)

    def _run_chunk_inference(
        self,
        chunk_idx: int,
        image_paths: list[str],
        global_start_idx: int,
        temp_dir: str,
        target_size: tuple[int, int],
    ) -> list[dict]:
        batch, metas = self._load_chunk_batch_resized(image_paths, target_size)

        with torch.no_grad(), self.accelerator.autocast():
            predictions = self.model.predict(batch, self.accelerator)

        cameras = predictions["cameras"][0]
        camera_poses = self._to_4x4_c2w(cameras.camera_to_worlds)
        intrinsics = self._squeeze_intrinsics(cameras.get_intrinsics_matrices())
        depth = self._squeeze_depth(predictions["depths"][0])
        world_points = self._squeeze_points(predictions["world_points"][0])
        depth_conf = self._extract_depth_conf(predictions, depth.shape)
        world_points_conf = depth_conf.astype(np.float32)

        chunk_dir = os.path.join(temp_dir, "chunks", f"chunk_{chunk_idx:04d}")
        ensure_dir(chunk_dir)

        records = []
        for local_idx, meta in enumerate(metas):
            frame_name = os.path.splitext(os.path.basename(str(meta["image_path"])))[0]
            save_path = os.path.join(chunk_dir, f"{frame_name}.npz")
            np.savez_compressed(
                save_path,
                depth=depth[local_idx].astype(np.float32),
                depth_conf=depth_conf[local_idx].astype(np.float32),
                world_points=world_points[local_idx].astype(np.float32),
                world_points_conf=world_points_conf[local_idx].astype(np.float32),
                intrinsic=intrinsics[local_idx].astype(np.float32),
                camera_pose_c2w=camera_poses[local_idx].astype(np.float32),
                image_path=np.array(meta["image_path"]),
                **{key: np.array(value) for key, value in meta.items() if key != "image_path"},
            )
            records.append(
                {
                    "frame_idx": global_start_idx + local_idx,
                    "frame_name": frame_name,
                    "chunk_idx": chunk_idx,
                    "chunk_local_idx": local_idx,
                    "pred_path": save_path,
                    **{key: meta[key] for key in meta if key != "image_path"},
                }
            )

        del batch
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
            result = {
                "depth": prediction["depth"],
                "depth_conf": prediction["depth_conf"],
                "world_points": prediction["world_points"],
                "world_points_conf": prediction["world_points_conf"],
                "intrinsic": prediction["intrinsic"],
                "camera_pose_c2w": prediction["camera_pose_c2w"],
                "image_path": str(prediction["image_path"]),
            }
            for key in (
                "orig_h",
                "orig_w",
                "resized_h",
                "resized_w",
                "target_h",
                "target_w",
                "crop_top",
                "crop_left",
                "pad_top",
                "pad_left",
                "pad_bottom",
                "pad_right",
                "infer_h",
                "infer_w",
            ):
                result[key] = self._scalar_int(prediction[key])
        return self._remove_padding(result)

    @staticmethod
    def _remove_padding(prediction: dict[str, Any]) -> dict[str, Any]:
        pad_top = int(prediction["pad_top"])
        pad_left = int(prediction["pad_left"])
        target_h = int(prediction["target_h"])
        target_w = int(prediction["target_w"])
        row_slice = slice(pad_top, pad_top + target_h)
        col_slice = slice(pad_left, pad_left + target_w)

        unpadded = dict(prediction)
        for key in ("depth", "depth_conf", "world_points", "world_points_conf"):
            unpadded[key] = prediction[key][row_slice, col_slice].astype(np.float32)
        intrinsic = prediction["intrinsic"].astype(np.float32).copy()
        intrinsic[0, 2] -= float(pad_left)
        intrinsic[1, 2] -= float(pad_top)
        unpadded["intrinsic"] = intrinsic
        return unpadded

    def _resolve_output_geometry(
        self,
        prediction: dict[str, Any],
        output_size: tuple[int, int] | None,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        target_size = (int(prediction["target_h"]), int(prediction["target_w"]))
        if output_size is None or tuple(output_size) == target_size:
            return target_size, target_size, (0, 0), target_size

        original_size = (int(prediction["orig_h"]), int(prediction["orig_w"]))
        if tuple(output_size) != original_size:
            return tuple(output_size), target_size, (0, 0), tuple(output_size)

        scale_y = float(prediction["orig_h"]) / float(prediction["resized_h"])
        scale_x = float(prediction["orig_w"]) / float(prediction["resized_w"])
        crop_top = int(round(float(prediction["crop_top"]) * scale_y))
        crop_left = int(round(float(prediction["crop_left"]) * scale_x))
        crop_h = max(1, int(round(float(prediction["target_h"]) * scale_y)))
        crop_w = max(1, int(round(float(prediction["target_w"]) * scale_x)))
        crop_h = min(crop_h, original_size[0] - crop_top)
        crop_w = min(crop_w, original_size[1] - crop_left)
        return original_size, target_size, (crop_top, crop_left), (crop_h, crop_w)

    def _apply_sim3_and_export_frame(
        self,
        prediction: dict[str, Any],
        sim3: dict[str, Any],
        output_size: tuple[int, int] | None = None,
        source_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        del source_size
        scale = float(sim3["s"])
        rotation = sim3["R"]
        translation = sim3["t"]
        target_size = (int(prediction["target_h"]), int(prediction["target_w"]))
        resolved_output_size, _, offset, paste_size = self._resolve_output_geometry(prediction, output_size)

        depth = (prediction["depth"].astype(np.float32) * scale)
        conf = prediction["depth_conf"].astype(np.float32)
        intrinsic = prediction["intrinsic"].astype(np.float32)

        if resolved_output_size == target_size:
            output_depth = depth
            output_conf = conf
            output_intrinsic = intrinsic
            valid_mask = np.ones(output_depth.shape, dtype=bool)
        elif tuple(output_size or ()) == (int(prediction["orig_h"]), int(prediction["orig_w"])):
            crop_top, crop_left = offset
            crop_h, crop_w = paste_size
            crop_depth = self._resize_frame_output(depth, paste_size, interpolation=cv2.INTER_LINEAR)
            crop_conf = self._resize_frame_output(conf, paste_size, interpolation=cv2.INTER_LINEAR)
            output_depth = np.zeros(resolved_output_size, dtype=np.float32)
            output_conf = np.zeros(resolved_output_size, dtype=np.float32)
            valid_mask = np.zeros(resolved_output_size, dtype=bool)
            output_depth[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w] = crop_depth
            output_conf[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w] = crop_conf
            valid_mask[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w] = True
            output_intrinsic = self._rescale_intrinsics(intrinsic, target_size, paste_size)
            output_intrinsic[0, 2] += float(crop_left)
            output_intrinsic[1, 2] += float(crop_top)
        else:
            output_depth = self._resize_frame_output(depth, resolved_output_size, interpolation=cv2.INTER_LINEAR)
            output_conf = self._resize_frame_output(conf, resolved_output_size, interpolation=cv2.INTER_LINEAR)
            output_intrinsic = self._rescale_intrinsics(intrinsic, target_size, resolved_output_size)
            valid_mask = np.ones(output_depth.shape, dtype=bool)

        pointmap = self._depth_to_pointmap(output_depth, output_intrinsic).astype(np.float32)
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
            raise ValueError("DVLTLongDepthEstimator only supports batch mode.")

        image_paths, frame_names, orig_sizes = self._collect_image_paths(
            path,
            interval=interval,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        if not image_paths:
            print("No images loaded. Cannot run DVLT depth estimation.")
            return None

        target_h, target_w = self._compute_target_size(
            orig_sizes[0][0],
            orig_sizes[0][1],
            pixel_limit,
            max_long_edge,
        )

        print(
            f"{self._log_prefix()} Processing {len(image_paths)} images with DVLT "
            f"resize-only input size ({target_w}, {target_h}), patch_size={self.patch_size}, "
            f"chunk size {self.chunk_size}, and overlap {self.overlap}."
        )

        if self.temp_root:
            ensure_dir(self.temp_root)
        temp_dir = tempfile.mkdtemp(prefix="dvlt_long_", dir=self.temp_root)
        try:
            chunk_indices = self._build_chunk_indices(len(image_paths))
            chunk_records = []
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_indices):
                print(
                    f"{self._log_prefix()} Running DVLT chunk {chunk_idx + 1}/{len(chunk_indices)}: "
                    f"frames {chunk_start} -> {chunk_end - 1}"
                )
                chunk_records.append(
                    self._run_chunk_inference(
                        chunk_idx=chunk_idx,
                        image_paths=image_paths[chunk_start:chunk_end],
                        global_start_idx=chunk_start,
                        temp_dir=temp_dir,
                        target_size=(target_h, target_w),
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
                "type": "dvlt_long_manifest",
                "image_paths": image_paths,
                "frame_names": frame_names,
                "orig_sizes": orig_sizes,
                "chunk_indices": chunk_indices,
                "chunk_transforms": self._accumulate_chunk_sim3(relative_transforms),
                "frame_records": self._select_best_records(chunk_records, chunk_indices),
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
        restore_to_original_resolution = config.depth_output_resolution != "native"
        for output_idx, record in enumerate(sorted(results["frame_records"], key=lambda item: item["frame_idx"])):
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
        print(f"All {len(results['frame_records'])} DVLT visualizations saved to {output_dir}")

    def visualize_depth(
        self,
        image_path: str,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
    ) -> np.ndarray:
        image = cv2.imread(image_path)
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
        output_resolution_mode = str(config.depth_output_resolution)
        frame_records = sorted(results["frame_records"], key=lambda item: item["frame_idx"])

        for output_idx, record in enumerate(frame_records):
            prediction = self._load_frame_prediction(record["pred_path"])
            output_size = (
                (record["orig_h"], record["orig_w"])
                if restore_to_original_resolution
                else (prediction["target_h"], prediction["target_w"])
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
                camera_params = build_camera_params(
                    exported["intrinsics"],
                    exported["extrinsics"],
                    output_size,
                    inference_size=inference_size,
                    original_size=original_size,
                    output_resolution_mode=output_resolution_mode,
                    preprocess_mode="resize_only",
                )
                camera_params["preprocess"].update(
                    {
                        "mode": "resize_only",
                        "patch_size": self.patch_size,
                        "pixel_limit": config.depth_input_pixel_limit,
                        "max_long_edge": config.depth_input_max_long_edge,
                        "resized_h": int(record["resized_h"]),
                        "resized_w": int(record["resized_w"]),
                        "target_h": int(record["target_h"]),
                        "target_w": int(record["target_w"]),
                        "crop_top": int(record["crop_top"]),
                        "crop_left": int(record["crop_left"]),
                        "pad_top": int(record["pad_top"]),
                        "pad_left": int(record["pad_left"]),
                        "pad_bottom": int(record["pad_bottom"]),
                        "pad_right": int(record["pad_right"]),
                    }
                )
                with open(os.path.join(camera_params_dir, f"{frame_prefix}.json"), "w") as file:
                    json.dump(camera_params, file, indent=2)

        print(f"All {len(frame_records)} DVLT frames saved to {output_dir}")
        if self.delete_temp_files:
            self._cleanup_temp(results.get("temp_dir"))
