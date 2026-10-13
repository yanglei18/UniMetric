import importlib
import json
import math
import os
import os.path as osp
import shutil
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

from metric_scale.config.config import config
from metric_scale.config.path_config import ensure_dir
from metric_scale.models.output_metadata import build_camera_params
from metric_scale.utils.utils import file_utils


class Scal3rLongDepthEstimator:
    """Long-sequence Metric-Depth wrapper around the adjacent Scal3R release project."""

    def __init__(
        self,
        project_root: str,
        config_path: str,
        checkpoint: str | None = None,
        device: str | None = None,
        block_size: int = 60,
        overlap_size: int = 30,
        preprocess_workers: int = 4,
        use_loop: int = 1,
        use_xyz_align: int = 0,
        save_xyz: int = 0,
        use_native_confidence: bool = False,
        delete_temp_files: bool = True,
        temp_root: str | None = None,
    ) -> None:
        self.project_root = osp.abspath(project_root)
        self.config_path = osp.abspath(config_path)
        self.checkpoint = osp.abspath(checkpoint) if checkpoint else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.block_size = int(block_size)
        self.overlap_size = int(overlap_size)
        self.preprocess_workers = int(preprocess_workers)
        self.use_loop = int(use_loop)
        self.use_xyz_align = int(use_xyz_align)
        self.save_xyz = int(save_xyz)
        self.use_native_confidence = bool(use_native_confidence)
        self.delete_temp_files = bool(delete_temp_files)
        self.temp_root = temp_root

        self._import_scal3r()
        self.config = self.load_config_fn(self.config_path)

    def _import_scal3r(self) -> None:
        try:
            inference_module = importlib.import_module("scal3r.pipelines.inference")
            config_module = importlib.import_module("scal3r.engine")
        except ImportError:
            if not osp.isdir(self.project_root):
                raise ImportError(
                    f"Scal3R project root does not exist: {self.project_root}"
                ) from None
            if self.project_root not in sys.path:
                sys.path.insert(0, self.project_root)
            inference_module = importlib.import_module("scal3r.pipelines.inference")
            config_module = importlib.import_module("scal3r.engine")

        self.InferenceRequest = getattr(inference_module, "InferenceRequest")
        self.run_inference = getattr(inference_module, "run_inference")
        self.load_config_fn = getattr(config_module, "load_config")

    @staticmethod
    def _frame_prefix(
        frame_idx: int,
        frame_ids: list[str] | None = None,
        start_frame: int = 0,
    ) -> str:
        if frame_ids and frame_idx < len(frame_ids):
            return f"{frame_ids[frame_idx]}_camera1"
        return f"{start_frame + frame_idx:05d}_camera1"

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
    def _next_multiple(value: int, divisor: int) -> int:
        return int(math.ceil(float(value) / divisor) * divisor)

    @staticmethod
    def _compute_target_size(
        orig_height: int,
        orig_width: int,
        pixel_limit: int | None,
        max_long_edge: int | None = None,
    ) -> tuple[int, int]:
        if max_long_edge is not None and max_long_edge > 0:
            max_side = max(orig_width, orig_height)
            scale = min(1.0, float(max_long_edge) / float(max_side)) if max_side > 0 else 1.0
            target_w = max(1, int(math.floor((orig_width * scale) / 14.0))) * 14
            target_h = max(1, int(math.floor((orig_height * scale) / 14.0))) * 14
            return target_h, target_w

        if pixel_limit is None or pixel_limit <= 0:
            return (
                Scal3rLongDepthEstimator._next_multiple(orig_height, 14),
                Scal3rLongDepthEstimator._next_multiple(orig_width, 14),
            )

        scale = (
            math.sqrt(float(pixel_limit) / float(orig_width * orig_height))
            if orig_width * orig_height > 0
            else 1.0
        )
        target_w = orig_width * scale
        target_h = orig_height * scale
        width_steps = max(1, round(target_w / 14))
        height_steps = max(1, round(target_h / 14))
        while (width_steps * 14) * (height_steps * 14) > pixel_limit:
            if width_steps / max(height_steps, 1) > target_w / max(target_h, 1.0):
                width_steps -= 1
            else:
                height_steps -= 1
        return max(1, height_steps) * 14, max(1, width_steps) * 14

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
    def _depth_to_pointmap(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        height, width = depth.shape
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.stack([u_coords, v_coords, np.ones_like(depth)], axis=-1)
        intrinsic_inv = np.linalg.inv(intrinsics)
        camera_dirs = pixels @ intrinsic_inv.T
        return (camera_dirs * depth[..., None]).astype(np.float32)

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

    @staticmethod
    def _safe_symlink_or_copy(src_path: str, dst_path: str) -> None:
        try:
            os.symlink(src_path, dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)

    def _stage_input_images(
        self,
        image_paths: list[str],
        stage_dir: str,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[str, list[str]]:
        input_dir = ensure_dir(osp.join(stage_dir, "input"))
        staged_paths: list[str] = []
        for index, image_path in enumerate(image_paths):
            suffix = osp.splitext(image_path)[1].lower() or ".png"
            staged_path = osp.join(input_dir, f"{index:06d}{suffix}")
            if target_size is None:
                self._safe_symlink_or_copy(image_path, staged_path)
            else:
                target_h, target_w = target_size
                with Image.open(image_path) as image:
                    resized = image.convert("RGB").resize(
                        (target_w, target_h),
                        Image.Resampling.LANCZOS,
                    )
                    resized.save(staged_path)
            staged_paths.append(staged_path)
        return input_dir, staged_paths

    @staticmethod
    def _read_depth_exr(depth_path: str) -> np.ndarray:
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise FileNotFoundError(f"Failed to read Scal3R depth file: {depth_path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth.astype(np.float32)

    @staticmethod
    def _read_intrinsics_yaml(intri_path: str, num_frames: int) -> list[np.ndarray]:
        storage = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise FileNotFoundError(f"Failed to open Scal3R intrinsics file: {intri_path}")

        intrinsics: list[np.ndarray] = []
        for index in range(num_frames):
            key = f"K_{index:06d}"
            node = storage.getNode(key)
            matrix = node.mat()
            if matrix is None:
                storage.release()
                raise KeyError(f"Missing intrinsic matrix '{key}' in {intri_path}")
            intrinsics.append(np.asarray(matrix, dtype=np.float32))
        storage.release()
        return intrinsics

    @staticmethod
    def _read_c2w_matrices(mat_path: str) -> np.ndarray:
        matrices = np.loadtxt(mat_path, dtype=np.float32)
        if matrices.ndim == 1:
            matrices = matrices[None, :]
        return matrices.reshape(-1, 4, 4).astype(np.float32)

    def _load_processed_images(self, image_paths: list[str]) -> np.ndarray:
        processed_images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                processed_images.append(np.array(image.convert("RGB"), dtype=np.uint8))
        return np.stack(processed_images, axis=0) if processed_images else np.empty((0,), dtype=np.uint8)

    def _cleanup(self, temp_dir: str) -> None:
        if self.delete_temp_files and osp.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    def load_images(
        self,
        path: str,
        interval: int = 1,
        pixel_limit: int | None = 255000,
        max_long_edge: int | None = None,
        start_frame: int = 0,
        frame_count: int | None = None,
        return_metadata: bool = False,
    ) -> Any:
        image_paths, orig_sizes = self._collect_image_paths(
            path,
            interval=interval,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        target_size = None
        if orig_sizes:
            target_size = self._compute_target_size(
                orig_sizes[0][0],
                orig_sizes[0][1],
                pixel_limit,
                max_long_edge,
            )
        if not return_metadata:
            return image_paths
        return image_paths, {
            "orig_sizes": orig_sizes,
            "infer_size": target_size,
            "image_paths": image_paths,
            "preprocess_mode": "resize_only",
            "pixel_limit": pixel_limit,
            "max_long_edge": max_long_edge,
        }

    def estimate_depth(
        self,
        image_paths: list[str],
        orig_sizes: list[tuple[int, int]],
        target_size: tuple[int, int] | None,
    ) -> dict[str, Any]:
        temp_dir = tempfile.mkdtemp(prefix="scal3r_", dir=self.temp_root)
        try:
            staged_input_dir, staged_paths = self._stage_input_images(
                image_paths,
                temp_dir,
                target_size=target_size,
            )
            result_dir = ensure_dir(osp.join(temp_dir, "result"))
            runtime_dir = ensure_dir(osp.join(result_dir, "runtime"))

            request = self.InferenceRequest(
                config_path=self.config_path,
                input_dir=staged_input_dir,
                output_dir=result_dir,
                runtime_dir=runtime_dir,
                checkpoint=self.checkpoint,
                device=self.device,
                preprocess_workers=self.preprocess_workers,
                block_size=self.block_size,
                overlap_size=self.overlap_size,
                use_loop=self.use_loop,
                use_xyz_align=self.use_xyz_align,
                save_dpt=1,
                save_xyz=self.save_xyz,
                dry_run=False,
            )
            self.run_inference(self.config, request)

            depth_dir = osp.join(result_dir, "depths")
            intri_path = osp.join(result_dir, "intri.yml")
            mat_path = osp.join(result_dir, "mat.txt")

            num_frames = len(image_paths)
            depth_maps = [
                self._read_depth_exr(osp.join(depth_dir, f"{index:06d}.exr"))
                for index in range(num_frames)
            ]
            depths = np.stack(depth_maps, axis=0).astype(np.float32)
            intrinsics = np.stack(self._read_intrinsics_yaml(intri_path, num_frames), axis=0).astype(np.float32)
            c2w_mats = self._read_c2w_matrices(mat_path)
            extrinsics = np.stack([np.linalg.inv(mat)[:3, :] for mat in c2w_mats], axis=0).astype(np.float32)

            if self.use_native_confidence:
                confidence_dir = osp.join(result_dir, "depth_confidence")
                confs = np.stack(
                    [
                        np.load(osp.join(confidence_dir, f"{index:06d}.npy"))
                        for index in range(num_frames)
                    ],
                    axis=0,
                ).astype(np.float32)
                if confs.shape != depths.shape:
                    raise ValueError(
                        f"Scal3R confidence shape {confs.shape} does not match depth shape {depths.shape}"
                    )
                confs[~np.isfinite(confs)] = 0.0
            else:
                confs = np.isfinite(depths).astype(np.float32)
            confs[~np.isfinite(depths) | (depths <= 0)] = 0.0
            point_maps = np.stack(
                [self._depth_to_pointmap(depths[index], intrinsics[index]) for index in range(num_frames)],
                axis=0,
            ).astype(np.float32)
            processed_images = self._load_processed_images(staged_paths)

            infer_size = tuple(int(value) for value in depths[0].shape[:2]) if num_frames > 0 else None
            return {
                "depth": depths,
                "conf": confs,
                "point_map": point_maps,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "processed_images": processed_images,
                "image_meta": {
                    "orig_sizes": orig_sizes,
                    "infer_size": infer_size,
                    "image_paths": image_paths,
                    "preprocess_mode": "resize_only",
                    "target_size": target_size,
                },
            }
        finally:
            self._cleanup(temp_dir)

    def process(
        self,
        path: str,
        interval: int = 1,
        pixel_limit: int | None = 255000,
        max_long_edge: int | None = None,
        mode: str = "batch",
        start_frame: int = 0,
        frame_count: int | None = None,
    ) -> dict[str, Any] | None:
        if mode != "batch":
            raise ValueError("Scal3rLongDepthEstimator only supports batch mode.")

        image_paths, image_meta = self.load_images(
            path,
            interval=interval,
            pixel_limit=pixel_limit,
            max_long_edge=max_long_edge,
            start_frame=start_frame,
            frame_count=frame_count,
            return_metadata=True,
        )
        if not image_paths:
            return None

        target_size = image_meta.get("infer_size")
        size_message = (
            f"resize-only input size ({target_size[1]}, {target_size[0]})"
            if target_size is not None
            else "original input size"
        )
        print(f"Loaded {len(image_paths)} images. Running Scal3R inference with {size_message}...")
        results = self.estimate_depth(image_paths, image_meta["orig_sizes"], target_size)
        print("Scal3R inference complete!")
        return results

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
        for frame_idx, depth_map in enumerate(results["depth"]):
            image = results["processed_images"][frame_idx]
            depth_min, depth_max = float(np.nanmin(depth_map)), float(np.nanmax(depth_map))
            normalized_depth = ((depth_map - depth_min) / (depth_max - depth_min + 1e-6) * 255).astype(np.uint8)
            colored_depth = cv2.cvtColor(cv2.applyColorMap(normalized_depth, colormap), cv2.COLOR_BGR2RGB)
            if colored_depth.shape[:2] != image.shape[:2]:
                colored_depth = cv2.resize(
                    colored_depth,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            blended = cv2.addWeighted(image, 1 - alpha, colored_depth, alpha, 0)
            frame_prefix = self._frame_prefix(frame_idx, frame_ids=frame_ids, start_frame=start_frame)
            cv2.imwrite(
                osp.join(output_dir, f"depth_visualization_{frame_prefix}.png"),
                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
            )

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
        ensure_dir(output_dir)
        if save_images:
            images_dir = ensure_dir(osp.join(output_dir, "images"))
        if save_depth:
            depth_dir = ensure_dir(osp.join(output_dir, "depth"))
        if save_conf:
            conf_dir = ensure_dir(osp.join(output_dir, "conf"))
        if save_pointmap:
            pointmap_dir = ensure_dir(osp.join(output_dir, "pointmap"))
        if save_camera_params:
            camera_params_dir = ensure_dir(osp.join(output_dir, "camera_params"))

        image_meta = results.get("image_meta", {})
        orig_sizes = image_meta.get("orig_sizes", [])
        infer_size = image_meta.get("infer_size")
        num_frames = results["depth"].shape[0]
        restore_to_original_resolution = config.depth_output_resolution != "native"

        for frame_idx in range(num_frames):
            frame_prefix = self._frame_prefix(frame_idx, frame_ids=frame_ids, start_frame=start_frame)
            output_size = (
                tuple(map(int, orig_sizes[frame_idx]))
                if restore_to_original_resolution and frame_idx < len(orig_sizes)
                else tuple(map(int, results["depth"][frame_idx].shape[:2]))
            )
            original_size = (
                tuple(map(int, orig_sizes[frame_idx]))
                if frame_idx < len(orig_sizes)
                else output_size
            )
            frame_depth = results["depth"][frame_idx].astype(np.float32)
            frame_conf = results["conf"][frame_idx].astype(np.float32)
            frame_intrinsics = results["intrinsics"][frame_idx].astype(np.float32)
            frame_extrinsics = results["extrinsics"][frame_idx].astype(np.float32)

            if frame_depth.shape[:2] != output_size:
                frame_depth = self._resize_frame_output(frame_depth, output_size)
            if frame_conf.shape[:2] != output_size:
                frame_conf = self._resize_frame_output(frame_conf, output_size)
            if infer_size is not None and tuple(map(int, infer_size)) != output_size:
                frame_intrinsics = self._rescale_intrinsics(frame_intrinsics, infer_size, output_size)

            frame_pointmap = self._depth_to_pointmap(frame_depth, frame_intrinsics)

            if save_images:
                frame_image = results["processed_images"][frame_idx]
                if frame_image.shape[:2] != output_size:
                    frame_image = cv2.resize(
                        frame_image,
                        (output_size[1], output_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                image_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(osp.join(images_dir, f"{frame_prefix}.png"), image_bgr)
            if save_depth:
                np.save(osp.join(depth_dir, f"{frame_prefix}.npy"), frame_depth)
            if save_conf:
                np.save(osp.join(conf_dir, f"{frame_prefix}.npy"), frame_conf)
            if save_pointmap:
                np.save(osp.join(pointmap_dir, f"{frame_prefix}.npy"), frame_pointmap.astype(np.float32))
            if save_camera_params:
                with open(osp.join(camera_params_dir, f"{frame_prefix}.json"), "w", encoding="utf-8") as file:
                    json.dump(
                        build_camera_params(
                            frame_intrinsics,
                            frame_extrinsics,
                            output_size,
                            inference_size=None if infer_size is None else tuple(map(int, infer_size)),
                            original_size=original_size,
                            output_resolution_mode=config.depth_output_resolution,
                        ),
                        file,
                        indent=2,
                    )

        print(f"All {num_frames} frames saved to {output_dir}")
