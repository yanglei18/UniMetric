from typing import Any

import torch

from metric_scale.models.dust3r_long_depth_estimator import Dust3rLongDepthEstimator


class Must3rLongDepthEstimator(Dust3rLongDepthEstimator):
    """Chunked long-sequence MUSt3R estimator with overlap-based Sim3 alignment."""

    def __init__(
        self,
        model_name: str = "must3r",
        retrieval_ckpt_path: str | None = None,
        device: str | None = None,
        dtype=None,
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
        self.retrieval_ckpt_path = retrieval_ckpt_path
        self.resolution_set = int(resolution_set)
        self.norm_type = str(norm_type)
        self.preprocess_mode = str(preprocess_mode or "resize_only").strip().lower()
        if self.preprocess_mode != "resize_only":
            raise ValueError(
                f"Unsupported MUSt3R preprocess mode: {self.preprocess_mode}. "
                "Expected 'resize_only' so MUSt3R uses the same Metric-Depth "
                "input resizing path as Pi3."
            )
        self.patch_size = int(patch_size)
        self.torch_hub_force_reload = bool(torch_hub_force_reload)
        self.quat_order = str(quat_order).strip().lower()
        if self.quat_order not in {"xyzw", "wxyz"}:
            raise ValueError(f"Unsupported quaternion order: {quat_order}")
        if not self.retrieval_ckpt_path:
            raise ValueError("MUSt3R requires a retrieval_ckpt_path.")

        self._import_external_factory()
        self.model = self.model_factory(
            "must3r",
            name="must3r",
            ckpt_path=self.model_name,
            retrieval_ckpt_path=self.retrieval_ckpt_path,
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
        results = super().process(
            path,
            interval=interval,
            pixel_limit=pixel_limit,
            max_long_edge=max_long_edge,
            mode=mode,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        if results is not None:
            results["type"] = "must3r_long_manifest"
        return results
