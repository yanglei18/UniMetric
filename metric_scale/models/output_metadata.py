import numpy as np


def rescale_intrinsics(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_size
    target_h, target_w = target_size
    if (source_h, source_w) == (target_h, target_w):
        return intrinsics.astype(np.float32)

    scaled = intrinsics.astype(np.float32).copy()
    scale_x = float(target_w) / max(float(source_w), 1.0)
    scale_y = float(target_h) / max(float(source_h), 1.0)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def build_camera_params(
    intrinsics_prediction: np.ndarray,
    extrinsics: np.ndarray,
    output_size: tuple[int, int],
    inference_size: tuple[int, int] | None = None,
    original_size: tuple[int, int] | None = None,
    intrinsics_original: np.ndarray | None = None,
    output_resolution_mode: str = "original",
    preprocess_mode: str = "resize_only",
) -> dict:
    output_h, output_w = output_size
    if inference_size is None:
        inference_size = output_size
    if original_size is None:
        original_size = output_size
    if intrinsics_original is None:
        intrinsics_original = rescale_intrinsics(
            intrinsics_prediction,
            output_size,
            original_size,
        )

    scale_x = float(output_w) / max(float(original_size[1]), 1.0)
    scale_y = float(output_h) / max(float(original_size[0]), 1.0)

    return {
        "intrinsics": intrinsics_prediction.astype(np.float32).tolist(),
        "intrinsics_original": intrinsics_original.astype(np.float32).tolist(),
        "intrinsics_prediction": intrinsics_prediction.astype(np.float32).tolist(),
        "extrinsics": extrinsics.astype(np.float32).tolist(),
        "original_resolution": [int(original_size[0]), int(original_size[1])],
        "inference_resolution": [int(inference_size[0]), int(inference_size[1])],
        "output_resolution": [int(output_h), int(output_w)],
        "prediction_resolution": [int(output_h), int(output_w)],
        "output_resolution_mode": output_resolution_mode,
        "preprocess": {
            "mode": preprocess_mode,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "crop_x": 0.0,
            "crop_y": 0.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        },
    }
