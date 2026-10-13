#!/usr/bin/env python3
"""Visualize Humanoid-Everyday raw depth projected onto RGB images."""

from __future__ import annotations

import argparse
import json
import logging
import lzma
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "metric_depth_matplotlib"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


LOGGER = logging.getLogger("humanoid_depth_projection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project Humanoid-Everyday RealSense D435 raw depth into the color "
            "camera view and save overlay visualizations."
        )
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to an episode folder such as humanoid-everyday-sample/episode_0, "
            "or to a sample root that contains episode_0."
        ),
    )
    parser.add_argument(
        "--calib-path",
        default=None,
        help="Path to calib_d435.json. If omitted, the script searches near --data-path.",
    )
    parser.add_argument("--robot", choices=("H1", "G1"), default="H1", help="Calibration profile to use.")
    parser.add_argument("--frame-index", type=int, default=100, help="Frame index in data.json.")
    parser.add_argument(
        "--frame-id",
        default=None,
        help="Optional frame id such as 000100. Overrides --frame-index when provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <sample-root>/depth_projection_visualizations.",
    )
    parser.add_argument(
        "--min-depth-mm",
        type=float,
        default=0.0,
        help="Minimum valid depth in millimeters. Values <= this are ignored.",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=float,
        default=0.0,
        help="Maximum valid depth in millimeters. Use <=0 to infer from valid pixels.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay opacity for depth colorization.",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap name for depth visualization.",
    )
    parser.add_argument(
        "--save-aligned-depth",
        action="store_true",
        help="Also save the color-aligned depth map as .npy.",
    )
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Process a frame range and save an overlay video.",
    )
    parser.add_argument("--frame-start", type=int, default=0, help="First frame index for --sequence.")
    parser.add_argument(
        "--frame-count",
        type=int,
        default=0,
        help="Number of frames for --sequence. Use <=0 for all frames from --frame-start.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS for --sequence.")
    parser.add_argument(
        "--video-name",
        default=None,
        help=(
            "Output video filename. Default: <episode>_<robot>_depth_projection.avi. "
            "Use .mp4 only if your OpenCV build writes a playable MP4 codec."
        ),
    )
    parser.add_argument(
        "--video-panel",
        choices=("aligned", "raw", "side_by_side", "triple"),
        default="aligned",
        help="Video content: aligned overlay only, raw overlay only, RGB/raw/aligned side-by-side, or RGB/raw/aligned triple.",
    )
    parser.add_argument(
        "--save-video-frames",
        action="store_true",
        help="Save per-frame video panels as PNG files under <output-dir>/frames.",
    )
    return parser.parse_args()


def resolve_episode_path(data_path: Path) -> Path:
    if (data_path / "data.json").exists():
        return data_path
    candidate = data_path / "episode_0"
    if (candidate / "data.json").exists():
        return candidate
    raise FileNotFoundError(f"Could not find episode data.json under: {data_path}")


def resolve_calib_path(episode_path: Path, calib_path: str | None) -> Path:
    if calib_path:
        path = Path(calib_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Calibration file does not exist: {path}")

    candidates = [
        episode_path.parent / "calib_d435.json",
        episode_path.parent.parent / "calib_d435.json",
        Path.cwd() / "humanoid-everyday-sample" / "calib_d435.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not infer calib_d435.json path; please pass --calib-path.")


def load_calibration(calib_path: Path, robot: str) -> dict[str, np.ndarray]:
    payload = json.loads(calib_path.read_text(encoding="utf-8"))
    if robot not in payload:
        raise KeyError(f"Robot calibration profile {robot!r} not found in {calib_path}")
    robot_payload = payload[robot]
    return {
        "K_depth": np.asarray(robot_payload["depth_intrinsics"]["K"], dtype=np.float64),
        "K_color": np.asarray(robot_payload["color_intrinsics"]["K"], dtype=np.float64),
        "R": np.asarray(robot_payload["depth_to_color_extrinsics"]["R"], dtype=np.float64),
        "t_m": np.asarray(robot_payload["depth_to_color_extrinsics"]["t_m"], dtype=np.float64),
    }


def load_frame_records(episode_path: Path) -> list[dict[str, Any]]:
    with (episode_path / "data.json").open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Expected data.json to contain a list, got {type(records).__name__}")
    return records


def select_frame_record(records: list[dict[str, Any]], frame_index: int, frame_id: str | None) -> tuple[int, dict[str, Any]]:
    if frame_id is not None:
        normalized = frame_id.zfill(6)
        image_suffix = f"frame_{normalized}.jpg"
        for index, record in enumerate(records):
            if str(record.get("image", "")).endswith(image_suffix):
                return index, record
        raise KeyError(f"Frame id {frame_id!r} not found in data.json")

    if frame_index < 0 or frame_index >= len(records):
        raise IndexError(f"frame-index {frame_index} is outside [0, {len(records) - 1}]")
    return frame_index, records[frame_index]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_depth_lzma(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    raw = lzma.decompress(path.read_bytes())
    height, width = image_shape
    expected_values = height * width

    depth = np.frombuffer(raw, dtype=np.uint16)
    if depth.size == expected_values:
        return depth.reshape(height, width)

    try:
        loaded = np.load(BytesIO(raw))
    except Exception as exc:
        raise ValueError(
            f"Depth payload has {depth.size} uint16 values, expected {expected_values}, "
            "and could not be loaded as an npy file."
        ) from exc
    if loaded.shape != (height, width):
        raise ValueError(f"Unexpected depth shape {loaded.shape}; expected {(height, width)}")
    return loaded.astype(np.uint16, copy=False)


def load_frame_rgb_depth(episode_path: Path, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = load_rgb(episode_path / record["image"])
    depth_raw_mm = load_depth_lzma(episode_path / record["depth"], image_shape=rgb.shape[:2])
    return rgb, depth_raw_mm


def warp_depth_to_color(depth_mm: np.ndarray, calibration: dict[str, np.ndarray]) -> np.ndarray:
    height, width = depth_mm.shape
    k_depth = calibration["K_depth"]
    k_color = calibration["K_color"]
    rotation = calibration["R"]
    translation_m = calibration["t_m"]

    out = np.zeros((height, width), dtype=np.float32)
    rows, cols = np.where(depth_mm > 0)
    if rows.size == 0:
        return out

    z_depth = depth_mm[rows, cols].astype(np.float64) / 1000.0
    x_depth = (cols - k_depth[0, 2]) * z_depth / k_depth[0, 0]
    y_depth = (rows - k_depth[1, 2]) * z_depth / k_depth[1, 1]
    points_depth = np.stack([x_depth, y_depth, z_depth], axis=1)

    points_color = points_depth @ rotation.T + translation_m
    z_color = points_color[:, 2]
    positive = z_color > 0
    points_color = points_color[positive]
    z_color = z_color[positive]

    projected_cols = (k_color[0, 0] * points_color[:, 0] / z_color + k_color[0, 2]).round().astype(np.int64)
    projected_rows = (k_color[1, 1] * points_color[:, 1] / z_color + k_color[1, 2]).round().astype(np.int64)
    inside = (
        (projected_cols >= 0)
        & (projected_cols < width)
        & (projected_rows >= 0)
        & (projected_rows < height)
    )
    projected_cols = projected_cols[inside]
    projected_rows = projected_rows[inside]
    z_color = z_color[inside]

    # Far-to-near assignment leaves the nearest projected depth in each color pixel.
    order = np.argsort(-z_color)
    flat = out.reshape(-1)
    flat[projected_rows[order] * width + projected_cols[order]] = (z_color[order] * 1000.0).astype(np.float32)
    return out


def colorize_depth(
    depth_mm: np.ndarray,
    min_depth_mm: float,
    max_depth_mm: float,
    cmap_name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    valid = depth_mm > min_depth_mm
    if max_depth_mm > 0:
        valid &= depth_mm <= max_depth_mm

    if not np.any(valid):
        rgba = np.zeros((*depth_mm.shape, 4), dtype=np.float32)
        return rgba, valid, (min_depth_mm, max_depth_mm)

    valid_values = depth_mm[valid].astype(np.float32)
    vmin = float(max(min_depth_mm, np.percentile(valid_values, 1)))
    vmax = float(max_depth_mm if max_depth_mm > 0 else np.percentile(valid_values, 99))
    if vmax <= vmin:
        vmax = float(valid_values.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    normalized = np.clip((depth_mm.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(normalized).astype(np.float32)
    rgba[..., 3] = valid.astype(np.float32)
    return rgba, valid, (vmin, vmax)


def blend_overlay(rgb: np.ndarray, depth_mm: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float]]:
    rgba, valid, depth_range = colorize_depth(
        depth_mm=depth_mm,
        min_depth_mm=args.min_depth_mm,
        max_depth_mm=args.max_depth_mm,
        cmap_name=args.cmap,
    )
    rgb_float = rgb.astype(np.float32) / 255.0
    alpha = np.clip(float(args.alpha), 0.0, 1.0) * rgba[..., 3:4]
    blended = rgb_float * (1.0 - alpha) + rgba[..., :3] * alpha
    stats = {
        "valid_ratio": float(valid.sum() / valid.size),
        "depth_min_mm": float(depth_mm[valid].min()) if np.any(valid) else 0.0,
        "depth_max_mm": float(depth_mm[valid].max()) if np.any(valid) else 0.0,
        "visual_min_mm": depth_range[0],
        "visual_max_mm": depth_range[1],
    }
    return np.clip(blended, 0.0, 1.0), stats


def save_visualization(
    rgb: np.ndarray,
    raw_overlay: np.ndarray,
    aligned_overlay: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    panels = [
        ("RGB", rgb),
        ("Raw depth overlay", raw_overlay),
        ("D2C aligned depth overlay", aligned_overlay),
    ]
    for axis, (panel_title, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(panel_title)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def make_video_panel(
    rgb: np.ndarray,
    raw_overlay: np.ndarray,
    aligned_overlay: np.ndarray,
    panel_mode: str,
) -> np.ndarray:
    rgb_uint8 = to_uint8_image(rgb)
    raw_uint8 = to_uint8_image(raw_overlay)
    aligned_uint8 = to_uint8_image(aligned_overlay)

    if panel_mode == "aligned":
        return aligned_uint8
    if panel_mode == "raw":
        return raw_uint8
    if panel_mode == "side_by_side":
        return np.concatenate([rgb_uint8, aligned_uint8], axis=1)
    if panel_mode == "triple":
        return np.concatenate([rgb_uint8, raw_uint8, aligned_uint8], axis=1)
    raise ValueError(f"Unsupported video panel: {panel_mode}")


def video_fourcc_for_path(video_path: Path) -> str:
    suffix = video_path.suffix.lower()
    if suffix == ".avi":
        return "MJPG"
    if suffix in {".mp4", ".m4v"}:
        return "avc1"
    return "MJPG"


def open_video_writer(video_path: Path, frame_shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    codec = video_fourcc_for_path(video_path)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path} with codec {codec}")
    return writer


def verify_video_readable(video_path: Path, expected_frames: int) -> dict[str, int | float | bool]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"opened": False, "frames_meta": 0, "frames_read": 0, "fps": 0.0, "width": 0, "height": 0}

    frames_meta = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_read = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames_read += 1
    capture.release()
    return {
        "opened": True,
        "frames_meta": frames_meta,
        "frames_read": frames_read,
        "fps": fps,
        "width": width,
        "height": height,
        "matches_expected": frames_read == expected_frames,
    }


def process_frame(
    episode_path: Path,
    record: dict[str, Any],
    calibration: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rgb, depth_raw_mm = load_frame_rgb_depth(episode_path, record)
    depth_aligned_mm = warp_depth_to_color(depth_raw_mm, calibration)
    raw_overlay, raw_stats = blend_overlay(rgb, depth_raw_mm.astype(np.float32), args)
    aligned_overlay, aligned_stats = blend_overlay(rgb, depth_aligned_mm, args)
    return {
        "rgb": rgb,
        "depth_raw_mm": depth_raw_mm,
        "depth_aligned_mm": depth_aligned_mm,
        "raw_overlay": raw_overlay,
        "aligned_overlay": aligned_overlay,
        "raw_stats": raw_stats,
        "aligned_stats": aligned_stats,
    }


def sequence_indices(records: list[dict[str, Any]], frame_start: int, frame_count: int) -> range:
    if frame_start < 0 or frame_start >= len(records):
        raise IndexError(f"frame-start {frame_start} is outside [0, {len(records) - 1}]")
    end = len(records) if frame_count <= 0 else min(len(records), frame_start + frame_count)
    return range(frame_start, end)


def process_sequence(
    episode_path: Path,
    records: list[dict[str, Any]],
    calibration: dict[str, np.ndarray],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    indices = sequence_indices(records, args.frame_start, args.frame_count)
    video_name = args.video_name or f"{episode_path.name}_{args.robot.lower()}_depth_projection.avi"
    video_path = output_dir / video_name
    frames_dir = output_dir / "frames"
    if args.save_video_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    aligned_valid_ratios: list[float] = []
    raw_valid_ratios: list[float] = []

    for processed_count, frame_index in enumerate(indices, start=1):
        record = records[frame_index]
        frame_name = Path(record["image"]).stem
        frame_data = process_frame(episode_path, record, calibration, args)
        panel = make_video_panel(
            frame_data["rgb"],
            frame_data["raw_overlay"],
            frame_data["aligned_overlay"],
            args.video_panel,
        )

        if writer is None:
            writer = open_video_writer(video_path, panel.shape, args.fps)

        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        if args.save_video_frames:
            frame_path = frames_dir / f"{frame_index:06d}_{frame_name}_{args.video_panel}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        raw_valid_ratios.append(frame_data["raw_stats"]["valid_ratio"])
        aligned_valid_ratios.append(frame_data["aligned_stats"]["valid_ratio"])
        if processed_count == 1 or processed_count % 25 == 0:
            LOGGER.info("Processed %d/%d frames; current=%s", processed_count, len(indices), frame_name)

    if writer is not None:
        writer.release()
    else:
        raise RuntimeError("No frames were processed; video was not created.")

    video_check = verify_video_readable(video_path, expected_frames=len(indices))
    if not video_check.get("matches_expected", False):
        raise RuntimeError(f"Video verification failed for {video_path}: {video_check}")

    metadata = {
        "episode_path": str(episode_path),
        "robot": args.robot,
        "frame_start": args.frame_start,
        "frame_count": len(indices),
        "fps": args.fps,
        "video_panel": args.video_panel,
        "video_path": str(video_path),
        "video_codec": video_fourcc_for_path(video_path),
        "video_check": video_check,
        "raw_valid_ratio_mean": float(np.mean(raw_valid_ratios)),
        "aligned_valid_ratio_mean": float(np.mean(aligned_valid_ratios)),
    }
    metadata_path = output_dir / f"{Path(video_name).stem}.json"
    write_metadata(metadata_path, metadata)
    LOGGER.info("Saved depth projection video: %s", video_path)
    LOGGER.info("Saved video metadata: %s", metadata_path)
    LOGGER.info("Video verification: %s", video_check)


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    episode_path = resolve_episode_path(Path(args.data_path))
    calib_path = resolve_calib_path(episode_path, args.calib_path)
    records = load_frame_records(episode_path)
    calibration = load_calibration(calib_path, args.robot)

    sample_root = episode_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else sample_root / "depth_projection_visualizations"

    if args.sequence:
        process_sequence(episode_path, records, calibration, output_dir, args)
        return

    frame_index, record = select_frame_record(records, args.frame_index, args.frame_id)
    frame_data = process_frame(episode_path, record, calibration, args)
    rgb = frame_data["rgb"]
    depth_aligned_mm = frame_data["depth_aligned_mm"]
    raw_overlay = frame_data["raw_overlay"]
    aligned_overlay = frame_data["aligned_overlay"]
    raw_stats = frame_data["raw_stats"]
    aligned_stats = frame_data["aligned_stats"]

    frame_name = Path(record["image"]).stem
    output_path = output_dir / f"{frame_name}_{args.robot.lower()}_depth_projection.png"
    title = f"{episode_path.name}/{frame_name} ({args.robot}), D435 depth-to-color projection"
    save_visualization(rgb, raw_overlay, aligned_overlay, output_path, title)

    metadata = {
        "episode_path": str(episode_path),
        "calib_path": str(calib_path),
        "robot": args.robot,
        "frame_index": frame_index,
        "frame_record": {
            "time": record.get("time"),
            "image": record.get("image"),
            "depth": record.get("depth"),
            "lidar": record.get("lidar"),
        },
        "image_shape": list(rgb.shape),
        "raw_depth": raw_stats,
        "aligned_depth": aligned_stats,
        "output": str(output_path),
    }
    metadata_path = output_dir / f"{frame_name}_{args.robot.lower()}_depth_projection.json"
    write_metadata(metadata_path, metadata)

    if args.save_aligned_depth:
        aligned_path = output_dir / f"{frame_name}_{args.robot.lower()}_depth_aligned.npy"
        np.save(aligned_path, depth_aligned_mm)
        LOGGER.info("Saved aligned depth: %s", aligned_path)

    LOGGER.info("Saved depth projection visualization: %s", output_path)
    LOGGER.info("Saved metadata: %s", metadata_path)
    LOGGER.info(
        "Raw valid %.1f%%, aligned valid %.1f%%",
        raw_stats["valid_ratio"] * 100.0,
        aligned_stats["valid_ratio"] * 100.0,
    )


if __name__ == "__main__":
    main()
