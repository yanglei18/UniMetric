#!/usr/bin/env python3
"""Convert KITTI-Raw synced drives to this project's unified dataset format.

Expected KITTI-Raw input layout:
  raw_root/
    2011_09_26/
      calib_cam_to_cam.txt
      calib_velo_to_cam.txt
      calib_imu_to_velo.txt
      2011_09_26_drive_0001_sync/
        image_02/data/*.png
        velodyne_points/data/*.bin
        oxts/data/*.txt

Output layout:
  output_dir/
    2011_09_26_drive_0001_sync/
      0000000000_camera1.jpg
      0000000000.pcd
      0000000000.yaml

Each output PCD is a single-frame raw sparse Velodyne scan copied from the
matching KITTI .bin file. The converter never accumulates point clouds across
frames and never transforms the PCD points into a global coordinate system.

KITTI-Raw synced data is recorded at 10 Hz. By default this converter keeps a
2 Hz temporal subset, i.e. every fifth synchronized frame, without imposing a
maximum frame count per drive. If a drive would have fewer than 20 output frames
at 2 Hz, the converter automatically keeps a 5 Hz subset for that drive.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image


EARTH_RADIUS_METERS = 6378137.0


@dataclass(frozen=True)
class KittiCalibration:
    camera_from_lidar: np.ndarray
    intrinsic: np.ndarray
    velo_from_imu: np.ndarray | None


@dataclass(frozen=True)
class DrivePaths:
    drive_dir: Path
    date_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KITTI-Raw synced data to Metric-Depth unified format."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data2/yanglei/metric_data/official-data/kitti-raw"),
        help="KITTI-Raw root containing date folders and synced drives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output dataset root in unified format.",
    )
    parser.add_argument(
        "--camera",
        default="image_02",
        choices=("image_00", "image_01", "image_02", "image_03"),
        help="KITTI camera folder to export. image_02 is the left color camera.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=1,
        help="Unified camera id used in output filenames, e.g. camera1.",
    )
    parser.add_argument(
        "--drive",
        action="append",
        default=None,
        help="Optional drive directory name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-frames-per-drive",
        type=int,
        default=0,
        help="Optional debug cap after temporal downsampling. 0 means no per-drive frame-count limit.",
    )
    parser.add_argument(
        "--source-hz",
        type=float,
        default=10.0,
        help="Input KITTI-Raw synced frame rate.",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=2.0,
        help="Output frame rate. The default converts KITTI-Raw 10 Hz to 2 Hz.",
    )
    parser.add_argument(
        "--min-target-frames",
        type=int,
        default=20,
        help="Use --short-sequence-target-hz when the requested target rate would produce fewer frames.",
    )
    parser.add_argument(
        "--short-sequence-target-hz",
        type=float,
        default=5.0,
        help="Fallback output frame rate for short drives. Set <= --target-hz to disable.",
    )
    parser.add_argument(
        "--prune-extra-frames",
        action="store_true",
        help="Remove existing output frames that are not selected by the current temporal sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted files.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only rewrite YAML metadata and validate existing raw sparse PCD files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered drives and frame counts.",
    )
    parser.add_argument(
        "--skip-raw-check",
        action="store_true",
        help="Skip the raw point-count consistency check between KITTI .bin and output PCD.",
    )
    return parser.parse_args()


def read_calib_file(path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values = []
        for token in value.strip().split():
            try:
                values.append(float(token))
            except ValueError:
                values = []
                break
        if values:
            data[key.strip()] = values
    return data


def transform_from_rot_trans(rotation: Iterable[float], translation: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(list(rotation), dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(list(translation), dtype=np.float64).reshape(3)
    return transform


def camera_suffix(camera: str) -> str:
    return camera.split("_", 1)[1]


def load_calibration(date_dir: Path, camera: str) -> KittiCalibration:
    cam_to_cam = read_calib_file(date_dir / "calib_cam_to_cam.txt")
    velo_to_cam = read_calib_file(date_dir / "calib_velo_to_cam.txt")
    imu_to_velo_path = date_dir / "calib_imu_to_velo.txt"
    imu_to_velo = read_calib_file(imu_to_velo_path) if imu_to_velo_path.exists() else {}

    suffix = camera_suffix(camera)
    p_key = f"P_rect_{suffix}"
    if p_key not in cam_to_cam:
        raise KeyError(f"{p_key} not found in {date_dir / 'calib_cam_to_cam.txt'}")
    if "R" not in velo_to_cam or "T" not in velo_to_cam:
        raise KeyError(f"R/T not found in {date_dir / 'calib_velo_to_cam.txt'}")

    projection = np.asarray(cam_to_cam[p_key], dtype=np.float64).reshape(3, 4)
    intrinsic = projection[:, :3].copy()

    rect_key = "R_rect_00"
    rect = np.eye(4, dtype=np.float64)
    if rect_key in cam_to_cam:
        rect[:3, :3] = np.asarray(cam_to_cam[rect_key], dtype=np.float64).reshape(3, 3)

    velo_to_cam0 = transform_from_rot_trans(velo_to_cam["R"], velo_to_cam["T"])
    cam_rect_from_cam0_rect = np.eye(4, dtype=np.float64)
    cam_rect_from_cam0_rect[:3, 3] = np.linalg.solve(intrinsic, projection[:, 3])
    camera_from_lidar = cam_rect_from_cam0_rect @ rect @ velo_to_cam0

    velo_from_imu = None
    if "R" in imu_to_velo and "T" in imu_to_velo:
        velo_from_imu = transform_from_rot_trans(imu_to_velo["R"], imu_to_velo["T"])

    return KittiCalibration(
        camera_from_lidar=camera_from_lidar,
        intrinsic=intrinsic,
        velo_from_imu=velo_from_imu,
    )


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def oxts_to_world_from_imu(packet: list[float], origin: np.ndarray, scale: float) -> np.ndarray:
    lat, lon, alt, roll, pitch, yaw = packet[:6]
    tx = scale * math.radians(lon) * EARTH_RADIUS_METERS
    ty = scale * EARTH_RADIUS_METERS * math.log(math.tan(math.pi * (90.0 + lat) / 360.0))
    translation = np.asarray([tx, ty, alt], dtype=np.float64) - origin
    rotation = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def pose_from_matrix(transform: np.ndarray) -> list[float]:
    rotation = transform[:3, :3]
    pitch = math.asin(float(np.clip(rotation[2, 0], -1.0, 1.0)))
    cp = math.cos(pitch)
    if abs(cp) < 1e-8:
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
        roll = 0.0
    else:
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        roll = math.atan2(-rotation[2, 1], rotation[2, 2])
    return [
        float(transform[0, 3]),
        float(transform[1, 3]),
        float(transform[2, 3]),
        math.degrees(roll),
        math.degrees(yaw),
        math.degrees(pitch),
    ]


def load_oxts_packets(oxts_dir: Path, frame_stems: list[str]) -> dict[str, list[float]]:
    packets: dict[str, list[float]] = {}
    for stem in frame_stems:
        path = oxts_dir / f"{stem}.txt"
        if not path.exists():
            continue
        values = [float(value) for value in path.read_text(encoding="utf-8").split()]
        packets[stem] = values
    return packets


def write_binary_pcd(points: np.ndarray, output_path: Path) -> None:
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected KITTI Velodyne points with shape Nx4, got {points.shape}")
    points = np.asarray(points[:, :4], dtype=np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {points.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {points.shape[0]}\n"
        "DATA binary\n"
    )
    with output_path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(points.tobytes(order="C"))


def raw_velodyne_point_count(path: Path) -> int:
    byte_count = path.stat().st_size
    point_stride = 4 * np.dtype(np.float32).itemsize
    if byte_count % point_stride != 0:
        raise ValueError(f"Invalid KITTI Velodyne file size for {path}: {byte_count} bytes")
    return byte_count // point_stride


def read_pcd_header_point_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("POINTS "):
                return int(line.split()[1])
            if line == "DATA binary":
                break
    raise ValueError(f"POINTS header not found in {path}")


def assert_raw_sparse_pcd_matches_bin(raw_bin_path: Path, pcd_path: Path) -> None:
    raw_count = raw_velodyne_point_count(raw_bin_path)
    pcd_count = read_pcd_header_point_count(pcd_path)
    if raw_count != pcd_count:
        raise ValueError(
            "Output PCD is not a one-to-one raw sparse LiDAR conversion: "
            f"{raw_bin_path} has {raw_count} points, but {pcd_path} declares {pcd_count} points"
        )


def convert_image(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    with Image.open(input_path) as image:
        image.convert("RGB").save(output_path, format="JPEG", quality=95)


def discover_drives(raw_root: Path, drive_filters: list[str] | None) -> list[DrivePaths]:
    candidates = sorted(path for path in raw_root.rglob("*_drive_*_sync") if path.is_dir())
    filters = set(drive_filters or [])
    drives = []
    for drive_dir in candidates:
        if filters and drive_dir.name not in filters:
            continue
        date_dir = find_date_dir(drive_dir, raw_root)
        if date_dir is None:
            continue
        drives.append(DrivePaths(drive_dir=drive_dir, date_dir=date_dir))
    return drives


def find_date_dir(drive_dir: Path, raw_root: Path) -> Path | None:
    for candidate in [drive_dir.parent, *drive_dir.parents]:
        if raw_root not in [candidate, *candidate.parents] and candidate != raw_root:
            break
        if (candidate / "calib_cam_to_cam.txt").exists() and (candidate / "calib_velo_to_cam.txt").exists():
            return candidate
    return None


def common_frame_stems(drive_dir: Path, camera: str) -> list[str]:
    image_dir = drive_dir / camera / "data"
    lidar_dir = drive_dir / "velodyne_points" / "data"
    oxts_dir = drive_dir / "oxts" / "data"
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    lidar_stems = {path.stem for path in lidar_dir.glob("*.bin")}
    oxts_stems = {path.stem for path in oxts_dir.glob("*.txt")}
    return sorted(image_stems & lidar_stems & oxts_stems)


def sampling_stride(source_hz: float, target_hz: float) -> int:
    if source_hz <= 0:
        raise ValueError(f"--source-hz must be positive, got {source_hz}")
    if target_hz <= 0:
        raise ValueError(f"--target-hz must be positive, got {target_hz}")
    if target_hz > source_hz:
        raise ValueError(f"--target-hz ({target_hz}) cannot exceed --source-hz ({source_hz})")

    ratio = source_hz / target_hz
    stride = int(round(ratio))
    if stride < 1:
        stride = 1
    if not math.isclose(ratio, stride, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "Only integer temporal downsampling strides are supported: "
            f"--source-hz / --target-hz = {ratio}"
        )
    return stride


def stride_subset(frame_stems: list[str], stride: int) -> list[str]:
    if stride == 1:
        return frame_stems
    return frame_stems[::stride]


def select_temporal_subset(
    frame_stems: list[str],
    source_hz: float,
    target_hz: float,
    min_target_frames: int = 0,
    short_sequence_target_hz: float = 0.0,
) -> tuple[list[str], int, float, bool]:
    stride = sampling_stride(source_hz, target_hz)
    sampled = stride_subset(frame_stems, stride)
    fallback_enabled = min_target_frames > 0 and short_sequence_target_hz > target_hz
    if fallback_enabled and len(sampled) < min_target_frames and frame_stems:
        fallback_stride = sampling_stride(source_hz, short_sequence_target_hz)
        return stride_subset(frame_stems, fallback_stride), fallback_stride, short_sequence_target_hz, True
    return sampled, stride, target_hz, False


def prune_extra_output_frames(sequence_dir: Path, selected_stems: set[str], camera_id: int) -> int:
    if not sequence_dir.exists():
        return 0
    removed = 0
    for path in sequence_dir.iterdir():
        if path.suffix not in {".jpg", ".pcd", ".yaml"}:
            continue
        if path.name.endswith(f"_camera{camera_id}.jpg"):
            stem = path.name[: -len(f"_camera{camera_id}.jpg")]
        else:
            stem = path.stem
        if stem not in selected_stems:
            path.unlink()
            removed += 1
    return removed


def convert_drive(drive: DrivePaths, output_dir: Path, args: argparse.Namespace) -> int:
    calibration = load_calibration(drive.date_dir, args.camera)
    sequence_dir = output_dir / drive.drive_dir.name
    sequence_dir.mkdir(parents=True, exist_ok=True)

    all_frame_stems = common_frame_stems(drive.drive_dir, args.camera)
    sampled_frame_stems, stride, effective_target_hz, fallback_used = select_temporal_subset(
        all_frame_stems,
        args.source_hz,
        args.target_hz,
        args.min_target_frames,
        args.short_sequence_target_hz,
    )
    frame_stems = sampled_frame_stems
    if args.max_frames_per_drive > 0:
        frame_stems = frame_stems[: args.max_frames_per_drive]
    if not frame_stems:
        print(f"[WARN] No synchronized frames found: {drive.drive_dir}")
        return 0
    if args.prune_extra_frames:
        removed = prune_extra_output_frames(sequence_dir, set(frame_stems), args.camera_id)
        if removed:
            print(f"[PRUNE] {drive.drive_dir.name}: removed_extra_files={removed}")

    packets = load_oxts_packets(drive.drive_dir / "oxts" / "data", frame_stems)
    first_packet = packets[frame_stems[0]]
    scale = math.cos(math.radians(first_packet[0]))
    origin = np.asarray(
        [
            scale * math.radians(first_packet[1]) * EARTH_RADIUS_METERS,
            scale * EARTH_RADIUS_METERS * math.log(math.tan(math.pi * (90.0 + first_packet[0]) / 360.0)),
            first_packet[2],
        ],
        dtype=np.float64,
    )

    imu_from_lidar = None if calibration.velo_from_imu is None else np.linalg.inv(calibration.velo_from_imu)
    camera_from_lidar = calibration.camera_from_lidar
    lidar_from_camera = np.linalg.inv(camera_from_lidar)

    for stem in frame_stems:
        image_path = drive.drive_dir / args.camera / "data" / f"{stem}.png"
        lidar_path = drive.drive_dir / "velodyne_points" / "data" / f"{stem}.bin"
        raw_point_count = raw_velodyne_point_count(lidar_path)
        world_from_imu = oxts_to_world_from_imu(packets[stem], origin, scale)
        world_from_lidar = world_from_imu @ imu_from_lidar if imu_from_lidar is not None else world_from_imu
        world_from_camera = world_from_lidar @ lidar_from_camera

        frame_id = stem
        image_out = sequence_dir / f"{frame_id}_camera{args.camera_id}.jpg"
        pcd_out = sequence_dir / f"{frame_id}.pcd"
        yaml_out = sequence_dir / f"{frame_id}.yaml"

        if args.metadata_only:
            if not image_out.exists() or not pcd_out.exists():
                raise FileNotFoundError(f"metadata-only mode requires existing outputs for frame {frame_id}")
        else:
            convert_image(image_path, image_out, args.overwrite)
        if not args.metadata_only and (args.overwrite or not pcd_out.exists()):
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            if points.shape[0] != raw_point_count:
                raise ValueError(f"Unexpected raw point count for {lidar_path}: {points.shape[0]} != {raw_point_count}")
            write_binary_pcd(points, pcd_out)
        if not args.skip_raw_check:
            assert_raw_sparse_pcd_matches_bin(lidar_path, pcd_out)
        if args.metadata_only or args.overwrite or not yaml_out.exists():
            payload = {
                f"camera{args.camera_id}": {
                    "cords": pose_from_matrix(world_from_camera),
                    "extrinsic": camera_from_lidar.tolist(),
                    "intrinsic": calibration.intrinsic.tolist(),
                },
                "lidar_pose": pose_from_matrix(world_from_lidar),
                "true_ego_pose": pose_from_matrix(world_from_lidar),
                "vehicles": {},
                "pedestrians": {},
                "cyclists": {},
                "source": {
                    "dataset": "kitti-raw",
                    "drive": drive.drive_dir.name,
                    "frame": stem,
                    "camera": args.camera,
                    "image_file": str(image_path.relative_to(drive.date_dir)),
                    "lidar_file": str(lidar_path.relative_to(drive.date_dir)),
                    "point_cloud": {
                        "type": "raw_sparse_lidar",
                        "coordinate_frame": "kitti_velodyne_lidar",
                        "point_count": raw_point_count,
                        "accumulated": False,
                    },
                    "frame_sampling": {
                        "source_hz": args.source_hz,
                        "requested_target_hz": args.target_hz,
                        "effective_target_hz": effective_target_hz,
                        "stride": stride,
                        "short_sequence_fallback_used": fallback_used,
                        "min_target_frames": args.min_target_frames,
                        "short_sequence_target_hz": args.short_sequence_target_hz,
                        "original_synchronized_frames": len(all_frame_stems),
                        "downsampled_synchronized_frames": len(sampled_frame_stems),
                        "exported_frames": len(frame_stems),
                        "max_frames_per_drive": args.max_frames_per_drive,
                    },
                },
            }
            yaml_out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    return len(frame_stems)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    drives = discover_drives(raw_root, args.drive)
    if not drives:
        raise FileNotFoundError(f"No KITTI-Raw synced drives found under {raw_root}")

    print(f"[INFO] Raw root: {raw_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Camera: {args.camera} -> camera{args.camera_id}")
    print(f"[INFO] Temporal sampling: {args.source_hz:g} Hz -> {args.target_hz:g} Hz, stride={sampling_stride(args.source_hz, args.target_hz)}")
    if args.min_target_frames > 0 and args.short_sequence_target_hz > args.target_hz:
        fallback_stride = sampling_stride(args.source_hz, args.short_sequence_target_hz)
        print(
            "[INFO] Short-drive fallback: "
            f"<{args.min_target_frames} frames at {args.target_hz:g} Hz -> "
            f"{args.short_sequence_target_hz:g} Hz, stride={fallback_stride}"
        )
    print(f"[INFO] Drives: {len(drives)}")
    for drive in drives:
        frame_stems = common_frame_stems(drive.drive_dir, args.camera)
        frame_count = len(frame_stems)
        sampled_stems, stride, effective_target_hz, fallback_used = select_temporal_subset(
            frame_stems,
            args.source_hz,
            args.target_hz,
            args.min_target_frames,
            args.short_sequence_target_hz,
        )
        sampled_count = len(sampled_stems)
        planned = min(sampled_count, args.max_frames_per_drive) if args.max_frames_per_drive > 0 else sampled_count
        fallback_label = f", fallback={effective_target_hz:g}hz" if fallback_used else ""
        print(
            f"[DRIVE] {drive.drive_dir.name}: frames={frame_count}, sampled={sampled_count}, "
            f"planned={planned}, effective_hz={effective_target_hz:g}, stride={stride}{fallback_label}"
        )
        if args.dry_run:
            continue
        converted = convert_drive(drive, output_dir, args)
        print(f"[DONE] {drive.drive_dir.name}: converted={converted}")

    if args.dry_run:
        return
    print("[DONE] KITTI-Raw conversion complete.")


if __name__ == "__main__":
    main()
