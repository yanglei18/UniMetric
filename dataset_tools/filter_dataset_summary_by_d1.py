#!/usr/bin/env python3
"""Build a filtered nuScenes subset from dataset_summary.json or nohup.out."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "outputs/nuScenes/metric_results/pi3_long/dataset_summary.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_RE = re.compile(r"^Sequence:\s*(?P<sequence>[\w.-]+)\s*$")
MODEL_RE = re.compile(r"^Model:\s*(?P<model>[\w.-]+)\s*$")

VARIANT_ORDER = (
    ("relative_scale", ("relative", "relative_scale")),
    ("consistent_scale", ("consistent", "consistent_scale")),
    ("optimized_scale_legacy", ("optimized_legacy", "optimized_scale_legacy")),
    ("optimized_scale_original", ("optimized_original", "optimized_scale_original")),
    ("optimized_scale_weighted_bi", ("optimized_weighted_bi", "optimized_scale_weighted_bi")),
    ("optimized_scale_weighted_ifpa", ("optimized_weighted_ifpa", "optimized_scale_weighted_ifpa")),
    ("selected_scale", ("selected", "selected_scale")),
)

VARIANT_ALIASES = {
    alias: (label, keys)
    for label, keys in VARIANT_ORDER
    for alias in (label, *keys)
}

METRIC_KEYS = ("rel", "delta1", "mae", "rmse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Metric-Depth dataset_summary.json or nohup.out by "
            "optimized_scale_original Depth d1, recompute Dataset Summary on the "
            "selected sequences, and copy the selected sequence folders into "
            "data/official-data/nuScenes-New and outputs/nuScenes-New."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Path to dataset_summary.json or nohup.out. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "json", "nohup"),
        default="auto",
        help="Input format. Default: auto.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Keep sequences whose filter variant Depth d1 is greater than this value.",
    )
    parser.add_argument(
        "--filter-variant",
        default="optimized_scale_original",
        help=(
            "Variant used for filtering. Default: optimized_scale_original. "
            "This script is intended to build nuScenes-New from optimized_scale_original."
        ),
    )
    parser.add_argument(
        "--include-equal",
        action="store_true",
        help="Use >= threshold instead of > threshold.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the filtered summary JSON.",
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="Optional path to save the rendered filtered summary report.",
    )
    parser.add_argument(
        "--list-sequences",
        action="store_true",
        help="Print selected sequence names after the summary table.",
    )
    parser.add_argument(
        "--copy-selected",
        dest="copy_selected",
        action="store_true",
        default=True,
        help=(
            "Copy selected sequence folders from source data/output roots to "
            "target data/output roots. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-copy-selected",
        dest="copy_selected",
        action="store_false",
        help="Only print/save the filtered summary; do not copy sequence folders.",
    )
    parser.add_argument(
        "--source-data-root",
        default="data/official-data/nuScenes",
        help="Source dataset root. Default: data/official-data/nuScenes.",
    )
    parser.add_argument(
        "--target-data-root",
        default="data/official-data/nuScenes-New",
        help="Target dataset root. Default: data/official-data/nuScenes-New.",
    )
    parser.add_argument(
        "--source-output-root",
        default="outputs/nuScenes",
        help="Source output root. Default: outputs/nuScenes.",
    )
    parser.add_argument(
        "--target-output-root",
        default="outputs/nuScenes-New",
        help="Target output root. Default: outputs/nuScenes-New.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing target sequence directories. Enabled by default.",
    )
    parser.add_argument(
        "--reset-targets",
        action="store_true",
        help=(
            "Remove target data/output roots before copying. Disabled by default."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Keep existing target sequence directories instead of replacing them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print copy actions without copying files.",
    )
    parser.add_argument(
        "--copy-workers",
        type=int,
        default=8,
        help="Number of threads used for copying sequence directories. Default: 8.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset_summary.json does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Input JSON must be an object: {path}")
    return payload


def parse_number(value: str) -> float | int | None:
    cleaned = str(value).strip().replace(" m", "")
    if cleaned in {"", "-"}:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def parse_metric_row(line: str) -> tuple[str, dict[str, Any]] | None:
    if not line.startswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 11 or cells[0] in {"Scale", ""}:
        return None

    scale_name = cells[0]
    depth_frames = parse_number(cells[1])
    point_frames = parse_number(cells[6])
    return scale_name, {
        "frames_requested": int(depth_frames or 0),
        "frames_evaluated": int(depth_frames or 0),
        "depth_frame_mean": {
            "frame_count": int(depth_frames or 0),
            "rel": parse_number(cells[2]),
            "delta1": parse_number(cells[3]),
            "mae": parse_number(cells[4]),
            "rmse": parse_number(cells[5]),
        },
        "point_frame_mean": {
            "frame_count": int(point_frames or 0),
            "rel": parse_number(cells[7]),
            "delta1": parse_number(cells[8]),
            "mae": parse_number(cells[9]),
            "rmse": parse_number(cells[10]),
        },
    }


def parse_nohup(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input nohup.out does not exist: {path}")

    sequence_results: dict[str, dict[str, Any]] = {}
    current_sequence: str | None = None
    current_model: str | None = None
    default_model: str | None = None
    in_frame_mean_table = False

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

            if line.startswith("Dataset Summary"):
                current_sequence = None
                in_frame_mean_table = False
                continue

            sequence_match = SEQUENCE_RE.match(line.strip())
            if sequence_match:
                current_sequence = sequence_match.group("sequence")
                sequence_results.setdefault(current_sequence, {"sequence_name": current_sequence})
                in_frame_mean_table = False
                continue

            model_match = MODEL_RE.match(line.strip())
            if model_match and current_sequence is not None:
                current_model = model_match.group("model")
                default_model = default_model or current_model
                continue

            if current_sequence is None:
                continue

            if line.strip() == "Metric geometry (frame-mean)":
                in_frame_mean_table = True
                continue

            if in_frame_mean_table and line.startswith("+"):
                continue

            if in_frame_mean_table and not line.startswith("|"):
                in_frame_mean_table = False
                continue

            if not in_frame_mean_table:
                continue

            parsed = parse_metric_row(line)
            if parsed is None:
                continue
            scale_name, metrics = parsed
            sequence_results[current_sequence][scale_name] = metrics

    parsed_sequences = list(sequence_results.values())
    if not parsed_sequences:
        raise ValueError(f"No per-sequence metric tables were parsed from nohup.out: {path}")

    return {
        "dataset": "parsed_from_nohup",
        "depth_model": default_model or "unknown_model",
        "source_log": str(path),
        "total_sequences": len(parsed_sequences),
        "processed_sequences": len(parsed_sequences),
        "sequence_results": parsed_sequences,
    }


def load_input(path: Path, input_format: str) -> dict[str, Any]:
    format_name = input_format
    if format_name == "auto":
        format_name = "json" if path.suffix.lower() == ".json" else "nohup"
    if format_name == "json":
        return load_json(path)
    if format_name == "nohup":
        return parse_nohup(path)
    raise ValueError(f"Unsupported input format: {input_format}")


def resolve_variant_keys(variant_name: str) -> tuple[str, tuple[str, ...]]:
    normalized = str(variant_name or "").strip()
    if normalized not in VARIANT_ALIASES:
        supported = ", ".join(label for label, _ in VARIANT_ORDER)
        raise ValueError(f"Unsupported variant: {variant_name}. Supported variants: {supported}")
    return VARIANT_ALIASES[normalized]


def resolve_filter_variants(raw_value: str) -> list[tuple[str, tuple[str, ...]]]:
    variants = []
    seen = set()
    for item in str(raw_value or "").split(","):
        if not item.strip():
            continue
        label, keys = resolve_variant_keys(item)
        if label in seen:
            continue
        variants.append((label, keys))
        seen.add(label)

    if not variants:
        raise ValueError("At least one --filter-variant must be provided.")
    return variants


def get_variant_metrics(sequence_result: dict[str, Any], variant_keys: tuple[str, ...]) -> dict[str, Any] | None:
    for key in variant_keys:
        value = sequence_result.get(key)
        if isinstance(value, dict):
            return value
    return None


def get_depth_d1(sequence_result: dict[str, Any], variant_keys: tuple[str, ...]) -> float | None:
    metrics = get_variant_metrics(sequence_result, variant_keys)
    if not metrics:
        return None

    frame_mean = metrics.get("depth_frame_mean")
    if isinstance(frame_mean, dict) and frame_mean.get("delta1") is not None:
        return float(frame_mean["delta1"])

    sample_weighted = metrics.get("depth")
    if isinstance(sample_weighted, dict) and sample_weighted.get("delta1") is not None:
        return float(sample_weighted["delta1"])

    return None


def sequence_name(sequence_result: dict[str, Any]) -> str:
    return str(sequence_result.get("sequence_name") or sequence_result.get("name") or "<unknown>")


def filter_sequences(
    sequence_results: list[dict[str, Any]],
    filter_variants: list[tuple[str, tuple[str, ...]]],
    threshold: float,
    include_equal: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    rejected = []
    for sequence_result in sequence_results:
        d1_by_variant = {
            label: get_depth_d1(sequence_result, variant_keys)
            for label, variant_keys in filter_variants
        }
        matched_variants = [
            label
            for label, d1 in d1_by_variant.items()
            if d1 is not None and (d1 >= threshold if include_equal else d1 > threshold)
        ]
        passed = bool(matched_variants)
        record = {
            "sequence_name": sequence_name(sequence_result),
            "depth_d1_by_variant": d1_by_variant,
            "matched_variants": matched_variants,
            "passed": bool(passed),
        }
        if passed:
            selected.append(sequence_result)
        else:
            rejected.append(record)
    return selected, rejected


def metric_weight(metric_summary: dict[str, Any], fallback_frames: int) -> int:
    if not isinstance(metric_summary, dict):
        return 0
    value = metric_summary.get("frame_count")
    if value in (None, ""):
        value = fallback_frames
    return max(0, int(value or 0))


def weighted_mean(items: list[tuple[float | None, int]]) -> float | None:
    valid_items = [(float(value), int(weight)) for value, weight in items if value is not None and weight > 0]
    total_weight = sum(weight for _, weight in valid_items)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in valid_items) / total_weight


def aggregate_frame_mean(sequence_results: list[dict[str, Any]], variant_keys: tuple[str, ...]) -> dict[str, Any]:
    frames_requested = 0
    frames_evaluated = 0
    depth_values: dict[str, list[tuple[float | None, int]]] = {key: [] for key in METRIC_KEYS}
    point_values: dict[str, list[tuple[float | None, int]]] = {key: [] for key in METRIC_KEYS}
    depth_frame_count = 0
    point_frame_count = 0

    for sequence_result in sequence_results:
        metrics = get_variant_metrics(sequence_result, variant_keys)
        if not metrics:
            continue

        frames_requested += int(metrics.get("frames_requested", 0) or 0)
        frames_evaluated += int(metrics.get("frames_evaluated", 0) or 0)

        depth_frame_mean = metrics.get("depth_frame_mean", {})
        point_frame_mean = metrics.get("point_frame_mean", {})
        depth_weight = metric_weight(depth_frame_mean, int(metrics.get("frames_evaluated", 0) or 0))
        point_weight = metric_weight(point_frame_mean, int(metrics.get("frames_evaluated", 0) or 0))
        depth_frame_count += depth_weight
        point_frame_count += point_weight

        for key in METRIC_KEYS:
            depth_values[key].append((depth_frame_mean.get(key), depth_weight))
            point_values[key].append((point_frame_mean.get(key), point_weight))

    return {
        "frames_requested": frames_requested,
        "frames_evaluated": frames_evaluated,
        "depth_frame_mean": {
            "frame_count": depth_frame_count,
            **{key: weighted_mean(depth_values[key]) for key in METRIC_KEYS},
        },
        "point_frame_mean": {
            "frame_count": point_frame_count,
            **{key: weighted_mean(point_values[key]) for key in METRIC_KEYS},
        },
    }


def available_variants(sequence_results: list[dict[str, Any]]) -> list[tuple[str, tuple[str, ...]]]:
    variants = []
    for label, keys in VARIANT_ORDER:
        if any(get_variant_metrics(sequence_result, keys) for sequence_result in sequence_results):
            variants.append((label, keys))
    return variants


def aggregate_dataset(sequence_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        label: aggregate_frame_mean(sequence_results, keys)
        for label, keys in available_variants(sequence_results)
    }


def format_count(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(value))


def format_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}{suffix}"


def build_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows)) if rows else len(str(header))
        for index, header in enumerate(headers)
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(values: tuple[str, ...]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(values)) + " |"

    lines = [separator, render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def render_metric_rows(dataset_metrics: dict[str, dict[str, Any]]) -> list[tuple[str, ...]]:
    rows = []
    for label, _ in VARIANT_ORDER:
        metrics = dataset_metrics.get(label)
        if not metrics:
            continue
        depth = metrics["depth_frame_mean"]
        point = metrics["point_frame_mean"]
        rows.append(
            (
                label,
                format_count(depth.get("frame_count")),
                format_value(depth.get("rel")),
                format_value(depth.get("delta1")),
                format_value(depth.get("mae"), " m"),
                format_value(depth.get("rmse"), " m"),
                format_count(point.get("frame_count")),
                format_value(point.get("rel")),
                format_value(point.get("delta1")),
                format_value(point.get("mae"), " m"),
                format_value(point.get("rmse"), " m"),
            )
        )
    return rows


def render_report(
    payload: dict[str, Any],
    selected_sequences: list[dict[str, Any]],
    dataset_metrics: dict[str, dict[str, Any]],
    filter_labels: list[str],
    threshold: float,
    include_equal: bool,
) -> str:
    depth_model = payload.get("depth_model", "<unknown>")
    comparison = ">=" if include_equal else ">"
    filter_description = " OR ".join(f"{label}.Depth d1" for label in filter_labels)
    primary_label = next(iter(dataset_metrics), None)
    requested_frames = dataset_metrics.get(primary_label, {}).get("frames_requested", 0) if primary_label else 0
    variant_items = [(label, metrics) for label, metrics in dataset_metrics.items()]

    headers = (
        "Scale",
        "Depth Frames",
        "Depth Rel↓*",
        "Depth d1↑*",
        "Depth MAE↓",
        "Depth RMSE↓",
        "Point Frames",
        "Point Rel↓",
        "Point d1↑",
        "Point MAE↓",
        "Point RMSE↓",
    )

    lines = [
        "Dataset Summary",
        f"Model: {depth_model}",
        f"Filter: ({filter_description}) {comparison} {threshold:.4f}",
        f"Sequences: {len(selected_sequences)}",
        f"Requested frames: {requested_frames}",
        "Requested frames by scale: "
        + ", ".join(f"{label}={metrics['frames_requested']}" for label, metrics in variant_items),
        "Frames evaluated: "
        + ", ".join(f"{label}={metrics['frames_evaluated']}" for label, metrics in variant_items),
        "Metric geometry (frame-mean)",
        build_table(headers, render_metric_rows(dataset_metrics)),
        "* marks primary metrics.",
    ]
    return "\n".join(lines) + "\n"


def build_output_payload(
    source_payload: dict[str, Any],
    selected_sequences: list[dict[str, Any]],
    rejected_sequences: list[dict[str, Any]],
    dataset_metrics: dict[str, dict[str, Any]],
    filter_labels: list[str],
    threshold: float,
    include_equal: bool,
    copy_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "source_depth_model": source_payload.get("depth_model"),
        "source_total_sequences": source_payload.get("total_sequences"),
        "source_processed_sequences": source_payload.get("processed_sequences"),
        "filter": {
            "variants": filter_labels,
            "metric": "depth_frame_mean.delta1",
            "threshold": threshold,
            "comparison": ">=" if include_equal else ">",
            "match_mode": "any",
        },
        "processed_sequences": len(selected_sequences),
        "selected_sequence_names": [sequence_name(sequence) for sequence in selected_sequences],
        "rejected_sequences": rejected_sequences,
        "dataset": dataset_metrics,
        "sequence_results": selected_sequences,
    }
    if copy_summary is not None:
        payload["copy_summary"] = copy_summary
    return payload


def validate_sequence_dir_name(name: str) -> str:
    sequence = str(name).strip()
    if not sequence or sequence in {".", ".."}:
        raise ValueError(f"Invalid sequence name: {name!r}")
    if Path(sequence).name != sequence:
        raise ValueError(f"Invalid sequence name with path separators: {name!r}")
    return sequence


def copy_sequence_dir(
    sequence: str,
    source_root: Path,
    target_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    safe_sequence = validate_sequence_dir_name(sequence)
    source_dir = source_root / safe_sequence
    target_dir = target_root / safe_sequence

    record: dict[str, Any] = {
        "sequence_name": safe_sequence,
        "source": str(source_dir),
        "target": str(target_dir),
    }
    if not source_dir.exists():
        record["status"] = "missing_source"
        return record
    if not source_dir.is_dir():
        record["status"] = "source_not_directory"
        return record
    if target_dir.exists() and not overwrite:
        record["status"] = "skipped_existing"
        return record

    record["status"] = "copied"
    if dry_run:
        record["status"] = "dry_run"
        return record

    target_root.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    return record


def progress_iter(items: list[str], description: str):
    try:
        from tqdm import tqdm
    except ImportError:
        total = len(items)
        for index, item in enumerate(items, start=1):
            print(f"{description}: {index}/{total} {item}")
            yield item
        return

    yield from tqdm(items, desc=description, unit="seq")


def progress_completed(futures, description: str):
    futures = list(futures)
    try:
        from tqdm import tqdm
    except ImportError:
        total = len(futures)
        for index, future in enumerate(as_completed(futures), start=1):
            print(f"{description}: {index}/{total}")
            yield future
        return

    yield from tqdm(as_completed(futures), total=len(futures), desc=description, unit="seq")


def copy_sequence_dirs_parallel(
    sequence_names: list[str],
    source_root: Path,
    target_root: Path,
    overwrite: bool,
    dry_run: bool,
    workers: int,
    description: str,
) -> list[dict[str, Any]]:
    if not sequence_names:
        return []

    max_workers = max(1, int(workers or 1))
    if max_workers == 1:
        return [
            copy_sequence_dir(name, source_root, target_root, overwrite, dry_run)
            for name in progress_iter(sequence_names, description)
        ]

    records_by_sequence: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sequence = {
            executor.submit(copy_sequence_dir, name, source_root, target_root, overwrite, dry_run): name
            for name in sequence_names
        }
        for future in progress_completed(future_to_sequence, description):
            sequence = future_to_sequence[future]
            records_by_sequence[sequence] = future.result()

    return [records_by_sequence[name] for name in sequence_names]


def reset_target_roots(
    target_data_root: Path,
    target_output_root: Path,
    reset_targets: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if not reset_targets:
        return {"enabled": False, "reason": "default_preserve_targets"}

    records = []
    for target_root in (target_data_root, target_output_root):
        record = {"target": str(target_root), "existed": target_root.exists()}
        if dry_run:
            record["status"] = "dry_run"
        else:
            if target_root.exists():
                shutil.rmtree(target_root)
            target_root.mkdir(parents=True, exist_ok=True)
            record["status"] = "reset"
        records.append(record)

    return {"enabled": True, "targets": records}


def validate_copy_roots(
    source_data_root: Path,
    target_data_root: Path,
    source_output_root: Path,
    target_output_root: Path,
) -> None:
    resolved_pairs = (
        (source_data_root.resolve(), target_data_root.resolve()),
        (source_output_root.resolve(), target_output_root.resolve()),
    )
    for source_root, target_root in resolved_pairs:
        if source_root == target_root:
            raise ValueError(f"Source and target roots must be different: {source_root}")
        if source_root in target_root.parents:
            continue
        if target_root in source_root.parents:
            raise ValueError(
                f"Target root must not be a parent of source root: source={source_root}, target={target_root}"
            )


def copy_selected_sequences(
    selected_sequences: list[dict[str, Any]],
    source_data_root: Path,
    target_data_root: Path,
    source_output_root: Path,
    target_output_root: Path,
    overwrite: bool,
    reset_targets: bool,
    dry_run: bool,
    copy_workers: int,
) -> dict[str, Any]:
    sequence_names = [sequence_name(sequence) for sequence in selected_sequences]
    validate_copy_roots(source_data_root, target_data_root, source_output_root, target_output_root)

    reset_summary = reset_target_roots(
        target_data_root,
        target_output_root,
        reset_targets=reset_targets,
        dry_run=dry_run,
    )
    if not dry_run:
        target_data_root.mkdir(parents=True, exist_ok=True)
        target_output_root.mkdir(parents=True, exist_ok=True)

    data_records = copy_sequence_dirs_parallel(
        sequence_names,
        source_data_root,
        target_data_root,
        overwrite,
        dry_run,
        copy_workers,
        "Copy data",
    )
    output_records = copy_sequence_dirs_parallel(
        sequence_names,
        source_output_root,
        target_output_root,
        overwrite,
        dry_run,
        copy_workers,
        "Copy outputs",
    )

    return {
        "source_data_root": str(source_data_root),
        "target_data_root": str(target_data_root),
        "source_output_root": str(source_output_root),
        "target_output_root": str(target_output_root),
        "overwrite": bool(overwrite),
        "preserve_existing": not bool(overwrite),
        "reset_targets": reset_summary,
        "dry_run": bool(dry_run),
        "copy_workers": int(copy_workers),
        "data": data_records,
        "outputs": output_records,
    }


def print_copy_summary(copy_summary: dict[str, Any]) -> None:
    print("Copy Summary")
    print(f"Data: {copy_summary['source_data_root']} -> {copy_summary['target_data_root']}")
    print(f"Outputs: {copy_summary['source_output_root']} -> {copy_summary['target_output_root']}")
    print(f"Copy workers: {copy_summary.get('copy_workers', 1)}")
    reset_summary = copy_summary.get("reset_targets", {})
    if reset_summary.get("enabled"):
        print("Target roots reset before copying: yes")
    else:
        print("Target roots reset before copying: no")
    print(f"Preserve existing target dirs: {copy_summary.get('preserve_existing', True)}")
    for key, title in (("data", "data"), ("outputs", "outputs")):
        counts: dict[str, int] = {}
        for record in copy_summary[key]:
            status = str(record.get("status"))
            counts[status] = counts.get(status, 0) + 1
        counts_text = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        print(f"{title}: {counts_text or 'none'}")


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input_path)
    payload = load_input(input_path, args.input_format)
    sequence_results = payload.get("sequence_results", [])
    if not isinstance(sequence_results, list):
        raise ValueError("dataset_summary.json must contain a list field: sequence_results")

    filter_variants = resolve_filter_variants(args.filter_variant)
    filter_labels = [label for label, _ in filter_variants]
    selected_sequences, rejected_sequences = filter_sequences(
        sequence_results,
        filter_variants,
        args.threshold,
        args.include_equal,
    )
    dataset_metrics = aggregate_dataset(selected_sequences)
    report = render_report(
        payload,
        selected_sequences,
        dataset_metrics,
        filter_labels,
        args.threshold,
        args.include_equal,
    )
    print(report, end="")

    copy_summary = None
    if args.copy_selected:
        source_data_root = resolve_project_path(args.source_data_root)
        target_data_root = resolve_project_path(args.target_data_root)
        source_output_root = resolve_project_path(args.source_output_root)
        target_output_root = resolve_project_path(args.target_output_root)
        copy_summary = copy_selected_sequences(
            selected_sequences=selected_sequences,
            source_data_root=source_data_root,
            target_data_root=target_data_root,
            source_output_root=source_output_root,
            target_output_root=target_output_root,
            overwrite=args.overwrite,
            reset_targets=args.reset_targets,
            dry_run=args.dry_run,
            copy_workers=args.copy_workers,
        )
        print_copy_summary(copy_summary)
    else:
        target_output_root = resolve_project_path(args.target_output_root)

    output_payload = build_output_payload(
        payload,
        selected_sequences,
        rejected_sequences,
        dataset_metrics,
        filter_labels,
        args.threshold,
        args.include_equal,
        copy_summary=copy_summary,
    )

    depth_model = str(payload.get("depth_model") or "unknown_model")
    default_summary_dir = target_output_root / "metric_results" / depth_model
    output_json = resolve_project_path(args.output_json) if args.output_json else default_summary_dir / "dataset_summary.json"
    output_txt = resolve_project_path(args.output_txt) if args.output_txt else default_summary_dir / "dataset_summary.txt"

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved filtered summary JSON: {output_json}")

    if output_txt:
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(report, encoding="utf-8")
        print(f"Saved filtered summary report: {output_txt}")

    if args.list_sequences:
        print("Selected sequences:")
        for sequence in selected_sequences:
            print(sequence_name(sequence))


if __name__ == "__main__":
    main()
