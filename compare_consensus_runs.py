#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any


IGNORED_FLATTEN_KEYS = {"per_frame", "anchor_measurements", "bi_segments"}


@dataclass(frozen=True)
class RunArtifacts:
    label: str
    root_path: str
    dataset_summary_path: str | None
    sequence_results: dict[str, str]
    debug_results: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare consensus-solver runs on the same batch of sequences."
    )
    parser.add_argument("--run-a", required=True, help="Path to the first run root, metrics dir, or dataset_summary.json")
    parser.add_argument("--run-b", required=True, help="Path to the second run root, metrics dir, or dataset_summary.json")
    parser.add_argument("--label-a", default="run_a", help="Display label for the first run")
    parser.add_argument("--label-b", default="run_b", help="Display label for the second run")
    parser.add_argument("--metrics-a", default=None, help="Optional explicit dataset_summary.json or metric_results dir for run A")
    parser.add_argument("--metrics-b", default=None, help="Optional explicit dataset_summary.json or metric_results dir for run B")
    parser.add_argument("--debug-a", default=None, help="Optional explicit debug root for run A")
    parser.add_argument("--debug-b", default=None, help="Optional explicit debug root for run B")
    parser.add_argument("--output-json", default=None, help="Optional output JSON report path")
    parser.add_argument("--output-md", default=None, help="Optional output Markdown report path")
    return parser.parse_args()


def read_json(file_path: str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_parent_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def find_first_file(root_path: str, filename: str) -> str | None:
    if not os.path.exists(root_path):
        return None
    if os.path.isfile(root_path):
        return root_path if os.path.basename(root_path) == filename else None
    for current_root, _, files in os.walk(root_path):
        if filename in files:
            return os.path.join(current_root, filename)
    return None


def find_json_files(root_path: str, predicate: Any) -> list[str]:
    if not root_path or not os.path.exists(root_path):
        return []
    if os.path.isfile(root_path):
        return [root_path] if predicate(root_path) else []

    matched_paths: list[str] = []
    for current_root, _, files in os.walk(root_path):
        for name in files:
            candidate = os.path.join(current_root, name)
            if predicate(candidate):
                matched_paths.append(candidate)
    return sorted(matched_paths)


def infer_sequence_name(file_path: str, payload: dict[str, Any]) -> str:
    sequence_name = payload.get("sequence_name")
    if sequence_name:
        return str(sequence_name)

    parts = os.path.normpath(file_path).split(os.sep)
    if "semantic_scale_debug" in parts:
        index = parts.index("semantic_scale_debug")
        if index >= 2:
            return parts[index - 2]
    return "unknown_sequence"


def flatten_metrics(data: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key in IGNORED_FLATTEN_KEYS:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_metrics(value, next_prefix))
        return flattened

    if isinstance(data, (int, float)) and not isinstance(data, bool):
        flattened[prefix] = float(data)
    return flattened


def interval_width(interval: Any) -> float | None:
    if not isinstance(interval, list) or len(interval) != 2:
        return None
    try:
        left = float(interval[0])
        right = float(interval[1])
    except (TypeError, ValueError):
        return None
    return right - left


def resolve_dataset_summary_path(root_path: str, override: str | None) -> str | None:
    candidate = override or root_path
    if not candidate:
        return None
    if os.path.isfile(candidate):
        return candidate
    return find_first_file(candidate, "dataset_summary.json")


def collect_sequence_results(root_path: str) -> dict[str, str]:
    paths = find_json_files(
        root_path,
        lambda path: path.endswith(".json") and os.path.basename(os.path.dirname(path)) == "sequences",
    )
    results: dict[str, str] = {}
    for path in paths:
        sequence_name = os.path.splitext(os.path.basename(path))[0]
        results[sequence_name] = path
    return results


def collect_debug_results(root_path: str) -> dict[str, str]:
    paths = find_json_files(
        root_path,
        lambda path: path.endswith(".json") and os.path.basename(os.path.dirname(path)) == "semantic_scale_debug",
    )
    results: dict[str, str] = {}
    for path in paths:
        payload = read_json(path)
        sequence_name = infer_sequence_name(path, payload)
        key = f"{sequence_name}/{os.path.basename(path)}"
        results[key] = path
    return results


def resolve_run_artifacts(
    label: str,
    run_path: str,
    metrics_override: str | None,
    debug_override: str | None,
) -> RunArtifacts:
    dataset_summary_path = resolve_dataset_summary_path(run_path, metrics_override)
    metrics_root = os.path.dirname(dataset_summary_path) if dataset_summary_path else (metrics_override or run_path)
    debug_root = debug_override or run_path
    return RunArtifacts(
        label=label,
        root_path=os.path.abspath(run_path),
        dataset_summary_path=os.path.abspath(dataset_summary_path) if dataset_summary_path else None,
        sequence_results=collect_sequence_results(metrics_root),
        debug_results=collect_debug_results(debug_root),
    )


def build_numeric_delta_map(a_data: dict[str, Any], b_data: dict[str, Any]) -> dict[str, dict[str, float]]:
    flat_a = flatten_metrics(a_data)
    flat_b = flatten_metrics(b_data)
    deltas: dict[str, dict[str, float]] = {}
    for key in sorted(set(flat_a) & set(flat_b)):
        deltas[key] = {
            "a": flat_a[key],
            "b": flat_b[key],
            "delta_b_minus_a": flat_b[key] - flat_a[key],
        }
    return deltas


def compare_dataset_summaries(run_a: RunArtifacts, run_b: RunArtifacts) -> dict[str, Any]:
    if not run_a.dataset_summary_path or not run_b.dataset_summary_path:
        return {"matched": False}

    payload_a = read_json(run_a.dataset_summary_path)
    payload_b = read_json(run_b.dataset_summary_path)
    return {
        "matched": True,
        "path_a": run_a.dataset_summary_path,
        "path_b": run_b.dataset_summary_path,
        "dataset_name_a": payload_a.get("dataset"),
        "dataset_name_b": payload_b.get("dataset"),
        "depth_model_a": payload_a.get("depth_model"),
        "depth_model_b": payload_b.get("depth_model"),
        "processed_sequences_a": payload_a.get("processed_sequences"),
        "processed_sequences_b": payload_b.get("processed_sequences"),
        "dataset_metric_deltas": build_numeric_delta_map(payload_a.get("dataset", {}), payload_b.get("dataset", {})),
    }


def compare_sequence_results(run_a: RunArtifacts, run_b: RunArtifacts) -> dict[str, Any]:
    matched_sequences = sorted(set(run_a.sequence_results) & set(run_b.sequence_results))
    sequence_deltas: dict[str, Any] = {}

    for sequence_name in matched_sequences:
        payload_a = read_json(run_a.sequence_results[sequence_name])
        payload_b = read_json(run_b.sequence_results[sequence_name])
        sequence_deltas[sequence_name] = build_numeric_delta_map(payload_a, payload_b)

    return {
        "matched_sequence_count": len(matched_sequences),
        "sequence_names": matched_sequences,
        "sequence_metric_deltas": sequence_deltas,
    }


def summarize_debug_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "sequence_name": payload.get("sequence_name"),
        "frame_prefix": payload.get("frame_prefix"),
        "consistent_scale": payload.get("consistent_scale"),
        "optimized_scale": payload.get("optimized_scale"),
        "consensus_solver": payload.get("consensus_solver"),
        "consensus_scale": payload.get("consensus_scale"),
        "scale_interval_count": payload.get("scale_interval_count"),
        "bi_scale": payload.get("bi_scale"),
        "bi_keep_threshold": payload.get("bi_keep_threshold"),
        "bi_total_weight": payload.get("bi_total_weight"),
        "bi_kept_segment_count": payload.get("bi_kept_segment_count"),
        "bi_all_segment_count": payload.get("bi_all_segment_count"),
        "ifpa_scale": payload.get("ifpa_scale"),
        "ifpa_best_score": payload.get("ifpa_best_score"),
        "ifpa_grid_size": payload.get("ifpa_grid_size"),
        "ifpa_use_weight": payload.get("ifpa_use_weight"),
        "ifpa_top_point_count": payload.get("ifpa_top_point_count"),
    }
    summary["consensus_interval_width"] = interval_width(payload.get("consensus_interval"))
    summary["bi_interval_width"] = interval_width(payload.get("bi_interval"))
    summary["ifpa_interval_width"] = interval_width(payload.get("ifpa_interval"))
    return summary


def aggregate_debug_deltas(per_item: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for _, metrics in per_item.items():
        for key, delta_info in metrics.items():
            buckets.setdefault(key, []).append(delta_info["delta_b_minus_a"])

    return {
        key: {
            "count": float(len(values)),
            "mean_delta_b_minus_a": sum(values) / len(values),
            "min_delta_b_minus_a": min(values),
            "max_delta_b_minus_a": max(values),
        }
        for key, values in sorted(buckets.items())
        if values
    }


def compare_debug_results(run_a: RunArtifacts, run_b: RunArtifacts) -> dict[str, Any]:
    matched_keys = sorted(set(run_a.debug_results) & set(run_b.debug_results))
    per_item: dict[str, dict[str, dict[str, float]]] = {}
    solver_counts_a: dict[str, int] = {}
    solver_counts_b: dict[str, int] = {}

    for key in matched_keys:
        payload_a = summarize_debug_payload(read_json(run_a.debug_results[key]))
        payload_b = summarize_debug_payload(read_json(run_b.debug_results[key]))
        solver_a = str(payload_a.get("consensus_solver"))
        solver_b = str(payload_b.get("consensus_solver"))
        solver_counts_a[solver_a] = solver_counts_a.get(solver_a, 0) + 1
        solver_counts_b[solver_b] = solver_counts_b.get(solver_b, 0) + 1
        per_item[key] = build_numeric_delta_map(payload_a, payload_b)

    return {
        "matched_debug_count": len(matched_keys),
        "debug_keys": matched_keys,
        "solver_counts_a": solver_counts_a,
        "solver_counts_b": solver_counts_b,
        "aggregated_metric_deltas": aggregate_debug_deltas(per_item),
        "per_debug_deltas": per_item,
    }


def make_default_output_path(args: argparse.Namespace, suffix: str) -> str:
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(root_dir, exist_ok=True)
    file_name = f"consensus_compare_{args.label_a}_vs_{args.label_b}.{suffix}"
    return os.path.join(root_dir, file_name)


def render_markdown_report(report: dict[str, Any]) -> str:
    run_a = report["runs"]["a"]
    run_b = report["runs"]["b"]
    dataset_summary = report["dataset_summary_comparison"]
    sequence_summary = report["sequence_comparison"]
    debug_summary = report["debug_comparison"]

    lines = [
        "# Consensus Solver Comparison",
        "",
        f"- Run A: `{run_a['label']}` -> `{run_a['root_path']}`",
        f"- Run B: `{run_b['label']}` -> `{run_b['root_path']}`",
        "",
    ]

    if dataset_summary.get("matched"):
        lines.extend(
            [
                "## Dataset Metrics",
                "",
                f"- Dataset summary A: `{dataset_summary['path_a']}`",
                f"- Dataset summary B: `{dataset_summary['path_b']}`",
                f"- Processed sequences: A={dataset_summary['processed_sequences_a']}, B={dataset_summary['processed_sequences_b']}",
                "",
            ]
        )
        for key, delta in list(dataset_summary["dataset_metric_deltas"].items())[:24]:
            lines.append(f"- `{key}`: A={delta['a']:.6f}, B={delta['b']:.6f}, Δ={delta['delta_b_minus_a']:.6f}")
        lines.append("")

    lines.extend(
        [
            "## Sequence Coverage",
            "",
            f"- Matched sequence count: {sequence_summary['matched_sequence_count']}",
            f"- Matched debug file count: {debug_summary['matched_debug_count']}",
            f"- Solver counts A: `{debug_summary['solver_counts_a']}`",
            f"- Solver counts B: `{debug_summary['solver_counts_b']}`",
            "",
            "## Aggregated Debug Deltas",
            "",
        ]
    )
    for key, stats in list(debug_summary["aggregated_metric_deltas"].items())[:24]:
        lines.append(
            f"- `{key}`: count={int(stats['count'])}, meanΔ={stats['mean_delta_b_minus_a']:.6f}, "
            f"minΔ={stats['min_delta_b_minus_a']:.6f}, maxΔ={stats['max_delta_b_minus_a']:.6f}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_a = resolve_run_artifacts(args.label_a, args.run_a, args.metrics_a, args.debug_a)
    run_b = resolve_run_artifacts(args.label_b, args.run_b, args.metrics_b, args.debug_b)

    report = {
        "runs": {
            "a": {
                "label": run_a.label,
                "root_path": run_a.root_path,
                "dataset_summary_path": run_a.dataset_summary_path,
                "sequence_result_count": len(run_a.sequence_results),
                "debug_result_count": len(run_a.debug_results),
            },
            "b": {
                "label": run_b.label,
                "root_path": run_b.root_path,
                "dataset_summary_path": run_b.dataset_summary_path,
                "sequence_result_count": len(run_b.sequence_results),
                "debug_result_count": len(run_b.debug_results),
            },
        },
        "dataset_summary_comparison": compare_dataset_summaries(run_a, run_b),
        "sequence_comparison": compare_sequence_results(run_a, run_b),
        "debug_comparison": compare_debug_results(run_a, run_b),
    }

    output_json = args.output_json or make_default_output_path(args, "json")
    ensure_parent_dir(output_json)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    output_md = args.output_md or make_default_output_path(args, "md")
    ensure_parent_dir(output_md)
    with open(output_md, "w", encoding="utf-8") as file:
        file.write(render_markdown_report(report))

    print(f"Saved JSON report to: {output_json}")
    print(f"Saved Markdown report to: {output_md}")


if __name__ == "__main__":
    main()
