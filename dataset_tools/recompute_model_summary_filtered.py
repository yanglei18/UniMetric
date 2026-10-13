#!/usr/bin/env python3
"""Recompute model dataset summaries after removing a sequence list.

The script parses per-sequence metric logs (dataset_summary.txt), removes selected
sequences, then rebuilds:
- per-model *_dataset_summary.txt
- aggregated all_models_dataset_summary.txt

It supports the three known evaluation tables:
- Metric geometry (frame-mean)
- Sparse 3D Reconstruction Proxy (subsequence-mean)
- TUM Pose (subsequence)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALES = ("relative_scale", "consistent_scale", "optimized_scale_original")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute filtered Dataset Summary files for specific depth models."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing original *_dataset_summary.txt files.",
    )
    parser.add_argument(
        "--remove-list",
        type=Path,
        required=True,
        help="Text file with one sequence ID per line (remove these sequences).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for filtered summaries.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model name(s) to process. Defaults to all *_dataset_summary.txt files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned output files without writing.",
    )
    return parser.parse_args()


def parse_number(value: str) -> float | None:
    cleaned = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+\.\d+|-?\d+", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def fmt_float(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.4f}{suffix}"


def resolve_project_path(path_value: Path) -> Path:
    value = path_value.expanduser()
    if value.is_absolute():
        return value.resolve()
    resolved = (Path.cwd() / value).resolve()
    if resolved.exists():
        return resolved
    return (PROJECT_ROOT / value).resolve()


def load_remove_set(path: Path) -> set[str]:
    remove_set: set[str] = set()
    resolved = resolve_project_path(path)
    for line in resolved.read_text(encoding="utf-8").splitlines():
        seq_id = line.strip()
        if seq_id:
            remove_set.add(seq_id)
    return remove_set


def parse_scale_block(line: str, mode: str) -> dict[str, float] | None:
    if not line.startswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 2:
        return None

    scale = cells[0]
    if scale not in SCALES:
        return None

    if mode == "metric":
        if len(cells) < 11:
            return None
        return {
            "scale": scale,
            "depth_frames": parse_number(cells[1]),
            "depth_rel": parse_number(cells[2]),
            "depth_d1": parse_number(cells[3]),
            "depth_mae": parse_number(cells[4]),
            "depth_rmse": parse_number(cells[5]),
            "point_frames": parse_number(cells[6]),
            "point_rel": parse_number(cells[7]),
            "point_d1": parse_number(cells[8]),
            "point_mae": parse_number(cells[9]),
            "point_rmse": parse_number(cells[10]),
        }

    if mode == "sparse":
        if len(cells) < 7:
            return None
        return {
            "scale": scale,
            "groups": parse_number(cells[1]),
            "acc": parse_number(cells[2]),
            "comp": parse_number(cells[3]),
            "chamfer": parse_number(cells[4]),
            "pairwise_mae": parse_number(cells[5]),
            "pairwise_rmse": parse_number(cells[6]),
        }

    if mode == "pose":
        if len(cells) < 8:
            return None
        return {
            "scale": scale,
            "groups": parse_number(cells[1]),
            "frames": parse_number(cells[2]),
            "ate_rmse": parse_number(cells[3]),
            "ate_mean": parse_number(cells[4]),
            "rpe_pairs": parse_number(cells[5]),
            "rpe_trans": parse_number(cells[6]),
            "rpe_rot": parse_number(cells[7]),
        }
    return None


def parse_model_summary(path: Path) -> dict[str, Any]:
    sequence_results: dict[str, dict[str, Any]] = {}
    current_seq: str | None = None
    mode: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip("\n")
            if line.startswith("Sequence:"):
                current_seq = line.split(":", 1)[1].strip()
                sequence_results[current_seq] = {}
                mode = None
                continue
            if line.startswith("Dataset Summary") or line.startswith("Sequence Difficulty Summary"):
                break
            if line.startswith("Metric geometry (frame-mean)"):
                mode = "metric"
                continue
            if line.startswith("Sparse 3D Reconstruction Proxy"):
                mode = "sparse"
                continue
            if line.startswith("TUM Pose"):
                mode = "pose"
                continue
            if current_seq is None or mode is None or not line.startswith("|"):
                continue
            if line.startswith("+") or "| Scale" in line:
                continue

            row = parse_scale_block(line, mode)
            if row is None:
                continue
            scale = str(row["scale"])
            record = sequence_results[current_seq]
            if mode == "metric":
                record.setdefault("metric", {})[scale] = row
            elif mode == "sparse":
                record.setdefault("sparse", {})[scale] = row
            else:
                record.setdefault("pose", {})[scale] = row

    if not sequence_results:
        raise ValueError(f"No sequence tables parsed from {path}")

    return sequence_results


def aggregate(sequence_results: dict[str, dict[str, Any]], removed: set[str]) -> dict[str, Any]:
    kept = {sid: data for sid, data in sequence_results.items() if sid not in removed}

    metric: dict[str, dict[str, float]] = {
        scale: {
            "depth_frames": 0.0,
            "point_frames": 0.0,
            "depth_rel_num": 0.0,
            "depth_d1_num": 0.0,
            "depth_mae_num": 0.0,
            "depth_rmse_num": 0.0,
            "point_rel_num": 0.0,
            "point_d1_num": 0.0,
            "point_mae_num": 0.0,
            "point_rmse_num": 0.0,
        }
        for scale in SCALES
    }
    sparse: dict[str, dict[str, float]] = {
        scale: {
            "groups": 0.0,
            "acc_num": 0.0,
            "comp_num": 0.0,
            "cham_num": 0.0,
            "pairwise_mae_num": 0.0,
            "pairwise_rmse_num": 0.0,
        }
        for scale in SCALES
    }
    pose: dict[str, dict[str, float]] = {
        scale: {
            "groups": 0.0,
            "frames": 0.0,
            "ate_rmse_num": 0.0,
            "ate_mean_num": 0.0,
            "rpe_pairs": 0.0,
            "rpe_trans_num": 0.0,
            "rpe_rot_num": 0.0,
        }
        for scale in SCALES
    }

    for seq_data in kept.values():
        for scale in SCALES:
            metric_row = seq_data.get("metric", {}).get(scale)
            if metric_row:
                dfr = metric_row["depth_frames"] or 0.0
                pfr = metric_row["point_frames"] or 0.0
                agg = metric[scale]
                agg["depth_frames"] += dfr
                agg["point_frames"] += pfr
                agg["depth_rel_num"] += (metric_row["depth_rel"] or 0.0) * dfr
                agg["depth_d1_num"] += (metric_row["depth_d1"] or 0.0) * dfr
                agg["depth_mae_num"] += (metric_row["depth_mae"] or 0.0) * dfr
                agg["depth_rmse_num"] += (metric_row["depth_rmse"] or 0.0) * dfr
                agg["point_rel_num"] += (metric_row["point_rel"] or 0.0) * pfr
                agg["point_d1_num"] += (metric_row["point_d1"] or 0.0) * pfr
                agg["point_mae_num"] += (metric_row["point_mae"] or 0.0) * pfr
                agg["point_rmse_num"] += (metric_row["point_rmse"] or 0.0) * pfr

            sparse_row = seq_data.get("sparse", {}).get(scale)
            if sparse_row:
                g = sparse_row["groups"] or 0.0
                agg = sparse[scale]
                agg["groups"] += g
                agg["acc_num"] += (sparse_row["acc"] or 0.0) * g
                agg["comp_num"] += (sparse_row["comp"] or 0.0) * g
                agg["cham_num"] += (sparse_row["chamfer"] or 0.0) * g
                agg["pairwise_mae_num"] += (sparse_row["pairwise_mae"] or 0.0) * g
                agg["pairwise_rmse_num"] += (sparse_row["pairwise_rmse"] or 0.0) * g

            pose_row = seq_data.get("pose", {}).get(scale)
            if pose_row:
                f = pose_row["frames"] or 0.0
                agg = pose[scale]
                agg["groups"] += pose_row["groups"] or 0.0
                agg["frames"] += f
                agg["ate_rmse_num"] += ((pose_row["ate_rmse"] or 0.0) ** 2) * f
                agg["ate_mean_num"] += (pose_row["ate_mean"] or 0.0) * f
                agg["rpe_pairs"] += pose_row["rpe_pairs"] or 0.0
                rpe_pairs = pose_row["rpe_pairs"] or 0.0
                agg["rpe_trans_num"] += ((pose_row["rpe_trans"] or 0.0) ** 2) * rpe_pairs
                agg["rpe_rot_num"] += ((pose_row["rpe_rot"] or 0.0) ** 2) * rpe_pairs

    summary: dict[str, Any] = {
        "sequences": len(kept),
        "metric": {},
        "sparse": {},
        "pose": {},
    }

    for scale in SCALES:
        m = metric[scale]
        dfr = m["depth_frames"] or 0.0
        pfr = m["point_frames"] or 0.0
        if dfr > 0:
            rel = m["depth_rel_num"] / dfr
            d1 = m["depth_d1_num"] / dfr
            mae = m["depth_mae_num"] / dfr
            rmse = m["depth_rmse_num"] / dfr
        else:
            rel = d1 = mae = rmse = None
        if pfr > 0:
            prel = m["point_rel_num"] / pfr
            pd1 = m["point_d1_num"] / pfr
            pmae = m["point_mae_num"] / pfr
            prmse = m["point_rmse_num"] / pfr
        else:
            prel = pd1 = pmae = prmse = None
        summary["metric"][scale] = {
            "depth_frames": dfr,
            "depth_rel": rel,
            "depth_d1": d1,
            "depth_mae": mae,
            "depth_rmse": rmse,
            "point_frames": pfr,
            "point_rel": prel,
            "point_d1": pd1,
            "point_mae": pmae,
            "point_rmse": prmse,
        }

        s = sparse[scale]
        sg = s["groups"] or 0.0
        if sg > 0:
            summary["sparse"][scale] = {
                "groups": sg,
                "acc": s["acc_num"] / sg,
                "comp": s["comp_num"] / sg,
                "chamfer": s["cham_num"] / sg,
                "pairwise_mae": s["pairwise_mae_num"] / sg,
                "pairwise_rmse": s["pairwise_rmse_num"] / sg,
            }
        else:
            summary["sparse"][scale] = {"groups": 0.0, "acc": None, "comp": None, "chamfer": None, "pairwise_mae": None, "pairwise_rmse": None}

        p = pose[scale]
        pf = p["frames"] or 0.0
        if pf > 0:
            summary["pose"][scale] = {
                "groups": p["groups"] or 0.0,
                "frames": pf,
                "ate_rmse": (p["ate_rmse_num"] / pf) ** 0.5,
                "ate_mean": p["ate_mean_num"] / pf,
                "rpe_pairs": p["rpe_pairs"],
                "rpe_trans": (p["rpe_trans_num"] / p["rpe_pairs"]) ** 0.5 if p["rpe_pairs"] > 0 else None,
                "rpe_rot": (p["rpe_rot_num"] / p["rpe_pairs"]) ** 0.5 if p["rpe_pairs"] > 0 else None,
            }
        else:
            summary["pose"][scale] = {"groups": 0.0, "frames": 0.0, "ate_rmse": None, "ate_mean": None, "rpe_pairs": 0.0, "rpe_trans": None, "rpe_rot": None}

    return summary


def build_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(header) for header in headers]
    for col in range(len(headers)):
        cell_max = max((len(row[col]) for row in rows if len(row) > col), default=0)
        widths[col] = max(widths[col], cell_max)

    sep = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def row(values: tuple[str, ...]) -> str:
        return "| " + " | ".join(str(values[idx]).ljust(widths[idx]) for idx in range(len(values))) + " |"

    out = [sep, row(headers), sep]
    out.extend(row(r) for r in rows)
    out.append(sep)
    return "\n".join(out)


def render_model_summary(model: str, summary: dict[str, Any], source: str) -> str:
    metric_headers = (
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
    metric_rows = []
    for scale in SCALES:
        m = summary["metric"][scale]
        metric_rows.append(
            (
                scale,
                f"{int(m['depth_frames'])}",
                fmt_float(m["depth_rel"]),
                fmt_float(m["depth_d1"]),
                fmt_float(m["depth_mae"], " m"),
                fmt_float(m["depth_rmse"], " m"),
                f"{int(m['point_frames'])}",
                fmt_float(m["point_rel"]),
                fmt_float(m["point_d1"]),
                fmt_float(m["point_mae"], " m"),
                fmt_float(m["point_rmse"], " m"),
            )
        )

    sparse_headers = (
        "Scale",
        "Groups",
        "Acc↓*",
        "Comp↓*",
        "Chamfer↓",
        "Pairwise MAE↓",
        "Pairwise RMSE↓",
    )
    sparse_rows = []
    for scale in SCALES:
        s = summary["sparse"][scale]
        sparse_rows.append(
            (
                scale,
                f"{int(s['groups'])}",
                fmt_float(s["acc"], " m"),
                fmt_float(s["comp"], " m"),
                fmt_float(s["chamfer"], " m"),
                fmt_float(s["pairwise_mae"], " m"),
                fmt_float(s["pairwise_rmse"], " m"),
            )
        )

    pose_headers = (
        "Scale",
        "Groups",
        "Frames",
        "ATE RMSE↓*",
        "ATE Mean↓",
        "RPE Pairs",
        "RPE Trans RMSE↓*",
        "RPE Rot RMSE↓",
    )
    pose_rows = []
    for scale in SCALES:
        p = summary["pose"][scale]
        pose_rows.append(
            (
                scale,
                f"{int(p['groups'])}",
                f"{int(p['frames'])}",
                fmt_float(p["ate_rmse"], " m"),
                fmt_float(p["ate_mean"], " m"),
                f"{int(p['rpe_pairs'])}",
                fmt_float(p["rpe_trans"], " m"),
                fmt_float(p["rpe_rot"], " deg"),
            )
        )

    requested = summary["metric"][SCALES[0]]["depth_frames"]
    requested_text = f"{int(requested)}"
    requested_by_scale = ", ".join(
        f"{scale}={int(summary['metric'][scale]['depth_frames'])}" for scale in SCALES
    )
    evaluated_by_scale = ", ".join(
        f"{scale}={int(summary['pose'][scale]['frames'])}" for scale in SCALES
    )

    lines = [
        "================================================================================",
        "Dataset Summary",
        f"Model: {model}",
        f"Source: {source}",
        f"Sequences: {summary['sequences']}",
        f"Requested frames: {requested_text}",
        f"Requested frames by scale: {requested_by_scale}",
        f"Frames evaluated: {evaluated_by_scale}",
        "Metric geometry (frame-mean)",
        build_table(metric_headers, metric_rows),
        "Sparse 3D Reconstruction Proxy (point-cloud global SE3 aligned, subsequence-mean)",
        build_table(sparse_headers, sparse_rows),
        "TUM Pose (subsequence, global SE3 alignment, scale=1.0000)",
        build_table(pose_headers, pose_rows),
        "* marks primary metrics.",
    ]
    return "\n".join(lines) + "\n"


def build_all_models_report(model_summaries: dict[str, tuple[dict[str, Any], str]]) -> str:
    lines = []
    for idx, (model, (summary, source)) in enumerate(model_summaries.items()):
        model_lines = [
            "================================================================================",
            f"Model: {model}",
            f"Source: {source}",
            "--------------------------------------------------------------------------------",
        ]
        model_lines.append(render_model_summary(model, summary, source).replace(
            "================================================================================\nDataset Summary\nModel: " + model + "\n", ""
        ))
        block = "\n".join(model_lines).rstrip()
        lines.append(block)
    return "\n".join(lines) + "\n"


def resolve_models(source_dir: Path, model_names: list[str] | None) -> list[str]:
    if model_names:
        return model_names

    models = []
    for path in sorted(source_dir.glob("*_dataset_summary.txt")):
        if path.name.endswith("_dataset_summary.txt"):
            model = path.name.removesuffix("_dataset_summary.txt")
            if model in {"all_models", "all_models_dataset_summary"}:
                continue
            models.append(model)
    return models


def main() -> None:
    args = parse_args()
    source_dir = resolve_project_path(args.source_dir)
    output_dir = resolve_project_path(args.output_dir)
    removed = load_remove_set(args.remove_list)
    models = resolve_models(source_dir, args.model)

    print(f"[INFO] Source dir: {source_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Models: {len(models)}")
    print(f"[INFO] Removed IDs: {len(removed)}")

    model_summaries: dict[str, tuple[dict[str, Any], str]] = {}

    for model in models:
        source_file = source_dir / f"{model}_dataset_summary.txt"
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source file: {source_file}")
        parsed = parse_model_summary(source_file)
        summary = aggregate(parsed, removed)
        model_summaries[model] = (summary, str(source_file))
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{model}_dataset_summary.txt").write_text(
                render_model_summary(model, summary, str(source_file)),
                encoding="utf-8",
            )
        print(f"[INFO] {model}: kept={summary['sequences']}")

    if not args.dry_run:
        (output_dir / "all_models_dataset_summary.txt").write_text(
            build_all_models_report(model_summaries),
            encoding="utf-8",
        )
        (output_dir / "removed_sequence_ids.txt").write_text("\n".join(sorted(removed)) + "\n", encoding="utf-8")
    print("[DONE] Recompute complete.")


if __name__ == "__main__":
    main()
