"""Combine five measured Argoverse reports with all three complete metric tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = (
    "vggt_omega_long", "da3_long", "scal3r_long", "4rc_long", "stream3r_long",
)
TABLES = (
    "Metric geometry (frame-mean)",
    "Sparse 3D Reconstruction Proxy (",
    "TUM Pose (",
)
SCALES = ("relative_scale", "optimized_scale_original")


def extract(path: Path, model: str) -> str:
    raw = path.read_text(encoding="utf-8-sig")
    if "Dataset Summary\n" not in raw:
        raise ValueError(f"{path}: missing Dataset Summary")
    section = "Dataset Summary\n" + raw.rsplit("Dataset Summary\n", 1)[1]
    end = section.find("* marks primary metrics.")
    if end < 0:
        raise ValueError(f"{path}: missing metric-table ending")
    lines = section[: end + len("* marks primary metrics.")].splitlines()
    required = (f"Model: {model}", "Sequences: 70", "Requested frames: 6613")
    if not all(item in lines for item in required):
        raise ValueError(f"{path}: wrong model or coverage")
    frames = next((line for line in lines if line.startswith("Frames evaluated: ")), "")
    if not all(f"{scale}=6613" in frames for scale in SCALES):
        raise ValueError(f"{path}: incomplete frames")
    if any(sum(line.startswith(title) for line in lines) != 1 for title in TABLES):
        raise ValueError(f"{path}: all three metric tables are required")
    for scale in (*SCALES, "consistent_scale"):
        if sum(line.startswith("| " + scale) for line in lines) != 3:
            raise ValueError(f"{path}: expected {scale} in all three tables")
    # Preserve all headings, columns and measured values. Only omit the row
    # that the user previously asked to remove from each table.
    return "\n".join(
        line for line in lines if not line.startswith("| consistent_scale")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs=5, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for path in args.sources:
        raw = path.read_text(encoding="utf-8-sig")
        section = raw.rsplit("Dataset Summary\n", 1)[-1]
        model = next(
            (line.split(": ", 1)[1] for line in section.splitlines() if line.startswith("Model: ")),
            None,
        )
        if model not in MODELS or model in reports:
            raise ValueError(f"{path}: unexpected or duplicate model {model}")
        reports[model] = extract(path, model)
    if set(reports) != set(MODELS):
        raise ValueError(f"Missing models: {set(MODELS) - set(reports)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n--------------------------------------------------------------------------------\n".join(reports[model] for model in MODELS) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
