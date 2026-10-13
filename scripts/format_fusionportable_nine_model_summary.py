"""Keep the original nine-model dataset-summary layout, omitting one table row.

This is a formatting pass over measured dataset_summary.txt reports. It never
substitutes a different scale variant or changes a metric value.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = (
    "pi3_long", "vggt_long", "vggt_omega_long", "da3_long", "scal3r_long",
    "lingbot_map_long", "4rc_long", "dvlt_long", "stream3r_long",
)
TABLE_TITLES = (
    "Metric geometry (frame-mean)",
    "Sparse 3D Reconstruction Proxy (",
    "TUM Pose (",
)
SCALE_ROWS = ("relative_scale", "optimized_scale_original")


def extract_report(text: str, model: str) -> str:
    marker = "Dataset Summary\n"
    if marker not in text:
        raise ValueError(f"{model}: missing Dataset Summary")
    section = text.rsplit(marker, 1)[1]
    end = section.find("* marks primary metrics.")
    if end < 0:
        raise ValueError(f"{model}: missing end of metric tables")
    section = marker + section[: end + len("* marks primary metrics.")]
    lines = section.splitlines()
    if f"Model: {model}" not in lines:
        raise ValueError(f"{model}: model name does not match source report")
    if "Sequences: 5" not in lines or "Requested frames: 3847" not in lines:
        raise ValueError(f"{model}: expected five sequences and 3847 requested frames")
    evaluated = next((line for line in lines if line.startswith("Frames evaluated: ")), "")
    if "relative_scale=3847" not in evaluated or "optimized_scale_original=3847" not in evaluated:
        raise ValueError(f"{model}: the two required variants must each evaluate 3847 frames")
    if sum(any(line.startswith(title) for title in TABLE_TITLES) for line in lines) != 3:
        raise ValueError(f"{model}: expected the original three metric tables")

    rows = {name: [] for name in (*SCALE_ROWS, "consistent_scale")}
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        label = line.split("|", 2)[1].strip()
        if label in rows:
            rows[label].append(index)
    for label in SCALE_ROWS:
        if len(rows[label]) != 3:
            raise ValueError(f"{model}: expected {label} in all three tables")
    if len(rows["consistent_scale"]) not in (0, 3):
        raise ValueError(f"{model}: incomplete consistent_scale rows")

    return "\n".join(
        line for index, line in enumerate(lines)
        if index not in rows["consistent_scale"]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True,
                        help="One combined report or nine measured dataset_summary.txt files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for path in args.sources:
        source_text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        sections = source_text.split("Dataset Summary\n")[1:]
        if not sections:
            raise ValueError(f"{path}: missing Dataset Summary")
        for section in sections:
            model_lines = [line for line in section.splitlines() if line.startswith("Model: ")]
            if not model_lines:
                raise ValueError(f"{path}: missing model name")
            model = model_lines[0].split(": ", 1)[1]
            if model not in MODELS or model in reports:
                raise ValueError(f"{path}: unexpected or duplicate model {model}")
            reports[model] = extract_report("Dataset Summary\n" + section, model)
    if set(reports) != set(MODELS):
        raise ValueError(f"Missing models: {set(MODELS) - set(reports)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(reports[model].rstrip() for model in MODELS) + "\n",
                           encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

