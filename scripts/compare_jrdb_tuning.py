"""Compare all 17 measured optimized-scale metrics across JRDB nine-model reports."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MODELS = (
    "pi3_long", "vggt_long", "vggt_omega_long", "da3_long", "scal3r_long",
    "lingbot_map_long", "4rc_long", "dvlt_long", "stream3r_long",
)
FIELDS = (
    ("Depth Rel", 0, 2, False), ("Depth d1", 0, 3, True),
    ("Depth MAE", 0, 4, False), ("Depth RMSE", 0, 5, False),
    ("Point Rel", 0, 7, False), ("Point d1", 0, 8, True),
    ("Point MAE", 0, 9, False), ("Point RMSE", 0, 10, False),
    ("Acc", 1, 2, False), ("Comp", 1, 3, False),
    ("Chamfer", 1, 4, False), ("Pairwise MAE", 1, 5, False),
    ("Pairwise RMSE", 1, 6, False), ("ATE RMSE", 2, 3, False),
    ("ATE Mean", 2, 4, False), ("RPE Trans RMSE", 2, 6, False),
    ("RPE Rot RMSE", 2, 7, False),
)


def read_report(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text(encoding="utf-8-sig")
    result = {}
    for section in text.split("Dataset Summary\n")[1:]:
        lines = section.splitlines()
        model = next((line[7:] for line in lines if line.startswith("Model: ")), None)
        if model not in MODELS:
            continue
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in lines if line.startswith("| optimized_scale_original")
        ]
        if len(rows) != 3:
            raise ValueError(f"{path}: {model} has {len(rows)} optimized rows")
        result[model] = {}
        for name, table, column, _ in FIELDS:
            match = re.search(r"-?\d+(?:\.\d+)?", rows[table][column])
            if not match:
                raise ValueError(f"{path}: {model} missing {name}")
            result[model][name] = float(match.group())
    if set(result) != set(MODELS):
        raise ValueError(f"{path}: missing models {set(MODELS) - set(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    reference = read_report(args.reference)
    candidate = read_report(args.candidate)
    print("model\tbetter\tworse\tequal\tDepth Rel\tDepth d1\tChamfer\tATE Mean")
    for model in MODELS:
        better = worse = equal = 0
        for name, _, _, higher_better in FIELDS:
            target = reference[model][name]
            actual = candidate[model][name]
            improvement = (actual - target) if higher_better else (target - actual)
            if abs(improvement) < 1e-4:
                equal += 1
            elif improvement > 0:
                better += 1
            else:
                worse += 1
        values = [
            f"{candidate[model][name]:.4f}/{reference[model][name]:.4f}"
            for name in ("Depth Rel", "Depth d1", "Chamfer", "ATE Mean")
        ]
        print(f"{model}\t{better}\t{worse}\t{equal}\t" + "\t".join(values))


if __name__ == "__main__":
    main()
