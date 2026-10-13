#!/usr/bin/env python3
"""Remove selected sequence folders from data and outputs roots.

By default this script only prints a dry-run plan. Pass --execute to delete.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTLIER_SEQUENCES = (
    "scene-0784",
    "scene-0218",
    "scene-0059",
    "scene-0029",
    "scene-0100",
    "scene-0770",
    "scene-0771",
    "scene-0295",
    "scene-0260",
    "scene-0187",
)


@dataclass(frozen=True)
class RemovalTarget:
    root: Path
    sequence_name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove outlier sequence folders from Metric-Depth data and outputs "
            "directories. Runs as dry-run unless --execute is set."
        )
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help=(
            "Sequence folder name to remove. Can be provided multiple times. "
            "Defaults to the 10 Acc outlier sequences."
        ),
    )
    parser.add_argument(
        "--sequence-file",
        type=Path,
        default=None,
        help="Optional text file with one sequence name per line.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Dataset root containing sequence folders. Can be provided multiple "
            "times. Default: data/official-data/nuScenes."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Outputs root containing sequence folders. Can be provided multiple "
            "times. Default: outputs/nuScenes."
        ),
    )
    parser.add_argument(
        "--include-nuscenes-new",
        action="store_true",
        help=(
            "Also scan data/official-data/nuScenes-New and outputs/nuScenes-New "
            "when explicit roots are not provided."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matched sequence folders. Without this flag only prints a plan.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: Path) -> Path:
    path = path_value.expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()


def load_sequence_file(sequence_file: Path | None) -> tuple[str, ...]:
    if sequence_file is None:
        return tuple()

    path = resolve_project_path(sequence_file)
    if not path.exists():
        raise FileNotFoundError(f"Sequence file does not exist: {path}")

    sequence_names = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sequence_names.append(line)
    return tuple(sequence_names)


def resolve_sequences(args: argparse.Namespace) -> tuple[str, ...]:
    raw_sequences = list(args.sequence or DEFAULT_OUTLIER_SEQUENCES)
    raw_sequences.extend(load_sequence_file(args.sequence_file))

    sequence_names = []
    seen = set()
    for raw_sequence in raw_sequences:
        sequence_name = str(raw_sequence).strip()
        if not sequence_name or sequence_name in seen:
            continue
        sequence_names.append(sequence_name)
        seen.add(sequence_name)
    return tuple(sequence_names)


def default_roots(include_nuscenes_new: bool) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    data_roots = [Path("data/official-data/nuScenes")]
    output_roots = [Path("outputs/nuScenes")]
    if include_nuscenes_new:
        data_roots.append(Path("data/official-data/nuScenes-New"))
        output_roots.append(Path("outputs/nuScenes-New"))
    return tuple(data_roots), tuple(output_roots)


def resolve_roots(args: argparse.Namespace) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    default_data_roots, default_output_roots = default_roots(args.include_nuscenes_new)
    raw_data_roots = args.data_root or list(default_data_roots)
    raw_output_roots = args.outputs_root or list(default_output_roots)

    data_roots = tuple(resolve_project_path(path) for path in raw_data_roots)
    output_roots = tuple(resolve_project_path(path) for path in raw_output_roots)
    return data_roots, output_roots


def collect_targets(roots: tuple[Path, ...], sequence_names: tuple[str, ...]) -> tuple[RemovalTarget, ...]:
    targets = []
    for root in roots:
        for sequence_name in sequence_names:
            sequence_path = root / sequence_name
            if sequence_path.is_dir():
                targets.append(
                    RemovalTarget(
                        root=root,
                        sequence_name=sequence_name,
                        path=sequence_path,
                    )
                )
    return tuple(targets)


def print_plan(
    sequence_names: tuple[str, ...],
    data_roots: tuple[Path, ...],
    output_roots: tuple[Path, ...],
    targets: tuple[RemovalTarget, ...],
    execute: bool,
) -> None:
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"[{mode}] Sequences requested: {len(sequence_names)}")
    print("  " + ", ".join(sequence_names))
    print(f"[{mode}] Data roots: {len(data_roots)}")
    for root in data_roots:
        print(f"  {root}")
    print(f"[{mode}] Output roots: {len(output_roots)}")
    for root in output_roots:
        print(f"  {root}")

    print(f"\n[{mode}] Matched sequence folders: {len(targets)}")
    if not targets:
        print("  No matched folders found.")
        return

    for target in targets:
        print(f"  remove: {target.path}")


def delete_targets(targets: tuple[RemovalTarget, ...]) -> None:
    for target in targets:
        shutil.rmtree(target.path)


def main() -> None:
    args = parse_args()
    sequence_names = resolve_sequences(args)
    data_roots, output_roots = resolve_roots(args)
    targets = collect_targets(data_roots + output_roots, sequence_names)

    print_plan(
        sequence_names=sequence_names,
        data_roots=data_roots,
        output_roots=output_roots,
        targets=targets,
        execute=args.execute,
    )

    if not args.execute:
        print("\nNo folders were deleted. Re-run with --execute to apply this plan.")
        return

    delete_targets(targets)
    print("\nRemoval complete.")


if __name__ == "__main__":
    main()
