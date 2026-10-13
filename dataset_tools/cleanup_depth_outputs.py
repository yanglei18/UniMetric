#!/usr/bin/env python3
"""Clean per-sequence depth-model outputs under an outputs directory.

The script is intentionally dry-run by default. Pass --execute to delete files.
It skips dataset summaries, metric result folders, and run logs, and only targets
known depth-model output directories such as da3_outputs or pi3_outputs.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_OUTPUT_DIRS = (
    "da3_outputs",
    "pi3_outputs",
    "pi3x_outputs",
    "vggt_outputs",
    "vggt_omega_outputs",
    "scal3r_outputs",
    "stream3r_outputs",
    "lingbot_map_outputs",
    "dvlt_outputs",
    "4rc_outputs",
)

EXCLUDED_DIR_NAMES = frozenset(
    {
        "dataset_summaries",
        "metric_results",
        "model_summaries",
        "run_logs",
    }
)
EXCLUDED_DIR_PREFIXES = ("model_summary",)


@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    dataset: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove per-sequence depth-model output folders under outputs/. "
            "Dry-run by default; pass --execute to delete."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Outputs root. Default resolves to ./outputs from the current project.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset folder name under outputs root. Can be provided multiple times.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Sequence folder name to include. Can be provided multiple times.",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help="Model output directory name to clean, e.g. pi3_outputs. Can be repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=("model-dirs", "children"),
        default="model-dirs",
        help=(
            "model-dirs removes each matched depth-model output directory. "
            "children removes child directories inside each model output directory."
        ),
    )
    parser.add_argument(
        "--preserve-child",
        action="append",
        default=(),
        help="In --mode children, preserve child directory names. Can be repeated.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only inspect outputs/<dataset>/<sequence>/<model_dir> instead of recursive search.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matched paths. Without this flag only prints a dry-run plan.",
    )
    return parser.parse_args()


def resolve_outputs_root(raw_outputs_root: Path | None) -> Path:
    if raw_outputs_root is not None:
        return raw_outputs_root.expanduser().resolve()

    project_root = Path(__file__).resolve().parents[1]
    for candidate in (Path.cwd() / "outputs", project_root / "outputs"):
        if candidate.exists():
            return candidate.resolve()
    return (project_root / "outputs").resolve()


def should_skip_path(path: Path, stop_at: Path) -> bool:
    try:
        relative_parts = path.relative_to(stop_at).parts
    except ValueError:
        relative_parts = path.parts
    for part in relative_parts:
        if part.startswith("."):
            return True
        if part in EXCLUDED_DIR_NAMES:
            return True
        if any(part.startswith(prefix) for prefix in EXCLUDED_DIR_PREFIXES):
            return True
    return False


def iter_dataset_dirs(outputs_root: Path, dataset_names: set[str] | None) -> list[Path]:
    if not outputs_root.exists():
        print(f"[WARN] outputs root does not exist: {outputs_root}")
        return []
    if dataset_names is None:
        return sorted(path for path in outputs_root.iterdir() if path.is_dir() and not path.name.startswith("."))

    dataset_dirs = []
    for dataset_name in sorted(dataset_names):
        dataset_dir = outputs_root / dataset_name
        if dataset_dir.is_dir():
            dataset_dirs.append(dataset_dir)
        else:
            print(f"[WARN] dataset directory not found, skipped: {dataset_dir}")
    return dataset_dirs


def iter_search_roots(dataset_dir: Path, sequence_names: set[str] | None) -> list[Path]:
    if sequence_names is None:
        return [dataset_dir]

    roots = []
    for sequence_name in sorted(sequence_names):
        sequence_dir = dataset_dir / sequence_name
        if sequence_dir.is_dir():
            roots.append(sequence_dir)
        else:
            print(f"[WARN] sequence directory not found, skipped: {sequence_dir}")
    return roots


def is_model_output_dir(path: Path, model_dir_names: set[str]) -> bool:
    if path.name in model_dir_names:
        return True
    if path.name.endswith("_outputs") and path.name in model_dir_names:
        return True
    return False


def collect_model_dirs(
    dataset_dir: Path,
    sequence_names: set[str] | None,
    model_dir_names: set[str],
    recursive: bool,
) -> list[Path]:
    model_dirs: list[Path] = []
    seen: set[Path] = set()
    for root in iter_search_roots(dataset_dir, sequence_names):
        if should_skip_path(root, dataset_dir):
            continue
        candidates = root.rglob("*") if recursive else root.glob("*/*")
        for candidate in candidates:
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            if should_skip_path(candidate, dataset_dir):
                continue
            if not is_model_output_dir(candidate, model_dir_names):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                model_dirs.append(candidate)
    return sorted(model_dirs)


def collect_candidates(
    outputs_root: Path,
    dataset_names: set[str] | None,
    sequence_names: set[str] | None,
    model_dir_names: set[str],
    mode: str,
    preserve_children: set[str],
    recursive: bool,
) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []
    for dataset_dir in iter_dataset_dirs(outputs_root, dataset_names):
        model_dirs = collect_model_dirs(dataset_dir, sequence_names, model_dir_names, recursive)
        for model_dir in model_dirs:
            if mode == "model-dirs":
                candidates.append(CleanupCandidate(model_dir, dataset_dir.name, "model output directory"))
                continue

            for child in sorted(model_dir.iterdir()):
                if not child.is_dir() or child.is_symlink() or child.name in preserve_children:
                    continue
                candidates.append(CleanupCandidate(child, dataset_dir.name, f"child of {model_dir.name}"))
    return candidates


def print_plan(outputs_root: Path, candidates: list[CleanupCandidate], execute: bool) -> None:
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"[{mode}] outputs root: {outputs_root}")
    print(f"[{mode}] matched paths: {len(candidates)}")

    by_dataset: dict[str, int] = {}
    for candidate in candidates:
        by_dataset[candidate.dataset] = by_dataset.get(candidate.dataset, 0) + 1
    for dataset, count in sorted(by_dataset.items()):
        print(f"[{mode}] {dataset}: {count}")

    for candidate in candidates:
        print(f"remove: {candidate.path}  ({candidate.reason})")


def delete_candidates(candidates: list[CleanupCandidate]) -> None:
    for candidate in candidates:
        shutil.rmtree(candidate.path)


def main() -> None:
    args = parse_args()
    outputs_root = resolve_outputs_root(args.outputs_root)
    dataset_names = None if args.dataset is None else set(args.dataset)
    sequence_names = None if args.sequence is None else set(args.sequence)
    model_dir_names = set(DEFAULT_MODEL_OUTPUT_DIRS) if args.model_dir is None else set(args.model_dir)
    preserve_children = set(args.preserve_child)

    candidates = collect_candidates(
        outputs_root=outputs_root,
        dataset_names=dataset_names,
        sequence_names=sequence_names,
        model_dir_names=model_dir_names,
        mode=args.mode,
        preserve_children=preserve_children,
        recursive=not args.no_recursive,
    )
    print_plan(outputs_root, candidates, execute=args.execute)

    if not args.execute:
        print("\nNo files were deleted. Re-run with --execute to apply this plan.")
        return

    delete_candidates(candidates)
    print("\nCleanup complete.")


if __name__ == "__main__":
    main()
