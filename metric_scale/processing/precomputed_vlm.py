"""Restore bundled VLM results without importing model dependencies."""

import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Iterable, Optional, Union
import zipfile


ARCHIVE_RELATIVE_PATH = Path("precomputed/vlm_results_12datasets.zip")
PathLike = Union[str, os.PathLike]


def restore_precomputed_vlm_results(
    project_root: PathLike,
    output_root: PathLike,
    dataset_name: str,
    sequence_names: Optional[Iterable[str]] = None,
) -> int:
    """Fill missing result JSONs for a dataset and return the restored count.

    ``output_root`` is the outputs directory; archive paths already contain the
    dataset and sequence directories. Existing files are always preserved.
    Missing archives are supported for installations predating bundled results.
    Unsafe archive paths raise ``ValueError`` before any results are restored.
    """
    archive_path = Path(project_root) / ARCHIVE_RELATIVE_PATH
    if not archive_path.is_file():
        return 0

    selected_sequences = None if sequence_names is None else set(sequence_names)
    output_path = Path(output_root).resolve()
    restored = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = []
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if (relative.is_absolute() or ".." in relative.parts
                    or "\\" in member.filename or ":" in member.filename):
                raise ValueError(f"Unsafe bundled VLM archive path: {member.filename!r}")
            if member.is_dir():
                continue
            parts = relative.parts
            if (len(parts) != 5 or parts[2:4] != ("vlm_outputs", "results")
                    or not parts[4].endswith("_camera1.json")):
                continue
            if parts[0] != dataset_name:
                continue
            if selected_sequences is not None and parts[1] not in selected_sequences:
                continue
            members.append((member, relative))

        for member, relative in members:
            destination = output_path.joinpath(*relative.parts)
            if destination.exists() or destination.is_symlink():
                continue
            if not destination.resolve().is_relative_to(output_path):
                raise ValueError(f"Bundled VLM output escapes output root: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    with archive.open(member) as source:
                        shutil.copyfileobj(source, temporary)
                # A hard link publishes the complete file atomically and fails
                # if another process has already filled this cache entry.
                try:
                    os.link(temporary_path, destination)
                except FileExistsError:
                    continue
                restored += 1
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    return restored
