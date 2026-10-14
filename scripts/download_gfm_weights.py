#!/usr/bin/env python3
"""Download only the pinned files needed by the selected UniMetric adapter."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from install_gfm import ROOT, load_profiles, select_model


def download(spec: dict) -> None:
    from huggingface_hub import hf_hub_download

    directory = ROOT / spec["directory"]
    for filename in spec["files"]:
        target = directory / filename
        if target.exists():
            # Never silently overwrite a checkpoint supplied by the user.
            if target.stat().st_size == 0:
                raise RuntimeError(f"Existing checkpoint is empty: {target}")
            print(f"Using existing {target}; its contents are not certified by the pinned download revision.")
            continue
        print(f"Downloading {spec['repo_id']}/{filename} @ {spec['revision']}", flush=True)
        hf_hub_download(repo_id=spec["repo_id"], filename=filename,
                        revision=spec["revision"], local_dir=directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()
    profiles = load_profiles()
    try:
        model = select_model(args.model, profiles)
        download(profiles["sam2_weights"])
        download(profiles["models"][model]["weights"])
    except Exception as exc:
        from huggingface_hub.errors import GatedRepoError
        if isinstance(exc, GatedRepoError):
            print("Checkpoint access is gated. Request access on the model's Hugging Face page and run `hf auth login` in this environment, then rerun the installer.", file=sys.stderr)
        else:
            print(f"Checkpoint download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
