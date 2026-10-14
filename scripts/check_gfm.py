#!/usr/bin/env python3
"""Check a selected environment without calling a VLM or evaluating a dataset."""
from __future__ import annotations

import argparse
import importlib
import importlib.resources
import json
import os
import sys

from install_gfm import ROOT, load_profiles, select_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--skip-cuda-check", action="store_true")
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    profiles = load_profiles()
    name = select_model(args.model, profiles)
    profile = profiles["models"][name]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # Config parses argv on import; pass the selected YAML, not this CLI's flags.
    sys.argv = ["check_gfm", "--config", str(ROOT / "configs" / "jrdb" / (profile["config_key"] + ".yaml"))]
    import numpy as np
    import torch
    import torchvision
    import open3d as o3d
    import cv2
    from scipy.spatial.transform import Rotation
    import sam2

    points = np.zeros((2, 3), dtype=np.float64)
    assert torch.from_numpy(points).numpy().shape == (2, 3)
    assert np.asarray(o3d.utility.Vector3dVector(points)).shape == (2, 3)
    assert cv2.resize(np.zeros((4, 4), dtype=np.uint8), (2, 2)).shape == (2, 2)
    assert Rotation.identity().as_matrix().shape == (3, 3)
    sam_config = importlib.resources.files("sam2").joinpath("configs/sam2.1/sam2.1_hiera_l.yaml")
    if not sam_config.is_file():
        raise RuntimeError("Installed SAM 2 package is missing the SAM 2.1 large config.")
    importlib.import_module("metric_scale.models.models")
    for module in profile["imports"]:
        importlib.import_module(module)
    importlib.import_module(profile["adapter"])
    if not args.skip_cuda_check:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and installed torch wheel.")
        result = (torch.ones((2, 2), device="cuda") @ torch.ones((2, 2), device="cuda")).cpu()
        assert bool(torch.all(result == 2))
    if args.weights or args.load_model:
        for spec in (profiles["sam2_weights"], profile["weights"]):
            for filename in spec["files"]:
                path = ROOT / spec["directory"] / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Missing or empty checkpoint file: {path}")
    if args.load_model:
        if args.skip_cuda_check:
            raise ValueError("--load-model requires CUDA")
        from metric_scale.models.models import model_manager
        model_manager.initialize_depth_estimator(model_type=profile["config_key"], device="cuda:0")
        model_manager.release_all()
    print(json.dumps({"model": name, "torch": torch.__version__, "torchvision": torchvision.__version__,
                      "numpy": np.__version__, "open3d": o3d.__version__, "cuda": torch.version.cuda,
                      "imports_and_array_interop": "passed", "cuda_checked": not args.skip_cuda_check,
                      "checkpoint_files_checked": args.weights or args.load_model,
                      "model_loaded": args.load_model}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
