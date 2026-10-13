"""CPU-only guard for the nine public model keys and 12 production config sets."""
import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "pi3_long",
    "vggt_long",
    "vggt_omega_long",
    "da3_long",
    "scal3r_long",
    "lingbot_map_long",
    "4rc_long",
    "dvlt_long",
    "stream3r_long",
}
DATASETS = {
    "3eed_dataset",
    "argoverse",
    "cear",
    "dair_v2x_i",
    "droid",
    "fusionportable",
    "humanoid_everyday",
    "jrdb",
    "kitti",
    "nuscenes",
    "v2x_radar_i",
    "waymo",
}

source = (ROOT / "metric_scale/models/models.py").read_text(encoding="utf-8")
tree = ast.parse(source)
for class_name, method_name in (
    ("ModelManager", "initialize_depth_estimator"),
    ("ModelFactory", "create_depth_estimator"),
):
    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    branches = {
        test.comparators[0].value
        for test in (node for node in ast.walk(method) if isinstance(node, ast.Compare))
        if isinstance(test.left, ast.Name)
        and test.left.id == "model_type"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and isinstance(test.comparators[0].value, str)
    }
    assert branches == MODELS, (class_name, method_name, sorted(branches ^ MODELS))

config_root = ROOT / "configs"
assert {p.name for p in config_root.iterdir() if p.is_dir()} == DATASETS
for dataset in DATASETS:
    folder = config_root / dataset
    assert {p.name for p in folder.glob("*.yaml")} == {
        "default.yaml", *(f"{model}.yaml" for model in MODELS)
    }, dataset
    for model in MODELS:
        config = yaml.safe_load((folder / f"{model}.yaml").read_text(encoding="utf-8"))
        assert config["runtime"]["depth_model"] == model, (dataset, model)

for removed in ("dust3r_long_depth_estimator.py", "must3r_long_depth_estimator.py",
                "moge_depth_estimator.py"):
    assert not (ROOT / "metric_scale/models" / removed).exists(), removed

print("PASS: exactly nine model keys and 12 complete production config sets")
