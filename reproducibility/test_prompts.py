"""Ensure each dataset's configured VLM prompt routes compile without API calls."""
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS = ('3eed_dataset', 'argoverse', 'cear', 'dair_v2x_i', 'droid',
            'fusionportable', 'humanoid_everyday', 'jrdb', 'kitti',
            'nuscenes', 'v2x_radar_i', 'waymo')
checked = 0
for dataset in DATASETS:
    module_name = 'metric_scale.prompts.' + dataset
    if importlib.util.find_spec(module_name) is None:
        module_name = 'metric_scale.prompts.default'
    module = importlib.import_module(module_name)
    get_prompt = module.get_prompt
    accepts_domain = len(inspect.signature(get_prompt).parameters) >= 2
    for file in sorted((ROOT / 'configs' / dataset).glob('*.yaml')):
        cfg = yaml.safe_load(file.read_text(encoding='utf-8')) or {}
        source = (cfg.get('semantic_scale') or {}).get('height_anchor_source', 'both')
        tasks = ['scene_planes']
        if source in ('dynamic', 'both'):
            tasks.append('dynamic_height_anchors')
        if source in ('static', 'both'):
            tasks.append('static_height_anchors')
        domain = (cfg.get('runtime') or {}).get('scene_domain', 'indoor')
        for task in tasks:
            prompt = get_prompt(task, domain) if accepts_domain else get_prompt(task)
            assert isinstance(prompt, str) and len(prompt) > 20, (file, task)
            checked += 1
from metric_scale.prompts import argoverse, default
for task in ('dynamic_height_anchors', 'static_height_anchors', 'height_anchors', 'scene_planes'):
    assert argoverse.get_prompt(task) == default.get_prompt(task), task
assert not hasattr(argoverse, 'get_anchor_type_priors')
print(f'PASS: {checked} configured prompt routes across {len(DATASETS)} datasets')
