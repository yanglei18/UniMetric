"""Freeze public source/config bytes; optionally audit local third-party code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('3eed_dataset', 'argoverse', 'cear', 'dair_v2x_i', 'droid',
            'fusionportable', 'humanoid_everyday', 'jrdb', 'kitti',
            'nuscenes', 'v2x_radar_i', 'waymo')
SUFFIXES = {'.py', '.pyi', '.sh', '.bash', '.c', '.cc', '.cpp', '.cu',
            '.h', '.hpp', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.json', '.md'}
SPECIAL = {'requirements.txt', 'Makefile', 'Dockerfile'}
SKIP = {'.git', '__pycache__', '.pytest_cache', 'assets', 'examples',
        'example', 'images', 'outputs', 'tmp', 'data', 'checkpoints',
        'pretrained_model', 'p3_images'}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--include-vendor', action='store_true',
                    help='write a local-only manifest that also hashes ignored vendor sources')
args = parser.parse_args()

paths = [ROOT / 'get_metric_scale.py', ROOT / '.gitignore', ROOT / 'README.md']
for dataset in DATASETS:
    paths.extend(sorted((ROOT / 'configs' / dataset).glob('*.yaml')))
paths.append(ROOT / 'configs/default.yaml')
paths.extend(sorted((ROOT / 'configs/kitti/scale_profile_reference').glob('*.json')))
paths.extend(sorted((ROOT / 'reproducibility').glob('*.py')))
paths.append(ROOT / 'reproducibility/README.md')
for optional in (ROOT / 'requirements-base.txt', ROOT / 'LICENSING.md', ROOT / 'vendor/README.md'):
    if optional.is_file():
        paths.append(optional)
subtrees = [ROOT / 'metric_scale', ROOT / 'dataset_tools', ROOT / 'tests', ROOT / 'docs', ROOT / 'scripts']
if args.include_vendor:
    subtrees.append(ROOT / 'vendor')
for subtree in subtrees:
    for current, dirs, files in os.walk(subtree, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP and not (Path(current) / d).is_symlink()]
        for name in files:
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() in SUFFIXES or name in SPECIAL or name.startswith('requirements'):
                paths.append(path)
items = []
for path in sorted(set(paths)):
    items.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                  'bytes': path.stat().st_size})
output = ROOT / 'reproducibility' / (
    'local-source-and-config-manifest.json' if args.include_vendor
    else 'source-and-config-manifest.json'
)
output.write_text(json.dumps({'dataset_dirs': DATASETS, 'files': items}, indent=2), encoding='utf-8')
print('wrote', len(items), 'file hashes to', output)
