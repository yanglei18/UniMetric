"""Record referenced external model paths and hashes of directly named files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from metric_scale.config.config import Config

DATASETS = ('3eed_dataset', 'argoverse', 'cear', 'dair_v2x_i', 'droid',
            'fusionportable', 'humanoid_everyday', 'jrdb', 'kitti',
            'nuscenes', 'v2x_radar_i', 'waymo')
SUFFIXES = ('_project_root', '_weights', '_checkpoint', '_config_path',
            '_model_path', '_ckpt_path', '_ckpt')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


references: dict[str, dict] = {}
datasets: dict[str, set[str]] = {}
for dataset in DATASETS:
    for file in sorted((ROOT / 'configs' / dataset).glob('*.yaml')):
        raw = yaml.safe_load(file.read_text(encoding='utf-8')) or {}
        sys.argv = [sys.argv[0], '--config', str(file)]
        cfg = Config()
        data_path = Path(cfg.data_path)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        datasets.setdefault(dataset, set()).add(str(data_path.resolve()))
        for key, value in (raw.get('models') or {}).items():
            if not key.endswith(SUFFIXES) or value in (None, '', 'none', 'None'):
                continue
            path = Path(str(cfg._resolve_template(str(value))))
            if not path.is_absolute():
                path = ROOT / path
            resolved = path.resolve()
            record = references.setdefault(str(resolved), {'path': str(resolved), 'keys': set(), 'configs': set()})
            record['keys'].add(key)
            record['configs'].add(str(file.relative_to(ROOT)))
sam = (ROOT / 'checkpoints/sam2.1_hiera_large.pt').resolve()
references.setdefault(str(sam), {'path': str(sam), 'keys': {'sam_checkpoint'}, 'configs': set()})

assets = []
for record in sorted(references.values(), key=lambda x: x['path']):
    path = Path(record['path'])
    item = {'path': record['path'], 'keys': sorted(record['keys']),
            'used_by': sorted(record['configs']), 'exists': path.exists()}
    if path.is_file():
        item.update(kind='file', bytes=path.stat().st_size, sha256=sha(path))
    elif path.is_dir():
        item.update(kind='directory', git_head=git_head(path))
    else:
        item.update(kind='missing')
    assets.append(item)
output = ROOT / 'reproducibility/external-assets-manifest.json'
output.write_text(json.dumps({'datasets': {k: sorted(v) for k, v in datasets.items()},
                              'assets': assets}, indent=2, ensure_ascii=False), encoding='utf-8')
print('wrote', len(assets), 'external asset records to', output)
