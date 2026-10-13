"""Preflight all 12 production datasets without invoking VLMs or depth models."""
from __future__ import annotations

import importlib.resources
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from metric_scale.config.config import Config

DATASETS = ('3eed_dataset', 'argoverse', 'cear', 'dair_v2x_i', 'droid',
            'fusionportable', 'humanoid_everyday', 'jrdb', 'kitti',
            'nuscenes', 'v2x_radar_i', 'waymo')
PRODUCTION_MODELS = ('pi3_long', 'vggt_long', 'vggt_omega_long',
                     'da3_long', 'scal3r_long', 'lingbot_map_long',
                     '4rc_long', 'dvlt_long', 'stream3r_long')
EXPECTED_FILES = {'default.yaml'} | {model + '.yaml' for model in PRODUCTION_MODELS}
PATH_KEYS = ('_project_root', '_weights', '_checkpoint', '_config_path',
             '_model_path', '_ckpt_path', '_ckpt')


def absolute_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    report = {'repo': str(ROOT), 'datasets': {}, 'configs_checked': 0,
              'sequence_overrides_checked': 0, 'missing': [], 'errors': [],
              'api_credentials_present': {
                  'DOUBAO_API_KEY': bool(os.environ.get('DOUBAO_API_KEY')),
                  'MODEL_KEY': bool(os.environ.get('MODEL_KEY')),
                  'BASE_URL': bool(os.environ.get('BASE_URL')),
              }}
    for dataset in DATASETS:
        directory = ROOT / 'configs' / dataset
        configs = sorted(directory.glob('*.yaml')) if directory.is_dir() else []
        report['datasets'][dataset] = len(configs)
        present = {file.name for file in configs}
        if present != EXPECTED_FILES:
            report['errors'].append(f'{dataset}: YAML set differs from default + nine production models; '
                                    f'missing={sorted(EXPECTED_FILES - present)}, extra={sorted(present - EXPECTED_FILES)}')
        for file in configs:
            name = str(file.relative_to(ROOT))
            try:
                raw = yaml.safe_load(file.read_text(encoding='utf-8')) or {}
                sys.argv = [sys.argv[0], '--config', str(file)]
                cfg = Config()
                data_path = absolute_path(cfg.data_path)
                if not data_path.is_dir():
                    report['missing'].append(f'{name}: data_path={data_path}')
                output_root = Path(cfg.get_path('output_root'))
                if not output_root.is_dir():
                    report['missing'].append(f'{name}: output_root={output_root}')
                for key, value in (raw.get('models') or {}).items():
                    if key.endswith(PATH_KEYS) and value not in (None, '', 'none', 'None'):
                        target = absolute_path(str(cfg._resolve_template(str(value))))
                        if not target.exists():
                            report['missing'].append(f'{name}: {key}={target}')
                for sequence in (raw.get('semantic_scale') or {}).get('sequence_overrides', {}):
                    cfg.activate_semantic_scale_sequence_profile(sequence)
                    cfg.activate_semantic_scale_sequence_override(sequence)
                    cfg.clear_semantic_scale_sequence_profile()
                    report['sequence_overrides_checked'] += 1
                _ = cfg.sequence_difficulty_profiles
                _ = cfg.sequence_difficulty_hard_sequences
                _ = cfg.scale_camera_translation
                _ = cfg.fallback_degenerate_plane_mask
                _ = cfg.fallback_to_all_dynamic_anchors_if_subsequence_empty
                if cfg.depth_model == 'vggt_omega_long':
                    _ = cfg.vggt_omega_preprocess_mode, cfg.vggt_omega_geometry_intrinsics_source
                if cfg.depth_model == 'scal3r_long':
                    _ = cfg.scal3r_offload_batches, cfg.scal3r_offload_outputs
                report['configs_checked'] += 1
            except Exception as exc:
                report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    # The public Git repository does not redistribute third-party source.
    # Report the optional local cache, while the YAML path checks above remain
    # authoritative for implementations that need a project root.
    report['local_vendor_source_present'] = {
        relative: (ROOT / relative).is_file()
        for relative in ('vendor/Pi3/pi3/models/pi3x.py',
                         'vendor/Scal3R/scal3r/pipelines/inference.py',
                         'vendor/LingBot-Map/lingbot_map/models/gct_stream.py')
    }
    try:
        import sam2
        sam_config = importlib.resources.files('sam2').joinpath('configs/sam2.1/sam2.1_hiera_l.yaml')
        if not sam_config.is_file():
            report['missing'].append('sam2 package config: configs/sam2.1/sam2.1_hiera_l.yaml')
    except Exception as exc:
        report['errors'].append(f'sam2 package: {type(exc).__name__}: {exc}')
    if not (ROOT / 'checkpoints/sam2.1_hiera_large.pt').is_file():
        report['missing'].append('checkpoints/sam2.1_hiera_large.pt')
    try:
        import torch
        report['torch'] = {'version': torch.__version__, 'cuda_version': torch.version.cuda,
                           'cuda_available': bool(torch.cuda.is_available())}
    except Exception as exc:
        report['errors'].append(f'torch: {type(exc).__name__}: {exc}')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['missing'] or report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
