# UniMetric

UniMetric estimates metric scale for long-sequence depth and 3D reconstruction outputs using VLM height anchors, SAM masks, and scene planes. It evaluates depth, reconstruction, and camera pose. The main entry point is `get_metric_scale.py`.

This repository contains project code and experiment configurations. It does **not** include datasets, model weights, API credentials, or third-party model source. The server's `vendor/` directory is a local dependency cache excluded from Git; [vendor/README.md](vendor/README.md) lists upstream sources and license considerations. The licensing authority for the original project code has not yet been confirmed, so there is no root `LICENSE`. Public visibility is not a grant of an open-source license; see [LICENSING.md](LICENSING.md).

## Supported models and datasets

The public code supports exactly these nine depth-model keys:

| Model | Configuration key |
|---|---|
| Pi3 | `pi3_long` |
| VGGT | `vggt_long` |
| VGGT-Omega | `vggt_omega_long` |
| Depth Anything 3 | `da3_long` |
| Scal3R | `scal3r_long` |
| LingBot-Map | `lingbot_map_long` |
| 4RC | `4rc_long` |
| DVLT | `dvlt_long` |
| STream3R | `stream3r_long` |

The 12 dataset directories are `3eed_dataset`, `argoverse`, `cear`, `dair_v2x_i`, `droid`, `fusionportable`, `humanoid_everyday`, `jrdb`, `kitti`, `nuscenes`, `v2x_radar_i`, and `waymo`. Each has a `default.yaml` and one YAML per supported model, for 120 dataset configurations. See the [configuration matrix](docs/config-matrix.md).

- `metric_scale/`: configuration parsing, VLM/SAM/depth/scale pipeline, nine model adapters, metrics, and visualization.
- `configs/`: documented production parameters. Three KITTI reference JSON files are retained because configurations refer to them.
- `dataset_tools/`: dataset conversion and validation utilities.
- `tests/` and `reproducibility/`: regression checks, environment preflight, and version records.

The code retains VGGT-Omega `balanced` preprocessing and dataset-intrinsic modes, the older `depth_only` workflow, JRDB per-sequence `sequence_overrides`, and the full VLM → SAM → depth → scale → metrics pipeline. Argoverse has an explicit prompt module that forwards the historical `default.py` fallback text. Existing metric reports are **not** reused across runs unless `runtime.resume_existing_results: true` is explicitly enabled after confirming that code, YAML, input data, and caches match.

## Environment and external assets

Python 3.11 is recommended. Install a PyTorch/torchvision pair compatible with your CUDA driver, then the base dependencies:

```bash
python -m pip install -r requirements-base.txt
```

This is not a one-command installation for all nine models. Prepare each model, SAM2, and the required weights according to [vendor/README.md](vendor/README.md). `reproducibility/env/` contains package-version snapshots from a tested server for comparison, not a portable lockfile. Weight licenses may differ from code licenses.

Configuration paths reflect the original experiment environment. On another machine, replace paths without changing algorithmic settings:

| Configuration field | Local resource |
|---|---|
| `runtime.data_path` | Dataset images and annotations |
| `models.*_project_root`, `models.*_checkpoint`, `models.*_weights` | Model source and weights |
| `paths.sam_checkpoint`, `paths.sam_config` | SAM2 checkpoint and package configuration |
| `depth.long_temp_root`, `paths.output_root`, `paths.p3_images` | Writable temporary, result, and P3-image directories |

Doubao configurations read `DOUBAO_API_KEY`; Humanoid Everyday's Qwen API configuration reads `MODEL_KEY` and `BASE_URL`. Never put credential values in YAML or commit them to Git.

## Validate and run

After updating paths and installing dependencies, run the preflight and CPU/static checks:

```bash
python reproducibility/check_setup.py
python reproducibility/test_active_features.py
python reproducibility/test_prompts.py
python reproducibility/test_supported_models.py
python -m pytest -q tests/test_omega_balanced_preprocess.py tests/test_omega_dataset_intrinsics.py
```

For a smoke run, copy a production YAML **outside** this repository and change only the copy: set `runtime.all_sequences: false`, `runtime.sequence_name: [your_sequence]`, `runtime.frame_count: 10`, and `parallel.enable: false`. Then run:

```bash
python get_metric_scale.py --config /path/to/smoke.yaml --disable-parallel
```

For a full run, use a production configuration without frame or sequence overrides:

```bash
python get_metric_scale.py --config configs/jrdb/pi3_long.yaml
```

`runtime.cleanup_depth_outputs_after_metrics: true` removes the current model's depth intermediates after metrics are saved; it does not remove VLM/SAM caches. See the [runbook](docs/runbook.md) and [reproducibility guide](reproducibility/README.md). Static checks of configuration parsing, prompt routing, and Omega regressions do not replace full GPU evaluation, which also requires a working CUDA environment and VLM API credentials.
