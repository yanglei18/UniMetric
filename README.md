# UniMetric

UniMetric is the central experiment repository in `Metric-Depth-Project`. It turns outputs from multiple depth and 3D foundation models into measurable metric geometry. It uses VLM semantic anchors, SAM masks, scene planes, sparse LiDAR ground truth, and pose metrics for scale calibration and evaluation.

The project has grown beyond its initial dataset-format validation scripts into a long-sequence geometry evaluation platform for autonomous driving and robotics. Model implementations live in sibling repositories; this repository standardizes inputs, outputs, scale estimation, and metric reports.

## What it does

The core task is to compare geometric outputs from depth and 3D reconstruction models on long visual sequences, then use semantic scale anchors to convert relative geometry to metric scale.

- Unifies depth maps, point maps, and camera parameters from multiple models.
- Uses a VLM to identify dynamic and static height anchors and scene planes.
- Uses SAM to generate object and plane masks from VLM bounding boxes.
- Produces `consistent_scale` and several `optimized_scale_*` variants with a semantic scale solver.
- Evaluates sparse depth and point metrics against projected LiDAR, sparse reconstruction proxies, and TUM-style pose ATE/RPE.
- Provides a dataset-by-model YAML configuration matrix.

## Related repositories

Sibling repositories under `Metric-Depth-Project/` provide background and local dependencies:

- `Depth-Anything-3/`: DA3 and DA3-Streaming.
- `VGGT-Long/`: long-sequence chunking, overlap, and alignment for VGGT.
- `Pi-Long/`: long-sequence adaptation for Pi3/Pi3X.
- `Scal3R/`: large-scale test-time-training reconstruction.
- `vggt-omega/`: source of the VGGT-Omega wrapper.
- `lingbot-map/`: source of the streaming/windowed reconstruction wrapper.
- `STream3R/`: research background for streaming reconstruction; no direct wrapper has been identified here.

These sibling repositories are not included in this repository. You must provide the required model code and checkpoints separately.

## Repository layout

```text
UniMetric/
├── get_metric_scale.py                  # Main entry point
├── configs/                             # Dataset/model YAML configuration matrix
├── dataset_tools/                       # Conversion, validation, cleanup, visualization
├── metric_scale/
│   ├── config/                          # YAML configuration and path resolution
│   ├── processing/                      # VLM/SAM/depth/scale pipeline
│   ├── models/                          # Depth/3D/VLM/SAM/scale model wrappers
│   ├── metrics/                         # Geometry, reconstruction, and pose evaluation
│   ├── prompts/                         # Dataset-specific VLM prompts
│   ├── utils/
│   └── visualization/
├── docs/                                # Usage, configuration, scale, and evaluation guides
└── plan/                                # Project inventory and architecture notes
```

## Supported models

`metric_scale.models.models.ModelManager` supports:

- `pi3_long`
- `vggt_long`
- `vggt_omega_long`
- `da3_long`
- `dust3r_long`
- `must3r_long`
- `scal3r_long`
- `lingbot_map_long`
- `moge`

Long-sequence wrappers generally split input using `long_chunk_size` and `long_overlap`, then save standardized outputs under `pointmap/`, `conf/`, and `camera_params/`.

## Quick start

Set the API key for your VLM backend before running. The default configuration uses the `doubao` backend and reads its key from an environment variable:

```bash
export DOUBAO_API_KEY=...
```

Basic command:

```bash
python get_metric_scale.py --config configs/nuscenes/lingbot_map_long.yaml
```

Example overrides:

```bash
python get_metric_scale.py \
  --config configs/nuscenes/pi3_long.yaml \
  --sequence-name scene-0092 \
  --frame-count 100 \
  --depth-model pi3_long
```

Currently, `ProcessingPipeline._process_sequence_task()` enables only the semantic scale stage by default. The VLM, SAM, P3 image preparation, and depth inference stages are commented out in that function. To regenerate intermediates end to end, enable the relevant stages or follow the staged workflow in the [runbook](docs/runbook.md).

## Configuration matrix

Configurations live at `configs/{dataset}/{model}.yaml`. The main datasets include `argoverse`, `dair_v2x_i`, `kitti`, `nuscenes`, `v2x_radar_i`, and `waymo`. See the [configuration matrix](docs/config-matrix.md) for details.

## Outputs

The usual output root is:

```text
outputs/{dataset_name}/{sequence_name}/{model}_outputs/
```

Important scale variants include:

```text
relative_scale/
consistent_scale/
optimized_scale/
optimized_scale_legacy/
optimized_scale_original/
optimized_scale_weighted_bi/
optimized_scale_weighted_ifpa/
selected_scale/
absolute_scale/
```

`relative_scale` is the model's original or relative-scale output. The semantic scale stage produces the other variants. See [semantic scale](docs/semantic-scale.md).

## Evaluation

`get_metric_scale.py` invokes `MetricCalculator` to produce sequence-level JSON and Markdown reports, dataset-level `dataset_summary.json` and Markdown reports, sparse depth and point metrics, a sparse reconstruction proxy, TUM-style pose ATE/RPE, and optional distance-bin metrics. See [evaluation](docs/evaluation.md) for metric definitions.

## Dataset format

A sequence usually contains:

```text
{frame_id}.pcd
{frame_id}.yaml
{frame_id}_camera0.jpg
{frame_id}_camera1.jpg
{frame_id}_camera2.jpg
```

The YAML contains camera `cords`, `extrinsic`, and `intrinsic` values, plus `lidar_pose`, `vehicles`, `pedestrians`, and `cyclists`. The main pipeline uses `camera1` by default. Dataset validation and visualization tools are in `dataset_tools/varify_dataset.py`.

## Credentials and local configuration

Do not put API keys, tokens, or credentials in YAML or source files. The current configuration reads the key through an environment variable:

```yaml
vision:
  api_key_env: "DOUBAO_API_KEY"
```

For a different backend, use a local Git-ignored configuration or a shell environment override.

## Further reading

- [Runbook](docs/runbook.md)
- [Configuration matrix](docs/config-matrix.md)
- [Semantic scale](docs/semantic-scale.md)
- [Evaluation](docs/evaluation.md)
- [Documentation inventory](plan/documentation-inventory.md)
- [Codebase map](plan/metric-depth-codebase-map.md)
