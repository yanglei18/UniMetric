# Configuration Matrix

Date: 2026-05-26

## Layout

The configuration tree is organized as a dataset-by-model matrix:

```text
configs/
  default.yaml
  argoverse/
  dair_v2x_i/
  kitti/
  nuscenes/
  v2x_radar_i/
  waymo/
```

Each dataset directory usually contains:

```text
default.yaml
pi3_long.yaml
vggt_long.yaml
vggt_omega_long.yaml
da3_long.yaml
dust3r_long.yaml
must3r_long.yaml
scal3r_long.yaml
lingbot_map_long.yaml
```

`configs/nuscenes/pi3x_long.yaml` is an additional Pi3X-named variant.

## Main sections

### `runtime`

Controls the dataset, sequence, frame count, evaluation switches, and pipeline behavior:

```yaml
runtime:
  dataset: "nuscenes"
  data_path: "data/official-data/nuScenes/"
  depth_model: "lingbot_map_long"
  use_shared_scale: true
  all_sequences: false
  sequence_name: []
  frame_count: 1000
  subsequence_size: 1000
```

Important fields:

- `dataset`: used for prompt routing, output paths, and reports.
- `data_path`: root of the input sequences.
- `depth_model`: must match a key supported by `ModelManager`.
- `use_shared_scale`: selects a shared subsequence scale or per-frame scales.
- `frame_count`: processes the first N frames; `null` or a nonpositive value disables truncation.
- `subsequence_size`: chunk size for long-sequence stages.

### `parallel`

Controls multi-GPU scheduling and threaded VLM API calls:

```yaml
parallel:
  enable: true
  max_gpus: 10
  vlm_api_enable: true
  max_vlm_workers: 1
  metric_workers: 8
```

### `vision`

Controls the VLM backend. Do not place API keys in YAML:

```yaml
vision:
  model_path: "{qwen_model}"
  model_type: "doubao"
  api_key_env: "DOUBAO_API_KEY"
  model_version: "seed-1.8-max"
```

A local Qwen model generally does not need an API key. API backends read the environment variable named by `api_key_env`.

### `depth`

Controls the depth model's input size, long-sequence chunks, and output resolution:

```yaml
depth:
  input_pixel_limit: 255000
  input_max_long_edge: null
  long_chunk_size: 60
  long_overlap: 20
  output_resolution: "native"
  native_sampling_mode: "nearest"
```

The meaning of `native` and `original` output resolution varies by model and can affect metric sampling. Confirm that compared runs use consistent settings.

### `semantic_scale`

Controls anchor sources, filters, optimization variants, and sequence difficulty:

```yaml
semantic_scale:
  height_anchor_source: "dynamic"
  default_optimized_variant: "optimized_scale_original"
  enable_selected_scale: false
  sequence_difficulty:
    enabled: true
```

## Dataset-specific differences

Long-sequence public or official dataset configurations generally use:

- `frame_count: 1000`
- `subsequence_size: 1000`
- `sparse_reconstruction_mode: sequence`
- `enable_pose_evaluation: true`
- `height_anchor_source: dynamic`

Earlier or default configurations retain shorter test settings:

- `frame_count: 60`
- `subsequence_size: 60`
- `sparse_reconstruction_mode: frame`
- `height_anchor_source: both`

Some V2X and DAIR configurations still use frame-level sparse reconstruction because scene organization and available ground truth may differ.

## Recommended usage

Select a complete YAML file:

```bash
python get_metric_scale.py --config configs/nuscenes/lingbot_map_long.yaml
```

Use CLI flags only for light overrides:

```text
--sequence-name
--frame-count
--parallel / --disable-parallel
--enable-visualization / --disable-visualization
```

Avoid switching models using only `--depth-model`: the checkpoint path, output resolution, and wrapper parameters often differ as well.
