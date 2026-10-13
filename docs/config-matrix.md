# Config Matrix

日期：2026-05-26

## 配置布局

当前配置树采用 dataset × model matrix：

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

每个 dataset 目录通常包含：

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

`configs/nuscenes/pi3x_long.yaml` 是当前额外存在的 Pi3X 命名变体。

## 核心 section

### runtime

控制 dataset、sequence、frame 数、评测开关和 pipeline 语义：

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

重点字段：

- `dataset`：prompt routing、output path 和报告字段使用。
- `data_path`：sequence root。
- `depth_model`：必须与 `ModelManager` 支持的 key 对齐。
- `use_shared_scale`：shared subsequence scale 或 per-frame scale。
- `frame_count`：处理前 N 帧；`null` 或非正数表示不截断。
- `subsequence_size`：long-sequence stage 的切分单位。

### parallel

控制多 GPU 与 threaded VLM API：

```yaml
parallel:
  enable: true
  max_gpus: 10
  vlm_api_enable: true
  max_vlm_workers: 1
  metric_workers: 8
```

### vision

控制 VLM backend。API key 不应写在 YAML 中：

```yaml
vision:
  model_path: "{qwen_model}"
  model_type: "doubao"
  api_key_env: "DOUBAO_API_KEY"
  model_version: "seed-1.8-max"
```

本地 Qwen 模型通常不需要 API key。API backend 会从 `api_key_env` 指定的环境变量读取。

### depth

控制 depth model 输入尺寸、long chunk 和输出分辨率：

```yaml
depth:
  input_pixel_limit: 255000
  input_max_long_edge: null
  long_chunk_size: 60
  long_overlap: 20
  output_resolution: "native"
  native_sampling_mode: "nearest"
```

注意：不同模型的 `native` / `original` 输出会影响 metric sampling。评测比较时需要确认一致性。

### semantic_scale

控制 anchor source、filters、optimization variants 和 sequence difficulty：

```yaml
semantic_scale:
  height_anchor_source: "dynamic"
  default_optimized_variant: "optimized_scale_original"
  enable_selected_scale: false
  sequence_difficulty:
    enabled: true
```

## Dataset 差异

当前长序列公开/官方数据配置一般使用：

- `frame_count: 1000`
- `subsequence_size: 1000`
- `sparse_reconstruction_mode: sequence`
- `enable_pose_evaluation: true`
- `height_anchor_source: dynamic`

早期/默认配置保留更短的测试风格：

- `frame_count: 60`
- `subsequence_size: 60`
- `sparse_reconstruction_mode: frame`
- `height_anchor_source: both`

V2X / DAIR 类数据里，部分配置仍使用 frame-level sparse reconstruction，这是因为场景组织和可用 GT 可能不同。

## 推荐使用方式

优先选择完整 YAML：

```bash
python get_metric_scale.py --config configs/nuscenes/lingbot_map_long.yaml
```

CLI 只用于轻量覆盖：

```bash
--sequence-name
--frame-count
--parallel / --disable-parallel
--enable-visualization / --disable-visualization
```

不建议只用 `--depth-model` 切模型，因为不同模型需要不同 checkpoint path、输出分辨率和 wrapper 参数。

