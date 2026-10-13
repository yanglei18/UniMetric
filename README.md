# Metric-Depth

Metric-Depth 是 `Metric-Depth-Project` 的中心实验仓库，用于把多种 depth / 3D foundation model 的输出统一到可评测的 metric geometry，并通过 VLM semantic anchors、SAM masks、scene planes、LiDAR sparse GT 和 pose metrics 做尺度标定与评测。

当前仓库已经不只是早期的数据格式验证脚本。它更像一个自动驾驶/机器人长序列几何评测平台：外部模型来自父级目录中的 sibling repositories，本仓库负责统一输入、输出、尺度求解和指标报告。

## 项目定位

核心问题：

> 在长序列视觉场景中，比较不同 foundation depth / 3D reconstruction 模型的几何输出，并用语义尺度锚点把相对尺度转成 metric-scale 结果。

当前能力：

- 多模型 depth / pointmap / camera 参数统一输出。
- VLM 挖掘 dynamic/static height anchors 和 scene planes。
- SAM 根据 VLM bbox 生成 object/plane masks。
- Semantic scale solver 生成 `consistent_scale` 和多个 `optimized_scale_*` variants。
- LiDAR 投影 sparse depth / point metrics、sparse reconstruction proxy、TUM-style pose ATE / RPE。
- Dataset × model YAML 配置矩阵。

## 父级项目关系

`Metric-Depth-Project/` 下的 sibling repositories 是当前仓库的重要背景和本地依赖来源：

- `Depth-Anything-3/`：DA3 / DA3-Streaming。
- `VGGT-Long/`：VGGT long-sequence chunk / overlap / alignment framework。
- `Pi-Long/`：Pi3/Pi3X long-sequence adaptation。
- `Scal3R/`：large-scale test-time-training reconstruction。
- `vggt-omega/`：VGGT-Omega wrapper 来源。
- `lingbot-map/`：streaming/windowed long-sequence reconstruction wrapper 来源。
- `STream3R/`：streaming reconstruction 研究背景，当前未发现直接 wrapper。

## 代码结构

```text
Metric-Depth/
├── get_metric_scale.py                  # 主入口
├── configs/                             # dataset/model YAML 配置矩阵
├── dataset_tools/                       # 数据转换、验证、清理与可视化工具
├── metric_scale/
│   ├── config/                          # YAML 配置与路径解析
│   ├── processing/                      # VLM/SAM/depth/scale pipeline
│   ├── models/                          # depth/3D/VLM/SAM/scale model wrappers
│   ├── metrics/                         # geometry、reconstruction、pose 评测
│   ├── prompts/                         # dataset-specific VLM prompts
│   ├── utils/
│   └── visualization/
├── docs/                                # 运行、配置、尺度、评测文档
└── plan/                                # 项目梳理与规划记录
```

## 支持模型

`metric_scale.models.models.ModelManager` 当前支持：

- `pi3_long`
- `vggt_long`
- `vggt_omega_long`
- `da3_long`
- `dust3r_long`
- `must3r_long`
- `scal3r_long`
- `lingbot_map_long`
- `moge`

长序列 wrapper 通常会按 `long_chunk_size` / `long_overlap` 切分输入，并把不同模型输出统一保存为 `pointmap/`、`conf/`、`camera_params/`。

## 快速运行

运行前根据 VLM backend 设置 API key。默认配置使用 `doubao` backend，并从环境变量读取：

```bash
export DOUBAO_API_KEY=...
```

基础命令：

```bash
python get_metric_scale.py --config configs/nuscenes/lingbot_map_long.yaml
```

常用覆盖：

```bash
python get_metric_scale.py \
  --config configs/nuscenes/pi3_long.yaml \
  --sequence-name scene-0092 \
  --frame-count 100 \
  --depth-model pi3_long
```

注意：当前 `ProcessingPipeline._process_sequence_task()` 默认只启用 semantic scale stage，VLM、SAM、P3 image prep 和 depth inference stage 在该函数中处于注释状态。若要端到端重新生成中间产物，需要在 pipeline 中重新启用对应阶段，或按 [runbook](docs/runbook.md) 分阶段运行。

## 配置矩阵

配置位于 `configs/{dataset}/{model}.yaml`。当前主要 dataset：

- `argoverse`
- `dair_v2x_i`
- `kitti`
- `nuscenes`
- `v2x_radar_i`
- `waymo`

更多配置说明见 [config-matrix](docs/config-matrix.md)。

## 输出结构

输出根目录通常为：

```text
outputs/{dataset_name}/{sequence_name}/{model}_outputs/
```

重要 scale variants：

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

`relative_scale` 是模型原始/相对尺度输出，其余目录由 semantic scale stage 生成。详情见 [semantic-scale](docs/semantic-scale.md)。

## 评测

评测入口由 `get_metric_scale.py` 调用 `MetricCalculator` 完成。主要报告：

- sequence-level JSON 和 Markdown report。
- dataset-level `dataset_summary.json` 和 Markdown report。
- sparse depth / point metrics。
- sparse reconstruction proxy。
- TUM-style pose ATE / RPE。
- 可选 distance-bin metrics。

指标口径见 [evaluation](docs/evaluation.md)。

## 数据集格式

每个 sequence 通常包含：

```text
{frame_id}.pcd
{frame_id}.yaml
{frame_id}_camera0.jpg
{frame_id}_camera1.jpg
{frame_id}_camera2.jpg
```

YAML 中包含：

- 相机 `cords`、`extrinsic`、`intrinsic`
- `lidar_pose`
- `vehicles`
- `pedestrians`
- `cyclists`

默认主流程使用 `camera1`。数据验证与可视化工具位于 `dataset_tools/varify_dataset.py`。

## 安全与本地配置

不要把 API keys、tokens、credentials 写入 YAML 或源码。当前配置通过：

```yaml
vision:
  api_key_env: "DOUBAO_API_KEY"
```

读取环境变量。需要不同 backend 时，优先在本地 gitignored config 或 shell 环境中覆盖。

## 更多文档

- [runbook](docs/runbook.md)
- [config-matrix](docs/config-matrix.md)
- [semantic-scale](docs/semantic-scale.md)
- [evaluation](docs/evaluation.md)
- [documentation-inventory](plan/documentation-inventory.md)
- [codebase-map](plan/metric-depth-codebase-map.md)
