# Runbook

日期：2026-05-26

## 运行前检查

1. 确认数据路径存在：

```bash
ls data/official-data/nuScenes
```

2. 如果启用 API-based VLM，设置环境变量：

```bash
export DOUBAO_API_KEY=...
```

3. 确认所选模型的本地 checkpoint / project root 已在 YAML 的 `models` section 中配置。

4. 确认当前 pipeline stage。当前默认 `ProcessingPipeline._process_sequence_task()` 只运行 semantic scale stage：

```python
self._process_scale_for_sequence(sequence_path, sequence_name, device=device)
```

如果要从零开始跑，需要重新启用 VLM、SAM、P3 image prep 和 depth inference。

## 复用已有中间产物重跑尺度

适用于已经存在：

```text
vlm_outputs/results/*.json
sam_outputs/
{model}_outputs/relative_scale/pointmap/*.npy
{model}_outputs/relative_scale/conf/*.npy
{model}_outputs/relative_scale/camera_params/*.json
```

运行：

```bash
python get_metric_scale.py --config configs/nuscenes/lingbot_map_long.yaml
```

这个模式会重新生成 scale variants，并在主脚本后半段计算 metrics。

## 端到端重新生成

在 `metric_scale/processing/processing.py` 的 `_process_sequence_task()` 中按需要启用：

```python
self._process_vlm_for_sequence(sequence_path, sequence_name, device=device)
self._process_sam_for_sequence(sequence_path, sequence_name, device=device)
self._prepare_p3_for_sequence(sequence_name, device=device)
self._process_depth_for_sequence(sequence_path, sequence_name, device=device)
self._process_scale_for_sequence(sequence_path, sequence_name, device=device)
```

建议先用单 sequence / 少量 frames 做 smoke run：

```bash
python get_metric_scale.py \
  --config configs/nuscenes/pi3_long.yaml \
  --sequence-name scene-0092 \
  --frame-count 10 \
  --disable-parallel \
  --disable-visualization
```

## 多 GPU 运行

关键配置：

```yaml
parallel:
  enable: true
  max_gpus: 10
  vlm_api_enable: true
  max_vlm_workers: 1
```

并行逻辑支持 sequence-level 和 subsequence-level scheduling。只有在 CUDA 可用、GPU 数量大于 1、任务数量足够时才会触发。

## 常见运行目标

### 单序列快速检查

```bash
python get_metric_scale.py \
  --config configs/kitti/pi3_long.yaml \
  --sequence-name 0000 \
  --frame-count 30 \
  --disable-parallel
```

### 全 dataset 批处理

确保配置里：

```yaml
runtime:
  all_sequences: true
  sequence_name: []
```

然后运行对应 YAML。

### 只换 depth model

优先换 config 文件，而不是只用 CLI 覆盖 `--depth-model`。原因是不同模型通常还需要不同 `models` path、`output_resolution` 和 wrapper 参数。

## 输出检查

运行后检查：

```text
outputs/{dataset}/{sequence}/{model}_outputs/
outputs/{dataset}/metric_results/{model}/
```

核心文件：

- `dataset_summary.json`
- `dataset_summary.md`
- `sequences/{sequence}.json`
- `sequences/{sequence}.md`

## 清理

只清理确认可再生成的输出。当前仓库有大量未提交改动，执行删除前先看 git status 和目标路径。

