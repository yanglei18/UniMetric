# Evaluation

日期：2026-05-26

## 评测目标

评测模块关注 metric geometry 是否正确，而不是只看视觉深度图是否平滑。当前指标主要围绕 LiDAR sparse GT、pointmap 几何、sequence reconstruction 和 camera pose。

入口：

```python
metric_scale.metrics.metrics.MetricCalculator
```

主脚本会在 pipeline 后调用：

```python
metric_calculator.calculate_all_metrics(...)
metric_calculator.save_sequence_result(...)
metric_calculator.save_dataset_summary(...)
```

## 评测对象

每个 scale variant 都可以被评测：

- `relative`
- `consistent`
- `optimized_legacy`
- `optimized_original`
- `optimized_weighted_bi`
- `optimized_weighted_ifpa`
- `selected`
- `absolute`

具体启用哪些 optimized variants 由配置决定。

## Sparse depth / point metrics

评测步骤：

1. 将 LiDAR points 投影到 camera image。
2. 在 projected pixels 处采样 predicted pointmap。
3. 与 GT camera-frame LiDAR points 对齐比较。

Depth metrics 包括：

- `rel`
- `mae`
- `rmse`
- `delta1`

Point metrics 包括：

- 3D point error
- relative point error
- valid sample counts

这些指标只在 LiDAR 可见投影点上评估，因此是 sparse proxy，不是 dense depth GT。

## Distance bins

可选启用：

```yaml
runtime:
  enable_metric_depth_bins: true
  metric_depth_bins: [20, 40, 60]
```

会形成：

- `0-20m`
- `20-40m`
- `40-60m`
- `60m+`

用于分析近中远距离的尺度/深度退化。

## Sparse reconstruction proxy

配置：

```yaml
runtime:
  enable_sparse_reconstruction: true
  sparse_reconstruction_mode: sequence
  sparse_reconstruction_alignment: point_cloud
  sparse_reconstruction_pose_source: pred
```

两种模式：

- `frame`：逐帧评估 point cloud。
- `sequence`：把 subsequence 融合到统一坐标系后评估。

Alignment source：

- `trajectory`：根据 camera trajectory 拟合 SE3。
- `point_cloud`：根据 sparse pred/GT point pairs 拟合 SE3。

Pose source：

- `pred`：用模型保存的 `camera_params/extrinsics` 将预测 pointmap 从每帧 camera frame 融合到 subsequence reference frame；GT samples 仍使用 dataset GT pose 融合。这是默认模式，可同时评估 pointmap 几何和预测 pose 的跨帧一致性。
- `gt`：预测 pointmap 和 GT samples 都使用 dataset GT pose 融合到同一个 reference frame。该模式去掉预测 pose 误差影响，更接近只评估 pointmap/scale 几何质量。

## Pose evaluation

配置：

```yaml
runtime:
  enable_pose_evaluation: true
  pose_evaluation_mode: subsequence
  pose_evaluation_pose_source: pred
  pose_eval_min_frames: 2
  pose_eval_scale: 1.0
```

当前 pose metrics 是 TUM-style：

- ATE 使用 SE3 alignment。
- RPE 使用相邻帧或固定 delta。
- 不做 Sim3 scale correction，因此更适合检查 metric-scale pose 是否可靠。

## 输出报告

输出位置通常是：

```text
outputs/{dataset}/metric_results/{depth_model}/
```

核心文件：

- `dataset_summary.json`
- `dataset_summary.md`
- `sequences/{sequence}.json`
- `sequences/{sequence}.md`
- sequence difficulty / group map 相关报告

`dataset_summary.json` 也会被下一轮 `sequence_difficulty` 读取，用来选择 easy / mid / hard profile。

## 解释指标时的注意事项

- `relative_scale` 是模型原始尺度 baseline，不代表 metric-correct。
- `absolute_scale` 当前通常是默认 optimized variant 的兼容输出，不一定代表 GT absolute oracle。
- Sparse LiDAR projection 只评估 LiDAR 可见点，不覆盖整张图。
- Pose metrics 依赖模型保存的 `camera_params/extrinsics` 语义，不同 wrapper 的坐标约定需要保持一致。
- `native_sampling_mode` 会影响 pointmap 采样，比较模型时要确认输出分辨率策略。
