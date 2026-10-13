# Semantic Scale

日期：2026-05-26

## 目标

许多 depth / 3D foundation models 输出的是相对尺度或弱 metric 尺度。Semantic scale stage 的目标是利用图像中可识别物体的真实高度先验和 predicted pointmap 的相对几何量，估计一个 metric scale，并把 pointmap 和 camera translation 统一缩放。

## 输入

Semantic scale stage 需要三类输入：

1. VLM results
   - `dynamic_height_anchors`
   - `static_height_anchors`
   - `semantic_planes`

2. SAM masks
   - 由 VLM bbox prompt 得到的 object/plane masks。

3. Depth model relative outputs
   - `relative_scale/pointmap/*.npy`
   - `relative_scale/conf/*.npy`
   - `relative_scale/camera_params/*.json`

## Anchor 类型

### Dynamic height anchors

典型对象：

- car / sedan / suv / van
- bus / truck
- pedestrian / cyclist / motorcycle

这些对象提供常识高度区间，例如小汽车高度、行人身高等。

### Static height anchors

典型对象：

- traffic cone
- traffic barrier
- curb
- guardrail
- bollard
- pole

静态设施在某些数据集上更稳定，但也更容易受类别标准差异影响。

### Scene planes

典型 plane：

- road
- ground

Planes 用于为 anchor 的 bottom/ground reference 提供局部或全局几何约束。

## 求解流程

1. VLM 给出 anchor bbox、height interval、confidence 和理由。
2. SAM 将 bbox 转成 mask。
3. Scale estimator 在 pointmap 中测量 anchor relative height。
4. 使用真实高度区间除以相对高度，得到 scale interval。
5. 过滤低质量 anchors、低置信度 anchors 和几何不可靠 measurements。
6. 聚合多个 scale intervals，求 `consistent_scale` 和 optimized variants。
7. 将 pointmap 与 camera extrinsics translation 按 scale 保存到对应输出目录。

## 主要配置

```yaml
semantic_scale:
  relative_height_mode: "peak_to_bottom"
  shared_plane_mode: "global_plane"
  height_anchor_source: "dynamic"
  use_depth_confidence: true
  use_anchor_quality_filter: true
  use_depth_geometry_filter: true
  use_vlm_anchor_confidence_filter: true
```

重要阈值：

- `depth_geometry_min_anchor_points`
- `depth_geometry_min_relative_height`
- `vlm_dynamic_anchor_confidence_threshold`
- `vlm_static_anchor_confidence_threshold`
- `anchor_quality_min_percentile`
- `anchor_quality_max_percentile`
- `plane_conf_percentile`

## Scale variants

- `relative_scale`：模型原始输出。
- `consistent_scale`：robust consensus 输出。
- `optimized_scale_legacy`：legacy optimized 求解。
- `optimized_scale_original`：当前常用默认 optimized variant。
- `optimized_scale_weighted_bi`：weighted BI variant。
- `optimized_scale_weighted_ifpa`：weighted IFPA variant。
- `optimized_scale`：兼容 alias，通常指向默认 optimized variant。
- `absolute_scale`：兼容 alias，通常与默认 optimized 输出一致。
- `selected_scale`：可选 gated final scale，多数配置当前未启用。

## Sequence difficulty profile

`sequence_difficulty` 可以读取上一轮 `dataset_summary.json`，根据相对尺度和候选 variants 的表现把 sequence 分成 easy / mid / hard，并动态覆盖部分 semantic-scale 参数。

当前常见规则：

- baseline：`relative_scale`
- candidates：`consistent_scale` + enabled optimized variants
- metric：`depth_frame_mean.delta1`
- easy：candidate 的 delta1 gain 超过阈值
- hard：candidate 相对 baseline 退化
- mid：其他情况

## 调试输出

当 `save_scale_intermediate_results: true` 时，会写出：

```text
semantic_scale_debug/
```

其中包含：

- scale intervals
- anchor measurements
- rejection summary
- consensus / BI / IFPA details
- selected-scale gate information

这些文件适合排查“为什么没有有效尺度锚点”或“optimized scale 为什么偏移过大”。

