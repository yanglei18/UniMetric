# Metric-Depth 当前代码地图

日期：2026-05-26

## 一句话理解

`Metric-Depth` 是 `Metric-Depth-Project` 中的中心实验仓库：它把多个 depth / 3D foundation model 的相对或弱 metric 输出转成统一的 pointmap / camera_params 结构，再用 VLM semantic anchors、SAM masks、ground/plane cues 和 LiDAR/pose GT 做 metric-scale 标定与评测。

## 输入数据契约

每个 sequence 通常包含：

- `{frame_id}.pcd`：LiDAR 点云。
- `{frame_id}.yaml`：相机、LiDAR、对象和场景元数据。
- `{frame_id}_camera{camera_id}.jpg`：相机图像，主流程默认使用 `camera1`。

`FrameLoader` 会读取图像、PCD、YAML，构造：

- `lidar2camera`
- `camera2world`
- `world2camera`
- `intrinsic`
- `object_list`
- image size / point cloud / projected bboxes / BEV 可视化输入

## 输出数据契约

输出根通常是：

```text
outputs/{dataset_name}/{sequence_name}/
```

不同 depth model 写到不同子目录：

```text
{model}_outputs/
  relative_scale/
    pointmap/
    conf/
    camera_params/
  consistent_scale/
  optimized_scale/
  optimized_scale_legacy/
  optimized_scale_original/
  optimized_scale_weighted_bi/
  optimized_scale_weighted_ifpa/
  selected_scale/
  absolute_scale/
  semantic_scale_debug/
```

其中 `relative_scale` 是 depth / 3D model 原始或相对尺度输出；后续 scale stage 会生成多种 metric-scale variants。

## Pipeline 阶段

代码里保留完整五阶段结构：

1. VLM Processing
   - 入口：`VLMProcessor.process_frames()`
   - 输出：`vlm_outputs/results/*.json`
   - 内容：dynamic height anchors、static height anchors、semantic planes。

2. Prepare P3 Images
   - 入口：`P3ImagePreparer.prepare_images()`
   - 输出：`p3_images/{sequence_name}/`
   - 内容：复制 VLM 结果中引用的图像，供 depth model 输入。

3. SAM Processing
   - 入口：`SAMProcessor.process_segmentation()`
   - 输出：`sam_outputs/{frame}/...mask...png`
   - 内容：根据 VLM bbox prompt 生成 masks。

4. Depth Estimation
   - 入口：`DepthProcessor.run_depth_estimation()`
   - 输出：`relative_scale/pointmap`、`relative_scale/conf`、`relative_scale/camera_params`。
   - 内容：运行指定 foundation model wrapper。

5. Semantic Plane Scale Estimation
   - 入口：`SemanticPlaneScaleProcessor.estimate_scale()`
   - 输出：`consistent_scale`、`optimized_scale_*`、`absolute_scale` 等。
   - 内容：用 semantic anchors 和 plane/ground geometry 求尺度。

当前实际默认：`ProcessingPipeline._process_sequence_task()` 只调用 `_process_scale_for_sequence()`，前四阶段被注释。后续运行前必须根据目标确认是否需要恢复端到端 pipeline。

## 支持的 depth / 3D models

当前 `ModelManager.initialize_depth_estimator()` 支持：

- Pi3 / Pi3X：`pi3_long`
- VGGT：`vggt_long`
- VGGT-Omega：`vggt_omega_long`
- Depth Anything 3：`da3_long`
- DUSt3R / MUSt3R：`dust3r_long`、`must3r_long`
- Scal3R：`scal3r_long`
- LingBot-Map：`lingbot_map_long`
- MoGe：`moge`

Long wrappers 的共同设计：

- 按 `long_chunk_size` / `long_overlap` 切 chunk。
- 将各模型输出统一保存为 pointmap、confidence、camera_params。
- 为不同模型处理 native/original resolution、intrinsics rescaling、camera extrinsics 方向差异。
- `temp_root` 默认指向大盘路径，`long_delete_temp_files` 控制清理。

## Semantic scale 机制

Semantic scale stage 的核心思想：

1. VLM 从图像中找可作为 metric height prior 的语义物体。
2. SAM 将 VLM bbox 转成 masks。
3. 在 predicted pointmap 中测量 anchor 的相对高度，结合 VLM 的真实高度 interval 得到 scale interval。
4. 使用 robust consensus / weighted optimization 聚合多个 anchors。
5. 将 pointmap 和 camera translation 按 scale 放大，形成不同 scale variants。

主要 scale variants：

- `relative`：原始模型尺度，不做 semantic scale。
- `consistent`：鲁棒 consensus 尺度。
- `optimized_scale_legacy`
- `optimized_scale_original`
- `optimized_scale_weighted_bi`
- `optimized_scale_weighted_ifpa`
- `optimized` / `absolute`：兼容 alias，通常指向默认 optimized variant。
- `selected`：可选 gated final scale，当前多数配置未启用。

当前配置常见策略：

- `height_anchor_source: dynamic` 或 `both`
- `default_optimized_variant: optimized_scale_original`
- `enable_selected_scale: false`
- `sequence_difficulty.enabled: true` 在长序列 dataset configs 中启用

## 评测体系

`MetricGeometryEvaluator` 和 `MetricCalculator` 覆盖：

- sparse LiDAR projection depth metrics：在 LiDAR 投影像素处采样 predicted pointmap。
- point metrics：比较 predicted points 与 GT camera-frame LiDAR points。
- sparse reconstruction proxy：frame 或 sequence 模式融合 sparse points。
- pose evaluation：TUM-style ATE / RPE，不做 Sim3 scale correction。
- distance bins：可选 `0-20m`、`20-40m`、`40-60m`、`60m+`。
- sequence difficulty：从已有 `dataset_summary.json` 推断 easy/mid/hard，并可动态调整 semantic scale profile。

报告输出包括：

- sequence JSON
- sequence Markdown report
- dataset_summary.json
- dataset_summary.md / dataset report
- sequence group map

## 配置使用心智模型

常用命令形态：

```bash
python get_metric_scale.py --config configs/nuscenes/lingbot_map_long.yaml
```

可以通过 CLI 覆盖：

```bash
python get_metric_scale.py \
  --config configs/nuscenes/pi3_long.yaml \
  --sequence-name scene-0092 \
  --frame-count 100 \
  --depth-model pi3_long
```

配置中的关键开关：

- `runtime.dataset`
- `runtime.data_path`
- `runtime.depth_model`
- `runtime.use_shared_scale`
- `runtime.all_sequences`
- `runtime.sequence_name`
- `runtime.frame_count`
- `runtime.subsequence_size`
- `runtime.enable_sparse_reconstruction`
- `runtime.enable_pose_evaluation`
- `runtime.sparse_reconstruction_mode`
- `depth.output_resolution`
- `semantic_scale.height_anchor_source`
- `semantic_scale.default_optimized_variant`
- `semantic_scale.sequence_difficulty`
- `paths.output_root`

## 当前文档缺口

- 没有说明当前 pipeline 只启用 scale stage。
- 没有说明 model wrapper 与 sibling repo 的关系。
- 没有说明 config tree 已从单文件 dataset config 演化为 dataset/model matrix。
- 没有说明 semantic-scale variants 的含义。
- 没有说明 metric report 中 depth / point / reconstruction / pose 指标的口径。
- 没有说明敏感配置应通过环境变量或本地覆盖注入。

## 后续优先任务

1. 先处理 hardcoded API key 风险：rotate，并迁移到环境变量。
2. 更新 README，使其反映当前真实 pipeline。
3. 将 `metrics.py` 拆成 depth/point/reconstruction/pose/reporting 子模块。
4. 给 config matrix 建表，明确每个 dataset/model 的推荐用途和默认 stage。
5. 为 semantic-scale solver 补最小单元测试，尤其是 scale interval aggregation 和 selected-scale gating。
