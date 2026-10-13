# Evaluation

Date: 2026-05-26

## Objective

The evaluation module measures whether metric geometry is correct, not just whether a visual depth map looks smooth. Metrics focus on sparse LiDAR ground truth, point-map geometry, sequence reconstruction, and camera pose.

The entry point is `metric_scale.metrics.metrics.MetricCalculator`. The pipeline saves per-sequence metrics, and the main script aggregates dataset summaries. Results already saved in the current run are not recomputed after depth intermediates have been cleaned. Core methods include:

```python
metric_calculator.calculate_all_metrics(...)
metric_calculator.save_sequence_result(...)
metric_calculator.save_dataset_summary(...)
```

## Evaluated scale variants

Enabled variants can include `relative`, `consistent`, `optimized_legacy`, `optimized_original`, `optimized_weighted_bi`, `optimized_weighted_ifpa`, `selected`, and `absolute`. The configuration determines which optimized variants are evaluated.

## Sparse depth and point metrics

Evaluation projects LiDAR points into the camera image, samples the predicted point map at projected pixels, and compares those samples with ground-truth LiDAR points in the camera frame.

Depth metrics include `rel`, `mae`, `rmse`, and `delta1`. Point metrics include 3D point error, relative point error, and valid-sample counts. Only LiDAR-visible projected pixels are evaluated, so these are sparse proxies rather than dense-depth ground truth.

## Distance bins

```yaml
runtime:
  enable_metric_depth_bins: true
  metric_depth_bins: [20, 40, 60]
```

This optional setting creates `0-20m`, `20-40m`, `40-60m`, and `60m+` bins to study scale and depth degradation by distance.

## Sparse reconstruction proxy

```yaml
runtime:
  enable_sparse_reconstruction: true
  sparse_reconstruction_mode: sequence
  sparse_reconstruction_alignment: point_cloud
  sparse_reconstruction_pose_source: pred
```

- `frame` evaluates each frame's point cloud independently; `sequence` fuses a subsequence into a common coordinate system first.
- `trajectory` alignment fits an SE3 transform from camera trajectories; `point_cloud` alignment fits one from sparse predicted/ground-truth point pairs.
- With pose source `pred`, model-saved `camera_params/extrinsics` fuse predicted point maps into a subsequence reference frame while ground-truth samples use dataset poses. This default tests point-map geometry and predicted-pose cross-frame consistency.
- With pose source `gt`, dataset ground-truth poses fuse both predictions and GT. This removes predicted-pose error and more directly tests point-map and scale geometry.

## Pose evaluation

```yaml
runtime:
  enable_pose_evaluation: true
  pose_evaluation_mode: subsequence
  pose_evaluation_pose_source: pred
  pose_eval_min_frames: 2
  pose_eval_scale: 1.0
```

Current pose metrics follow TUM conventions: ATE uses SE3 alignment, and RPE uses adjacent frames or a fixed delta. No Sim3 scale correction is applied, making these metrics suitable for checking metric-scale pose reliability.

## Reports

The usual report directory is `outputs/{dataset}/metric_results/{depth_model}/`. Key files include `dataset_summary.json`, `dataset_summary.md`, `sequences/{sequence}.json`, `sequences/{sequence}.md`, and sequence-difficulty/group-map reports. A later `sequence_difficulty` run can read `dataset_summary.json` to select easy, mid, or hard profiles.

## Interpretation notes

- `relative_scale` is the model's original-scale baseline; it is not necessarily metric-correct.
- `absolute_scale` is usually a compatibility output for the default optimized variant, not necessarily a ground-truth absolute oracle.
- Sparse LiDAR projection covers only LiDAR-visible pixels, not the whole image.
- Pose metrics depend on the coordinate semantics of each wrapper's `camera_params/extrinsics`; wrappers must use consistent conventions.
- `native_sampling_mode` affects point-map sampling. Check output-resolution policies before comparing models.
