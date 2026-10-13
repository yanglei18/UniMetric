# Evaluation

Date: 2026-05-26

## Objective

The evaluation module measures whether metric geometry is correct, not just whether a depth map looks smooth. Its metrics focus on sparse LiDAR ground truth, point-map geometry, sequence reconstruction, and camera pose.

The entry point is `metric_scale.metrics.metrics.MetricCalculator`. After the pipeline runs, the main script calls:

```python
metric_calculator.calculate_all_metrics(...)
metric_calculator.save_sequence_result(...)
metric_calculator.save_dataset_summary(...)
```

## Evaluated scale variants

Any enabled scale variant can be evaluated: `relative`, `consistent`, `optimized_legacy`, `optimized_original`, `optimized_weighted_bi`, `optimized_weighted_ifpa`, `selected`, and `absolute`. The configuration determines which optimized variants are enabled.

## Sparse depth and point metrics

Evaluation projects LiDAR points into the camera image, samples the predicted point map at those projected pixels, and compares the samples with ground-truth LiDAR points in the camera frame.

Depth metrics include `rel`, `mae`, `rmse`, and `delta1`. Point metrics include 3D point error, relative point error, and valid-sample counts. Since only projected LiDAR-visible points are evaluated, these are sparse proxies, not dense-depth ground truth.

## Distance bins

Optional configuration:

```yaml
runtime:
  enable_metric_depth_bins: true
  metric_depth_bins: [20, 40, 60]
```

This creates `0-20m`, `20-40m`, `40-60m`, and `60m+` bins for analyzing scale or depth degradation at different distances.

## Sparse reconstruction proxy

```yaml
runtime:
  enable_sparse_reconstruction: true
  sparse_reconstruction_mode: sequence
  sparse_reconstruction_alignment: point_cloud
  sparse_reconstruction_pose_source: pred
```

Two modes are available:

- `frame`: evaluates each frame's point cloud independently.
- `sequence`: fuses a subsequence into a common coordinate system before evaluation.

Alignment methods:

- `trajectory`: fits an SE3 transform from the camera trajectory.
- `point_cloud`: fits an SE3 transform from sparse predicted/ground-truth point pairs.

Pose sources:

- `pred`: uses model-saved `camera_params/extrinsics` to fuse predicted point maps from camera frames into a subsequence reference frame; ground-truth samples still use dataset ground-truth poses. This default mode tests both point-map geometry and cross-frame consistency of predicted poses.
- `gt`: uses dataset ground-truth poses to fuse both predicted point maps and ground-truth samples. Removing predicted-pose error makes this closer to a test of point-map and scale geometry alone.

## Pose evaluation

```yaml
runtime:
  enable_pose_evaluation: true
  pose_evaluation_mode: subsequence
  pose_evaluation_pose_source: pred
  pose_eval_min_frames: 2
  pose_eval_scale: 1.0
```

The current pose metrics follow TUM conventions: ATE uses SE3 alignment, and RPE uses adjacent frames or a fixed delta. No Sim3 scale correction is applied, making these metrics suitable for checking metric-scale pose reliability.

## Reports

The usual report directory is:

```text
outputs/{dataset}/metric_results/{depth_model}/
```

Key files are `dataset_summary.json`, `dataset_summary.md`, `sequences/{sequence}.json`, `sequences/{sequence}.md`, and sequence-difficulty/group-map reports. A later `sequence_difficulty` run can read `dataset_summary.json` to select easy, mid, or hard profiles.

## Interpretation notes

- `relative_scale` is the model's original-scale baseline; it is not necessarily metric-correct.
- `absolute_scale` is usually a compatibility output for the default optimized variant, not necessarily a ground-truth absolute oracle.
- Sparse LiDAR projection covers only LiDAR-visible pixels, not the whole image.
- Pose metrics depend on the coordinate semantics of the model's `camera_params/extrinsics`; wrappers must use consistent conventions.
- `native_sampling_mode` affects point-map sampling. Check output-resolution policies before comparing models.
