# UniMetric Codebase Map

Date: 2026-05-26. This is a historical codebase snapshot; later changes may have extended the model and dataset lists.

## In one sentence

UniMetric is the central experiment repository in `Metric-Depth-Project`: it normalizes relative or weakly metric outputs from depth and 3D foundation models into point maps and camera parameters, then uses VLM semantic anchors, SAM masks, ground/plane cues, and LiDAR/pose ground truth for metric-scale calibration and evaluation.

## Input contract

Each sequence usually contains:

- `{frame_id}.pcd`: LiDAR point cloud.
- `{frame_id}.yaml`: camera, LiDAR, object, and scene metadata.
- `{frame_id}_camera{camera_id}.jpg`: camera image; the main pipeline uses `camera1` by default.

`FrameLoader` reads images, PCD, and YAML and constructs `lidar2camera`, `camera2world`, `world2camera`, intrinsics, object lists, image size, point clouds, projected bounding boxes, and BEV visualization inputs.

## Output contract

The usual root is `outputs/{dataset_name}/{sequence_name}/`. Each depth model writes to its own subtree:

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

`relative_scale` is the original or relative-scale model output. The scale stage produces metric-scale variants.

## Pipeline stages

1. **VLM processing:** `VLMProcessor.process_frames()` writes `vlm_outputs/results/*.json` with dynamic/static height anchors and semantic planes.
2. **P3 image preparation:** `P3ImagePreparer.prepare_images()` copies images referenced by VLM results to `p3_images/{sequence_name}/` for depth-model input.
3. **SAM processing:** `SAMProcessor.process_segmentation()` uses VLM bounding-box prompts to write masks under `sam_outputs/{frame}/`.
4. **Depth estimation:** `DepthProcessor.run_depth_estimation()` runs the selected foundation-model wrapper and writes `relative_scale/pointmap`, `conf`, and `camera_params`.
5. **Semantic plane scale estimation:** `SemanticPlaneScaleProcessor.estimate_scale()` estimates scale from semantic anchors and plane/ground geometry, producing `consistent_scale`, `optimized_scale_*`, `absolute_scale`, and other variants.

At the time of this snapshot, `ProcessingPipeline._process_sequence_task()` called only `_process_scale_for_sequence()`; the first four stages were commented out. Decide whether to restore the end-to-end pipeline before running a new experiment.

## Supported depth and 3D models

`ModelManager.initialize_depth_estimator()` supported Pi3/Pi3X (`pi3_long`), VGGT (`vggt_long`), VGGT-Omega (`vggt_omega_long`), Depth Anything 3 (`da3_long`), DUSt3R/MUSt3R (`dust3r_long`, `must3r_long`), Scal3R (`scal3r_long`), LingBot-Map (`lingbot_map_long`), and MoGe (`moge`).

Long-sequence wrappers commonly:

- Split input with `long_chunk_size` and `long_overlap`.
- Save standardized point maps, confidence, and camera parameters.
- Handle model-specific native/original resolution, intrinsic rescaling, and camera-extrinsic direction.
- Use `temp_root` for large-disk temporary files and `long_delete_temp_files` for cleanup.

## Semantic-scale mechanism

1. A VLM identifies objects whose real-world height can serve as a metric prior.
2. SAM converts VLM boxes into masks.
3. The predicted point map supplies relative anchor heights; real-world height intervals yield scale intervals.
4. Robust consensus and weighted optimization combine multiple anchors.
5. The resulting scales are applied to point maps and camera translations.

Main variants:

- `relative`: original model scale without semantic correction.
- `consistent`: robust consensus scale.
- `optimized_scale_legacy`, `optimized_scale_original`, `optimized_scale_weighted_bi`, and `optimized_scale_weighted_ifpa`: optimized alternatives.
- `optimized`/`absolute`: compatibility aliases, usually for the default optimized variant.
- `selected`: optional gated final scale, disabled in most configurations at the time.

Common configuration choices were `height_anchor_source: dynamic` or `both`, `default_optimized_variant: optimized_scale_original`, `enable_selected_scale: false`, and `sequence_difficulty.enabled: true` for long-sequence datasets.

## Evaluation

`MetricGeometryEvaluator` and `MetricCalculator` cover:

- Sparse depth metrics sampled at LiDAR-projected pixels.
- Point metrics against ground-truth camera-frame LiDAR points.
- A frame- or sequence-level sparse reconstruction proxy.
- TUM-style pose ATE/RPE without Sim3 scale correction.
- Optional `0-20m`, `20-40m`, `40-60m`, and `60m+` distance bins.
- Easy/mid/hard sequence-difficulty inference from a prior `dataset_summary.json`, optionally changing semantic-scale profiles.

Reports include sequence JSON and Markdown files, `dataset_summary.json`, a dataset Markdown report, and a sequence group map.

## Working with configuration

Typical command:

```bash
python get_metric_scale.py --config configs/nuscenes/lingbot_map_long.yaml
```

Selected CLI overrides:

```bash
python get_metric_scale.py \
  --config configs/nuscenes/pi3_long.yaml \
  --sequence-name scene-0092 \
  --frame-count 100 \
  --depth-model pi3_long
```

Important keys include `runtime.dataset`, `runtime.data_path`, `runtime.depth_model`, `runtime.use_shared_scale`, `runtime.all_sequences`, `runtime.sequence_name`, `runtime.frame_count`, `runtime.subsequence_size`, `runtime.enable_sparse_reconstruction`, `runtime.enable_pose_evaluation`, `runtime.sparse_reconstruction_mode`, `depth.output_resolution`, `semantic_scale.height_anchor_source`, `semantic_scale.default_optimized_variant`, `semantic_scale.sequence_difficulty`, and `paths.output_root`.

## Documentation gaps identified at the time

The earlier README had not explained the scale-only default pipeline, model wrappers' dependence on sibling repositories, the move to a dataset/model configuration matrix, scale-variant meanings, depth/point/reconstruction/pose metric definitions, or safe injection of sensitive configuration via environment variables. Later README and `docs/` updates addressed several of these gaps.

## Follow-up priorities

1. Rotate any previously hard-coded API key and migrate to environment variables.
2. Keep the README aligned with the actual pipeline.
3. Split `metrics.py` into depth, point, reconstruction, pose, and reporting modules.
4. Document the recommended use and default stage for each dataset/model configuration.
5. Add basic semantic-scale solver tests, especially interval aggregation and selected-scale gating.
