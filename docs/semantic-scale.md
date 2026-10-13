# Semantic Scale

Date: 2026-05-26

## Goal

Many depth and 3D foundation models output relative scale or only weakly metric scale. The semantic scale stage estimates a metric scale from real-world height priors for recognizable objects and relative measurements in the predicted point map. It then applies the same scale to the point map and camera translations.

## Inputs

The stage uses three kinds of input:

1. VLM results: `dynamic_height_anchors`, `static_height_anchors`, and `semantic_planes`.
2. SAM masks: object and plane masks generated from VLM bounding-box prompts.
3. Relative outputs from the depth model: `relative_scale/pointmap/*.npy`, `relative_scale/conf/*.npy`, and `relative_scale/camera_params/*.json`.

## Anchor types

### Dynamic height anchors

Typical objects include cars, sedans, SUVs, vans, buses, trucks, pedestrians, cyclists, and motorcycles. They provide common-sense height ranges, such as passenger-car height or human height.

### Static height anchors

Typical objects include traffic cones, barriers, curbs, guardrails, bollards, and poles. Static infrastructure may be more stable in some datasets, but its dimensions can vary more across category definitions.

### Scene planes

Typical planes include roads and the ground. They provide local or global geometric constraints for the bottom or ground reference of an anchor.

## Solver workflow

1. The VLM supplies each anchor's bounding box, height interval, confidence, and rationale.
2. SAM converts the box into a mask.
3. The scale estimator measures the anchor's relative height in the point map.
4. Dividing the real-world height interval by the relative height yields a scale interval.
5. Low-quality or low-confidence anchors and geometrically unreliable measurements are filtered out.
6. The remaining intervals are aggregated into `consistent_scale` and optimized variants.
7. Scaled point maps and camera-extrinsic translations are saved in the corresponding output directories.

## Main configuration

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

Important thresholds include `depth_geometry_min_anchor_points`, `depth_geometry_min_relative_height`, `vlm_dynamic_anchor_confidence_threshold`, `vlm_static_anchor_confidence_threshold`, `anchor_quality_min_percentile`, `anchor_quality_max_percentile`, and `plane_conf_percentile`.

## Scale variants

- `relative_scale`: original model output.
- `consistent_scale`: robust consensus output.
- `optimized_scale_legacy`: legacy optimized solver.
- `optimized_scale_original`: commonly used default optimized variant.
- `optimized_scale_weighted_bi`: weighted BI variant.
- `optimized_scale_weighted_ifpa`: weighted IFPA variant.
- `optimized_scale`: compatibility alias, usually pointing to the default optimized variant.
- `absolute_scale`: compatibility alias, usually matching the default optimized output.
- `selected_scale`: optional gated final scale; currently disabled in most configurations.

## Sequence difficulty profiles

`sequence_difficulty` can read a previous `dataset_summary.json`, classify sequences as easy, mid, or hard from the performance of relative and candidate scales, and override some semantic-scale parameters.

Common rules use `relative_scale` as the baseline, `consistent_scale` and enabled optimized variants as candidates, and `depth_frame_mean.delta1` as the metric. A candidate gain above a threshold is classified as easy; regression against the baseline is hard; other cases are mid.

## Debug outputs

Setting `save_scale_intermediate_results: true` writes `semantic_scale_debug/`, including scale intervals, anchor measurements, rejection summaries, consensus/BI/IFPA details, and selected-scale gate information. These files help explain why no valid scale anchors were found or why an optimized scale shifted too far.
