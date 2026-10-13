# Semantic Scale

Date: 2026-05-26

## Goal

Many depth and 3D foundation models output relative or only weakly metric geometry. The semantic-scale stage estimates metric scale from real-world height priors for recognizable objects and relative measurements in the predicted point map. It normally scales point maps and camera translations together. Set `semantic_scale.scale_camera_translation: false` to preserve the original camera translations, as in the V2X Radar Scal3R configuration.

## Inputs

The stage uses:

1. VLM results: `dynamic_height_anchors`, `static_height_anchors`, and `semantic_planes`.
2. SAM object and plane masks generated from VLM bounding-box prompts.
3. Relative depth-model outputs: `relative_scale/pointmap/*.npy`, `relative_scale/conf/*.npy`, and `relative_scale/camera_params/*.json`.

## Anchor types

- **Dynamic height anchors:** cars, sedans, SUVs, vans, buses, trucks, pedestrians, cyclists, and motorcycles. These provide common-sense height ranges.
- **Static height anchors:** traffic cones, barriers, curbs, guardrails, bollards, and poles. Such infrastructure can be stable within a dataset, but dimensions vary across category definitions.
- **Scene planes:** road and ground surfaces provide local or global geometric constraints for an anchor's bottom or ground reference.

## Solver workflow

1. The VLM proposes an anchor bounding box, height interval, confidence, and rationale.
2. SAM converts the box into a mask.
3. The estimator measures the anchor's relative height in the predicted point map.
4. Dividing the real-world height interval by relative height produces a scale interval.
5. Low-quality or low-confidence anchors and geometrically unreliable measurements are filtered out.
6. The remaining intervals are combined into `consistent_scale` and optimized variants.
7. Scaled point maps and, unless disabled, camera-extrinsic translations are saved under their output variants.

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
- `optimized_scale_legacy`: legacy optimized solution.
- `optimized_scale_original`: commonly used default optimized variant.
- `optimized_scale_weighted_bi`: weighted BI variant.
- `optimized_scale_weighted_ifpa`: weighted IFPA variant.
- `optimized_scale`: compatibility alias, usually for the default optimized variant.
- `absolute_scale`: compatibility alias, usually matching the default optimized output.
- `selected_scale`: optional gated final scale, currently disabled in most configurations.

## Sequence difficulty profiles

`sequence_difficulty` can read an earlier `dataset_summary.json`, classify sequences as easy, mid, or hard from the performance of relative and candidate scales, and override selected semantic-scale parameters. Common rules use `relative_scale` as the baseline, `consistent_scale` plus enabled optimized variants as candidates, and `depth_frame_mean.delta1` as the metric. A candidate gain above a threshold is easy; regression against the baseline is hard; remaining cases are mid.

## Debug outputs

`save_scale_intermediate_results: true` writes `semantic_scale_debug/` with scale intervals, anchor measurements, rejection summaries, consensus/BI/IFPA details, and selected-scale gate information. These files help diagnose missing valid anchors or excessive optimized-scale shifts.
