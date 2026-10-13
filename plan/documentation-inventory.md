# Documentation Inventory and Project Context

Date: 2026-05-26. This is a historical inventory; some gaps identified here were addressed by later README and `docs/` updates.

## Scope

This review covered the then-current `Metric-Depth/` repository, closely related sibling repositories under `Metric-Depth-Project/`, and the configuration, pipeline, model adapters, evaluation, and utility scripts. It did not change existing code behavior.

Its outputs were this inventory, `plan/metric-depth-codebase-map.md`, a local-only project memory snapshot (not part of this public repository), and later updates to `README.md` and `docs/` covering the pipeline, configuration, semantic scale, and evaluation.

## Documentation at the time of review

At the start of the review, the only Markdown document in `Metric-Depth/` was a README describing an early dataset format: sequences containing PCD, YAML, and three camera JPG files; YAML camera intrinsics/extrinsics, LiDAR pose, and object metadata; and `dataset_tools/varify_dataset.py` for validation and visualization.

That README no longer covered the actual platform, which had grown to compare multiple depth and 3D reconstruction foundation models with VLM/SAM semantic anchors, long-sequence adapters, metric-scale calibration, multiple scale variants, LiDAR/pose evaluation, and cross-run comparisons. The README and `docs/` were expanded after this inventory.

No existing project-memory registry or Obsidian vault binding was found in the parent workspace. Context was reconstructed from the repository, sibling-repository READMEs, configuration tree, and workspace state.

## Parent-project context

`Metric-Depth-Project/` acts as a foundation-model integration workspace. `Metric-Depth/` is the central evaluation and calibration repository; sibling repositories supply local model implementations or background:

- `Depth-Anything-3/`: DA3/DA3-Streaming, including visual geometry, depth, confidence, pose, and point maps.
- `VGGT-Long/`: long RGB-sequence VGGT with chunking, overlap, alignment, and loop closure.
- `Pi-Long/`: adaptation of the long-sequence framework to Pi3/Pi3X, emphasizing stability and lower memory use.
- `Scal3R/`: large-scale 3D reconstruction using test-time training and block inference.
- `vggt-omega/`: VGGT-Omega 1B camera/depth/point/cloud outputs; this repository has a wrapper.
- `lingbot-map/`: streaming 3D reconstruction with KV cache and long/windowed sequences; this repository has a long-sequence wrapper.
- `STream3R/`: background for causal/streaming reconstruction; no direct wrapper was found in this repository at the time.

Together, these repositories supply candidate depth and 3D outputs. This repository unifies their I/O, estimates metric scale, evaluates against sparse LiDAR and poses, and compares models and scale strategies.

## Actual role of the code

The central question is not simply how to run one depth model. It is how to turn outputs from different foundation depth and 3D reconstruction models into comparable metric geometry for long autonomous-driving and mobile-robot sequences, using semantic anchors, ground/plane cues, sparse LiDAR ground truth, and pose metrics.

`get_metric_scale.py` invokes `metric_scale.processing.processing.pipeline.run()`. At review time, `ProcessingPipeline._process_sequence_task()` enabled only the semantic plane scale stage; VLM, SAM, P3 image preparation, and depth inference were commented out. The default run therefore reused existing intermediates for scale and metrics rather than running every stage end to end.

## Main code map

- `get_metric_scale.py`: `MainRunner.run()` prints configuration, runs the pipeline, then handles visualization and metrics. `process_visualization_and_metrics()` uses `ResultVisualizer` and `MetricCalculator` by sequence. `cleanup_post_metric_outputs()` removes postprocessing outputs other than `relative_scale` from depth-output directories.
- `metric_scale/config/config.py`: YAML loading, CLI overrides, template-path resolution, and the `runtime`, `parallel`, `vision`, `depth`, `semantic_scale`, `models`, and `paths` sections. Sequence-difficulty profiles can use an earlier `dataset_summary.json`.
- `metric_scale/processing/processing.py`: VLM anchor mining, copying referenced images to `p3_images/`, SAM segmentation, depth/3D inference, and semantic plane scale estimation. It produces `consistent_scale`, `optimized_scale_*`, `selected_scale`, and `absolute_scale` and supports multi-GPU sequence/subsequence scheduling.
- `metric_scale/models/models.py`: `ModelManager` and `ModelFactory` coordinate wrappers for `pi3_long`, `vggt_long`, `vggt_omega_long`, `da3_long`, `dust3r_long`, `must3r_long`, `scal3r_long`, `lingbot_map_long`, and `moge`.
- `metric_scale/models/semantic_scale_estimator.py`: calculates scale intervals from VLM anchors, semantic planes, masks, point maps, and confidence; includes robust consensus, weighted BI, IFPA, consensus priors, class allowlists, and anchor confidence/geometry filters.
- `metric_scale/models/vlm_semantic_scale_anchor.py`: extracts dynamic height anchors, static traffic-facility anchors, and scene planes. Prompts come from `metric_scale/prompts/{dataset}.py` or a default fallback. Boxes are normalized to [0,1000] before conversion to pixel coordinates.
- `metric_scale/metrics/metrics.py`: sparse projected-LiDAR depth/point metrics, sparse reconstruction proxy, TUM-style pose ATE/RPE, distance bins, and sequence/dataset reports and difficulty groups. It was already large enough to merit splitting into smaller modules.
- `dataset_tools/`: dataset conversion, validation, cleanup, sequence filtering, and camera-ego, BEV, and sparse-reconstruction visualization.

## Configuration tree at the time

The structure had changed from single-file dataset configurations to `configs/{dataset}/{model}.yaml`. Datasets included `argoverse`, `dair_v2x_i`, `kitti`, `nuscenes`, `v2x_radar_i`, and `waymo`. Most had configurations for Pi3, VGGT, VGGT-Omega, DA3, DUSt3R, MUSt3R, Scal3R, and LingBot-Map; some also had `default` or `pi3x_long`.

At review time, `config.py` defaulted to `configs/nuscenes/lingbot_map_long.yaml`. Most long-sequence configurations used `frame_count: 1000` and `subsequence_size: 1000`; older defaults retained 60-frame test settings. Model-dependent `original` versus `native` output resolution makes sampling mode important in comparisons.

## Risks identified

- Some YAML configurations had previously contained hard-coded API keys. They were migrated to `vision.api_key_env: "DOUBAO_API_KEY"`; any previously exposed credentials still needed rotation and revocation.
- The default pipeline enabled only the scale stage. Re-enable VLM/SAM/depth stages deliberately for an end-to-end run.
- The workspace contained many uncommitted changes and untracked directories; distinguish user work from generated outputs before cleaning up or committing.
- `metric_scale/metrics/metrics.py` exceeded 3,000 lines and would benefit from splitting.
- There was no repository-local `pyproject.toml` or `requirements.txt`; dependencies came from external environments and sibling repositories, making reproduction harder.
- Keep README claims aligned with actual end-to-end runs as the project evolves.

## Recommended follow-up

1. Rotate and revoke previously exposed API credentials.
2. Add focused tests for configuration loading and `vision.api_key_env`.
3. Add basic unit tests for scale-interval aggregation and selected-scale gating.
4. Split `metrics.py` into depth, point, reconstruction, pose, and reporting modules.
5. Add a repository-local dependency manifest to lower the cost of reproducing experiments.
