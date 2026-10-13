# Configuration Matrix

Production configurations live at `configs/<dataset>/<model>.yaml`. Each of the 12 dataset directories has a `default.yaml` and one configuration for each of the nine supported models: 120 dataset YAML files in total. Per-file hashes are recorded in the [source and configuration manifest](../reproducibility/source-and-config-manifest.json).

Dataset directories: `3eed_dataset`, `argoverse`, `cear`, `dair_v2x_i`, `droid`, `fusionportable`, `humanoid_everyday`, `jrdb`, `kitti`, `nuscenes`, `v2x_radar_i`, and `waymo`.

Supported model keys: `pi3_long`, `vggt_long`, `vggt_omega_long`, `da3_long`, `scal3r_long`, `lingbot_map_long`, `4rc_long`, `dvlt_long`, and `stream3r_long`. The former Humanoid Everyday `pi3x_long.yaml` also selected `depth_model: pi3_long`, but it belonged to a separate VLM/output-path experiment and is not part of the production matrix.

The retained 120 dataset YAMLs preserve algorithmic parameters from their source directories. Only 123 recorded path fields were changed during the server-checkout preparation: 120 `depth.long_temp_root` values were redirected to a dedicated temporary directory; the CEAR default data path was corrected to the existing `CEAR_indoor` directory; and the Argoverse LingBot source root and checkpoint were updated to verified paths. The LingBot checkpoint matched the new server's file by SHA-256. Explanatory comments were added for matching dataset/model configurations without changing non-comment parameter lines.

Important cases:

- `configs/3eed_dataset/vggt_omega_long.yaml` uses VGGT-Omega `balanced` preprocessing and dataset intrinsics. Both code paths are retained and regression-tested.
- Historical Argoverse runs used the outdoor fallback in `metric_scale.prompts.default`; neither the original run repository nor the new server had a dedicated Argoverse prompt. This checkout's `metric_scale/prompts/argoverse.py` explicitly forwards the same text. All four legacy prompt types were compared byte-for-byte.
- The JRDB Pi3, Scal3R, and 4RC configurations use `semantic_scale.sequence_overrides`, applied per sequence at runtime.
- Current code parses and executes the additional offload and scale-fallback switches for CEAR/Humanoid Scal3R, FusionPortable, and V2X Radar.
- `resume_existing_results` is off by default to avoid silently reusing metrics after configuration changes.

For path migration, upstream source/weights, and licensing, see the [runbook](runbook.md) and [vendor guide](../vendor/README.md). Third-party source is not committed to the public repository.
