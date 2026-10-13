# Ablation Study Reproduction Audit

Audit date: 2026-09-23. Source project: `ablation_study` on the source server. Scope: VGGT-Omega, nuScenes, and FusionPortable only. This was a read-only audit: the source project was not modified, no environment was installed, and the full experiment was not rerun.

## Conclusion

The project is not yet a self-contained package that can be installed on a new server and run from raw data with one command. The main project has no `requirements.txt`, `environment.yml`, `pyproject.toml`, `Dockerfile`, or `install.sh`. Installation files for other models under `vendor/` are not a dependency manifest for this project.

Existing launchers use the Conda `base` environment for ablation evaluation and `pi-long` for UniMetric scale recovery. This does not establish that the whole project used QWen. Scripts invoked with bare `python` depend on the caller's active environment. `QWen` is an environment name and does not by itself mean a Qwen model was loaded.

## Actual workflow and external dependencies

1. `scripts/run_relative_scale.py`: runs VGGT-Omega inference and writes `relative_scale`. It reads `vendor/vggt-omega/source`, which is actually a symlink to a location outside the project.
2. `scripts/run_unimetric.py`: hard-codes an import of the external `Metric-Depth-new` pipeline, reads VLM/SAM caches from a shared output tree, and runs scale estimation. It does not generate VLM/SAM results from images.
3. `scripts/run_ablation.py`: reads `relative_scale` and `semantic_scale_debug`, generates scale variants, and evaluates AbsRel, delta1, CD, and ATE. Outputs include per-sequence results, a dataset summary, and `ablation_table.txt`.
4. `final_8/*.sh`: resumption scripts tied to historical run IDs and existing Oracle result directories. They are not clean-install entry points.

To produce only the six VGGT-Omega result types, explicitly use:

```text
none,metric_model,midpoint_mean,midpoint_median,table_lookup,oracle
```

Omitting `--variants` also requests MoGe-2, MoGe-3, and UniDepth-v1/v2 calibration controls. Those four rows are outside this audit's model scope; there is no need to package their weights or the baseline_model, MoGe, UniDepth, or FlexGEMM source trees.

## Existing environment versions

These versions were observed on the source server:

| Package | base (evaluation launcher) | pi-long (UniMetric launcher) |
|---|---|---|
| Python | 3.11.3 | 3.10.18 |
| torch | 2.6.0+cu124 | 2.5.1+cu118 |
| torchvision | 0.21.0+cu124 | 0.20.1+cu118 |
| numpy | 1.24.4 | 1.26.4 |
| scipy | 1.10.1 | 1.15.3 |
| opencv-python | 4.11.0.86 | 4.11.0.86 |
| PyYAML | 6.0.2 | 6.0.3 |
| Pillow | 9.5.0 | 12.0.0 |
| einops | 0.8.2 | 0.8.1 |
| transformers | 4.49.0 | 5.0.0 |

This is a version audit, not a complete lockfile tested in a clean installation. A reproducibility package should export a complete manifest from a verified environment, record editable source locations, and test an installation script in an isolated environment. Do not mechanically copy every unrelated package from `base` into requirements.

Import tests for the VGGT-Omega model module, project wrapper, and ablation `scale_variants` passed in `base`, `pi-long`, and `QWen`. The QWen environment had torch 2.6.0+cu124. It is a candidate for a unified environment, but this does not show that historical experiments ran there or replace GPU inference and end-to-end tests.

The Omega `source` checkout on the source server was at commit `39a0cb8af88554f15ddcb5354cd52bde588fa014`. Preserve any local modifications as well; the commit alone does not guarantee identical source.

## Weights

The current configuration pointed to `vggt_omega_1b_512.pt` on the source server. The file existed and was 4,576,706,117 bytes.

Official [source code](https://github.com/facebookresearch/vggt-omega) and [weights](https://huggingface.co/facebook/VGGT-Omega) are available separately. The official instructions require access approval for the weights. After approval, download with:

```bash
hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints
```

Alternatively, transfer the same file from the existing server and record its SHA-256. To reproduce existing results, retain the 512 checkpoint, matching source, and existing preprocessing; do not silently substitute the newer 416-Reproduction weights.

Recomputing VLM/SAM also requires SAM2 source, matching configuration and checkpoint, and an explicit VLM API backend, model version, and credentials. Reusing existing scale-debug JSON for the ablation does not require VLM/SAM execution or Qwen weights.

## Cache coverage and packaging boundary

A joint check of the shared outputs and project outputs found nonempty relative point maps, `shared_scale_*.json`, VLM results, and SAM directories for 116 nuScenes sequences and five FusionPortable sequences. This checked existence, not frame-by-frame completeness. The older `PLAN.md` claim that only three sequences had predictions is outdated.

Three levels of reproduction are possible:

- **Recompute result tables:** package project source, raw data, full `relative_scale` structures (point maps, confidence, camera parameters), `semantic_scale_debug`, existing reference results, and configurations. This is easiest to compare with existing numbers but does not rerun model inference or semantic extraction.
- **Rerun Omega and scale recovery:** also provide Omega source and weights, VLM/SAM caches, and the `Metric-Depth-new` scale solver and dependencies. After regenerating `relative_scale`, recompute scale for the new predictions; do not apply old-prediction scales to them.
- **Rerun everything from images:** add a standalone VLM → SAM → Omega → scale recovery → evaluation entry point. `ablation_study` does not currently close this loop. Changes in VLM version or nondeterministic output may change the results; retain the original semantic caches for comparison.

Suggested package split:

1. `ablation-code`: main code, two YAML files, actual Omega source, and required scale-solver modules; record source versions and file hashes.
2. `omega-checkpoint`: existing 512 weights and SHA-256.
3. `semantic-cache`: VLM/SAM caches and debug JSON for only these two datasets; package actual files, not symlinks back to the old server.
4. `reference-results`: `result.txt`, corresponding sequence and dataset summaries, configurations, sequence/frame lists, and environment manifest.

The data ZIP was prepared separately; do not duplicate other datasets. For the table-only route, add a `relative_scale` data package. If rerunning inference, large prediction arrays need not be transferred.

## Changes needed for portability

- Parameterize data and source-output roots, Omega source and checkpoint paths, `run_unimetric.py`'s `MD_ROOT`/`SOURCE_ROOT`, and shell interpreter/root paths.
- Include required external source and resolve `vendor` and data symlinks. Do not dereference every output link and accidentally copy an entire shared drive.
- Provide an environment or locked dependency manifest, `install.sh`, `download_weights.sh` (with clear restricted-access failure), `check_assets.py`, a one-sequence smoke script, and run scripts for both datasets.
- Download weights during installation and verify all assets before offline execution. Existing launchers set `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`.
- Preflight each frame's JPG/PCD/YAML, point map, confidence, camera parameters, and cache frame IDs to avoid stale or incomplete outputs.
- Preserve chunk size 60, overlap 20, `resize_only`, `alignment=false`, the 100 m evaluation threshold, camera/pose conventions, and evaluation settings rather than changing them merely to make a run complete.
- Run a one-sequence smoke test before the full datasets. Check nuScenes (116 sequences/4,627 frames), FusionPortable (five sequences/3,847 frames), and identical aggregation rules.

Illustrative commands **after** paths and dependencies are fixed; these are not a claim of out-of-the-box execution:

```bash
python scripts/run_relative_scale.py --config configs/nuscenes/vggt_omega_long.yaml --sequence-name scene-0278
python scripts/run_unimetric.py --config configs/nuscenes/vggt_omega_long.yaml --sequence-name scene-0278
python scripts/run_ablation.py --config configs/nuscenes/vggt_omega_long.yaml --sequence-name scene-0278 --variants none,metric_model,midpoint_mean,midpoint_median,table_lookup,oracle
```

`handheld_room00` is a possible first FusionPortable check. The full workflow must run scale recovery for every sequence and then evaluate both datasets. Avoid directly invoking `final_8` scripts tied to historical run IDs.

## Reference results and limitations

The original `result.txt` reports UniMetric full values of 0.1670 / 0.8391 / 1.0028 / 4.9629 for nuScenes and 0.4178 / 0.5224 / 1.6907 / 14.8457 for FusionPortable (AbsRel/delta1/CD/ATE, respectively). These are existing report values, not values recalculated in this audit.

The original report says nuScenes `scene-0060`, `scene-0177`, `scene-0557`, `scene-0778`, and FusionPortable `ugv_parking00` had no valid scale intervals. The affected methods fell back to scale 1.0. Preserve and explicitly report this policy rather than silently omitting these sequences.
