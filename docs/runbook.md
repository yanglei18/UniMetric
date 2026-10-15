# Runbook

The [README](../README.md#quick-start) provides the default setup and run commands. Place each MetricBench dataset directly in `data/metric_bench/<dataset>/` (for example, `data/metric_bench/JRDB/`) and model assets in the [default directories](../vendor/README.md#default-directories), then run a supplied YAML from the repository root. No configuration edits are needed. `python reproducibility/check_setup.py` checks the entire configuration matrix, including models and datasets you may not have installed.

## Start from empty outputs

By default, `get_metric_scale.py` restores missing VLM JSON for the selected dataset/sequences from `precomputed/vlm_results_12datasets_20261013.zip`, preserves existing cache files, and reuses the responses. It then generates SAM masks, P3 images, depth/point maps, semantic scale, and metrics using the current configuration. Select a dataset and one of the nine supported models with its YAML:

```bash
python get_metric_scale.py --config configs/jrdb/pi3_long.yaml
```

Cached frames do not need VLM credentials. Missing frames use the configured API: Doubao reads `DOUBAO_API_KEY` and `DOUBAO_BASE_URL`; Humanoid Everyday's Qwen API configuration reads `MODEL_KEY` and `BASE_URL`. Do not put credential values in YAML or commit them.

## Resume and depth-only runs

`runtime.reuse_vlm_results` defaults to `true` and controls VLM JSON reuse independently. Set it to `false` to regenerate responses. `runtime.resume_existing_results` defaults to `false`; full reuse of SAM/depth/metric outputs remains opt-in. Enable it only when code, YAML, input data, and caches match; otherwise the run may reuse metrics computed with obsolete parameters. Disable both flags when deliberately regenerating VLM responses.

The legacy `runtime.depth_only: true` workflow remains supported. It requires VLM JSON for the corresponding frames and generates depth intermediates without computing scale or metrics. Leave it `false` or unset for a complete run.

`runtime.cleanup_depth_outputs_after_metrics: true` removes the current model's depth outputs after saving metrics. It does not remove VLM or SAM caches. Smoke-test one sequence and a small frame count before running the full dataset. A real run requires a working CUDA environment and the relevant model assets; API credentials are needed only for new VLM requests.

## Outputs and external assets

Datasets, model weights, checkpoints, generated outputs, and P3 images are excluded by `.gitignore`. The portable VLM response archive is included under `precomputed/`. Data is read from `data/metric_bench/<dataset>/`; model source and weights use `vendor/` and `pretrained_model/`; SAM weights use `checkpoints/`. Results, prepared images, and temporary files are written to `outputs/`, `p3_images/`, and `tmp/` respectively. Dataset paths resolve relative to the working directory, so launch commands from the repository root. Cached JSON image paths resolve against the project root and the selected data directory.

`reproducibility/freeze_external_assets.py` can generate a Git-ignored local manifest of asset paths and file hashes.
