# Runbook

The [README](../README.md#quick-start) provides the default setup and run commands. Place each MetricBench dataset directly in `data/metric_bench/<dataset>/` (for example, `data/metric_bench/JRDB/`) and model assets in the [default directories](../vendor/README.md#default-directories), then run a supplied YAML from the repository root. No configuration edits are needed. `python reproducibility/check_setup.py` checks the entire configuration matrix, including models and datasets you may not have installed.

## Start from empty outputs

By default, `get_metric_scale.py` generates VLM JSON, SAM masks, P3 images, depth/point maps, semantic scale, and metrics in sequence. A new run does not automatically reuse results from another run. Select a dataset and one of the nine supported models with its YAML:

```bash
python get_metric_scale.py --config configs/jrdb/pi3_long.yaml
```

Doubao configurations read `DOUBAO_API_KEY`; Humanoid Everyday's Qwen API configuration reads `MODEL_KEY` and `BASE_URL`. Do not put credential values in YAML or commit them.

## Resume and depth-only runs

`runtime.resume_existing_results` defaults to `false`. Set it to `true` only after verifying that code, YAML, input data, and caches match; otherwise the run may reuse metrics computed with obsolete parameters.

The legacy `runtime.depth_only: true` workflow remains supported. It requires VLM JSON for the corresponding frames and generates depth intermediates without computing scale or metrics. Leave it `false` or unset for a complete run.

`runtime.cleanup_depth_outputs_after_metrics: true` removes the current model's depth outputs after saving metrics. It does not remove VLM or SAM caches. Smoke-test one sequence and a small frame count before running the full dataset. A real run requires a working CUDA environment and the relevant model and API credentials.

## Outputs and external assets

Datasets, model weights, checkpoints, outputs, and P3 images are not part of the source commit. The default resource directories are excluded by `.gitignore`. Data is read from `data/metric_bench/<dataset>/`; model source and weights use `vendor/` and `pretrained_model/`; SAM weights use `checkpoints/`. Results, prepared images, and temporary files are written to `outputs/`, `p3_images/`, and `tmp/` respectively. Dataset paths resolve relative to the working directory, so launch commands from the repository root.

`reproducibility/freeze_external_assets.py` can generate a Git-ignored local manifest of asset paths and file hashes.
