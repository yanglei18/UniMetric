# Runbook

The [reproducibility guide](../reproducibility/README.md) is the full setup reference. First run `python reproducibility/check_setup.py` to check datasets, model source, checkpoints, and the SAM2 configuration.

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

Datasets, model weights, checkpoints, outputs, and P3 images are not part of the source commit. Local symbolic links and third-party `vendor/` source on the original server are excluded by `.gitignore`. On another machine, install the model source as described in the [vendor guide](../vendor/README.md) and update the YAML `data_path`, model roots, checkpoints, and output paths.

`reproducibility/freeze_external_assets.py` can generate a Git-ignored local manifest of asset paths and file hashes.
