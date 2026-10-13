# Runbook

Date: 2026-05-26

## Pre-run checklist

1. Confirm that the dataset path exists:

```bash
ls data/official-data/nuScenes
```

2. If using an API-based VLM, set its environment variable:

```bash
export DOUBAO_API_KEY=...
```

3. Confirm that the selected model's local checkpoint and project root are configured in the YAML `models` section.

4. Check which pipeline stages are enabled. By default, `ProcessingPipeline._process_sequence_task()` currently runs only the semantic scale stage:

```python
self._process_scale_for_sequence(sequence_path, sequence_name, device=device)
```

To run from scratch, re-enable VLM, SAM, P3 image preparation, and depth inference.

## Recompute scale from existing intermediates

This mode requires existing files such as:

```text
vlm_outputs/results/*.json
sam_outputs/
{model}_outputs/relative_scale/pointmap/*.npy
{model}_outputs/relative_scale/conf/*.npy
{model}_outputs/relative_scale/camera_params/*.json
```

Run:

```bash
python get_metric_scale.py --config configs/nuscenes/lingbot_map_long.yaml
```

This regenerates scale variants and then computes metrics in the second part of the main script.

## Regenerate end to end

Enable the required calls in `_process_sequence_task()` in `metric_scale/processing/processing.py`:

```python
self._process_vlm_for_sequence(sequence_path, sequence_name, device=device)
self._process_sam_for_sequence(sequence_path, sequence_name, device=device)
self._prepare_p3_for_sequence(sequence_name, device=device)
self._process_depth_for_sequence(sequence_path, sequence_name, device=device)
self._process_scale_for_sequence(sequence_path, sequence_name, device=device)
```

Start with a smoke run on one sequence and a few frames:

```bash
python get_metric_scale.py \
  --config configs/nuscenes/pi3_long.yaml \
  --sequence-name scene-0092 \
  --frame-count 10 \
  --disable-parallel \
  --disable-visualization
```

## Multi-GPU runs

Relevant configuration:

```yaml
parallel:
  enable: true
  max_gpus: 10
  vlm_api_enable: true
  max_vlm_workers: 1
```

Parallel scheduling works at the sequence and subsequence levels. It is used only when CUDA is available, more than one GPU is available, and there are enough tasks.

## Common workflows

### Quick single-sequence check

```bash
python get_metric_scale.py \
  --config configs/kitti/pi3_long.yaml \
  --sequence-name 0000 \
  --frame-count 30 \
  --disable-parallel
```

### Full-dataset batch

Set:

```yaml
runtime:
  all_sequences: true
  sequence_name: []
```

Then run the relevant YAML configuration.

### Switch depth models

Prefer switching to the model's full configuration file over only overriding `--depth-model`. Different models often need different `models` paths, `output_resolution` settings, and wrapper parameters.

## Inspect outputs

After a run, check:

```text
outputs/{dataset}/{sequence}/{model}_outputs/
outputs/{dataset}/metric_results/{model}/
```

Key reports:

- `dataset_summary.json`
- `dataset_summary.md`
- `sequences/{sequence}.json`
- `sequences/{sequence}.md`

## Cleanup

Delete only outputs you have confirmed can be regenerated. Check `git status` and the exact target paths before removing anything.
