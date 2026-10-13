"""CPU-only regression checks for active feature flags in the 12-dataset YAMLs."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metric_scale.config.config import Config


def load(relative):
    sys.argv = [sys.argv[0], '--config', str(ROOT / relative)]
    return Config()


omega = load('configs/3eed_dataset/vggt_omega_long.yaml')
assert omega.vggt_omega_preprocess_mode == 'balanced'
assert omega.vggt_omega_geometry_intrinsics_source == 'dataset'

pi3 = load('configs/jrdb/pi3_long.yaml')
pi3.activate_semantic_scale_sequence_profile('bytes-cafe-2019-02-07_0')
assert pi3.activate_semantic_scale_sequence_override('bytes-cafe-2019-02-07_0')
assert abs(pi3.dynamic_anchor_height_scale - 0.90) < 1e-9
pi3.clear_semantic_scale_sequence_profile()
assert not pi3.resume_existing_results

fusion = load('configs/fusionportable/4rc_long.yaml')
assert fusion.fallback_degenerate_plane_mask

v2x = load('configs/v2x_radar_i/scal3r_long.yaml')
assert not v2x.scale_camera_translation
dvlt = load('configs/v2x_radar_i/dvlt_long.yaml')
assert dvlt.fallback_to_all_dynamic_anchors_if_subsequence_empty

cear = load('configs/cear/scal3r_long.yaml')
assert cear.scal3r_offload_batches == 1
assert cear.scal3r_offload_outputs == 1

humanoid = load('configs/humanoid_everyday/pi3_long.yaml')
assert humanoid.vision_model_type == 'qwen-api'
assert humanoid.sequence_difficulty_hard_sequences
hard_profile = humanoid.resolve_semantic_scale_sequence_profile(
    humanoid.sequence_difficulty_hard_sequences[0]
)
assert hard_profile['profile_name'] == 'hard'

import numpy as np
from metric_scale.processing import processing
original_config = processing.config
try:
    processing.config = v2x
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1, 2, 3]
    unscaled = processing.SemanticPlaneScaleProcessor._scale_camera_extrinsics(pose, 2.0)
    assert np.array_equal(unscaled[:3, 3], [1, 2, 3])
    processing.config = pi3
    scaled = processing.SemanticPlaneScaleProcessor._scale_camera_extrinsics(pose, 2.0)
    assert np.array_equal(scaled[:3, 3], [2, 4, 6])
    assert np.array_equal(pose[:3, 3], [1, 2, 3])
finally:
    processing.config = original_config
assert processing.ProcessingPipeline()._existing_sequence_metric_path('nonexistent') is None

# A fresh run must mine VLM JSON before SAM/depth/scale, rather than assuming
# an existing cache from a previous checkout.
pipeline = processing.ProcessingPipeline()
with patch.object(pipeline, 'prepare_sequences', side_effect=lambda: setattr(pipeline, 'sequences', [('/unused', 'test_sequence')])), \
     patch.object(pipeline, '_should_parallelize_sequences', return_value=False), \
     patch.object(pipeline, 'process_vlm') as vlm_stage, \
     patch.object(pipeline, '_process_sequence_task') as sequence_stage, \
     patch.object(processing.model_manager, 'release_all'):
    pipeline.run()
    vlm_stage.assert_called_once()
    sequence_stage.assert_called_once_with('/unused', 'test_sequence')

import get_metric_scale
old_runtime = dict(get_metric_scale.config._config['runtime'])
try:
    get_metric_scale.config._config['runtime']['depth_only'] = True
    with patch.object(get_metric_scale.config, 'print_summary'), \
         patch.object(get_metric_scale.pipeline, 'run') as pipeline_run, \
         patch.object(get_metric_scale.MainRunner, 'process_visualization_and_metrics') as metrics_run:
        get_metric_scale.MainRunner.run()
        pipeline_run.assert_called_once()
        metrics_run.assert_not_called()
finally:
    get_metric_scale.config._config['runtime'] = old_runtime

# Pipeline-saved current-run results must be summarized without recomputing
# after depth intermediates have been cleaned.
with tempfile.TemporaryDirectory() as directory:
    saved = Path(directory) / 'sequence.json'
    saved.write_text('{"sequence_name":"test_sequence"}', encoding='utf-8')
    with patch.object(get_metric_scale.sequence_utils, 'get_sequences', return_value=[('/unused', 'test_sequence')]), \
         patch.object(get_metric_scale.metric_calculator, '_sequence_result_path', return_value=str(saved)), \
         patch.object(get_metric_scale.metric_calculator, 'calculate_all_metrics') as calculate, \
         patch.object(get_metric_scale.metric_calculator, 'save_dataset_summary') as summarize, \
         patch.object(get_metric_scale.result_visualizer, 'visualize_all_results') as visualize:
        get_metric_scale.MainRunner.process_visualization_and_metrics()
        calculate.assert_not_called()
        visualize.assert_not_called()
        summarize.assert_called_once()

print('PASS: Omega balanced/dataset-K, JRDB override, Fusion/V2X fallback, Scal offload,')
print('      Humanoid Qwen/hard-sequences, camera translation, depth-only, resume default,')
print('      and current-run metric reuse without stale cross-run metrics')
