# Visualizing Scale Recovery on nuScenes

This document records the state checked on 2026-09-16. The local `data/` directory was empty. A remote project configured through SSH was checked: its `outputs` link pointed to a shared output tree, raw nuScenes data were accessible under `data/official-data/nuScenes`, and JPG, PCD, and YAML files for `scene-0003` were confirmed.

The shared nuScenes output tree had 116 scene directories, but all 929 model-output directories were empty. A recursive scan found no NPY, NPZ, PLY, or PCD prediction arrays. SAM masks, VLM results, nine model evaluation reports, and ablation files remained. Another workspace cited by the reports pointed to the same output root.

Nine model reports were downloaded to `output/figures/nuscenes_scale/evidence/all_summaries.json` with source paths retained. `scripts/plot_nuscenes_available_metrics.py` generates real three-task metric plots and per-scene CSV files from those reports, averaging only scenes valid both before and after scale recovery with equal weight per scene. Eight models covered 116 scenes each; Scal3R covered only one. These metric plots are **not** visualizations of predicted depth, point clouds, or trajectories. Original predictions must be recovered or rerun. `nuscenes_layout_preview` remains a layout mockup only.

## Generate figures

```powershell
python scripts/plot_nuscenes_scale_triptych.py
python scripts/plot_nuscenes_scale_triptych.py --bundle path/to/scene.npz --title "nuScenes scene-XXXX | model"
```

The script writes PNG and PDF files. Without a bundle, it produces only a clearly labeled empty layout. A real bundle produces depth, reconstruction, and ego-pose figures.

## NPZ input contract

All arrays must be numeric; pickle is not allowed. The `gt`, `before`, and `after` suffixes mean ground-truth reference, before scale recovery, and after scale recovery.

| Key | Shape | Meaning |
|---|---|---|
| `rgb` | H,W,3 | RGB from the same frame, uint8 or float in [0,1] |
| `depth_gt / depth_before / depth_after` | H,W | Camera-z depth in meters; invalid values are NaN or <=0; GT remains LiDAR-sparse |
| `object_gt / object_before / object_after` | N,3 | Point cloud for the same object ROI; camera coordinates x right, y down, z forward |
| `endpoints_gt / endpoints_before / endpoints_after` | 2,3 | Corresponding 3D positions of the same physical endpoints, in meters |
| `cloud_gt / cloud_before / cloud_after` | N,3 | Fused point cloud in a shared coordinate frame, z up, in meters |
| `trajectory_gt / trajectory_before / trajectory_after` | T,3 | Time-synchronized ego translations in a shared coordinate frame, z up, in meters |

Point-cloud branches may contain different numbers of points. Trajectories must have equal lengths. By default, a displacement-scale ratio is computed every ten frames; segments with GT displacement <=0.5 m are omitted from the ratio. ATE is translational RMSE in the shared input frame; the script performs no alignment.

## Data preparation and fair comparisons

- Use the same scene, `CAM_FRONT`, and timestamps. Do not select the most favorable frames separately for before and after.
- Transform LiDAR through the ego pose at its sample time and the ego pose at camera exposure time into camera coordinates. Keep the nearest point with a z-buffer when projecting. Depth means camera-z, not LiDAR range.
- For single-frame depth and object-cloud figures, use one depth color scale and label the 3D Euclidean distance of the same endpoint pair. A nuScenes annotation-box size is not automatically the distance between arbitrary visible points; label it as an annotation-box extent if used.
- The current local object-cloud plot is an x-z projection with a shared range. A final rendering can use RGB point clouds from the same virtual viewpoint; compute dimensions from 3D endpoints.
- State whether fused reconstruction uses predicted or GT pose. GT pose is appropriate only when isolating the depth-scale effect. Filter or motion-compensate dynamic objects before fusion.
- Preserve scale in reconstruction and trajectory plots. A single rigid SE(3) alignment of the whole sequence is acceptable if its transform is recorded; Sim(3), per-frame, and per-segment scale alignment are not.
- Convert ego and camera poses using calibrated extrinsics before comparison. Do not mix them; apply corresponding transforms to point clouds and trajectories.
- If a model outputs arbitrary scale, state its unit convention. Do not label arbitrary units as meters or fit scale to GT and call that the unrecovered result.
- Save the scene, frame, model, scale variant, units, alignment transform, and cloud-filtering policy as provenance. Summary metrics alone cannot generate these real visualizations.

Once prediction paths are available, build bundles from this repository's `pointmap/*.npy`, `camera_params/*.json`, and frame YAML/PCD contracts. The plotting script is not already a complete raw-nuScenes data adapter.
