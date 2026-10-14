# Third-party model implementations

The `vendor/` subdirectories on the development server are a local dependency
cache. They are **not** part of the public Git repository: `.gitignore` excludes
them while keeping this README. The project wrappers and YAMLs remain in the
public repository; obtain each model implementation and checkpoint from its
upstream publisher and use the default directories below. The bundled YAMLs
already point to these locations.

See the [GFM environment guide](../docs/gfm-environments.md) for installation
commands and dependency constraints. Eight models share `unimetric`; DVLT
uses `unimetric-dvlt`. The default `bash scripts/install_gfm.sh` installs all
nine models, SAM 2, and their checkpoints into these two environments without
requiring model selection. VGGT-Omega weights require prior Hugging Face access
approval and authentication, as described in the guide. `python get_metric_scale.py`
defaults to LingBot-Map on nuScenes in `unimetric`. Each GFM runs in the
active Python environment; a source directory does not select another Conda
environment.

## Default directories

All paths are relative to the UniMetric repository root. Install the selected
upstream project's dependencies and place its source and weights as follows:

| Model | Source / import package | Weight location |
| --- | --- | --- |
| Pi3 / Pi3X | `vendor/Pi3/` | `pretrained_model/Pi3X/` |
| VGGT | `vendor/VGGT/`, installed as the `vggt` package | `pretrained_model/VGGT-1B/` |
| VGGT-Omega | `vendor/VGGT-Omega/` | `pretrained_model/VGGT-Omega/vggt_omega_1b_512.pt` |
| Scal3R | `vendor/Scal3R/` | `pretrained_model/Scal3R/scal3r.pt` |
| 4RC | `vendor/4RC/` | `pretrained_model/4RC/` |
| STream3R | `vendor/STream3R/` | `pretrained_model/STream3R/` |
| Depth Anything 3 | `vendor/Depth-Anything-3/`, installed as `depth_anything_3` | `pretrained_model/DA3-LARGE/` |
| LingBot-Map | `vendor/LingBot-Map/` | `pretrained_model/LingBot-Map/lingbot-map-long.pt` |
| DVLT | `vendor/DVLT/`, installed in editable mode (`src/dvlt/` layout) | `pretrained_model/DVLT/` |

For directory-based checkpoints, retain the upstream model files and directory
structure. Only the VGGT backend needs the `vggt` package; the derived adapters
reuse UniMetric's chunk utilities without importing that package. Scal3R reads
`configs/models/scal3r.yaml` from its source directory.

`bash scripts/install_gfm.sh <model>` installs SAM 2 from `vendor/SAM2/` into
each selected environment and downloads `checkpoints/sam2.1_hiera_large.pt`.
Its configuration remains the package resource
`configs/sam2.1/sam2.1_hiera_l.yaml`. Source and weight revisions are pinned in
[`scripts/gfm_profiles.json`](../scripts/gfm_profiles.json).

## Upstream references

| Model source | Upstream | Licensing note |
| --- | --- | --- |
| Pi3 / Pi3X | [yyfz/Pi3](https://github.com/yyfz/Pi3) | Code: BSD-3-Clause; model weights: separate non-commercial terms. |
| VGGT | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | See the upstream license and checkpoint terms. |
| VGGT-Omega | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) | See the upstream license and checkpoint access terms. |
| Scal3R | [zju3dv/Scal3R](https://github.com/zju3dv/Scal3R) | Project Registration License v1.0; organizational project use requires registration. |
| 4RC | [Luo-Yihang/4RC](https://github.com/Luo-Yihang/4RC) | S-Lab License 1.0; non-commercial redistribution/use terms. |
| STream3R | [NIRVANALAN/STream3R](https://github.com/NIRVANALAN/STream3R) | S-Lab License 1.0; non-commercial redistribution/use terms. |
| Depth Anything 3 | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Code: Apache-2.0; check each checkpoint's model card separately. |
| LingBot-Map | [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map) | Code: Apache-2.0; check checkpoint terms separately. |
| DVLT | [nv-tlabs/dvlt](https://github.com/nv-tlabs/dvlt) | Mixed source terms (Apache-2.0 plus VGGT-derived code); checkpoint has separate non-commercial terms. |

`LegacyQWenSite` is a mixed local runtime snapshot rather than a standalone
upstream model release; it is deliberately not published. The main project
license, once authorized, will not relicense any third-party source or model
weights. Review the upstream license and model card for the exact version you
install. For the original server environment, the local source/config manifest
and external-asset manifest record hashes and paths for audit; they do not
replace upstream permissions.
