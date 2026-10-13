# Third-party model implementations

The `vendor/` subdirectories on the development server are a local dependency
cache. They are **not** part of the public Git repository: `.gitignore` excludes
them while keeping this README. The project wrappers and YAMLs remain in the
public repository; obtain each model implementation and checkpoint from its
upstream publisher, then set the corresponding `models.*_project_root` and
`models.*_checkpoint` paths in your local YAML copy.

| Model source | Upstream | Licensing note |
| --- | --- | --- |
| Pi3 / Pi3X | [yyfz/Pi3](https://github.com/yyfz/Pi3) | Code: BSD-3-Clause; model weights: separate non-commercial terms. |
| VGGT-Long | [DengKaiCQ/VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) | Follows the custom [VGGT License](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt); checkpoints have separate terms. |
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
