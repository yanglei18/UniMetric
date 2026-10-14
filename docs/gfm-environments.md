# GFM environments

Use two **Python 3.11 Conda environments**: **`unimetric`** for eight GFMs, and **`unimetric-dvlt`** for DVLT. The installer prepares the UniMetric runtime, selected GFMs, and SAM 2 together. Data and checkpoints stay in the repository's shared default directories.

## Install and run

Requirements: Linux x86_64, glibc >= 2.31 (Ubuntu 20.04 or newer), Conda/Miniforge, Git, and an NVIDIA driver supporting the selected CUDA wheel channel. Run commands from the UniMetric repository root.

```bash
# Install all nine backends, SAM 2, and their checkpoints
bash scripts/install_gfm.sh
conda activate unimetric

# After preparing data/metric_bench/nuScenes and VLM API credentials
python get_metric_scale.py
```

With no arguments, the installer automatically prepares all nine backends and their checkpoints across `unimetric` and `unimetric-dvlt`. No model or environment selection is needed. VGGT-Omega weights require prior approval on the [model page](https://huggingface.co/facebook/VGGT-Omega) and authentication through `HF_TOKEN` or `hf auth login`; the installer cannot grant that access. Rerun the same command after obtaining access if a gated download fails.

The main program uses `configs/nuscenes/lingbot_map_long.yaml` without arguments. No YAML path edits are needed. The [README](../README.md#metricbench) describes extracting `autonomous_driving.zip` to prepare `data/metric_bench/nuScenes/`; [API credentials](../README.md#1-set-api-credentials) are supplied through environment variables.

## Optional model selection

The default installation includes every backend. To install only a subset instead, pass one or more names, such as `bash scripts/install_gfm.sh pi3 da3`. Run another installed backend using its supplied YAML with `--config`.

| Installer argument | YAML model key | Conda environment | PyTorch / torchvision | CUDA wheel channel |
| --- | --- | --- | --- | --- |
| `pi3` | `pi3_long` | `unimetric` | 2.6.0 / 0.21.0 | cu124 |
| `vggt` | `vggt_long` | `unimetric` | 2.6.0 / 0.21.0 | cu124 |
| `omega` | `vggt_omega_long` | `unimetric` | 2.6.0 / 0.21.0 | cu124 |
| `da3` | `da3_long` | `unimetric` | 2.6.0 / 0.21.0 | cu124 |
| `scal3r` | `scal3r_long` | `unimetric` | 2.6.0 / 0.21.0 | cu124 |
| `lingbot` | `lingbot_map_long` | `unimetric` | 2.6.0 / 0.21.0 | cu124 |
| `4rc` | `4rc_long` | `unimetric` | 2.6.0 / 0.21.0 | cu124 |
| `stream3r` | `stream3r_long` | `unimetric` | 2.6.0 / 0.21.0 | cu124 |
| `dvlt` | `dvlt_long` | `unimetric-dvlt` | 2.5.1 / 0.20.1 | cu124 |

The YAML model keys also work as installer arguments. Additional aliases include `pi3x`, `vggt-omega`, `depth-anything-3`, and `lingbot-map`.

`bash scripts/install_gfm.sh common` installs all eight shared models; `bash scripts/install_gfm.sh all` is equivalent to the default no-argument installation. Single-model commands remain supported. Explicit selections add only those models, and subsequent shared-model installations reuse `unimetric`. Switch to `unimetric-dvlt` only when running DVLT.

The shared environment uses NumPy 1.26.4. These groups follow the server's QWen/dvlt organization and core torch/NumPy versions. They are not complete replicas: the server's DVLT environment uses Python 3.12, whereas both installer environments use Python 3.11.

## What the installer prepares

1. Clones the selected model and SAM 2 at the exact Git revisions in [`gfm_profiles.json`](../scripts/gfm_profiles.json).
2. Creates the selected environment and installs its torch pair from the [official PyTorch wheel index](https://pytorch.org/get-started/previous-versions/).
3. Resolves UniMetric and selected backend requirements together with any previously recorded models in that environment, then installs the GFMs in editable mode. SAM 2 is installed normally so the two environments have separate compiled extensions.
4. Downloads only the checkpoint files needed by that adapter, using pinned Hugging Face revisions. SAM 2.1 large goes in `checkpoints/`; GFM weights go in the [default weight directories](../vendor/README.md#default-directories).
5. Runs `pip check`, runtime imports for selected and previously recorded models, NumPy/Torch/Open3D array interoperability checks, a small CUDA matrix operation, and checkpoint-file checks. A failure returns a nonzero exit status.

Pi3 uses its [package metadata](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/pyproject.toml) and required inference dependencies instead of its full upstream requirements file, which pins torch 2.5.1. 4RC and STream3R also use their inference dependencies without the upstream training/demo requirements. LingBot uses the existing SDPA setting, so FlashInfer is unnecessary for this configuration. DA3 installs `xformers==0.0.29.post3` and torch 2.6.0 from the same cu124 index to match their CUDA builds. Scal3R's default loop-disabled path does not require SALAD weights.

SAM 2 attempts to build its optional CUDA postprocessing extension. Its [upstream build](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/INSTALL.md) permits installation when that extension cannot be compiled. Build it with a matching CUDA toolkit if the additional postprocessing operations are required.

## DVLT dependency resolution

```bash
bash scripts/install_gfm.sh dvlt
conda activate unimetric-dvlt
python get_metric_scale.py --config configs/jrdb/dvlt_long.yaml
```

DVLT's [pinned upstream metadata](https://github.com/nv-tlabs/dvlt/blob/f59bb18259ab446a7585c4f47731a77cfdefa426/pyproject.toml) specifies a different dependency set. The installer replaces these seven base pins **only in the DVLT environment**:

| Package | DVLT version |
| --- | --- |
| NumPy | 2.4.4 |
| Open3D | 0.19.0 |
| SciPy | 1.16.0 |
| Matplotlib | 3.10.3 |
| Pillow | 11.2.1 |
| safetensors | 0.5.3 |
| huggingface_hub | 1.16.4 |

The shared chunking class now imports the VGGT model and pose decoder only when VGGT itself runs. DVLT, VGGT-Omega, 4RC, and STream3R therefore reuse these utilities without installing the `vggt` package or inheriting its NumPy < 2 requirement. DVLT is installed in editable mode so its `src/dvlt/` layout is importable.

The installer keeps normal dependency resolution and `pip check` enabled. Do not subsequently install the unmodified `requirements-base.txt` into the DVLT environment: use the installer-generated merged requirements instead.

## Options and reruns

```bash
# List model names, or inspect a plan without any downloads or changes
bash scripts/install_gfm.sh --list
bash scripts/install_gfm.sh dvlt --dry-run

# Install dependencies first, leaving checkpoint downloads for later
bash scripts/install_gfm.sh pi3 --no-weights
conda activate unimetric
python scripts/download_gfm_weights.py pi3
python scripts/check_gfm.py pi3 --weights

# Also load the GFM checkpoint; requires enough free GPU memory
bash scripts/install_gfm.sh pi3 --load-model

# Prepare on a Linux login node without a visible GPU
bash scripts/install_gfm.sh pi3 --skip-cuda-check
# Later, on a GPU node in the same environment:
python scripts/check_gfm.py pi3 --weights
```

The same installer command can resume after an interrupted download or dependency installation. Environments created by this installer for the same repository and dependency group can be reused, including when adding another shared model. For an existing environment created elsewhere, select a new `--env-name` or explicitly allow package updates with `--reuse-env`. A custom `--env-name` accepts models from only one dependency group; `--reuse-env` cannot cross an environment's recorded shared/DVLT group. Existing source at another revision, or with tracked edits, causes an error rather than a reset. Existing nonempty checkpoint files are retained; their contents are not certified by the pinned download revision.

VGGT-Omega weights require approval on the [model page](https://huggingface.co/facebook/VGGT-Omega). Supply `HF_TOKEN` or run `hf auth login` in `unimetric`, then rerun the installation. The script cannot grant access to gated weights.

Installation records are written to the ignored `reproducibility/environments/<env-name>/` directory: merged `requirements.txt`, `install-state.json`, and `pip-freeze.txt` after success. The environment's `.unimetric-environment.json` marker retains model membership, including pending installations, so retries and later additions resolve and check them together. Successful checks are recorded in `install-state.json`. Keep these records with experiment outputs. Source and weight revisions are fixed, but transitive dependencies are resolved by pip and recorded in the freeze; the profile is not a complete transitive lockfile.

## Validation scope

Validation on 2026-10-15: the eight shared models jointly resolved to 167 packages using the exact torch 2.6.0 / torchvision 0.21.0 / xformers cu124 wheel metadata, with no dependency conflicts. All 17 installer and dependency-isolation tests passed, including installing all nine models without arguments or prompts and appending a model without dropping existing backend requirements. Read-only inspection of the server's QWen environment also located the eight shared model modules; it did not run inference. The unchanged DVLT profile previously resolved against the same torch version on PyPI; its exact cu124 metadata endpoint returned an access error. Full installation of the new environments, SAM compilation, and model inference have not been executed on the macOS development host.

The installer performs the checks above on the user's target machine. `--load-model` additionally constructs the selected adapter and loads its weights; it does not evaluate a dataset. A full run still requires prepared MetricBench data, sufficient GPU memory, and VLM API credentials. These installation profiles are not a claim that the historical experiments used the same upstream revisions.

`reproducibility/check_setup.py` checks the entire model/dataset matrix. Use `scripts/check_gfm.py <model>` to check just the environment you installed. Installation logic and dependency isolation can be tested without a GPU using:

```bash
python -m unittest discover -s tests -p 'test_gfm_*.py' -v
```
