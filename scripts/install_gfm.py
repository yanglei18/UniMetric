#!/usr/bin/env python3
"""Install all nine GFMs with default settings into two Conda environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES = Path(__file__).with_name("gfm_profiles.json")


def load_profiles() -> dict[str, Any]:
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    for model, profile in profiles["models"].items():
        profiles["models"][model] = {**profiles["environments"][profile["group"]], **profile}
    return profiles


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def select_model(name: str, profiles: dict[str, Any]) -> str:
    for key, profile in profiles["models"].items():
        names = [key, profile["config_key"], *profile.get("aliases", [])]
        if canonical_name(name) in {canonical_name(item) for item in names}:
            return key
    raise ValueError(f"Unknown model {name!r}. Choose: {', '.join(profiles['models'])}")


def select_models(names: list[str], profiles: dict[str, Any]) -> list[str]:
    selected = []
    for name in names:
        if canonical_name(name) in {"common", "all"}:
            selected.extend(key for key, profile in profiles["models"].items()
                            if canonical_name(name) == "all" or profile["group"] == "common")
        else:
            selected.append(select_model(name, profiles))
    return list(dict.fromkeys(selected))


def merged_requirements(profile: dict[str, Any]) -> list[str]:
    """Replace base pins for this backend instead of installing conflicting sets."""
    requirements: dict[str, str] = {}
    for line in (ROOT / "requirements-base.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==", 1)
            requirements[canonical_name(name)] = f"{name}=={version}"
    for name, version in profile.get("base_overrides", {}).items():
        requirements[canonical_name(name)] = f"{name}=={version}"
    for requirement in profile.get("extra_requirements", []) + profile.get("cuda_extra_requirements", []):
        name, _ = requirement.split("==", 1)
        requirements[canonical_name(name)] = requirement
    requirements["torch"] = f"torch=={profile['torch']}"
    requirements["torchvision"] = f"torchvision=={profile['torchvision']}"
    return list(requirements.values())


def group_requirements(models: list[str], profiles: dict[str, Any]) -> list[str]:
    """Keep constraints for every model already added to the shared environment."""
    requirements: dict[str, str] = {}
    for model in models:
        for requirement in merged_requirements(profiles["models"][model]):
            name, _ = requirement.split("==", 1)
            name = canonical_name(name)
            if name in requirements and requirements[name] != requirement:
                raise ValueError(f"Conflicting shared environment requirements: {requirements[name]}, {requirement}")
            requirements[name] = requirement
    return list(requirements.values())


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def ensure_source(source: dict[str, str]) -> Path:
    """Never reset or overwrite an existing checkout, including a user's edits."""
    target = ROOT / source["directory"]
    revision = source["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Source revision must be a full Git SHA: {source['directory']}")
    if target.exists() or target.is_symlink():
        if not (target / ".git").exists():
            raise RuntimeError(f"{target} already exists and is not a Git checkout. Preserve it and move it aside before installing.")
        root = Path(git_output(target, "rev-parse", "--show-toplevel")).resolve()
        if root != target.resolve():
            raise RuntimeError(f"{target} is not a standalone checkout.")
        if git_output(target, "rev-parse", "HEAD") != revision:
            raise RuntimeError(f"{target} is not at required revision {revision}. Existing source was left unchanged.")
        if git_output(target, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"{target} has local changes. Existing source was left unchanged.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-install-", dir=target.parent) as staging:
        checkout = Path(staging) / "source"
        run(["git", "init", str(checkout)])
        run(["git", "-C", str(checkout), "remote", "add", "origin", source["url"]])
        run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", revision])
        run(["git", "-C", str(checkout), "checkout", "--detach", revision])
        checkout.rename(target)
    return target


def environment_prefix(conda: str, name: str) -> Path | None:
    info = json.loads(subprocess.check_output([conda, "env", "list", "--json"], text=True))
    for entry in info["envs"]:
        if Path(entry).name == name:
            return Path(entry)
    return None


def prepare_environment(conda: str, name: str, models: list[str], profiles: dict[str, Any],
                        reuse: bool) -> tuple[Path, list[str]]:
    prefix = environment_prefix(conda, name)
    profile = profiles["models"][models[0]]
    group = profile["group"]
    members = list(models)
    if prefix is None:
        run([conda, "create", "-y", "-n", name, "--override-channels", "-c", "conda-forge",
             f"python={profile['python']}", "pip"])
        prefix = environment_prefix(conda, name)
        if prefix is None:
            raise RuntimeError(f"Conda did not create environment {name}.")
    else:
        marker = prefix / ".unimetric-environment.json"
        previous = json.loads(marker.read_text()) if marker.is_file() else None
        if previous:
            # Accept markers from the earlier per-model installer as well.
            recorded = previous.get("models", [previous["model"]] if "model" in previous else [])
            if any(model not in profiles["models"] for model in recorded):
                raise RuntimeError(f"Environment {name} contains unknown model records; use a new --env-name.")
            previous_group = previous.get("group")
            if (previous_group and previous_group != group) or any(
                profiles["models"][model]["group"] != group for model in recorded
            ):
                raise RuntimeError("Common models and DVLT must use separate environments, even with --reuse-env.")
            members = list(dict.fromkeys([*recorded, *models]))
        if (not previous or previous.get("repository") != str(ROOT)) and not reuse:
            raise RuntimeError(f"Environment {name} already exists. Use a new --env-name, or --reuse-env to explicitly update its packages.")
    python = prefix / "bin" / "python"
    version = subprocess.check_output([str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
    if version != profile["python"]:
        raise RuntimeError(f"{name} uses Python {version}; this profile requires Python {profile['python']}.")
    # Membership includes pending installations so a failed attempt can be retried
    # without dropping dependencies that pip may already have installed.
    owner = {"repository": str(ROOT), "group": group, "models": members}
    (prefix / ".unimetric-environment.json").write_text(json.dumps(owner, indent=2) + "\n")
    return python, members


def print_plan(models: list[str], profiles: dict[str, Any], args: argparse.Namespace) -> None:
    profile = profiles["models"][models[0]]
    print(json.dumps({
        "models": models, "group": profile["group"], "environment": args.env_name or profile["environment"],
        "python": profile["python"], "torch_index": f"https://download.pytorch.org/whl/{profile['cuda']}",
        "requirements": group_requirements(models, profiles),
        "sources": [profiles["sam2"], *[profiles["models"][model]["source"] for model in models]],
        "upstream_requirements": [model for model in models if profiles["models"][model].get("install_requirements")],
        "weights": [profiles["models"][model]["weights"] for model in models] if not args.no_weights else "skipped",
        "sam_checkpoint": "checkpoints/sam2.1_hiera_large.pt" if not args.no_weights else "skipped",
        "reuse": "Also retain and recheck models recorded in this environment when installing",
        "checks": ["pip check", "runtime imports", "NumPy/Torch/Open3D interoperability"]
            + ([] if args.skip_cuda_check else ["CUDA tensor operation"])
            + (["model checkpoint load"] if args.load_model else []),
    }, indent=2))


def install_group(selected: list[str], profiles: dict[str, Any], args: argparse.Namespace) -> None:
    profile = profiles["models"][selected[0]]
    name = args.env_name or profile["environment"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("--env-name must be a Conda name, not a path or shell expression.")
    if args.dry_run:
        print_plan(selected, profiles, args)
        return
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("Installation targets Linux x86_64 with NVIDIA CUDA. Use --dry-run to inspect on another platform.")
    libc, libc_version = platform.libc_ver()
    if libc != "glibc" or tuple(int(part) for part in libc_version.split(".")[:2]) < (2, 31):
        raise RuntimeError("The pinned Linux wheels require glibc >= 2.31 (for example Ubuntu 20.04 or newer).")
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if not conda or not shutil.which("git"):
        raise RuntimeError("Install Conda/Miniforge and Git first, then rerun this command.")
    if not args.skip_cuda_check:
        run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    python, models = prepare_environment(conda, name, selected, profiles, args.reuse_env)
    sources = {model: ensure_source(profiles["models"][model]["source"]) for model in models}
    sam_source = ensure_source(profiles["sam2"])
    state_dir = ROOT / "reproducibility" / "environments" / name
    state_dir.mkdir(parents=True, exist_ok=True)
    requirements = state_dir / "requirements.txt"
    requirements.write_text("\n".join(group_requirements(models, profiles)) + "\n")
    state = {"selected_models": selected, "models": models, "group": profile["group"],
             "environment": name, "status": "installing", "profiles": {model: profiles["models"][model] for model in models},
             "sam2": profiles["sam2"], "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest()}
    receipt = state_dir / "install-state.json"
    receipt.write_text(json.dumps(state, indent=2) + "\n")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
               CONDA_PREFIX=str(python.parent.parent), CONDA_DEFAULT_ENV=name)
    env.pop("PYTHONPATH", None)
    pip = [str(python), "-m", "pip"]
    try:
        run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"], env=env)
        cuda_extras = list(dict.fromkeys(requirement for model in models
                                        for requirement in profiles["models"][model].get("cuda_extra_requirements", [])))
        run(pip + ["install", f"torch=={profile['torch']}", f"torchvision=={profile['torchvision']}"]
            + cuda_extras
            + ["--index-url", f"https://download.pytorch.org/whl/{profile['cuda']}"], env=env)
        # Solve base and backend requirements together; a conflict is an error.
        command = pip + ["install", "-c", str(requirements), "-r", str(requirements)]
        for model, source in sources.items():
            command += ["-e", str(source)]
            if profiles["models"][model].get("install_requirements"):
                command += ["-r", str(source / "requirements.txt")]
        run(command, env=env)
        # A regular install copies SAM's build into this environment; CUDA builds
        # from distinct torch environments must not share an editable extension.
        run(pip + ["install", "--no-build-isolation", "-c", str(requirements), str(sam_source)], env=env)
        run(pip + ["check"], env=env)
        for model in models:
            # Check every registered backend in a fresh process, but download or
            # load weights only for the selection in this command.
            if model in selected and not args.no_weights:
                run([str(python), str(ROOT / "scripts" / "download_gfm_weights.py"), model], env=env)
            check = [str(python), str(ROOT / "scripts" / "check_gfm.py"), model]
            if args.skip_cuda_check:
                check.append("--skip-cuda-check")
            if model in selected and not args.no_weights:
                check.append("--weights")
            if model in selected and args.load_model:
                check.append("--load-model")
            run(check, env=env)
        freeze = subprocess.check_output(pip + ["freeze"], text=True, env=env)
        (state_dir / "pip-freeze.txt").write_text(freeze)
        state.update(status="installed", weights_checked=selected if not args.no_weights else [],
                     cuda_checked=not args.skip_cuda_check, models_loaded=selected if args.load_model else [])
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        receipt.write_text(json.dumps(state, indent=2) + "\n")
    print(f"\nInstalled {', '.join(selected)} in {name}. Available models: {', '.join(models)}. Environment record: {receipt}")
    print(f"conda activate {name}")
    print(f"cd {shlex.quote(str(ROOT))}")
    for model in selected:
        print(f"python get_metric_scale.py --config configs/jrdb/{profiles['models'][model]['config_key']}.yaml")
    print("Prepare MetricBench and VLM API credentials before running. Installation checks do not reproduce benchmark results.")


def install(models: str | list[str], profiles: dict[str, Any], args: argparse.Namespace) -> None:
    if isinstance(models, str):
        models = [models]
    groups: dict[str, list[str]] = {}
    for model in models:
        groups.setdefault(profiles["models"][model]["group"], []).append(model)
    if args.env_name and len(groups) > 1:
        raise ValueError("--env-name can only be used with models from one environment group.")
    for selected in groups.values():
        install_group(selected, profiles, args)


def main() -> int:
    profiles = load_profiles()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Optional subset: model names, YAML keys, or common; default: all nine models")
    parser.add_argument("--list", action="store_true", help="List supported model names")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloads or changes")
    parser.add_argument("--env-name", help="Override the Conda name for a single environment group")
    parser.add_argument("--reuse-env", action="store_true", help="Allow package updates in an existing environment")
    parser.add_argument("--no-weights", action="store_true", help="Install the environment without downloading checkpoints")
    parser.add_argument("--skip-cuda-check", action="store_true", help="Prepare on a Linux host without a visible GPU; skip the CUDA check")
    parser.add_argument("--load-model", action="store_true", help="Also load the selected model checkpoint; requires GPU memory")
    args = parser.parse_args()
    if args.list:
        for key, item in profiles["models"].items():
            print(f"{key:12} {item['config_key']:20} {item['environment']}")
        print("common       All eight common models in unimetric")
        print("all          All nine models in two environments (default)")
        return 0
    if args.load_model and (args.no_weights or args.skip_cuda_check):
        parser.error("--load-model requires weights and the CUDA check")
    default_install = not args.models
    if default_install:
        args.models = ["all"]
    try:
        install(select_models(args.models, profiles), profiles, args)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        return 1
    if default_install and not args.dry_run:
        print("\nAll nine GFMs installed. After preparing MetricBench and VLM API credentials, run the default pipeline:")
        print(f"conda activate {profiles['environments']['common']['environment']}")
        print(f"cd {shlex.quote(str(ROOT))}")
        print("python get_metric_scale.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
