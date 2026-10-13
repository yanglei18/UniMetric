#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import time
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Metric-Depth twice with different consensus interval solvers and compare the outputs."
    )
    parser.add_argument(
        "--base-config",
        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
        help="Base YAML config used to derive the two temporary run configs.",
    )
    parser.add_argument(
        "--solver-a",
        default="bi",
        choices=["legacy", "bi", "ifpa"],
        help="Consensus interval solver for run A.",
    )
    parser.add_argument(
        "--solver-b",
        default="ifpa",
        choices=["legacy", "bi", "ifpa"],
        help="Consensus interval solver for run B.",
    )
    parser.add_argument("--label-a", default="bi", help="Display label for run A.")
    parser.add_argument("--label-b", default="ifpa", help="Display label for run B.")
    parser.add_argument("--work-root", default=None, help="Optional working root for generated configs, logs, outputs, and reports.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to run the two pipelines and the comparison script.")
    parser.add_argument("--keep-temp-configs", action="store_true", help="Keep generated temporary configs after completion.")
    return parser.parse_args()


def load_yaml(file_path: str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {file_path}")
    return data


def write_yaml(file_path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_work_root(args: argparse.Namespace) -> str:
    if args.work_root:
        return os.path.abspath(args.work_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "temp", "consensus_runs", timestamp))


def derive_output_root(work_root: str, label: str) -> str:
    return os.path.join(work_root, "outputs", label)


def deepcopy_via_yaml(payload: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def derive_config(
    base_config: dict[str, Any],
    solver: str,
    label: str,
    output_root: str,
) -> dict[str, Any]:
    config_copy = deepcopy_via_yaml(base_config)
    semantic_scale = config_copy.setdefault("semantic_scale", {})
    paths_cfg = config_copy.setdefault("paths", {})

    semantic_scale["consensus_interval_solver"] = solver
    paths_cfg["output_root"] = output_root
    config_copy.setdefault("compare_run", {})
    config_copy["compare_run"]["label"] = label
    return config_copy


def stream_command(cmd: list[str], cwd: str, log_path: str, prefix: str) -> int:
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            decorated = f"[{prefix}] {line}"
            print(decorated, end="")
            log_file.write(decorated)
        return process.wait()


def run_metric_depth(python_bin: str, repo_root: str, config_path: str, label: str, log_dir: str) -> str:
    log_path = os.path.join(log_dir, f"{label}.log")
    cmd = [python_bin, "get_metric_scale.py", "--config", config_path]
    exit_code = stream_command(cmd, cwd=repo_root, log_path=log_path, prefix=label)
    if exit_code != 0:
        raise RuntimeError(f"Run {label} failed with exit code {exit_code}. See log: {log_path}")
    return log_path


def run_comparison(
    python_bin: str,
    repo_root: str,
    run_root_a: str,
    run_root_b: str,
    label_a: str,
    label_b: str,
    report_dir: str,
) -> tuple[str, str]:
    output_json = os.path.join(report_dir, f"{label_a}_vs_{label_b}.json")
    output_md = os.path.join(report_dir, f"{label_a}_vs_{label_b}.md")
    cmd = [
        python_bin,
        "compare_consensus_runs.py",
        "--run-a",
        run_root_a,
        "--run-b",
        run_root_b,
        "--label-a",
        label_a,
        "--label-b",
        label_b,
        "--output-json",
        output_json,
        "--output-md",
        output_md,
    ]
    exit_code = stream_command(
        cmd,
        cwd=repo_root,
        log_path=os.path.join(report_dir, f"{label_a}_vs_{label_b}.compare.log"),
        prefix="compare",
    )
    if exit_code != 0:
        raise RuntimeError(f"Comparison failed with exit code {exit_code}.")
    return output_json, output_md


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    work_root = build_work_root(args)
    config_dir = ensure_dir(os.path.join(work_root, "configs"))
    log_dir = ensure_dir(os.path.join(work_root, "logs"))
    report_dir = ensure_dir(os.path.join(work_root, "reports"))

    base_config = load_yaml(os.path.abspath(args.base_config))

    output_root_a = derive_output_root(work_root, args.label_a)
    output_root_b = derive_output_root(work_root, args.label_b)

    config_a = derive_config(base_config, args.solver_a, args.label_a, output_root_a)
    config_b = derive_config(base_config, args.solver_b, args.label_b, output_root_b)

    config_path_a = os.path.join(config_dir, f"{args.label_a}.yaml")
    config_path_b = os.path.join(config_dir, f"{args.label_b}.yaml")
    write_yaml(config_path_a, config_a)
    write_yaml(config_path_b, config_b)

    print(f"Work root: {work_root}")
    print(f"Run A output root: {output_root_a}")
    print(f"Run B output root: {output_root_b}")

    log_a = ""
    log_b = ""
    report_json = ""
    report_md = ""
    try:
        log_a = run_metric_depth(args.python_bin, repo_root, config_path_a, args.label_a, log_dir)
        log_b = run_metric_depth(args.python_bin, repo_root, config_path_b, args.label_b, log_dir)
        report_json, report_md = run_comparison(
            args.python_bin,
            repo_root,
            output_root_a,
            output_root_b,
            args.label_a,
            args.label_b,
            report_dir,
        )
    finally:
        if not args.keep_temp_configs:
            for file_path in (config_path_a, config_path_b):
                if os.path.exists(file_path):
                    os.remove(file_path)

    print("")
    print("Dual-run comparison finished.")
    print(f"Run A log: {log_a}")
    print(f"Run B log: {log_b}")
    print(f"Comparison JSON: {report_json}")
    print(f"Comparison Markdown: {report_md}")


if __name__ == "__main__":
    main()
