#!/usr/bin/env python3
"""Run an East River distributed CONFLUENCE workflow from a YAML config.

This script mirrors the notebook flow but is deterministic and CLI-friendly.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import yaml

from CONFLUENCE import CONFLUENCE


DEFAULT_STEPS = [
    "setup_project",
    "discretize_domain",
    "process_observed_data",
    "run_model_agnostic_preprocessing",
    "preprocess_models",
    "run_models",
]


def _copy_tree_contents(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for src in source.glob("**/*"):
        if src.is_file():
            rel = src.relative_to(source)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _parse_steps(raw: str | None) -> List[str]:
    if not raw:
        return list(DEFAULT_STEPS)
    return [token.strip() for token in raw.split(",") if token.strip()]


def run_workflow(config_path: Path, steps: Iterable[str], reuse_domain: Path | None) -> None:
    confluence = CONFLUENCE(config_path)

    project_dir = None
    if "setup_project" in steps:
        project_dir = confluence.managers["project"].setup_project()
        confluence.managers["project"].create_pour_point()

    if reuse_domain and project_dir:
        # Reuse prior domain assets to speed up experimentation.
        _copy_tree_contents(reuse_domain / "shapefiles", project_dir / "shapefiles")
        _copy_tree_contents(reuse_domain / "attributes", project_dir / "attributes")

    if "discretize_domain" in steps:
        confluence.managers["domain"].discretize_domain()

    if "process_observed_data" in steps:
        confluence.managers["data"].process_observed_data()

    if "run_model_agnostic_preprocessing" in steps:
        confluence.managers["data"].run_model_agnostic_preprocessing()

    if "preprocess_models" in steps:
        confluence.managers["model"].preprocess_models()

    if "run_models" in steps:
        confluence.managers["model"].run_models()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run East River distributed workflow")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML (for example ess-project/0_config_files/config_East_River_distributed_seasonal_bigBuckt.yaml)",
    )
    parser.add_argument(
        "--steps",
        default=",".join(DEFAULT_STEPS),
        help="Comma-separated steps to execute",
    )
    parser.add_argument(
        "--reuse-domain",
        default="",
        help="Optional existing domain directory to copy shapefiles/attributes from",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    if config.get("FORCING_PATH") is None:
        raise ValueError("FORCING_PATH must be set in the config")

    reuse_domain = Path(args.reuse_domain).expanduser().resolve() if args.reuse_domain else None
    run_workflow(config_path, _parse_steps(args.steps), reuse_domain)


if __name__ == "__main__":
    main()
