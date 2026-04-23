#!/usr/bin/env python3
"""Run an East River distributed CONFLUENCE workflow from a YAML config.

This script mirrors the notebook flow but is deterministic and CLI-friendly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import yaml

# Add repo root explicitly so this script works when run outside the repository cwd.
def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "CONFLUENCE.py").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _bootstrap_runtime_env() -> None:
    """Set runtime paths required by geospatial and R-backed dependencies.

    This keeps nohup/non-interactive runs robust even when shell init scripts
    do not export all expected variables.
    """
    runtime_prefixes: List[Path] = []

    env_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if env_prefix:
        runtime_prefixes.append(Path(env_prefix))

    # Fallback when Python is launched by absolute path without conda activate,
    # or when CONDA_PREFIX points to a different environment than sys.executable.
    exe_prefix = Path(sys.executable).resolve().parents[1]
    if exe_prefix not in runtime_prefixes:
        runtime_prefixes.append(exe_prefix)

    if not os.environ.get("PROJ_LIB"):
        for prefix in runtime_prefixes:
            proj_lib = prefix / "share" / "proj"
            if (proj_lib / "proj.db").exists():
                os.environ["PROJ_LIB"] = str(proj_lib)
                break

    if not os.environ.get("R_HOME"):
        for prefix in runtime_prefixes:
            bundled_r_home = prefix / "lib" / "R"
            if bundled_r_home.exists():
                os.environ["R_HOME"] = str(bundled_r_home)
                break

    if not os.environ.get("R_HOME"):
        r_binary = shutil.which("R")
        if r_binary:
            try:
                r_home = subprocess.check_output(
                    [r_binary, "RHOME"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                r_home = ""

            if r_home:
                os.environ["R_HOME"] = r_home


_bootstrap_runtime_env()

from CONFLUENCE import CONFLUENCE


DEFAULT_STEPS = [
    "setup_project",
    "define_domain",
    "compute_aspect",
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
    step_list = list(steps)

    project_dir = None
    if "setup_project" in step_list:
        project_dir = confluence.managers["project"].setup_project()
        confluence.managers["project"].create_pour_point()

    if reuse_domain and project_dir:
        # Reuse prior domain assets to speed up experimentation.
        _copy_tree_contents(reuse_domain / "shapefiles", project_dir / "shapefiles")
        _copy_tree_contents(reuse_domain / "attributes", project_dir / "attributes")

    if "define_domain" in step_list and not reuse_domain:
        confluence.managers["domain"].define_domain()

    if "compute_aspect" in step_list:
        from utils.geospatial.discretization_utils import DomainDiscretizer
        domain_mgr = confluence.managers["domain"]
        if domain_mgr.domain_discretizer is None:
            domain_mgr.domain_discretizer = DomainDiscretizer(domain_mgr.config, domain_mgr.logger)
        aspect_path = domain_mgr.domain_discretizer.compute_aspect_raster()
        if aspect_path is None:
            raise RuntimeError("compute_aspect step failed — aspect raster could not be created.")

    if "discretize_domain" in step_list:
        confluence.managers["domain"].discretize_domain()

    if "process_observed_data" in step_list:
        confluence.managers["data"].process_observed_data()

    if "run_model_agnostic_preprocessing" in step_list:
        confluence.managers["data"].run_model_agnostic_preprocessing()

    if "preprocess_models" in step_list:
        confluence.managers["model"].preprocess_models()

    if "run_models" in step_list:
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
