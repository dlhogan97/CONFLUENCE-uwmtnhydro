#!/usr/bin/env python3
"""Lightweight East River distributed workflow runner for rapid local checks.

This runner is intended for quick functionality tests before a full distributed run.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Any

import yaml

from run_east_river_distributed_workflow import _parse_steps, run_workflow


DEFAULT_LIGHT_STEPS = [
    "setup_project",
    "define_domain",
    "compute_aspect",
    "discretize_domain",
    "process_observed_data",
    "run_model_agnostic_preprocessing",
    "preprocess_models",
]

PRESET_STEPS = {
    "minimal": [
        "setup_project",
        "define_domain",
        "compute_aspect",
        "discretize_domain",
        "preprocess_models",
    ],
    # Purpose-built prep preset for distributed elevation-band runs.
    # This creates HRUs and prepares remapped forcing without running SUMMA.
    "prep_hru": [
        "setup_project",
        "define_domain",
        "compute_aspect",
        "discretize_domain",
        "run_model_agnostic_preprocessing",
        "preprocess_models",
    ],
    "light": list(DEFAULT_LIGHT_STEPS),
    "full": [
        "setup_project",
        "define_domain",
        "compute_aspect",
        "discretize_domain",
        "process_observed_data",
        "run_model_agnostic_preprocessing",
        "preprocess_models",
        "run_models",
    ],
}


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _write_temp_config(config_path: Path, config: Dict[str, Any]) -> Path:
    temp_name = f"{config_path.stem}.light.{os.getpid()}.yaml"
    temp_path = config_path.parent / temp_name
    with temp_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)
    return temp_path


def _resolve_summa_binary(config: Dict[str, Any]) -> Path:
    install_path = Path(str(config.get("SUMMA_INSTALL_PATH", "default"))).expanduser()
    if str(install_path).lower() == "default":
        data_dir = Path(str(config.get("CONFLUENCE_DATA_DIR", ""))).expanduser()
        install_path = data_dir / "installs" / "summa" / "bin"

    parallel_exe = str(config.get("SETTINGS_SUMMA_PARALLEL_EXE", "summa")).strip()
    summa_exe = str(config.get("SUMMA_EXE", "summa")).strip()

    parallel_path = install_path / parallel_exe
    if parallel_path.exists():
        return parallel_path

    fallback_path = install_path / summa_exe
    if fallback_path.exists():
        return fallback_path

    return parallel_path


def _resolve_mpi_launcher(config: Dict[str, Any]) -> str:
    configured = str(config.get("SETTINGS_SUMMA_MPI_LAUNCHER", "auto")).strip()
    if configured and configured.lower() != "auto":
        launcher = shutil.which(configured)
        if launcher:
            return launcher
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return str(configured_path)
        return "<not found>"

    for candidate in ("mpirun", "mpiexec", "srun"):
        launcher = shutil.which(candidate)
        if launcher:
            return launcher
    return "<none detected>"


def _print_parallel_preflight(config: Dict[str, Any]) -> None:
    process_count = int(config.get("MPI_PROCESSES", 1) or 1)
    parallel_mode = str(config.get("SETTINGS_SUMMA_PARALLEL_MODE", "auto")).strip().lower()
    launcher = _resolve_mpi_launcher(config)
    summa_binary = _resolve_summa_binary(config)

    print("=== Lightweight Runner Preflight ===")
    print(f"parallel_mode: {parallel_mode}")
    print(f"mpi_processes: {process_count}")
    print(f"mpi_launcher: {launcher}")
    print(f"summa_binary: {summa_binary}")
    print(f"summa_binary_exists: {summa_binary.exists()}")


def _apply_light_overrides(
    config: Dict[str, Any],
    run_models: bool,
    use_parallel_summa: bool,
    hru_discretization: str,
    elevation_band_size: int | None,
) -> Dict[str, Any]:
    updated = dict(config)

    base_experiment = str(updated.get("EXPERIMENT_ID", "east_river_distributed"))
    if not base_experiment.endswith("_light"):
        updated["EXPERIMENT_ID"] = f"{base_experiment}_light"

    # Keep light runs workstation-friendly and predictable.
    updated["MPI_PROCESSES"] = 12
    updated["EASYMORE_MAX_CORES"] = 12
    updated["EASYMORE_BATCH_SIZE"] = 12
    updated["SETTINGS_SUMMA_PARALLEL_MODE"] = "local"

    # Optional explicit HRU override so prep runs are reproducible from CLI.
    if hru_discretization.strip():
        updated["DOMAIN_DISCRETIZATION"] = hru_discretization.strip()

    # Optional override for elevation band size used when elevation is part of discretization.
    if elevation_band_size is not None:
        updated["ELEVATION_BAND_SIZE"] = int(elevation_band_size)

    if run_models:
        updated["SETTINGS_SUMMA_USE_PARALLEL_SUMMA"] = bool(use_parallel_summa)
    else:
        # Model run is skipped by default in light mode.
        updated["SETTINGS_SUMMA_USE_PARALLEL_SUMMA"] = False

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight East River distributed workflow checks")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to distributed config YAML",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_STEPS.keys()),
        default="light",
        help="Step preset: minimal (fast smoke), prep_hru (build HRUs + forcing), light (default), or full",
    )
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated workflow steps (overrides --preset when provided)",
    )
    parser.add_argument(
        "--reuse-domain",
        default="",
        help="Optional existing domain directory to copy shapefiles/attributes from",
    )
    parser.add_argument(
        "--with-model-run",
        action="store_true",
        help="Include run_models in the lightweight execution",
    )
    parser.add_argument(
        "--with-parallel-summa",
        action="store_true",
        help="If --with-model-run is set, use local MPI SUMMA execution",
    )
    parser.add_argument(
        "--hru-discretization",
        default="",
        help=(
            "Override DOMAIN_DISCRETIZATION in the temp config. "
            "Example: elevation,soilclass,landclass"
        ),
    )
    parser.add_argument(
        "--elevation-band-size",
        type=int,
        default=None,
        help="Optional override for ELEVATION_BAND_SIZE in meters",
    )
    parser.add_argument(
        "--export-hru-attributes",
        action="store_true",
        help="Export HRU diagnostics CSVs after workflow completion",
    )
    parser.add_argument(
        "--export-elevation-bands",
        default="",
        help="Optional elevation band edges for export CSV, e.g. 2500,2800,3100,3400",
    )
    parser.add_argument(
        "--export-aspect-classes",
        action="store_true",
        help="Include cardinal aspect classes in exported HRU diagnostics",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Preserve config discretization unless explicit CLI override is provided.
    hru_discretization = args.hru_discretization

    config = _load_config(config_path)
    updated_config = _apply_light_overrides(
        config,
        run_models=bool(args.with_model_run),
        use_parallel_summa=bool(args.with_parallel_summa),
        hru_discretization=hru_discretization,
        elevation_band_size=args.elevation_band_size,
    )

    _print_parallel_preflight(updated_config)

    temp_config_path = _write_temp_config(config_path, updated_config)
    reuse_domain = Path(args.reuse_domain).expanduser().resolve() if args.reuse_domain else None

    try:
        if args.steps.strip():
            steps = _parse_steps(args.steps)
        else:
            steps = list(PRESET_STEPS[args.preset])

        if args.with_model_run and "run_models" not in steps:
            steps.append("run_models")
        if not args.with_model_run and "run_models" in steps:
            steps = [step for step in steps if step != "run_models"]

        run_workflow(temp_config_path, steps, reuse_domain)

        # Export a simple HRU diagnostics table so debugging can happen with plain CSVs.
        do_export = bool(args.export_hru_attributes) or args.preset == "prep_hru"
        if do_export:
            # Import only when needed so core prep can still run in lighter environments.
            try:
                from export_hru_attributes import export_hru_tables

                full_csv, summary_csv = export_hru_tables(
                    config_path=temp_config_path,
                    variables="",
                    elevation_bands=args.export_elevation_bands,
                    aspect_classes=bool(args.export_aspect_classes),
                    outdir="",
                )
                print("=== HRU diagnostics export ===")
                print(f"full_csv: {full_csv}")
                print(f"summary_csv: {summary_csv}")
            except ModuleNotFoundError as exc:
                print(
                    "WARNING: Skipping HRU diagnostics export because optional dependencies "
                    f"are missing ({exc})."
                )
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()


if __name__ == "__main__":
    main()
