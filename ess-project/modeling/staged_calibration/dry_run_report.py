#!/usr/bin/env python3
"""Generate a dry-run report for staged SUMMA optimization configs.

This does not run SUMMA. It reports:
- selected stage order
- HRU elevation + MODIS class mapping
- base parameter values from trialParams
- per-stage multiplier bounds
- per-HRU effective values at M=1 when spatial weights are present
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from parameter_manager import ParameterManager
from staged_optimizer import StageConfig, _build_spatial_weight_arrays


MODIS_NAMES: Dict[int, str] = {
    1: "evergreen needleleaf forest",
    7: "open shrublands",
    16: "barren/sparsely vegetated",
}


def build_report(config_path: Path) -> str:
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    run_cfg = cfg["run"]
    pm = ParameterManager(run_cfg["base_trial_params_nc"])

    calib_init = run_cfg.get("base_calib_bounds_json")
    if calib_init:
        pm.apply_calib_bounds_values(calib_init)

    hru_lc = {int(k): int(v) for k, v in run_cfg.get("hru_landcover", {}).items()}
    hru_elev = {int(k): float(v) for k, v in run_cfg.get("hru_elevations", {}).items()}
    n_hru = len(hru_lc)

    lines = []
    lines.append("DRY-RUN REPORT: staged optimization")
    lines.append(f"config: {config_path.resolve()}")
    lines.append(f"summa_settings_dir: {run_cfg.get('summa_settings_dir', '')}")
    lines.append(f"base_trial_params_nc: {run_cfg.get('base_trial_params_nc', '')}")
    lines.append(
        f"output_prefix: {run_cfg.get('output_prefix', '')}; "
        f"sim window: {run_cfg.get('sim_start', '')} -> {run_cfg.get('sim_end', '')}"
    )
    lines.append("")

    lines.append("HRU metadata (index, elevation_m, MODIS_code, MODIS_name):")
    for i in range(n_hru):
        code = hru_lc.get(i)
        elev = hru_elev.get(i, float("nan"))
        lines.append(
            f"  hru={i}: elev={elev:.1f}, code={code}, class={MODIS_NAMES.get(code, 'unknown')}"
        )

    stage_files = cfg.get("stages", [])
    lines.append("")
    lines.append(f"stages ({len(stage_files)}):")
    for sf in stage_files:
        lines.append(f"  - {sf}")

    base_ds = pm.base_dataset
    data_vars = list(base_ds.data_vars)
    lines.append("")
    lines.append(f"base_trial_params variables ({len(data_vars)}): {data_vars}")
    non_id_vars = [v for v in data_vars if v != "hruId"]
    if not non_id_vars:
        lines.append("WARNING: No calibratable parameter variables found in base_trial_params_nc.")

    for sf in stage_files:
        sc = StageConfig.from_yaml(sf)
        lines.append("")
        lines.append(f"[{sc.name}] {sc.description}")
        lines.append(f"  params: {', '.join(sc.params)}")
        lines.append(f"  bounds: {sc.multiplier_search_bounds}")

        stage_sw = {}
        if sc.spatial_weights:
            stage_sw = _build_spatial_weight_arrays(sc.spatial_weights, hru_lc, n_hru)

        for p in sc.params:
            if p not in base_ds:
                lines.append(f"  - {p}: NOT FOUND in base trial params")
                continue

            arr = np.asarray(base_ds[p].values)
            dims = base_ds[p].dims

            if "hru" in dims and arr.ndim == 1:
                lines.append(f"  - {p} (hru): base={np.round(arr, 6).tolist()}")
                if p in stage_sw:
                    eff = arr * stage_sw[p]  # M = 1.0
                    lines.append(f"    spatial_weights={np.round(stage_sw[p], 4).tolist()}")
                    lines.append(f"    effective_at_M1={np.round(eff, 6).tolist()}")
            elif "gru" in dims and arr.ndim == 1:
                lines.append(f"  - {p} (gru): base={np.round(arr, 6).tolist()}")
            else:
                lines.append(f"  - {p} ({dims}): shape={arr.shape}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a dry-run report for staged optimization configs")
    parser.add_argument("--config", required=True, help="Path to explicit optimization config YAML")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path. Defaults to <results_dir>/dry_run_report_<config_stem>.txt",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    report = build_report(config_path)

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(cfg["results_dir"]) / f"dry_run_report_{config_path.stem}.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote dry-run report: {out_path}")


if __name__ == "__main__":
    main()
