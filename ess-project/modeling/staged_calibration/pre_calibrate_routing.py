#!/usr/bin/env python3
"""
pre_calibrate_routing.py — Derive shared two-reservoir routing parameters
from a single uncalibrated SUMMA baseline run.

The output JSON file contains {k_fast, k_slow, f_fast} that all three physics
configurations (noXplict, bigBuckt, qTopmodel) load and apply identically.
Using a shared routing means subsequent KGE/NSE/CoM differences across
physics configs reflect actual subsurface physics — not compensating routing.

Usage
-----
    python pre_calibrate_routing.py --config optimization_config_bigBuckt.yaml \
        --output routing_params_shared.json

What this does
--------------
1. Copies the optimization config's `summa_settings_dir` to a temp run dir.
2. Patches modelDecisions: groundwatr=bigBuckt, bcLowrSoiH=drainage.
3. Patches fileManager.txt to use the configured sim_start/sim_end.
4. Runs SUMMA with the a-priori trialParams.nc (no calibrated values).
5. Reads basin__TotalRunoff (m/s) → multiplies by basin_area_m2 → m³/s.
6. Resamples to daily mean, trims spinup_days from the start.
7. Calls calibrate_reservoirs() against observed streamflow.
8. Saves {k_fast, k_slow, f_fast} + diagnostics to JSON.

Run this **after** the forcing fix is applied (so routing reflects corrected
precip + LW, not biased forcing).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Reuse trial-run plumbing
from summa_runner import (
    run_summa, read_basin_output, setup_trial_run_dir,
    patch_file_manager, patch_model_decisions,
)
from forcing_adjuster import write_adjusted_forcing, patch_forcing_path

# Import linear-reservoir calibrator from utils.custom
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from utils.custom.linear_reservoir import calibrate_reservoirs  # noqa: E402

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("pre_calibrate_routing")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="Optimization config YAML (used for paths and basin_area_m2)")
    p.add_argument("--output", required=True,
                   help="Output JSON path for routing params")
    p.add_argument("--metric", default="balanced",
                   help="Routing-calibration metric (nse | kge | log_nse | log_kge | balanced)")
    p.add_argument("--n-starts", type=int, default=200,
                   help="Number of multistart points for the routing calibrator")
    p.add_argument("--keep-run-dir", action="store_true",
                   help="Don't delete the baseline run directory after completion")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    run_cfg = cfg["run"]

    settings_src = Path(run_cfg["summa_settings_dir"])
    sim_start = run_cfg["sim_start"]
    sim_end = run_cfg["sim_end"]
    basin_area_m2 = float(run_cfg["basin_area_m2"])
    spinup_days = int(run_cfg.get("spinup_days", 365))
    obs_path = Path(
        run_cfg.get("streamflow_obs_path")
        or cfg.get("observations", {}).get("streamflow_obs_path")
        or ""
    )
    if not obs_path.exists():
        logger.error("Streamflow obs not found: %s", obs_path)
        return 2

    # Build a one-off run dir
    run_dir = Path(cfg["results_dir"]) / "_routing_baseline"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # Reuse trial setup helpers — this preserves trialParams.nc (a-priori values)
    settings_dir, output_dir = setup_trial_run_dir(
        base_settings_dir=settings_src,
        base_output_dir=str(run_dir),
        trial_id="routing_baseline",
        # Pass the existing trialParams from base settings — i.e., a-priori
        trial_params_nc=settings_src / run_cfg.get("param_nc_filename", "trialParams.nc"),
        param_nc_filename=run_cfg.get("param_nc_filename", "trialParams.nc"),
    )

    output_prefix = "routing_baseline"
    fm_path = patch_file_manager(
        settings_dir, output_dir,
        sim_start=sim_start,
        sim_end=sim_end,
        out_file_prefix=output_prefix,
        trial_param_filename=run_cfg.get("param_nc_filename", "trialParams.nc"),
    )
    # Force bigBuckt physics for the baseline (most complete subsurface storage)
    patch_model_decisions(settings_dir, {
        "groundwatr": "noXplict",
        "bcLowrSoiH": "drainage",
    })

    # If forcing multipliers are defined at config level (post-forcing-fix),
    # apply them to a baseline forcing dir.  Otherwise the trial reads the
    # base forcing directly via the existing forcingPath in fileManager.txt.
    base_forcing_dir = run_cfg.get("base_forcing_dir")
    if base_forcing_dir:
        baseline_forcing_dir = run_dir / "forcing"
        write_adjusted_forcing(
            base_forcing_dir=Path(base_forcing_dir),
            trial_forcing_dir=baseline_forcing_dir,
            sim_start=sim_start,
            sim_end=sim_end,
            n_hru=int(run_cfg.get("n_hru") or len(run_cfg.get("hru_elevations", {})) or 1),
            hru_precip_multipliers=None,   # baseline = no calibration adjustment
            basin_lw_multiplier=1.0,
        )
        patch_forcing_path(fm_path, baseline_forcing_dir)

    logger.info("Running SUMMA baseline (bigBuckt, a-priori params)…")
    t0 = time.perf_counter()
    ok = run_summa(
        exe_path=run_cfg.get("summa_exe", "summa"),
        file_manager=str(fm_path),
        run_id="routing_baseline",
        timeout_sec=run_cfg.get("timeout_sec", 3600),
    )
    elapsed = time.perf_counter() - t0
    if not ok:
        logger.error("SUMMA baseline run failed after %.1fs", elapsed)
        return 3
    logger.info("SUMMA baseline finished in %.1fs", elapsed)

    # Read basin__TotalRunoff (m/s) and convert to daily m³/s
    sim_q_rate = read_basin_output(
        output_dir, output_prefix, "basin__TotalRunoff",
        spinup_days=spinup_days,
        time_step_hours=float(run_cfg.get("time_step_hours", 1.0)),
    )
    sim_q_cms = sim_q_rate * basin_area_m2
    sim_daily = sim_q_cms.resample("D").mean()

    # Load and align observed streamflow
    obs = pd.read_csv(obs_path, parse_dates=["date"], index_col="date")["value"].astype(float)
    obs_daily = obs.resample("D").mean()

    sim_a, obs_a = sim_daily.align(obs_daily, join="inner")
    n_aligned = int(np.isfinite(sim_a.values).sum() & np.isfinite(obs_a.values).sum())
    logger.info("Aligned: %d daily values (%s → %s)",
                len(sim_a), sim_a.index.min(), sim_a.index.max())

    if len(sim_a) < 365:
        logger.error("Aligned series too short (n=%d) — check sim window vs obs coverage", len(sim_a))
        return 4

    # Calibrate two-reservoir
    logger.info("Calibrating two-reservoir routing (metric=%s, n_starts=%d)…",
                args.metric, args.n_starts)
    result = calibrate_reservoirs(
        q_in=sim_a.values,
        obs=obs_a.values,
        metric=args.metric,
        n_starts=args.n_starts,
        strategy="adaptive",
    )

    # Drop the actual sim array before saving JSON
    payload = {
        "k_fast": float(result["k_fast"]),
        "k_slow": float(result["k_slow"]),
        "f_fast": float(result["f_fast"]),
        "residence_fast_days": float(result["residence_fast_days"]),
        "residence_slow_days": float(result["residence_slow_days"]),
        "calibration_metric": args.metric,
        "calibration_score": float(result.get(args.metric, np.nan)),
        "scores": {k: float(result.get(k, np.nan)) for k in ("nse", "kge", "log_nse", "log_kge", "balanced")},
        "calibration_window": [str(sim_a.index.min()), str(sim_a.index.max())],
        "n_aligned_days": int(len(sim_a)),
        "baseline_run": {
            "config": str(args.config),
            "settings_src": str(settings_src),
            "physics": "bigBuckt + drainage (a-priori parameters)",
            "sim_start": sim_start,
            "sim_end": sim_end,
            "spinup_days": spinup_days,
            "elapsed_summa_seconds": round(elapsed, 1),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved routing params → %s", out_path)
    logger.info("k_fast=%.4f day⁻¹ (%.1fd) | k_slow=%.4f (%.1fd) | f_fast=%.2f",
                payload["k_fast"], payload["residence_fast_days"],
                payload["k_slow"], payload["residence_slow_days"],
                payload["f_fast"])
    logger.info("KGE=%.3f  NSE=%.3f  log-NSE=%.3f",
                payload["scores"]["kge"], payload["scores"]["nse"],
                payload["scores"]["log_nse"])

    if not args.keep_run_dir:
        shutil.rmtree(run_dir, ignore_errors=True)
        logger.info("Cleaned up baseline run dir")

    return 0


if __name__ == "__main__":
    sys.exit(main())
