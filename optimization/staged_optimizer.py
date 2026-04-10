#!/usr/bin/env python3
"""
staged_optimizer.py — Staged, process-based calibration for distributed SUMMA.

Philosophy
----------
Simultaneous global optimization of all parameters leads to compensating errors
(equifinality).  By isolating process groups and constraining each with the most
relevant observation type, we prevent the optimizer from trading errors between
snow, soil, groundwater, and routing (Clark et al. 2015; Beven 2006).

Stages (sequential, each freezes its parameters before the next begins):
  1. Snow / radiation  → constrained by SNOTEL SWE
  2. Soil hydraulics + ET → constrained by OpenET / rising-limb Q
  3. Groundwater / baseflow → constrained by separated baseflow from observed Q
  4. Routing → constrained by outlet streamflow

Each stage optimizes *multipliers* on the per-HRU base parameters, not raw values.
This preserves spatial heterogeneity from the a-priori data.

Usage
-----
    python staged_optimizer.py --config optimization/optimization_config.yaml

Or from Python:
    from staged_optimizer import run_all_stages
    run_all_stages("optimization/optimization_config.yaml")
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import differential_evolution

from parameter_manager import ParameterManager, MULTIPLIER_BOUNDS
from summa_runner import run_summa, read_hru_output, read_basin_output, setup_trial_run_dir, patch_file_manager
from objective_functions import (
    compute_anchor_snow,
    compute_anchor_et,
    compute_anchor_streamflow_rising,
    compute_anchor_baseflow,
    compute_anchor_streamflow,
    compute_coherence_snow,
    compute_coherence_et,
    compute_coherence_baseflow,
    combined_objective,
    eckhardt_baseflow,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("staged_optimizer")


# ---------------------------------------------------------------------------
# Stage configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    name: str                           # e.g. "stage1_snow"
    params: List[str]                   # multiplier parameter names for this stage
    anchor_variable: str                # SUMMA output variable for anchor
    anchor_obs_path: str                # path to observations CSV
    coherence_type: str                 # "snow" | "et" | "baseflow" | "streamflow"
    w_anchor: float = 0.4               # weight given to anchor vs coherence
    multiplier_search_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    max_iterations: int = 150
    popsize: int = 10                   # differential_evolution population multiplier
    tol: float = 0.01
    spinup_days: int = 365
    time_step_hours: float = 1.0
    description: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StageConfig":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        # Convert multiplier_search_bounds list-of-2 → tuple
        raw_bounds = data.get("multiplier_search_bounds", {})
        parsed_bounds = {k: tuple(v) for k, v in raw_bounds.items()}
        return cls(
            name=data["name"],
            params=data["params"],
            anchor_variable=data["anchor_variable"],
            anchor_obs_path=data["anchor_obs_path"],
            coherence_type=data["coherence_type"],
            w_anchor=data.get("w_anchor", 0.4),
            multiplier_search_bounds=parsed_bounds,
            max_iterations=data.get("max_iterations", 150),
            popsize=data.get("popsize", 10),
            tol=data.get("tol", 0.01),
            spinup_days=data.get("spinup_days", 365),
            time_step_hours=data.get("time_step_hours", 1.0),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Observation loaders
# ---------------------------------------------------------------------------

def load_obs_timeseries(csv_path: str, value_col: str = "value") -> pd.Series:
    """Load a CSV with columns [date, value] → pd.Series with DatetimeIndex."""
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    return df[value_col].astype(float).sort_index()


def load_obs_monthly(csv_path: str, value_col: str = "value") -> pd.Series:
    """Load monthly obs CSV → pd.Series with DatetimeIndex (month-start)."""
    return load_obs_timeseries(csv_path, value_col)


# ---------------------------------------------------------------------------
# Trial evaluator
# ---------------------------------------------------------------------------

class TrialEvaluator:
    """Wraps a single trial function (parameter vector → scalar cost) for one stage."""

    def __init__(
        self,
        stage_cfg: StageConfig,
        param_manager: ParameterManager,
        run_cfg: Dict[str, Any],
        log_path: Path,
    ):
        self.stage_cfg = stage_cfg
        self.pm = param_manager
        self.run_cfg = run_cfg
        self.log_path = log_path
        self._trial_counter = 0
        self._obs = self._load_observations()

        # CSV log header
        with open(self.log_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            header = (
                ["trial_id", "stage"]
                + [f"mult_{p}" for p in stage_cfg.params]
                + ["J_anchor", "J_coherence", "J_total", "runtime_sec", "converged"]
            )
            writer.writerow(header)

    def _load_observations(self) -> pd.Series:
        obs_path = self.stage_cfg.anchor_obs_path
        if not obs_path or not Path(obs_path).exists():
            logger.warning("Observation file not found: %s — anchor cost will be 0.5", obs_path)
            return pd.Series(dtype=float)
        if self.stage_cfg.coherence_type in ("et",):
            return load_obs_monthly(obs_path)
        return load_obs_timeseries(obs_path)

    def __call__(self, multiplier_vector: np.ndarray) -> float:
        """Evaluate a trial parameter set. Returns scalar cost."""
        self._trial_counter += 1
        # Use PID + counter so parallel worker processes never collide on trial dirs
        trial_id = f"trial_p{os.getpid()}_{self._trial_counter:04d}"
        t0 = time.perf_counter()

        # Build multiplier dict
        multipliers = {
            p: float(multiplier_vector[i])
            for i, p in enumerate(self.stage_cfg.params)
        }

        # Write trial params
        trial_ds = self.pm.apply_multipliers(multipliers)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name
        self.pm.write_trial_params(trial_ds, tmp_path)

        # Setup isolated run directory
        settings_dir, output_dir = setup_trial_run_dir(
            base_settings_dir=self.run_cfg["summa_settings_dir"],
            base_output_dir=self.run_cfg["trial_base_dir"],
            trial_id=trial_id,
            trial_params_nc=tmp_path,
            param_nc_filename=self.run_cfg.get("param_nc_filename", "trialParams.nc"),
        )
        fm_path = patch_file_manager(
            settings_dir, output_dir,
            sim_start=self.run_cfg.get("sim_start"),
            sim_end=self.run_cfg.get("sim_end"),
        )
        Path(tmp_path).unlink(missing_ok=True)

        # Run SUMMA
        converged = run_summa(
            exe_path=self.run_cfg.get("summa_exe", "summa"),
            file_manager=str(fm_path),
            run_id=trial_id,
            timeout_sec=self.run_cfg.get("timeout_sec", 1800),
        )

        elapsed = time.perf_counter() - t0

        if not converged:
            self._log_trial(trial_id, multipliers, 999.0, 999.0, 999.0, elapsed, False)
            self._cleanup_trial(self.run_cfg["trial_base_dir"], trial_id)
            return 999.0

        # Compute objectives
        try:
            j_anchor, j_coherence = self._compute_objectives(output_dir)
        except Exception as exc:
            logger.warning("[%s] Objective computation failed: %s", trial_id, exc)
            j_anchor, j_coherence = 999.0, 999.0

        j_total = combined_objective(j_anchor, j_coherence, w_anchor=self.stage_cfg.w_anchor)
        self._log_trial(trial_id, multipliers, j_anchor, j_coherence, j_total, elapsed, True)
        self._cleanup_trial(self.run_cfg["trial_base_dir"], trial_id)

        logger.info("[%s] J=%.4f  (anchor=%.4f  coherence=%.4f)  params=%s",
                    trial_id, j_total, j_anchor, j_coherence,
                    {p: f"{v:.3f}" for p, v in multipliers.items()})
        return j_total

    def _compute_objectives(self, output_dir: Path) -> Tuple[float, float]:
        cfg = self.stage_cfg
        prefix = self.run_cfg.get("output_prefix", "")

        if cfg.coherence_type == "snow":
            all_swe = read_hru_output(output_dir, prefix, "scalarSWE",
                                      cfg.spinup_days, cfg.time_step_hours)
            hru_elevs = self.run_cfg.get("hru_elevations", {i: 3000.0 for i in all_swe})
            hru_aspects = self.run_cfg.get("hru_aspects", None)
            if hru_aspects:
                hru_aspects = {int(k): v for k, v in hru_aspects.items()}
            all_swe_arr = {i: s.values for i, s in all_swe.items()}
            j_coh = compute_coherence_snow(all_swe_arr, hru_elevs, hru_aspects=hru_aspects)

            # Anchor: one KGE per SNOTEL site, averaged.
            # snotel_sites: list of {elevation, obs_path} from run_cfg.
            # Falls back to self._obs (single-site legacy) if not defined.
            snotel_sites = self.run_cfg.get("snotel_sites", None)
            if snotel_sites:
                site_kges = []
                for site in snotel_sites:
                    site_elev = site["elevation"]
                    site_obs = load_obs_timeseries(site["obs_path"])
                    # Select HRU closest to this site's elevation
                    best_hru = min(hru_elevs, key=lambda h: abs(hru_elevs[h] - site_elev))
                    hru_series = all_swe.get(best_hru, list(all_swe.values())[0])
                    sim_daily = hru_series.resample("D").mean()
                    obs_daily = site_obs.resample("D").mean()
                    sim_a, obs_a = sim_daily.align(obs_daily, join="inner")
                    if len(sim_a) >= 10:
                        site_kges.append(compute_anchor_snow(sim_a.values, obs_a.values))
                j_anc = float(np.mean(site_kges)) if site_kges else 0.5
            elif not self._obs.empty:
                anchor_swe_s = self._select_anchor_hru(all_swe)
                sim_daily = anchor_swe_s.resample("D").mean()
                obs_daily = self._obs.resample("D").mean()
                sim_a, obs_a = sim_daily.align(obs_daily, join="inner")
                j_anc = compute_anchor_snow(sim_a.values, obs_a.values)
            else:
                j_anc = 0.5

        elif cfg.coherence_type == "et":
            # scalarTotalET + scalarSnowSublimation — both negative (flux leaving surface)
            all_et = read_hru_output(output_dir, prefix, "scalarTotalET",
                                     cfg.spinup_days, cfg.time_step_hours)
            try:
                all_subl = read_hru_output(output_dir, prefix, "scalarSnowSublimation",
                                           cfg.spinup_days, cfg.time_step_hours)
                all_et = {i: all_et[i].add(all_subl[i], fill_value=0.0)
                          for i in all_et if i in all_subl}
            except Exception:
                pass
            # Coherence: pass raw negative Series values
            all_et_arr = {i: s.values for i, s in all_et.items()}
            hru_lc = self.run_cfg.get("hru_landcover", {})
            j_coh = compute_coherence_et(all_et_arr, hru_lc, cfg.time_step_hours)

            if not self._obs.empty:
                # Basin mean ET, negated to positive, resampled to monthly totals (mm/month)
                et_basin_s = pd.concat(list(all_et.values()), axis=1).mean(axis=1)
                et_basin_s = -et_basin_s  # negate → positive evaporation
                # Resample to calendar months, converting kg/m²/s × seconds → mm
                step_sec = cfg.time_step_hours * 3600.0
                sim_monthly = (et_basin_s * step_sec).resample("MS").sum()
                obs_m = self._obs.copy()
                obs_m.index = obs_m.index.to_period("M").to_timestamp()
                sim_a, obs_a = sim_monthly.align(obs_m, join="inner")
                j_anc = compute_anchor_et(sim_a.values, obs_a.values)
            else:
                try:
                    sim_q = read_basin_output(output_dir, prefix, "averageRoutedRunoff",
                                              cfg.spinup_days, cfg.time_step_hours)
                    obs_q = load_obs_timeseries(self.run_cfg.get("streamflow_obs_path", ""))
                    sim_daily = sim_q.resample("D").mean()
                    sim_a, obs_a = sim_daily.align(obs_q, join="inner")
                    j_anc = compute_anchor_streamflow_rising(sim_a.values, obs_a.values)
                except Exception:
                    j_anc = 0.5

        elif cfg.coherence_type == "baseflow":
            all_bf = read_hru_output(output_dir, prefix, "scalarAquiferBaseflow",
                                     cfg.spinup_days, cfg.time_step_hours)
            sim_q = read_basin_output(output_dir, prefix, "averageRoutedRunoff",
                                      cfg.spinup_days, cfg.time_step_hours)
            all_bf_arr = {i: s.values for i, s in all_bf.items()}
            sim_q_arr = sim_q.values
            j_coh = compute_coherence_baseflow(all_bf_arr, sim_q_arr)

            if not self._obs.empty:
                # Resample hourly sim to daily; align with daily obs
                sim_daily = sim_q.resample("D").mean()
                sim_a, obs_a = sim_daily.align(self._obs, join="inner")
                j_anc = compute_anchor_baseflow(sim_a.values, obs_a.values)
            else:
                j_anc = 0.5

        elif cfg.coherence_type in ("streamflow", "routing"):
            sim_q = read_basin_output(output_dir, prefix, "averageRoutedRunoff",
                                      cfg.spinup_days, cfg.time_step_hours)
            j_coh = 0.0
            if not self._obs.empty:
                sim_daily = sim_q.resample("D").mean()
                sim_a, obs_a = sim_daily.align(self._obs, join="inner")
                j_anc = compute_anchor_streamflow(sim_a.values, obs_a.values)
            else:
                j_anc = 0.5

        else:
            raise ValueError(f"Unknown coherence_type: {cfg.coherence_type!r}")

        return float(j_anc), float(j_coh)

    def _select_anchor_hru(self, all_swe: "Dict[int, pd.Series]") -> "pd.Series":
        """Return SWE Series for the HRU closest to the SNOTEL anchor elevation."""
        anchor_elev = self.run_cfg.get("anchor_hru_elevation", None)
        hru_elevs = self.run_cfg.get("hru_elevations", {})
        if anchor_elev and hru_elevs:
            best = min(hru_elevs, key=lambda h: abs(hru_elevs[h] - anchor_elev))
            return all_swe.get(best, list(all_swe.values())[0])
        return list(all_swe.values())[0]

    @staticmethod
    def _aggregate_to_monthly(hourly: np.ndarray, time_step_hours: float) -> np.ndarray:
        """Aggregate hourly ET (kg m-2 s-1) to monthly totals in mm/month.

        Conversion: kg/m²/s × 3600 s/hr × steps_in_month → mm/month
        (1 kg/m² water = 1 mm depth)
        """
        step_seconds = time_step_hours * 3600.0
        steps_per_day = 24.0 / time_step_hours
        steps_per_month = int(steps_per_day * 30.44)
        n_months = len(hourly) // steps_per_month
        monthly = np.array([
            # sum rates × step_length → total mm per month
            np.sum(hourly[i * steps_per_month:(i + 1) * steps_per_month]) * step_seconds
            for i in range(n_months)
        ])
        return monthly

    def _log_trial(
        self, trial_id: str, multipliers: Dict[str, float],
        j_anc: float, j_coh: float, j_tot: float,
        runtime: float, converged: bool,
    ) -> None:
        row = (
            [trial_id, self.stage_cfg.name]
            + [multipliers.get(p, np.nan) for p in self.stage_cfg.params]
            + [j_anc, j_coh, j_tot, round(runtime, 2), converged]
        )
        with open(self.log_path, "a", newline="") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                csv.writer(fh).writerow(row)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _cleanup_trial(self, trial_base: str, trial_id: str) -> None:
        trial_dir = Path(trial_base) / trial_id
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(
    stage_cfg: StageConfig,
    param_manager: ParameterManager,
    run_cfg: Dict[str, Any],
    results_dir: Path,
) -> Dict[str, float]:
    """Optimize a single stage; return best multipliers.

    Uses scipy.optimize.differential_evolution with `workers=-1` for
    automatic parallel evaluation across available CPUs.
    """
    logger.info("=" * 70)
    logger.info("Starting %s: %s", stage_cfg.name, stage_cfg.description)
    logger.info("Parameters: %s", stage_cfg.params)

    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{stage_cfg.name}_trials.csv"

    evaluator = TrialEvaluator(stage_cfg, param_manager, run_cfg, log_path)

    # Build bounds for differential_evolution
    bounds = []
    for p in stage_cfg.params:
        if p in stage_cfg.multiplier_search_bounds:
            bounds.append(stage_cfg.multiplier_search_bounds[p])
        elif p in MULTIPLIER_BOUNDS:
            bounds.append(MULTIPLIER_BOUNDS[p])
        else:
            bounds.append((0.5, 2.0))  # fallback: ±50% of base
            logger.warning("No multiplier bounds for '%s' — using (0.5, 2.0)", p)

    workers = run_cfg.get("parallel_workers", 1)
    logger.info("Running differential_evolution: maxiter=%d  popsize=%d  workers=%d",
                stage_cfg.max_iterations, stage_cfg.popsize, workers)

    # Convergence callback: called after each generation with the current population.
    # Logs best/mean cost and estimated KGE so progress is visible in the log file.
    _gen_counter = [0]
    _running_best = [np.inf]

    def _convergence_callback(xk: np.ndarray, convergence: float) -> bool:
        """Called by differential_evolution after each generation."""
        _gen_counter[0] += 1
        df = pd.read_csv(log_path)
        df_ok = df[df["converged"] == True]
        if df_ok.empty:
            return False
        best_so_far = df_ok["J_total"].min()
        best_anchor = df_ok.loc[df_ok["J_total"].idxmin(), "J_anchor"]
        improved = best_so_far < _running_best[0] - 1e-4
        _running_best[0] = min(_running_best[0], best_so_far)
        logger.info(
            "Gen %3d | best_J=%.4f  KGE=%+.3f  convergence=%.5f  %s",
            _gen_counter[0],
            best_so_far,
            1.0 - float(best_anchor),
            convergence,
            "↓ improved" if improved else "  plateau",
        )
        return False  # never force-stop via callback; let tol handle it

    result = differential_evolution(
        func=evaluator,
        bounds=bounds,
        maxiter=stage_cfg.max_iterations,
        popsize=stage_cfg.popsize,
        tol=stage_cfg.tol,
        seed=run_cfg.get("random_seed", 42),
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
        polish=False,
        disp=False,      # suppress scipy's own output; we log via callback
        callback=_convergence_callback,
    )

    best_multipliers = {p: float(result.x[i]) for i, p in enumerate(stage_cfg.params)}
    best_cost = float(result.fun)

    logger.info("%s complete — best cost=%.5f  converged=%s",
                stage_cfg.name, best_cost, result.success)
    logger.info("Best multipliers: %s", best_multipliers)

    # Save results
    out = {
        "stage": stage_cfg.name,
        "best_cost": best_cost,
        "converged": bool(result.success),
        "n_evaluations": int(result.nfev),
        "best_multipliers": best_multipliers,
    }
    with open(results_dir / f"{stage_cfg.name}_best_params.json", "w") as fh:
        json.dump(out, fh, indent=2)

    return best_multipliers


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------

def run_all_stages(config_path: str | Path) -> None:
    """Load config, run all four stages sequentially, freeze parameters between stages."""
    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    log_level = getattr(logging, cfg.get("log_level", "INFO").upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = cfg["run"]
    run_cfg["trial_base_dir"] = str(results_dir / "_trials")

    # Initialize parameter manager from the base trialParams.nc
    pm = ParameterManager(run_cfg["base_trial_params_nc"])

    stage_configs_dir = Path(cfg.get("stage_configs_dir", "stage_configs"))
    stage_files = cfg.get("stages", [
        str(stage_configs_dir / "stage1_snow.yaml"),
        str(stage_configs_dir / "stage2_soil_et.yaml"),
        str(stage_configs_dir / "stage3_groundwater.yaml"),
        str(stage_configs_dir / "stage4_routing.yaml"),
    ])

    all_best: Dict[str, float] = {}

    for stage_file in stage_files:
        stage_cfg = StageConfig.from_yaml(stage_file)

        # Resolve relative obs paths against the config file's directory
        if stage_cfg.anchor_obs_path and not Path(stage_cfg.anchor_obs_path).is_absolute():
            stage_cfg.anchor_obs_path = str(config_path.parent / stage_cfg.anchor_obs_path)

        best_mults = run_stage(stage_cfg, pm, run_cfg, results_dir)

        # Freeze this stage's parameters into the base before the next stage
        pm.freeze_params(best_mults, stage_cfg.params)

        # Persist updated base params to disk so a restart can resume from here
        frozen_path = results_dir / f"{stage_cfg.name}_frozen_params.nc"
        pm.save_base(frozen_path)
        logger.info("Frozen base params saved → %s", frozen_path)

        all_best.update(best_mults)

    # Write final combined parameter set
    final_params_path = results_dir / "final_best_params.json"
    with open(final_params_path, "w") as fh:
        json.dump(all_best, fh, indent=2)
    logger.info("All stages complete.  Final params → %s", final_params_path)

    # Write final trialParams.nc for use in evaluation runs
    final_nc_path = results_dir / "final_trialParams.nc"
    pm.save_base(final_nc_path)
    logger.info("Final trialParams.nc → %s", final_nc_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Staged distributed SUMMA calibration")
    p.add_argument("--config", required=True, help="Path to optimization_config.yaml")
    p.add_argument("--stage", default=None,
                   help="Run only this stage YAML file (for testing / resuming)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.stage:
        # Single-stage mode for development / resuming
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        run_cfg = cfg["run"]
        results_dir = Path(cfg["results_dir"])
        run_cfg["trial_base_dir"] = str(results_dir / "_trials")
        pm = ParameterManager(run_cfg["base_trial_params_nc"])
        stage_cfg = StageConfig.from_yaml(args.stage)
        run_stage(stage_cfg, pm, run_cfg, results_dir)
    else:
        run_all_stages(args.config)
