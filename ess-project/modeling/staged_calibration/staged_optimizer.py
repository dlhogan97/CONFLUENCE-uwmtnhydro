#!/usr/bin/env python3
"""
staged_optimizer.py — Staged, process-based calibration for distributed SUMMA.

Philosophy
----------
Simultaneous global optimization of all parameters leads to compensating errors
(equifinality).  By isolating process groups and constraining each with the most
relevant observation type, we prevent the optimizer from trading errors between
snow, soil, groundwater, and routing (Clark et al. 2015; Beven 2006).

Stages are config-driven and run sequentially; each stage freezes its
parameters before the next begins. Typical stages are snow, soil/ET,
groundwater/baseflow, and routing.

Each stage optimizes *multipliers* on the per-HRU base parameters, not raw values.
This preserves spatial heterogeneity from the a-priori data.

Usage
-----
    python staged_optimizer.py --config optimization/optimization_config_bigBuckt.yaml
    python staged_optimizer.py --config optimization/optimization_config_noXplicit.yaml

Or from Python:
    from staged_optimizer import run_all_stages
    run_all_stages("optimization/optimization_config_bigBuckt.yaml")
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
from summa_runner import run_summa, read_hru_output, read_basin_output, setup_trial_run_dir, patch_file_manager, patch_model_decisions
from objective_functions import (
    compute_anchor_snow,
    compute_anchor_et,
    compute_anchor_streamflow_rising,
    compute_anchor_baseflow,
    compute_anchor_runoff,
    compute_anchor_streamflow,
    compute_anchor_generic,
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
    coherence_type: str                 # "snow" | "et" | "runoff" | "baseflow" | "streamflow"
    anchor_obs_path: str = ""           # path to observations CSV (overridden by optimization config observations: block)
    anchor_metric: str = "KGE"         # metric for anchor cost: KGE | KGE_log | NSE | NRMSE
    w_anchor: float = 0.4               # weight given to anchor vs coherence
    multiplier_search_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    max_iterations: int = 150
    popsize: int = 10                   # differential_evolution population multiplier
    tol: float = 0.01
    spinup_days: int = 365
    time_step_hours: float = 1.0
    description: str = ""
    # Optional per-HRU spatial weights: {param_name: {landcover_class: weight, ...}}
    # or {param_name: {hru_index: weight, ...}}.
    # Weights carry directional spatial information; the global multiplier M carries
    # only magnitude.  actual_param_i = base_i × weight_i × M.
    spatial_weights: Dict[str, Dict] = field(default_factory=dict)

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
            anchor_obs_path=data.get("anchor_obs_path", ""),
            coherence_type=data["coherence_type"],
            anchor_metric=data.get("anchor_metric", "KGE"),
            w_anchor=data.get("w_anchor", 0.4),
            multiplier_search_bounds=parsed_bounds,
            max_iterations=data.get("max_iterations", 150),
            popsize=data.get("popsize", 10),
            tol=data.get("tol", 0.01),
            spinup_days=data.get("spinup_days", 365),
            time_step_hours=data.get("time_step_hours", 1.0),
            description=data.get("description", ""),
            spatial_weights=data.get("spatial_weights", {}),
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


def _validate_required_stage_params(pm: ParameterManager, stage_cfgs: List[StageConfig]) -> None:
    """Hard-fail if any stage parameter is missing from base trialParams.nc.

    This prevents long optimization runs from failing later due to a misbuilt
    trialParams file.
    """
    available = set(pm.base_dataset.data_vars)
    missing_by_stage: Dict[str, List[str]] = {}

    for sc in stage_cfgs:
        missing = [p for p in sc.params if p not in available]
        if missing:
            missing_by_stage[sc.name] = missing

    if missing_by_stage:
        details = "\n".join(
            f"  - {stage}: {', '.join(params)}"
            for stage, params in missing_by_stage.items()
        )
        raise ValueError(
            "Missing required stage parameters in base trialParams.nc.\n"
            f"Missing by stage:\n{details}\n"
            f"Available variables: {sorted(available)}"
        )


def _build_spatial_weight_arrays(
    raw_weights: Dict[str, Dict],
    hru_landcover: Dict[int, Any],
    n_hru: int,
) -> Dict[str, np.ndarray]:
    """Expand per-landcover or per-HRU weight dicts to per-HRU numpy arrays.

    Parameters
    ----------
    raw_weights:
        {param_name: {landcover_class: weight}} or {param_name: {hru_index: weight}}.
        Keys can be landcover strings ("barren", "forest", …) or integer HRU indices.
        Any HRU without an explicit weight defaults to 1.0.
    hru_landcover:
        {hru_idx: landcover_class} mapping from optimization_config_bigBuckt.yaml
        or optimization_config_noXplicit.yaml.
        Landcover class can be a descriptive string (e.g., "barren") or
        an integer/str code (e.g., 16 for MODIS barren).
    n_hru:
        Total number of HRUs.

    Returns
    -------
    {param_name: np.ndarray of shape (n_hru,)}
    """
    result: Dict[str, np.ndarray] = {}
    for param, weight_spec in raw_weights.items():
        w_arr = np.ones(n_hru, dtype=float)
        for hru_idx in range(n_hru):
            # Prefer explicit HRU-index keys (int or string-of-int)
            if hru_idx in weight_spec:
                w_arr[hru_idx] = float(weight_spec[hru_idx])
            elif str(hru_idx) in weight_spec:
                w_arr[hru_idx] = float(weight_spec[str(hru_idx)])
            else:
                lc = hru_landcover.get(hru_idx, "")
                # Match landcover labels or codes robustly.
                # Accept exact key, stringified key, and case-insensitive string labels.
                if lc in weight_spec:
                    w_arr[hru_idx] = float(weight_spec[lc])
                elif str(lc) in weight_spec:
                    w_arr[hru_idx] = float(weight_spec[str(lc)])
                elif isinstance(lc, str):
                    lc_norm = lc.strip().lower()
                    for k, v in weight_spec.items():
                        if isinstance(k, str) and k.strip().lower() == lc_norm:
                            w_arr[hru_idx] = float(v)
                            break
        result[param] = w_arr
    return result


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

        # Build per-HRU spatial weight arrays from the stage YAML spec.
        self._spatial_weights: Dict[str, np.ndarray] = {}
        if stage_cfg.spatial_weights:
            hru_lc = {int(k): v for k, v in run_cfg.get("hru_landcover", {}).items()}
            n_hru = len(hru_lc) or 1
            self._spatial_weights = _build_spatial_weight_arrays(
                stage_cfg.spatial_weights, hru_lc, n_hru
            )
            logger.info(
                "Spatial weights active for %s: %s",
                stage_cfg.name,
                {k: list(v.round(3)) for k, v in self._spatial_weights.items()},
            )

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

        # Write trial params (pass spatial weights so per-HRU weighting is applied)
        trial_ds = self.pm.apply_multipliers(
            multipliers,
            spatial_weights=self._spatial_weights if self._spatial_weights else None,
        )
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
        param_nc_filename = self.run_cfg.get("param_nc_filename", "trialParams.nc")
        fm_path = patch_file_manager(
            settings_dir, output_dir,
            sim_start=self.run_cfg.get("sim_start"),
            sim_end=self.run_cfg.get("sim_end"),
            out_file_prefix=self.run_cfg.get("output_prefix"),
            trial_param_filename=param_nc_filename,
        )
        model_decisions = self.run_cfg.get("model_decisions_override", {})
        if model_decisions:
            patch_model_decisions(settings_dir, model_decisions)
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
        anchor_var = str(getattr(cfg, "anchor_variable", "") or "").strip()
        # averageRoutedRunoff from SUMMA is in m s⁻¹ (depth per unit area per time).
        # Streamflow observations are in m³/s.  Multiply by basin area to convert.
        basin_area_m2 = float(self.run_cfg.get("basin_area_m2", 1.0))

        def _is_basin_volumetric_var(var_name: str) -> bool:
            """Return True when a variable is already basin volumetric flow (m3/s)."""
            v = str(var_name or "").strip()
            if not v:
                return False
            # Known exception: basin__TotalRunoff is a depth-rate (m/s), not volumetric flow.
            if v.lower() == "basin__totalrunoff":
                return False
            # Basin-prefixed fluxes (e.g., basin__ColumnOutflow) are volumetric in qTopmodl runs.
            return v.startswith("basin__")

        def _to_discharge_cms(series: pd.Series, var_name: str) -> pd.Series:
            """Convert a flow series to m3/s only when it is reported as depth rate (m/s)."""
            if _is_basin_volumetric_var(var_name):
                return series
            return series * basin_area_m2

        def _to_depth_rate_mps(series: pd.Series, var_name: str) -> pd.Series:
            """Convert a flow series to m/s only when it is reported as volumetric flow (m3/s)."""
            if _is_basin_volumetric_var(var_name):
                return series / basin_area_m2
            return series

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
                    sim_q = (
                        read_basin_output(output_dir, prefix, "averageRoutedRunoff",
                                          cfg.spinup_days, cfg.time_step_hours)
                        * basin_area_m2   # m/s → m³/s
                    )
                    obs_q = load_obs_timeseries(self.run_cfg.get("streamflow_obs_path", ""))
                    sim_daily = sim_q.resample("D").mean()
                    sim_a, obs_a = sim_daily.align(obs_q, join="inner")
                    j_anc = compute_anchor_streamflow_rising(sim_a.values, obs_a.values)
                except Exception:
                    j_anc = 0.5

        elif cfg.coherence_type == "baseflow":
            baseflow_var = anchor_var or "scalarAquiferBaseflow"
            all_bf = read_hru_output(output_dir, prefix, baseflow_var,
                                     cfg.spinup_days, cfg.time_step_hours)
            all_bf_arr = {
                i: _to_depth_rate_mps(s, baseflow_var).values
                for i, s in all_bf.items()
            }

            # Coherence uses BFI ratios in consistent units.
            # - bigBuckt: scalarAquiferBaseflow is HRU depth-rate, compare to HRU scalarTotalRunoff
            #             and compute mean BFI cost across HRUs.
            # - qTopmodel-style basin baseflow vars (e.g., basin__ColumnOutflow): compare
            #             area-normalized baseflow against basin__TotalRunoff.
            if _is_basin_volumetric_var(baseflow_var):
                runoff_var = "basin__TotalRunoff"
                sim_q_rate = read_basin_output(output_dir, prefix, runoff_var,
                                               cfg.spinup_days, cfg.time_step_hours)
                sim_q_for_coherence = _to_depth_rate_mps(sim_q_rate, runoff_var)
                j_coh = compute_coherence_baseflow(all_bf_arr, sim_q_for_coherence.values)
            else:
                runoff_var = "scalarTotalRunoff"
                all_runoff = read_hru_output(output_dir, prefix, runoff_var,
                                             cfg.spinup_days, cfg.time_step_hours)
                all_runoff_arr = {
                    i: _to_depth_rate_mps(s, runoff_var).values
                    for i, s in all_runoff.items()
                }
                j_coh = compute_coherence_baseflow(
                    all_bf_arr,
                    all_hru_total_runoff=all_runoff_arr,
                )

            if not self._obs.empty:
                # Anchor: compare simulated aquifer baseflow (m³/s) vs Eckhardt-separated
                # observed streamflow.  Use scalarAquiferBaseflow directly — comparing total
                # routed runoff to filtered baseflow diverges badly during snowmelt peaks.
                bf_series_list = list(all_bf.values())
                if bf_series_list:
                    df_bf = pd.concat(bf_series_list, axis=1)
                    # basin__ColumnOutflow is already volumetric discharge (m3/s);
                    # only depth-rate variables need basin-area conversion.
                    if _is_basin_volumetric_var(baseflow_var):
                        sim_bf_cms = df_bf.mean(axis=1)
                    else:
                        sim_bf_cms = df_bf.mean(axis=1) * basin_area_m2
                    sim_daily_bf = sim_bf_cms.resample("D").mean()
                    sim_a, obs_a = sim_daily_bf.align(self._obs, join="inner")
                    j_anc = compute_anchor_baseflow(sim_a.values, obs_a.values)
                else:
                    j_anc = 0.5
            else:
                j_anc = 0.5

        elif cfg.coherence_type == "runoff":
            # Soil stage: anchor to total basin streamflow; no HRU coherence.
            runoff_var = anchor_var or "averageRoutedRunoff"
            sim_q = _to_discharge_cms(
                read_basin_output(output_dir, prefix, runoff_var,
                                  cfg.spinup_days, cfg.time_step_hours),
                runoff_var,
            )
            j_coh = 0.0
            if not self._obs.empty:
                sim_daily = sim_q.resample("D").mean()
                sim_a, obs_a = sim_daily.align(self._obs, join="inner")
                j_anc = compute_anchor_runoff(sim_a.values, obs_a.values,
                                              metric=cfg.anchor_metric)
            else:
                j_anc = 0.5

        elif cfg.coherence_type in ("streamflow", "routing"):
            streamflow_var = anchor_var or "averageRoutedRunoff"
            sim_q = _to_discharge_cms(
                read_basin_output(output_dir, prefix, streamflow_var,
                                  cfg.spinup_days, cfg.time_step_hours),
                streamflow_var,
            )
            j_coh = 0.0
            if not self._obs.empty:
                sim_daily = sim_q.resample("D").mean()
                sim_a, obs_a = sim_daily.align(self._obs, join="inner")
                j_anc = compute_anchor_streamflow(sim_a.values, obs_a.values,
                                                  metric=cfg.anchor_metric)
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
    """Load config, run configured stages sequentially, and freeze parameters between stages."""
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
    calib_init = run_cfg.get("base_calib_bounds_json")
    if calib_init:
        pm.apply_calib_bounds_values(calib_init)

    stage_configs_dir = Path(cfg.get("stage_configs_dir", "stage_configs"))
    stage_files = cfg.get("stages", [
        str(stage_configs_dir / "stage1_snow.yaml"),
        str(stage_configs_dir / "stage2_soil_et.yaml"),
        str(stage_configs_dir / "stage3_groundwater.yaml"),
        str(stage_configs_dir / "stage4_routing.yaml"),
    ])

    # Fail fast if any stage parameter is missing in base trialParams.nc.
    stage_cfgs = [StageConfig.from_yaml(sf) for sf in stage_files]
    _validate_required_stage_params(pm, stage_cfgs)

    all_best: Dict[str, float] = {}

    # Pre-compute HRU landcover mapping (used by spatial weight expansion)
    hru_lc = {int(k): v for k, v in run_cfg.get("hru_landcover", {}).items()}
    n_hru = len(hru_lc) or 1

    # Build a lookup of coherence_type → obs path from the run config's
    # `observations:` block.  These override any anchor_obs_path baked into
    # the individual stage YAML files, making the stage configs basin-agnostic.
    _obs_override: Dict[str, str] = {}
    obs_block = cfg.get("observations", {})
    if obs_block.get("swe_obs_path"):
        _obs_override["snow"] = obs_block["swe_obs_path"]
    if obs_block.get("et_obs_path"):
        _obs_override["et"] = obs_block["et_obs_path"]
    if obs_block.get("streamflow_obs_path"):
        _obs_override["runoff"]      = obs_block["streamflow_obs_path"]
        _obs_override["baseflow"]    = obs_block["streamflow_obs_path"]
        _obs_override["streamflow"]  = obs_block["streamflow_obs_path"]
        _obs_override["routing"]     = obs_block["streamflow_obs_path"]

    for stage_cfg, stage_file in zip(stage_cfgs, stage_files):

        # Inject basin-level obs paths from run config (overrides stage yaml).
        if stage_cfg.coherence_type in _obs_override:
            stage_cfg.anchor_obs_path = _obs_override[stage_cfg.coherence_type]
        elif stage_cfg.anchor_obs_path and not Path(stage_cfg.anchor_obs_path).is_absolute():
            # Fall back: resolve relative path against config file location
            stage_cfg.anchor_obs_path = str(config_path.parent / stage_cfg.anchor_obs_path)

        best_mults = run_stage(stage_cfg, pm, run_cfg, results_dir)

        # Build spatial weight arrays for the freeze step so that the baked-in
        # base values reflect  base × weight × M  (not just base × M).
        stage_sw: Optional[Dict[str, np.ndarray]] = None
        if stage_cfg.spatial_weights:
            stage_sw = _build_spatial_weight_arrays(stage_cfg.spatial_weights, hru_lc, n_hru)

        # Freeze this stage's parameters into the base before the next stage
        pm.freeze_params(best_mults, stage_cfg.params, spatial_weights=stage_sw)

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
    p.add_argument(
        "--config",
        required=True,
        help=(
            "Path to an explicit optimization config file, e.g. "
            "optimization/optimization_config_bigBuckt.yaml or "
            "optimization/optimization_config_noXplicit.yaml"
        ),
    )
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
        calib_init = run_cfg.get("base_calib_bounds_json")
        if calib_init:
            pm.apply_calib_bounds_values(calib_init)
        stage_cfg = StageConfig.from_yaml(args.stage)
        _validate_required_stage_params(pm, [stage_cfg])
        run_stage(stage_cfg, pm, run_cfg, results_dir)
    else:
        run_all_stages(args.config)
