#!/usr/bin/env python3
"""
Autonomous parameter sweep agent for Tuolumne lumped noXplicit zeroFlux model.

Uses Nelder-Mead optimization starting from final_trialParams.nc. For each
candidate parameter set, runs SUMMA (~90s), auto-calibrates a two-reservoir
linear router against observed streamflow, then computes NSE/KGE/logNSE.
Generates 4 diagnostic plots (hydrograph, monthly timing, SWE, FDC) for the
best result.

Usage:
    python tuolumne_agent_sweep.py [--max-iter N] [--plot-only TRIAL_DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize

# ── paths ────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[3]
UTILS_CUSTOM = REPO_ROOT / "utils" / "custom"
sys.path.insert(0, str(UTILS_CUSTOM))
from linear_reservoir import calibrate_reservoirs, two_reservoir_daily, nse, kge, log_nse

DOMAIN_DIR      = Path("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_lumped")
TEMPLATE_SETTINGS = (
    DOMAIN_DIR
    / "optimization/staged_results_noXplict_zeroFlux"
    / "_trials/trial_p3256521_0001/settings"
)
FINAL_PARAMS    = (
    DOMAIN_DIR
    / "optimization/staged_results_noXplict_zeroFlux/final_trialParams.nc"
)
SWEEP_DIR       = DOMAIN_DIR / "optimization/agent_sweep"
SUMMA_BIN       = "/home/dlhogan/bin/summa"
BASIN_AREA_M2   = 774_508_608.0          # from catchment shapefile row 1

# METSIM forcing preprocessed by prep_metsim_forcing.py (single HRU, SUMMA encoding)
METSIM_FORCING_DIR = DOMAIN_DIR / "forcing/metsim_SUMMA_input"

OBS_Q_PATH      = DOMAIN_DIR / "observations/formatted/streamflow_obs.csv"
OBS_SWE_PATH    = DOMAIN_DIR / "observations/formatted/snotel_swe.csv"

SIM_START  = "2012-10-01 00:00"          # 1-year spinup before eval period
SIM_END    = "2017-09-30 23:00"
EVAL_START = "2013-10-01"
EVAL_END   = "2017-09-30"

# Monthly METSIM files covering the simulation period (spinup + eval)
_SIM_MONTHS = [
    f"{y}{m:02d}"
    for y in range(2012, 2018)
    for m in range(1, 13)
    if not (y == 2012 and m < 10) and not (y == 2017 and m > 9)
]
METSIM_FILE_LIST = [
    f"Tuolumne_River_lumped_METSIM_remapped_metsim_{ym}.nc"
    for ym in _SIM_MONTHS
]

# ── tunable parameters and bounds ────────────────────────────────────────────

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "albedoDecayRate":    (5e4,   2e6),
    "frozenPrecipMultip": (0.40,  1.50),
    "qSurfScale":         (0.30, 15.00),
    "k_soil":             (3e-4,  8e-3),
    "rootingDepth":       (0.05,  0.50),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())

# ── trial counter (shared mutable state so callback can log iteration) ────────
_trial_count = [0]
_eval_cache:  dict[tuple, float] = {}
_results_log: list[dict]         = []


# ── parameter helpers ─────────────────────────────────────────────────────────

def load_base_params() -> dict[str, float]:
    ds = xr.load_dataset(FINAL_PARAMS)
    params = {p: float(ds[p].values.ravel()[0]) for p in PARAM_NAMES if p in ds}
    ds.close()
    return params


def normalize(params: dict[str, float]) -> np.ndarray:
    return np.array(
        [(params[p] - lo) / (hi - lo) for p, (lo, hi) in PARAM_BOUNDS.items()],
        dtype=np.float64,
    )


def denormalize(x: np.ndarray) -> dict[str, float]:
    x = np.clip(x, 0.0, 1.0)
    return {
        p: lo + xi * (hi - lo)
        for xi, (p, (lo, hi)) in zip(x, PARAM_BOUNDS.items())
    }


# ── trial setup ──────────────────────────────────────────────────────────────

def write_trial_params(settings_dir: Path, new_params: dict[str, float]) -> None:
    base = xr.load_dataset(FINAL_PARAMS)
    for p, v in new_params.items():
        if p in base:
            arr = base[p].values.copy()
            arr[:] = v
            base[p] = xr.Variable(base[p].dims, arr, base[p].attrs)
    base.to_netcdf(settings_dir / "trialParams.nc", mode="w")
    base.close()


def write_file_manager(settings_dir: Path, output_dir: Path, run_id: str) -> None:
    fm = f"""controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'
simStartTime    '{SIM_START}'
simEndTime    '{SIM_END}'
tmZoneInfo           'utcTime'
outFilePrefix    '{run_id}'
settingsPath    '{settings_dir}/'
forcingPath          '{METSIM_FORCING_DIR}/'
outputPath    '{output_dir}/'
initConditionFile    'coldState.nc'
attributeFile        'attributes.nc'
trialParamFile    'trialParams.nc'
forcingListFile      'forcingFileList.txt'
decisionsFile        'modelDecisions.txt'
outputControlFile    'outputControl.txt'
globalHruParamFile   'localParamInfo.txt'
globalGruParamFile   'basinParamInfo.txt'
vegTableFile         'TBL_VEGPARM.TBL'
soilTableFile        'TBL_SOILPARM.TBL'
generalTableFile     'TBL_GENPARM.TBL'
noahmpTableFile      'TBL_MPTABLE.TBL'
"""
    (settings_dir / "fileManager.txt").write_text(fm)


def write_forcing_file_list(settings_dir: Path) -> None:
    missing = [f for f in METSIM_FILE_LIST if not (METSIM_FORCING_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} METSIM forcing files not found in {METSIM_FORCING_DIR}. "
            f"Run prep_metsim_forcing.py first. First missing: {missing[0]}"
        )
    (settings_dir / "forcingFileList.txt").write_text("\n".join(METSIM_FILE_LIST) + "\n")


def setup_trial(run_id: str, new_params: dict[str, float]) -> Path:
    trial_dir   = SWEEP_DIR / run_id
    settings_dir = trial_dir / "settings"
    output_dir   = trial_dir / "output"
    settings_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip = {"trialParams.nc", "trialParams.nc.bak", "fileManager.txt",
            "warmState.nc", "forcingFileList.txt"}
    for src in TEMPLATE_SETTINGS.iterdir():
        if src.is_file() and src.name not in skip and not src.name.startswith("trialParams_backup"):
            shutil.copy2(src, settings_dir / src.name)

    write_trial_params(settings_dir, new_params)
    write_forcing_file_list(settings_dir)
    write_file_manager(settings_dir, output_dir, run_id)
    return trial_dir


# ── SUMMA runner ──────────────────────────────────────────────────────────────

def run_summa(trial_dir: Path, run_id: str) -> Path | None:
    fm = trial_dir / "settings" / "fileManager.txt"
    t0 = time.time()
    result = subprocess.run(
        [SUMMA_BIN, "-m", str(fm)],
        capture_output=True, text=True, timeout=360,
    )
    elapsed = time.time() - t0
    print(f"  SUMMA finished in {elapsed:.0f}s  (return={result.returncode})")

    if result.returncode != 0:
        print(f"  SUMMA stderr tail:\n{result.stderr[-400:]}")
        return None

    out_dir = trial_dir / "output"
    # SUMMA names output as <prefix>_timestep.nc
    nc = out_dir / f"{run_id}_timestep.nc"
    if nc.exists():
        return nc
    # Fallback: any netCDF in output dir
    ncs = sorted(out_dir.glob("*.nc"))
    return ncs[0] if ncs else None


# ── output loading and conversion ─────────────────────────────────────────────

def load_summa_output(nc_path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    time_idx = pd.DatetimeIndex(ds["time"].values)
    runoff   = ds["scalarTotalRunoff"].values.ravel()
    swe      = ds["scalarSWE"].values.ravel()
    ds.close()

    df = pd.DataFrame(
        {"q_cms": runoff * BASIN_AREA_M2, "swe_mm": swe},
        index=time_idx,
    )
    return df.resample("1D").mean()


def load_observations() -> tuple[pd.Series, pd.Series]:
    obs_q   = pd.read_csv(OBS_Q_PATH,   parse_dates=["date"], index_col="date")["value"]
    obs_swe = pd.read_csv(OBS_SWE_PATH, parse_dates=["date"], index_col="date")["value"]

    # Mask bad SWE data — jumps > 300 mm in one day are sensor errors
    bad = obs_swe.diff().abs() > 300
    obs_swe[bad] = np.nan
    # Also mask implausible July/Aug nonzero SWE that jump from 0
    return obs_q, obs_swe


# ── routing and metrics ───────────────────────────────────────────────────────

def apply_routing(
    sim_daily: pd.DataFrame,
    obs_q: pd.Series,
) -> tuple[np.ndarray, dict]:
    eval_idx = sim_daily.loc[EVAL_START:EVAL_END].index
    q_in = sim_daily["q_cms"].reindex(eval_idx).values
    obs  = obs_q.reindex(eval_idx).values

    lr = calibrate_reservoirs(q_in=q_in, obs=obs, metric="balanced", n_starts=150, seed=42)
    routed = two_reservoir_daily(q_in, lr["k_fast"], lr["k_slow"], lr["f_fast"])
    return routed, lr


def compute_metrics(routed: np.ndarray, obs_q: pd.Series, sim_daily: pd.DataFrame) -> dict:
    eval_idx = sim_daily.loc[EVAL_START:EVAL_END].index
    obs = obs_q.reindex(eval_idx).values
    mask = np.isfinite(obs) & np.isfinite(routed)
    o, s = obs[mask], routed[mask]

    nse_v  = nse(o, s)
    kge_v  = kge(o, s)
    lnse_v = log_nse(o, s)
    comp   = (nse_v + kge_v + lnse_v) / 3.0
    return {"nse": nse_v, "kge": kge_v, "log_nse": lnse_v, "composite": comp}


# ── objective function ────────────────────────────────────────────────────────

def evaluate(x_norm: np.ndarray, _unused=None) -> float:
    _trial_count[0] += 1
    trial_num = _trial_count[0]
    params = denormalize(x_norm)

    cache_key = tuple(round(v, 10) for v in params.values())
    if cache_key in _eval_cache:
        print(f"  [trial_{trial_num:03d}] cache hit → {_eval_cache[cache_key]:.4f}")
        return _eval_cache[cache_key]

    run_id    = f"trial_{trial_num:03d}"
    param_str = "  ".join(f"{k}={v:.3g}" for k, v in params.items())
    print(f"\n[{run_id}] {param_str}")

    trial_dir = setup_trial(run_id, params)
    nc_path   = run_summa(trial_dir, run_id)

    if nc_path is None:
        _eval_cache[cache_key] = 1.0
        return 1.0

    try:
        sim_daily         = load_summa_output(nc_path)
        obs_q, _          = load_observations()
        routed, lr_result = apply_routing(sim_daily, obs_q)
        metrics           = compute_metrics(routed, obs_q, sim_daily)

        composite = metrics["composite"]
        print(
            f"  NSE={metrics['nse']:.3f}  KGE={metrics['kge']:.3f}  "
            f"logNSE={metrics['log_nse']:.3f}  composite={composite:.3f}\n"
            f"  LR: k_fast={lr_result['k_fast']:.4f}  "
            f"k_slow={lr_result['k_slow']:.2e}  f_fast={lr_result['f_fast']:.3f}"
        )

        obj = -composite
        _eval_cache[cache_key] = obj
        _results_log.append({"run_id": run_id, "params": params, "metrics": metrics, "obj": obj})
        _save_log()
        return obj

    except Exception as exc:
        print(f"  Eval error: {exc}")
        _eval_cache[cache_key] = 1.0
        return 1.0


def _save_log() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_DIR / "sweep_log.json", "w") as f:
        json.dump(_results_log, f, indent=2, default=float)


# ── diagnostic plots ──────────────────────────────────────────────────────────

def plot_diagnostics(
    sim_daily: pd.DataFrame,
    routed: np.ndarray,
    obs_q: pd.Series,
    obs_swe: pd.Series,
    params: dict,
    metrics: dict,
    lr_result: dict,
    out_path: Path,
) -> None:
    eval_idx    = sim_daily.loc[EVAL_START:EVAL_END].index
    obs_q_eval  = obs_q.reindex(eval_idx)
    swe_sim     = sim_daily["swe_mm"].reindex(eval_idx)
    obs_swe_eval = obs_swe.reindex(eval_idx)
    routed_s    = pd.Series(routed, index=eval_idx)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Tuolumne lumped noXplicit zeroFlux — Agent Sweep Best\n"
        f"NSE={metrics['nse']:.3f}  KGE={metrics['kge']:.3f}  "
        f"logNSE={metrics['log_nse']:.3f}  composite={metrics['composite']:.3f}",
        fontsize=11,
    )

    # 1. Hydrograph
    ax = axes[0, 0]
    ax.plot(obs_q_eval.index,  obs_q_eval.values,  "k-",  lw=1.2, label="Observed",       alpha=0.85)
    ax.plot(routed_s.index,    routed_s.values,    "r-",  lw=1.0, label="Sim (LR-routed)", alpha=0.85)
    ax.set_ylabel("Q (m³/s)")
    ax.set_title("Hydrograph")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 2. Monthly mean timing
    ax = axes[0, 1]
    obs_m_resampled = obs_q_eval.resample("MS").mean()
    sim_m_resampled = routed_s.resample("MS").mean()
    obs_monthly = obs_m_resampled.groupby(obs_m_resampled.index.month).mean()
    sim_monthly = sim_m_resampled.groupby(sim_m_resampled.index.month).mean()
    months = list(range(1, 13))
    labels = list("JFMAMJJASOND")
    x = np.array(months)
    ax.bar(x - 0.2, [obs_monthly.get(m, np.nan) for m in months], width=0.35,
           label="Observed", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, [sim_monthly.get(m, np.nan) for m in months], width=0.35,
           label="Simulated", color="salmon", alpha=0.8)
    ax.set_xticks(months)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Q (m³/s)")
    ax.set_title("Monthly Average Flow")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    # 3. SWE
    ax = axes[1, 0]
    ax.plot(obs_swe_eval.index, obs_swe_eval.values, "b-",  lw=1.2, label="SNOTEL (TUM)", alpha=0.85)
    ax.plot(swe_sim.index,       swe_sim.values,      "r--", lw=1.0, label="SUMMA SWE",   alpha=0.85)
    ax.set_ylabel("SWE (mm)")
    ax.set_title("Snow Water Equivalent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 4. Flow Duration Curve — low-flow tail (exceedance 50–100%)
    ax = axes[1, 1]
    for label, series, color in [("Observed", obs_q_eval.dropna(), "k"),
                                   ("Simulated", routed_s.dropna(), "r")]:
        sorted_q = np.sort(series.values)[::-1]
        exceed   = np.linspace(0, 100, len(sorted_q))
        ax.semilogy(exceed, sorted_q, color=color, lw=1.5, label=label)
    ax.set_xlim(50, 100)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Q (m³/s, log scale)")
    ax.set_title("Flow Duration Curve (low-flow)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Parameter annotation
    param_txt = "\n".join(f"{k}: {v:.3g}" for k, v in params.items())
    param_txt += f"\nLR k_fast={lr_result['k_fast']:.3f}  k_slow={lr_result['k_slow']:.2e}  f={lr_result['f_fast']:.2f}"
    fig.text(0.01, 0.01, param_txt, fontsize=7, family="monospace",
             verticalalignment="bottom")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous parameter sweep for Tuolumne noXplicit model")
    parser.add_argument("--max-iter", type=int, default=120,
                        help="Max Nelder-Mead iterations (default 120 ≈ 30–60 SUMMA runs)")
    parser.add_argument("--plot-only", type=str, default="",
                        help="Skip sweep; generate plots for an existing trial directory")
    args = parser.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        trial_dir = Path(args.plot_only)
        nc_path   = next(trial_dir.glob("output/*.nc"), None)
        if nc_path is None:
            print(f"No output nc found in {trial_dir}/output/"); return
        sim_daily          = load_summa_output(nc_path)
        obs_q, obs_swe     = load_observations()
        routed, lr_result  = apply_routing(sim_daily, obs_q)
        metrics            = compute_metrics(routed, obs_q, sim_daily)
        # Read params from trialParams.nc
        tp = xr.load_dataset(trial_dir / "settings" / "trialParams.nc")
        params = {p: float(tp[p].values.ravel()[0]) for p in PARAM_NAMES if p in tp}
        tp.close()
        plot_diagnostics(sim_daily, routed, obs_q, obs_swe, params, metrics, lr_result,
                         SWEEP_DIR / "diagnostics_manual.png")
        print(f"NSE={metrics['nse']:.3f}  KGE={metrics['kge']:.3f}  logNSE={metrics['log_nse']:.3f}")
        return

    base_params = load_base_params()
    print("Starting parameters:")
    for k, v in base_params.items():
        lo, hi = PARAM_BOUNDS[k]
        print(f"  {k}: {v:.4g}  [{lo:.3g} – {hi:.3g}]")

    x0 = normalize(base_params)

    print(f"\nStarting Nelder-Mead sweep (max_iter={args.max_iter})")
    print(f"Output directory: {SWEEP_DIR}\n")

    opt = minimize(
        evaluate,
        x0=x0,
        method="Nelder-Mead",
        options={
            "maxiter": args.max_iter,
            "xatol":   0.02,
            "fatol":   0.005,
            "disp":    True,
            "adaptive": True,
        },
    )

    best_params = denormalize(opt.x)
    print("\n=== Sweep complete ===")
    print(f"Best composite: {-opt.fun:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4g}")

    # Final run for best params + plots
    print("\nRunning SUMMA for best parameters and generating diagnostics...")
    trial_dir          = setup_trial("best", best_params)
    nc_path            = run_summa(trial_dir, "best")

    if nc_path is None:
        print("Best trial SUMMA run failed."); return

    sim_daily          = load_summa_output(nc_path)
    obs_q, obs_swe     = load_observations()
    routed, lr_result  = apply_routing(sim_daily, obs_q)
    metrics            = compute_metrics(routed, obs_q, sim_daily)

    print(f"\nNSE={metrics['nse']:.3f}  KGE={metrics['kge']:.3f}  logNSE={metrics['log_nse']:.3f}")
    print(f"LR: k_fast={lr_result['k_fast']:.4f}  k_slow={lr_result['k_slow']:.2e}  f_fast={lr_result['f_fast']:.3f}")

    plot_diagnostics(sim_daily, routed, obs_q, obs_swe, best_params, metrics, lr_result,
                     SWEEP_DIR / "diagnostics_best.png")

    summary = {
        "best_params":  best_params,
        "best_metrics": metrics,
        "lr_params":    {k: lr_result[k] for k in ("k_fast", "k_slow", "f_fast")},
        "n_evals":      _trial_count[0],
    }
    with open(SWEEP_DIR / "best_params.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary saved: {SWEEP_DIR / 'best_params.json'}")


if __name__ == "__main__":
    main()
