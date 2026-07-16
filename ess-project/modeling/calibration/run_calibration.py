#!/usr/bin/env python3
"""
run_calibration.py — Differential-evolution driver for the signature calibration.

Calibrates one basin to the aggregate-signature objective (see CALIBRATION_SPEC.md),
20-way parallel, with a per-run manifest and per-trial logs.  Produces the memory-free
baseline parameter set; experiments run on top of it afterward.

Usage
-----
    python run_calibration.py East_River   [--quick]
    python run_calibration.py Tuolumne_River

`--quick` runs a tiny DE (parallelism smoke test), not a real calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from calibration_runner import CalibrationRunner, PARAM_SPECS

# All calibration OUTPUT (run dirs, worker scratch, trial logs, seeds) lives on /scratch --
# never in the repo. Only code + spec live in git. Override with $CALIB_OUT if needed.
OUT_ROOT = Path(os.environ.get(
    "CALIB_OUT", "/scratch/dlhogan/ess-project-data/calibration"))

# per-basin run configuration.
# Calibrate WY1993-2002 (10 yr), spin-up WY1990-1992 (3 yr, dry -> drained baseline).
# Chosen for a wet/dry/average mix in BOTH basins (see CALIBRATION_SPEC.md forcing table).
# Deliberately BEFORE the WY2013-2024 window reserved for the IC experiments + evaluation.
BASIN_CFG = {
    "East_River": dict(
        sim_start="1989-10-01", sim_end="2002-09-30", analysis_start="1992-10-01",
        obs_csv="/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/"
                "observations/streamflow/preprocessed/East_River_USGS09112500_dailyQ_WY1970_2025.csv",
        precip_mm_yr=942.0, peak_swe_apriori=461.0),
    # Tuolumne: SAME window as East (WY1993-2002). Pre-2007 is scaled-HH, which is consistent
    # with the long-record lag analysis (also merged/scaled) and keeps the manuscript simple.
    "Tuolumne_River": dict(
        sim_start="1989-10-01", sim_end="2002-09-30", analysis_start="1992-10-01",
        obs_csv="/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/"
                "observations/streamflow/preprocessed/Tuolumne_River_merged_dailyQ_WY1971_2025.csv",
        precip_mm_yr=1143.0, peak_swe_apriori=772.0),
}


# Optional seed: a good physical parameter set to inject as DE's x0 (replaces the best
# member of the initial population). East seed = best from the first exploratory generation
# (objective 0.216, KGE 0.76, recession 23 d).
# East seed: best from the 8-worker run (objective 0.1224, tau 25.9 d, KGE 0.73), with
# frozenPrecipMultip recast as an elevation ramp. The old uniform best was 1.249; low=1.15 +
# delta=0.20 keeps roughly the same area-weighted mean (~1.25 => same total water) while
# redistributing it toward the high elevations, so DE starts at a known-good volume with the
# ramp already engaged.
SEEDS = {
    "East_River": {
        "k_soil": 0.396532, "qSurfScale": 3.70585, "zScale_TOPMODEL": 1.40251,
        "aquiferBaseflowRate": 1.38586e-07, "aquiferScaleFactor": 1.13114,
        "aquiferBaseflowExp": 2.94102,
        "frozenPrecipMultip_low": 1.15, "frozenPrecipMultip_delta": 0.20,
        "routingGammaScale": 45789.2,
    },
}


def load_seed(domain):
    """Seed precedence: seeds/<domain>.json -> hardcoded SEEDS.

    The box reboots uncleanly (lightning, no UPS) and a run can be lost mid-flight. Before
    relaunching, checkpoint the best point found so far with

        python best_so_far.py <domain> --write-seed

    which writes seeds/<domain>.json; this then warm-starts from it, so only the evals since
    the checkpoint are lost. (This used to be automated by a supervisor daemon. It misread
    driver liveness, relaunched 12x on top of itself, and put 200 SUMMA on 24 cores -- every
    trial then timed out. Removed 2026-07-16: a wrong restart costs far more than a manual one.)
    """
    p = OUT_ROOT / "seeds" / f"{domain}.json"
    if p.exists():
        try:
            s = json.loads(p.read_text())
            print(f"  seed loaded from {p.name}")
            return s
        except Exception as e:
            print(f"  [seed warning] {p.name} unreadable ({e}); falling back to SEEDS")
    return SEEDS.get(domain, {})


def migrate_seed(seed_physical, specs):
    """Carry an older seed forward onto the current parameter spec.

    Params keep their meaning across spec changes, so a seed from a previous objective is
    still a good STARTING POINT -- only the names/parameterisation move. Currently handles:
      uniform `frozenPrecipMultip` -> elevation ramp (`_low`, `_delta`)
        The ramp is fpm = low + delta*znorm with mean(znorm) ~ 0.5, so low = V - delta/2
        preserves the area-weighted mean (i.e. the same total water) while engaging the ramp.
    """
    names = {s.name for s in specs}
    s = dict(seed_physical)
    if "frozenPrecipMultip_low" in names and "frozenPrecipMultip" in s:
        v = s.pop("frozenPrecipMultip")
        delta = s.get("frozenPrecipMultip_delta", 0.20)
        s["frozenPrecipMultip_low"] = v - delta / 2.0
        s["frozenPrecipMultip_delta"] = delta
        print(f"  seed migrated: frozenPrecipMultip {v:.3f} (uniform) -> "
              f"low={s['frozenPrecipMultip_low']:.3f} + delta={delta:.2f} "
              f"(same area-weighted mean)", flush=True)
    s.pop("frozenPrecipMultip", None)          # drop anything the spec no longer uses
    return s


def seed_vector(runner, seed_physical):
    """Physical parameter dict -> optimized coordinate vector (x0), clipped to bounds.
    Returns None if the seed cannot be used -- and says WHY (a silent skip costs the warm
    start without anyone noticing)."""
    if not seed_physical:
        return None
    seed_physical = migrate_seed(seed_physical, runner.specs)
    missing = [s.name for s in runner.specs if s.name not in seed_physical]
    if missing:
        print(f"  [seed WARNING] unusable — missing {missing}; starting COLD (no warm start)",
              flush=True)
        return None
    x = []
    for s in runner.specs:
        v = seed_physical[s.name]
        xi = np.log10(v) if s.log else v
        lo, hi = s.bounds
        xc = float(np.clip(xi, lo, hi))
        if abs(xc - xi) > 1e-9:
            print(f"  [seed] {s.name} clipped to bounds", flush=True)
        x.append(xc)
    return np.array(x)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=list(PARAM_SPECS))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUT_ROOT / "runs" / f"{args.domain}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = BASIN_CFG[args.domain]

    runner = CalibrationRunner(domain=args.domain, work_root=str(outdir / "work"), **cfg)
    if args.quick:
        args.popsize, args.maxiter, args.workers = 2, 1, 8

    manifest = dict(domain=args.domain, timestamp=ts, git_sha=git_sha(),
                    config=cfg, params=[s.name for s in runner.specs],
                    bounds=runner.bounds, tau_target=runner.tau_target,
                    de=dict(popsize=args.popsize, maxiter=args.maxiter,
                            workers=args.workers, seed=args.seed))
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[{args.domain}] {len(runner.specs)} params, "
          f"popsize={args.popsize} maxiter={args.maxiter} workers={args.workers}", flush=True)
    print(f"  window: sim {cfg['sim_start']}..{cfg['sim_end']}, "
          f"analyze from {cfg['analysis_start']}  ->  {outdir}")

    hist = []

    def cb(*args, **kwargs):
        """Generation callback. scipy >=1.14 calls callback(intermediate_result=OptimizeResult);
        older scipy calls callback(xk, convergence=val). Accept both, and never let a callback
        error kill the optimisation (the trial logs are the real record)."""
        try:
            ir = kwargs.get("intermediate_result")
            if ir is not None:
                x, fun = np.asarray(ir.x), float(ir.fun)
            else:
                x, fun = np.asarray(args[0]), None
            rec = dict(t=time.time(), x=[float(v) for v in x], objective=fun)
            hist.append(rec)
            (outdir / "history.json").write_text(json.dumps(hist, indent=2, default=str))
            msg = f"  gen {len(hist):2d}:"
            msg += f" best objective={fun:.4f}" if fun is not None else " (best x recorded)"
            print(msg, flush=True)
        except Exception as e:                      # never abort the run over logging
            print(f"  [callback warning] {type(e).__name__}: {e}", flush=True)

    de_kwargs = dict(bounds=runner.bounds, workers=args.workers,
                     popsize=args.popsize, maxiter=args.maxiter, seed=args.seed,
                     updating="deferred", polish=False, tol=0.01, callback=cb, disp=False)
    seed = load_seed(args.domain)
    x0 = seed_vector(runner, seed)
    if x0 is not None:
        de_kwargs["x0"] = x0
        print(f"  seeded x0 (physical): {seed}", flush=True)

    t0 = time.time()
    result = differential_evolution(runner.evaluate, **de_kwargs)
    dt = time.time() - t0

    best = runner.evaluate_full(result.x)
    out = dict(x=list(result.x), objective=float(result.fun),
               physical_params=best.get("params", {}),
               parts=best.get("parts", {}), diagnostics=best.get("diagnostics", {}),
               nfev=int(result.nfev), runtime_hr=dt / 3600.0, success=bool(result.success))
    (outdir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nDONE ({dt/3600:.2f} h, {result.nfev} evals): objective={result.fun:.3f}")
    print("  best params:", {k: round(v, 5) if abs(v) > 1e-4 else f"{v:.2e}"
                             for k, v in best.get("params", {}).items()})
    print("  diagnostics:", {k: round(v, 2) for k, v in best.get("diagnostics", {}).items()})


if __name__ == "__main__":
    main()
