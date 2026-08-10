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

from calibration_runner import CalibrationRunner, PARAM_SPECS, SUMMA_EXE

# Rich outputControl for the per-generation snapshot runs (variable names + flags copied from
# the domain's own outputControl.txt, so they are known to run). Enough to reproduce every
# diagnostic -- hydrograph, ET, storage, soil moisture, runoff partition -- as the fit evolves.
GEN_OUTPUT_CONTROL = (
    "hruId                     | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n"
    "pptrate                   | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
    "scalarSWE                 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
    "scalarTotalET             | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
    "scalarSnowSublimation     | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
    "scalarSurfaceRunoff       | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
    "scalarAquiferStorage      | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n"
    "scalarAquiferBaseflow     | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
    "scalarTotalSoilLiq        | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
    "averageRoutedRunoff       | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n")


def save_generation_output(runner, x, gen, outdir):
    """Snapshot the generation's best member: stage it, then launch SUMMA DETACHED and nice'd
    so it writes full output while the calibration keeps running. Non-blocking; failures here
    must never touch the optimisation, so the caller wraps this in try/except.

    Returns immediately -- the netCDF appears in gen_NN/run/out/ when SUMMA finishes (~20 min).
    """
    dest = Path(outdir) / "generations" / f"gen_{gen:02d}"
    work = dest / "run"
    runner._stage(work)
    _pv = runner._vector_to_params(np.asarray(x))
    runner._apply_params(work, _pv)
    runner._wet_soil(work)
    # snapshots must use the SAME per-trial bias-corrected forcing the trial was scored with,
    # or the saved output will not reproduce the objective it is meant to illustrate
    _fdir = runner._apply_forcing(work, _pv)
    runner._write_filemanager(work, "gen", forcing=_fdir)
    (work / "outputControl.txt").write_text(GEN_OUTPUT_CONTROL)   # override the minimal one
    (dest / "params.json").write_text(json.dumps(
        {k: v[1] for k, v in runner._vector_to_params(np.asarray(x)).items()}, indent=2))
    logf = open(dest / "summa.log", "w")
    subprocess.Popen(["nice", "-n", "19", SUMMA_EXE, "-m", str(work / "fileManager.txt")],
                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "OMP_NUM_THREADS": "1"})

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


def migrate_seed(seed_physical, specs, apriori=None):
    """Carry an older seed forward onto the current parameter spec.

    Params keep their meaning across spec changes, so a seed from a previous objective is
    still a good STARTING POINT -- only the names/parameterisation move. Handles:
      * uniform `frozenPrecipMultip` -> elevation ramp (`_low`, `_delta`)
          The ramp is fpm = low + delta*znorm with mean(znorm) ~ 0.5, so low = V - delta/2
          preserves the area-weighted mean (i.e. the same total water) while engaging the ramp.
      * params ADDED to the spec since the seed was written (e.g. kAnisotropic) -> seed them at
          their a-priori value ('mult' params at 1.0, i.e. the unscaled a-priori field). Without
          this a single added param makes the whole seed unusable and the run starts cold.
      * params DROPPED from the spec (e.g. aquiferBaseflowExp under qTopmodl) -> discarded.
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
    for spec in specs:                          # fill params added since the seed was written
        if spec.name in s:
            continue
        fill = 1.0 if spec.kind == "mult" else (apriori or {}).get(spec.name)
        if fill is None:
            continue                            # nothing sensible to fill -> seed_vector warns
        s[spec.name] = fill
        print(f"  seed migrated: {spec.name} not in seed -> a-priori {fill:g}", flush=True)
    dropped = [k for k in s if k not in names]
    for k in dropped:
        s.pop(k)
    if dropped:
        print(f"  seed migrated: dropped {dropped} (no longer in spec)", flush=True)
    return s


def seed_vector(runner, seed_physical):
    """Physical parameter dict -> optimized coordinate vector (x0), clipped to bounds.
    Returns None if the seed cannot be used -- and says WHY (a silent skip costs the warm
    start without anyone noticing)."""
    if not seed_physical:
        return None
    seed_physical = migrate_seed(seed_physical, runner.specs, getattr(runner, "apriori", None))
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
        # Clip just INSIDE the bounds, not onto them. scipy's differential_evolution rescales
        # x0 to [0,1] as (x - mid)/range + 0.5; a value sitting exactly on a bound can land at
        # -4e-16 through floating point and trip its "entries in x0 lay outside the specified
        # bounds" check. A 1e-6 inset of the range is physically negligible and removes it.
        pad = 1e-6 * (hi - lo)
        xc = float(np.clip(xi, lo + pad, hi - pad))
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
            # snapshot this generation's best with full output (detached, non-blocking)
            try:
                save_generation_output(runner, x, len(hist), outdir)
                print(f"       -> gen_{len(hist):02d} full-output run launched (detached)", flush=True)
            except Exception as e:
                print(f"       [gen-output warning] {type(e).__name__}: {e}", flush=True)
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
