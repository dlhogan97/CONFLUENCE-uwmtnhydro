#!/usr/bin/env python3
"""
run_calibration_aquifer.py — DE calibration of ONLY the aquifer + routing params for East River,
with the aquifer initialised full (10 m storage).

Everything else is FROZEN at the current best (East run 20260720_094546, objective 0.3369, KGE
0.897). The DE varies only four dimensions:

    aquiferBaseflowRate   (abs, log)   Qb = rate*(S/scale)^exp
    aquiferScaleFactor    (abs)        storage scale
    aquiferBaseflowExp    (abs)        recession non-linearity  (was PEGGED; re-opened here)
    routingGammaScale     (abs)        hillslope routing scale

Rationale (store-memory experiment): calibrate the aquifer's release behaviour under a CHARGED
initial aquifer (10 m), so the fitted recession/storage params describe a basin that actually
carries multi-year groundwater memory, then compare against the memory-free baseline.

Usage:
    python run_calibration_aquifer.py --workers 8            # real run (default popsize/maxiter below)
    python run_calibration_aquifer.py --workers 8 --quick    # tiny smoke test
    python run_calibration_aquifer.py --stage-check          # stage ONE eval, verify frozen/10m, exit
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
from datetime import datetime
from pathlib import Path

import numpy as np
import netCDF4 as nc
from scipy.optimize import differential_evolution

from calibration_runner import CalibrationRunner, ParamSpec, _routing_scale_bounds, SUMMA_EXE
from run_calibration import (BASIN_CFG, OUT_ROOT, git_sha, seed_vector,
                             save_generation_output)

DOMAIN = "East_River"
AQUIFER_INIT_STORAGE_M = 10.0

# Current best of the NON-calibrated params (East 20260720_094546, objective 0.3369, KGE 0.897).
# These are held FIXED for every trial; only the four aquifer/routing dims below move.
FROZEN = dict(k_soil_mult=2.49408, albedoDecayRate=409426.0,
              fpm_low=1.06303, fpm_delta=0.0275356, tempOffset=-1.34363)

# Seed (x0) for the four calibrated dims = their current best (Exp = the value it was pegged at).
BEST4 = dict(aquiferBaseflowRate=9.43986e-06, aquiferScaleFactor=2.65777,
             aquiferBaseflowExp=2.056814, routingGammaScale=51629.2)


class AquiferRunner(CalibrationRunner):
    """CalibrationRunner restricted to 4 aquifer/routing dims, everything else frozen at best,
    aquifer initialised to 10 m."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.specs = [
            ParamSpec("aquiferBaseflowRate", "abs", 1e-9, 1e-4, log=True),
            ParamSpec("aquiferScaleFactor",  "abs", 5.0, 30.0),
            ParamSpec("aquiferBaseflowExp",  "abs", 0.3, 4.0),
            ParamSpec("routingGammaScale",   "abs", *_routing_scale_bounds(16, 36)),
        ]
        # a-priori k_soil field (frozen k_soil = this field * k_soil_mult)
        with nc.Dataset(self.settings / "trialParams.nc") as d:
            self._base_ksoil = np.array(d["k_soil"][:]).copy()
        # pre-build the fixed-tempOffset forcing ONCE; trials (no forcing params) reuse it
        ff = self.work_root / "_fixed_forcing"
        self.forcing = self._apply_forcing(ff, {"tempOffset": ("forcing", FROZEN["tempOffset"])})
        print(f"  frozen forcing built (tempOffset={FROZEN['tempOffset']:+.3f}) -> {self.forcing}")

    def _apply_params(self, work, params):
        super()._apply_params(work, params)                 # writes the 4 aquifer/routing dims
        with nc.Dataset(work / "trialParams.nc", "a") as ds:
            ds["k_soil"][:] = self._base_ksoil * FROZEN["k_soil_mult"]
            ds["albedoDecayRate"][:] = FROZEN["albedoDecayRate"]
            ds["frozenPrecipMultip"][:] = FROZEN["fpm_low"] + FROZEN["fpm_delta"] * self.elev_norm

    def _wet_soil(self, work):
        super()._wet_soil(work)                             # field-capacity soil moisture
        with nc.Dataset(work / "coldState.nc", "a") as ds:
            ds["scalarAquiferStorage"][:] = AQUIFER_INIT_STORAGE_M


def stage_check(runner):
    """Stage one evaluation and read back trialParams / coldState / forcing to prove the frozen
    params, the four calibrated seeds, and the 10 m aquifer landed correctly."""
    work = runner.work_root / "_stagecheck"
    x0phys = BEST4
    x = seed_vector(runner, dict(x0phys))
    params = runner._vector_to_params(x)
    runner._stage(work); runner._apply_params(work, params); runner._wet_soil(work)
    with nc.Dataset(work / "trialParams.nc") as d:
        g = lambda v: np.unique(np.round(np.array(d[v][:]).ravel(), 8))
        print("\n  --- staged trialParams (frozen) ---")
        print("    k_soil          :", g("k_soil"), f"(= base * {FROZEN['k_soil_mult']})")
        print("    albedoDecayRate :", g("albedoDecayRate"))
        print("    frozenPrecipMult: [", round(float(np.array(d['frozenPrecipMultip'][:]).min()),4),
              "..", round(float(np.array(d['frozenPrecipMultip'][:]).max()),4), "] (ramp)")
        print("  --- staged trialParams (CALIBRATED, seeded at best) ---")
        for v in ["aquiferBaseflowRate","aquiferScaleFactor","aquiferBaseflowExp","routingGammaScale"]:
            print(f"    {v:20s}:", g(v))
    with nc.Dataset(work / "coldState.nc") as d:
        print("  --- coldState ---")
        print("    scalarAquiferStorage:", np.unique(np.round(np.array(d["scalarAquiferStorage"][:]).ravel(),4)), "m")
    import shutil; shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage-check", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUT_ROOT / "runs" / f"{DOMAIN}_aquifer10m_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = BASIN_CFG[DOMAIN]
    runner = AquiferRunner(domain=DOMAIN, work_root=str(outdir / "work"), **cfg)

    if args.stage_check:
        stage_check(runner); return
    if args.quick:
        args.popsize, args.maxiter, args.workers = 2, 1, 8

    manifest = dict(domain=DOMAIN, timestamp=ts, git_sha=git_sha(), experiment="aquifer10m",
                    aquifer_init_storage_m=AQUIFER_INIT_STORAGE_M, frozen=FROZEN, seed4=BEST4,
                    config=cfg, params=[s.name for s in runner.specs], bounds=runner.bounds,
                    de=dict(popsize=args.popsize, maxiter=args.maxiter, workers=args.workers, seed=args.seed))
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[{DOMAIN} aquifer10m] {len(runner.specs)} params, popsize={args.popsize} "
          f"maxiter={args.maxiter} workers={args.workers}  ->  {outdir}", flush=True)
    print(f"  window: sim {cfg['sim_start']}..{cfg['sim_end']}, analyze from {cfg['analysis_start']}")

    hist = []
    def cb(*a, **kw):
        try:
            ir = kw.get("intermediate_result")
            x, fun = (np.asarray(ir.x), float(ir.fun)) if ir is not None else (np.asarray(a[0]), None)
            hist.append(dict(t=time.time(), x=[float(v) for v in x], objective=fun))
            (outdir / "history.json").write_text(json.dumps(hist, indent=2, default=str))
            print(f"  gen {len(hist):2d}: " + (f"best objective={fun:.4f}" if fun is not None else "(best x)"), flush=True)
            try:
                save_generation_output(runner, x, len(hist), outdir)
            except Exception as e:
                print(f"       [gen-output warning] {type(e).__name__}: {e}", flush=True)
        except Exception as e:
            print(f"  [callback warning] {type(e).__name__}: {e}", flush=True)

    x0 = seed_vector(runner, dict(BEST4))
    de_kwargs = dict(bounds=runner.bounds, workers=args.workers, popsize=args.popsize,
                     maxiter=args.maxiter, seed=args.seed, updating="deferred", polish=False,
                     tol=0.01, callback=cb, disp=False)
    if x0 is not None:
        de_kwargs["x0"] = x0; print(f"  seeded x0 at current best: {BEST4}", flush=True)

    t0 = time.time()
    result = differential_evolution(runner.evaluate, **de_kwargs)
    dt = time.time() - t0
    best = runner.evaluate_full(result.x)
    (outdir / "result.json").write_text(json.dumps(dict(
        x=list(result.x), objective=float(result.fun), physical_params=best.get("params", {}),
        parts=best.get("parts", {}), diagnostics=best.get("diagnostics", {}),
        nfev=int(result.nfev), runtime_hr=dt/3600.0, success=bool(result.success)), indent=2, default=str))
    print(f"\nDONE ({dt/3600:.2f} h, {result.nfev} evals): objective={result.fun:.4f}")
    print("  best:", {k: (round(v,5) if abs(v)>1e-4 else f'{v:.3e}') for k,v in best.get("params",{}).items()})
    print("  diagnostics:", {k: round(v,3) for k,v in best.get("diagnostics",{}).items()})


if __name__ == "__main__":
    main()
