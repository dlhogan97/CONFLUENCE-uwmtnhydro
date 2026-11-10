#!/usr/bin/env python3
"""
run_calibration_east_routing.py -- East River no-storage CONTROL calibrated on ROUTING and TIMING only (2026-10-22).

Design (user, 2026-10-22):
  * Water volume is taken as right: precipitation (frozenPrecipMultip_low) and tempOffset are FROZEN at the previous control best
    (East_River_control_20261020_154016, objective 0.853). Calibrated instead: the FAST HRU aquifer buckets and hillslope routing,
    with k_soil and the snow-albedo floor/decay rate.
  * Objective = signature_objective_plus.signature_objective_routing: peak timing, peak volume, falling limb (days from peak to 25%),
    centre-of-mass timing, monthly bias over Apr-Oct (Nov-Mar excluded). NO low-flow and NO memory terms.
  * No deep store (aquiferLeakRate 0). The buckets must stay FAST so they cannot become a hidden multi-year store: trials whose
    effective bucket e-folding time (mean storage / mean baseflow, area-weighted, scoring window) exceeds TAU_MAX_D are rejected,
    as are trials whose annual-mean storage drifts by more than DRIFT_MAX_M over the window, and trials with KGE < KGE_MIN (0.80).
The deep-store TREATMENT then reuses these parameters unchanged and adds a linear leak (fraction f) and a linear deep store
(residence time tau_d), both fixed from observation constraints, not fitted (see the plan in the 2026-10-22 notes).

usage: python run_calibration_east_routing.py --workers 14 --popsize 6 --maxiter 15     (42 trials/generation = 3 full rounds)
       python run_calibration_east_routing.py --time-one
"""
from __future__ import annotations
import argparse, json, os, subprocess, time, uuid, shutil
from datetime import datetime
import numpy as np, pandas as pd, netCDF4 as nc
from scipy.optimize import differential_evolution
from calibration_runner import ParamSpec, FAIL_PENALTY, _routing_scale_bounds
from signature_objective import guard_runoff_ratio, guard_snow
import signature_objective_plus as P
import deepstore_build as DB
from run_calibration import OUT_ROOT, git_sha, seed_vector
import run_calibration_control as C

DOM = "East_River"
SPECS = [ParamSpec("k_soil", "mult", 0.1, 10.0, log=True), ParamSpec("albedoMinVisible", "abs", 0.20, 0.60),
         ParamSpec("albedoDecayRate", "abs", 3e5, 3e6, log=True),
         # aquifer bounds narrowed 2026-10-22 around the feasible region: in the first launch (bounds 1e-7..1e-4, 0.5..8, 1..6) 94% of trials failed the
         # bucket-time / drift / KGE guards (median bucket e-folding time 852 d vs the 120 d cap); the 4 accepted trials had exponent 1.2-2.1 and log10 rate -5.7..-4.5
         ParamSpec("aquiferBaseflowRate", "abs", 1.58e-6, 3.98e-5, log=True), ParamSpec("aquiferScaleFactor", "abs", 1.0, 8.0),
         ParamSpec("aquiferBaseflowExp", "abs", 1.0, 2.6), ParamSpec("routingGammaScale", "abs", *_routing_scale_bounds(16, 36))]
# frozen: water-volume terms at the previous control best + the stage-A forcing corrections
FROZEN = dict(frozenPrecipMultip_low=1.0358598, frozenPrecipMultip_delta=0.02754, tempOffset=1.6110816, tempLapse=-2.607, swScale=0.915, lwOffset=-5.0)
SEED = dict(k_soil=3.2959351, albedoMinVisible=0.4742087, albedoDecayRate=4.503868e5,              # previous control best (J 0.853)
            aquiferBaseflowRate=9.43986e-06, aquiferScaleFactor=2.65777, aquiferBaseflowExp=2.056814, routingGammaScale=51629.17)
TAU_MAX_D = 120.0       # d; the previous control best sits at 102 d -- still well short of the ~1 yr deep-store residence time
KGE_MIN = 0.80          # user, 2026-10-22: anything below this over the scoring window is rejected
DRIFT_MAX_M = 0.15


class RunnerRouting(C.RunnerControl):
    def __init__(self, **cfg):
        super().__init__(DOM, **cfg)
        self.specs = SPECS; self.frozen = {}
        self.apriori = dict(SEED)

    def _vector_to_params(self, x):
        p = {s.name: (s.kind, s.to_physical(xi)) for s, xi in zip(self.specs, x)}
        for k, v in FROZEN.items():
            p[k] = ("elev", v) if k.startswith("frozenPrecipMultip") else ("forcing", v)
        return p

    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing)
        with open(work/"outputControl.txt", "a") as fh: fh.write("scalarAquiferBaseflow | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n")

    def evaluate_full(self, x):
        work = self.work_root/f"w_{os.getpid()}_{uuid.uuid4().hex[:8]}"; t0 = time.time()
        try:
            params, proc = self.run_summa(x, work); out = work/"out"/"cal_timestep.nc"; pv = {k: v[1] for k, v in params.items()}
            if proc.returncode != 0 or not out.exists(): return {"objective": FAIL_PENALTY, "reason": "summa_failed", "params": pv}
            d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
            sel = (t >= self.analysis_start) & (t <= f"{self.sim_end} 23:00")
            q = pd.Series(np.array(d["averageRoutedRunoff"][:, 0]), index=t)
            swe = pd.DataFrame(np.array(d["scalarSWE"][:]), index=t, columns=self.hruid).resample("D").mean()[self.analysis_start:self.sim_end]
            S = np.array(d["scalarAquiferStorage_mean"][:]); Qb = np.array(d["scalarAquiferBaseflow_mean"][:]); d.close()
            sim = (q*self.area).resample("D").mean()[self.analysis_start:self.sim_end]
            sim_mm_yr = (q*86400*1000).resample("D").mean()[self.analysis_start:self.sim_end].sum()/(len(sim)/365.25)
            wy = (swe.index.year + (swe.index.month >= 10))
            peak_swe = float(pd.Series((swe.values*self.hru_wt[None, :]).sum(1), index=swe.index).groupby(wy).max().mean())
            Sb = (S[sel]*self.hru_wt[None, :]).sum(1); Qbb = (Qb[sel]*self.hru_wt[None, :]).sum(1)
            tau_d = float(Sb.mean()/max(Qbb.mean(), 1e-15)/86400.0)
            ann = pd.Series(Sb, index=t[sel]).groupby(t[sel].year + (t[sel].month >= 10)).mean()
            drift = float(np.polyfit(np.arange(len(ann)), ann.values, 1)[0]*(len(ann) - 1)) if len(ann) >= 4 else 0.0
            base = {"params": pv, "runoff_mm": float(sim_mm_yr), "peak_swe": peak_swe, "drift_m": drift, "tau_bucket_d": tau_d, "eval_s": time.time() - t0}
            if not guard_runoff_ratio(sim_mm_yr, self.precip_mm_yr): return {"objective": FAIL_PENALTY, "reason": "runoff_ratio", **base}
            if not guard_snow(peak_swe, self.peak_swe_apriori): return {"objective": FAIL_PENALTY, "reason": "snow_guard", **base}
            res = P.signature_objective_routing(sim, self.obs[self.analysis_start:self.sim_end])
            # GRADED penalty (FAIL_PENALTY + how far outside the limit): a flat penalty gave DE no direction in the first launch. Every trial that
            # violates a guard still scores above any accepted one (accepted objectives are <= 3).
            viol = {"storage_drift": max(abs(drift)/DRIFT_MAX_M - 1.0, 0.0), "slow_bucket": max(np.log(max(tau_d, 1e-9)/TAU_MAX_D), 0.0),
                    "kge_guard": max((KGE_MIN - res.diagnostics["KGE"])*10.0, 0.0)}
            bad = {k: v for k, v in viol.items() if v > 0}
            if bad: return {"objective": FAIL_PENALTY + sum(bad.values()), "reason": max(bad, key=bad.get), "violations": bad, "diagnostics": res.diagnostics, **base}
            return {"objective": res.objective, "parts": res.parts, "diagnostics": res.diagnostics, **base}
        except subprocess.TimeoutExpired: return {"objective": FAIL_PENALTY, "reason": "timeout"}
        except Exception as e: return {"objective": FAIL_PENALTY, "reason": f"exc:{type(e).__name__}:{e}"}
        finally: shutil.rmtree(work, ignore_errors=True)

    def _log_trial(self, x, r):
        logdir = self.work_root/"logs"; logdir.mkdir(exist_ok=True)
        rec = {"t": time.time(), "domain": DOM, "x": [float(v) for v in np.asarray(x)], "objective": r["objective"], "reason": r.get("reason", "ok"),
               **{k: r.get(k) for k in ("parts", "diagnostics", "params", "runoff_mm", "peak_swe", "drift_m", "tau_bucket_d", "eval_s", "violations")}}
        with open(logdir/f"trials_{os.getpid()}.jsonl", "a") as fh: fh.write(json.dumps(rec, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=7); ap.add_argument("--popsize", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=15); ap.add_argument("--seed", type=int, default=11); ap.add_argument("--time-one", action="store_true")
    a = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); outdir = OUT_ROOT/"runs"/f"{DOM}_routing_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    runner = RunnerRouting(work_root=str(outdir/"work"), **C.CFG[DOM]); x0 = seed_vector(runner, dict(SEED))
    if a.time_one:
        r = runner.evaluate_full(x0); runner._log_trial(x0, r)
        print(json.dumps({k: r.get(k) for k in ("objective", "reason", "parts", "diagnostics", "tau_bucket_d", "drift_m", "eval_s")}, indent=1, default=str)); return
    (outdir/"manifest.json").write_text(json.dumps(dict(domain=DOM, ts=ts, git=git_sha(), cfg=C.CFG[DOM], exe=DB.SUMMA_DS, free=[s.name for s in SPECS],
        bounds=runner.bounds, frozen=FROZEN, seed=SEED, tau_max_d=TAU_MAX_D, kge_min=KGE_MIN, drift_max_m=DRIFT_MAX_M, weights=P.DEFAULT_WEIGHTS_ROUTING,
        bias_months=P.ROUTING_BIAS_MONTHS, de=vars(a), objective="signature_objective_routing"), indent=2, default=str))
    print(f"[{DOM} routing] {len(SPECS)} free params popsize={a.popsize} maxiter={a.maxiter} workers={a.workers} -> {outdir}", flush=True); hist = []
    def cb(intermediate_result):
        hist.append(dict(t=time.time(), x=[float(v) for v in intermediate_result.x], objective=float(intermediate_result.fun)))
        (outdir/"history.json").write_text(json.dumps(hist, indent=2)); print(f"  gen {len(hist):2d}: best objective={intermediate_result.fun:.4f}", flush=True)
    res = differential_evolution(runner.evaluate, bounds=runner.bounds, workers=a.workers, popsize=a.popsize, maxiter=a.maxiter, seed=a.seed,
                                 updating="deferred", polish=False, tol=0.005, callback=cb, disp=False, x0=x0)
    (outdir/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev)), indent=2))
    print("DONE", res.fun, flush=True)


if __name__ == "__main__":
    main()
