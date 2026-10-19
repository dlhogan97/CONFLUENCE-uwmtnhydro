#!/usr/bin/env python3
"""
run_calibration_stageA.py -- East River STAGE-A recalibration (snow / energy / soil), rebuilt after the stage-B emulator test showed rounds 1-2 were not equilibrium-consistent.

What is different from run_calibration_v2.py
  * Aquifer + routing are FROZEN at the old calibrated values (rate 9.44e-6, scale 2.658, exp 2.057, routing 51629 s). That aquifer equilibrates in ~2 months (tau = S/(n*R) ~ 62 d), so
    the stage-A trajectory is equilibrium-consistent and the emulator (stage B) is valid afterwards.
  * The aquifer starts at its equilibrium storage (0.08 m), not 0.15 m.
  * A STORAGE-DRIFT GUARD rejects trials whose area-weighted annual-mean aquifer storage trends by more than 0.15 m over the scoring window (water parked in storage is not streamflow).
  * Only 4 free parameters (the ones that showed a relationship with the timing/volume terms): tempOffset, albedoDecayRate, k_soil, frozenPrecipMultip_low.
    Fixed: lapse at the measured value (no rotation), precip-ramp delta at the old 0.0275, soft forcing corrections at HALF the SAIL point-measurement bias (swScale 0.915, lwOffset -5 W/m2).
  * Objective = STAGE-A features only: centre-of-mass timing, peak timing, rising-limb timing (day of 10% and 25% of Mar-Jul volume), total volume, + soft Butte SWE terms (0.5 each).
    Not scored here (stage B / reported only): monthly volume, low flows, recession, peak-flow magnitude.
Window WY2009-2020 (spin-up from 2005-10-01); WY2021-2025 held out.
usage: python run_calibration_stageA.py [--time-one] [--workers 20] [--popsize 5] [--maxiter 20]
"""
from __future__ import annotations
import argparse, json, os, subprocess, time, uuid, shutil
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc
from scipy.optimize import differential_evolution
from calibration_runner import ParamSpec, SUMMA_EXE, FAIL_PENALTY
from signature_objective import guard_runoff_ratio, guard_snow, guard_lowflow, min7_median
import signature_objective_plus as P
from run_calibration import OUT_ROOT, git_sha, seed_vector
from run_calibration_v2 import RunnerV2, CFG

DOM = "East_River"
SPECS_A = [ParamSpec("frozenPrecipMultip_low", "elev", 1.00, 1.20), ParamSpec("tempOffset", "forcing", -4.0, 2.0),
           ParamSpec("albedoDecayRate", "abs", 1e5, 5e6, log=True), ParamSpec("k_soil", "mult", 0.1, 10.0, log=True)]
FROZEN = dict(frozenPrecipMultip_delta=0.02754, aquiferBaseflowRate=9.43986e-06, aquiferScaleFactor=2.65777, aquiferBaseflowExp=2.056814, routingGammaScale=51629.17,
              tempLapse=-2.607, swScale=0.915, lwOffset=-5.0)
AQ_INIT_M = 0.08
SEED_A = dict(frozenPrecipMultip_low=1.0630, tempOffset=-1.3436, albedoDecayRate=409426.0, k_soil=2.4941)
W_A = {"com_timing": 1.0, "monthly_volume": 0.0, "total_volume": 1.0, "peak_flow": 0.0, "peak_timing": 1.0, "lowflow_sepnov": 0.0, "recession": 0.0, "swe_peak": 0.5, "swe_melt": 0.5}
W_RISE = 1.0
DRIFT_MAX_M = 0.15


def _doy_frac(g, y, months, frac):
    s = g[g.index.month.isin(months)]; c = s.cumsum()/s.sum()
    return float((c.index[np.argmax(c.values >= frac)] - pd.Timestamp(int(y)-1, 10, 1)).days)
def sig_rising_limb(sim, obs, norm_days=7.0):
    """mean over water years and over the 10% / 25% Mar-Jul-volume dates of |sim date - obs date| / 7 d, clipped to [0, 3]."""
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna(); j = j.assign(wy=j.index.year+(j.index.month >= 10)); errs = []
    for y, g in j.groupby("wy"):
        if len(g) < 360: continue
        for fr in (0.10, 0.25): errs.append(abs(_doy_frac(g.s, y, [3, 4, 5, 6, 7], fr) - _doy_frac(g.o, y, [3, 4, 5, 6, 7], fr)))
    return 3.0 if not errs else float(np.clip(np.mean(errs)/norm_days, 0, 3))


class RunnerA(RunnerV2):
    def __init__(self, **cfg):
        super().__init__(DOM, **cfg); self.specs = SPECS_A
        with xr.open_dataset(self.settings/"trialParams.nc") as tp:
            self.base_vals = {s.name: tp[s.name].values.copy() for s in self.specs if s.kind == "mult" and s.name in tp}
            self.apriori = {s.name: float(np.ravel(tp[s.name].values)[0]) for s in self.specs if s.name in tp}
        for s in self.specs:
            if s.kind == "forcing" and s.name not in self.apriori: self.apriori[s.name] = 0.0

    def _vector_to_params(self, x):
        p = super()._vector_to_params(x)
        for k in ("tempLapse", "swScale", "lwOffset"): p[k] = ("forcing", FROZEN[k])
        p["frozenPrecipMultip_delta"] = ("elev", FROZEN["frozenPrecipMultip_delta"])          # the base class builds the ramp from the _low/_delta pair
        return p
    def _apply_params(self, work, params):
        super()._apply_params(work, params)
        with nc.Dataset(work/"trialParams.nc", "a") as ds:
            lo = params["frozenPrecipMultip_low"][1]; ds["frozenPrecipMultip"][:] = lo + FROZEN["frozenPrecipMultip_delta"]*self.elev_norm
            for k in ("aquiferBaseflowRate", "aquiferScaleFactor", "aquiferBaseflowExp", "routingGammaScale"): ds[k][:] = FROZEN[k]
    def _wet_soil(self, work):
        super()._wet_soil(work)
        with nc.Dataset(work/"coldState.nc", "a") as ds: ds["scalarAquiferStorage"][:] = AQ_INIT_M
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId                | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\naverageRoutedRunoff  | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
                                              "scalarSWE            | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\nscalarAquiferStorage | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n")
    def _log_trial(self, x, r):
        logdir = self.work_root/"logs"; logdir.mkdir(exist_ok=True)
        rec = {"t": time.time(), "domain": DOM, "x": [float(v) for v in np.asarray(x)], "objective": r["objective"], "reason": r.get("reason", "ok"), "parts": r.get("parts", {}),
               "diagnostics": r.get("diagnostics", {}), "params": r.get("params", {}), "runoff_mm": r.get("runoff_mm"), "peak_swe": r.get("peak_swe"), "aso_eval": r.get("aso_eval"),
               "drift_m": r.get("drift_m"), "aq_mean_m": r.get("aq_mean_m")}
        with open(logdir/f"trials_{os.getpid()}.jsonl", "a") as fh: fh.write(json.dumps(rec, default=str)+"\n")

    def evaluate_full(self, x):
        work = self.work_root/f"w_{os.getpid()}_{uuid.uuid4().hex[:8]}"; prefix = "cal"
        try:
            self._stage(work); params = self._vector_to_params(np.asarray(x)); self._apply_params(work, params); self._wet_soil(work)
            fdir = self._apply_forcing(work, params); self._write_filemanager(work, prefix, forcing=fdir)
            t0 = time.time(); env = {**os.environ, "OMP_NUM_THREADS": "1"}
            proc = subprocess.run([SUMMA_EXE, "-m", str(work/"fileManager.txt")], capture_output=True, text=True, timeout=4800, env=env)
            out = work/"out"/f"{prefix}_timestep.nc"
            if proc.returncode != 0 or not out.exists(): return {"objective": FAIL_PENALTY, "reason": "summa_failed", "params": {k: v[1] for k, v in params.items()}}
            ds = xr.open_dataset(out); t = pd.DatetimeIndex(ds["time"].values); q_all = pd.Series(ds["averageRoutedRunoff"].isel(gru=0).values, index=t)
            sim = (q_all*self.area).resample("D").mean()[self.analysis_start:self.sim_end]
            swe_hru = pd.DataFrame(ds["scalarSWE"].values, index=t, columns=self.hruid).resample("D").mean()[self.analysis_start:self.sim_end]
            aq = pd.Series((ds["scalarAquiferStorage_mean"].values*self.hru_wt[None, :]).sum(1), index=t).resample("D").mean()[self.analysis_start:self.sim_end]
            peak_swe = pd.Series((swe_hru.values*self.hru_wt[None, :]).sum(1), index=swe_hru.index).groupby(swe_hru.index.year+(swe_hru.index.month >= 10).astype(int)).max().mean()
            sim_mm_yr = (q_all*86400*1000).resample("D").mean()[self.analysis_start:self.sim_end].sum()/(len(sim)/365.25); ds.close()
            ann = aq.groupby(aq.index.year+(aq.index.month >= 10)).mean(); ann = ann[aq.groupby(aq.index.year+(aq.index.month >= 10)).size() >= 360]
            drift = float(np.polyfit(np.arange(len(ann)), ann.values, 1)[0]*(len(ann)-1)) if len(ann) >= 4 else 0.0
            base = {"runoff_mm": sim_mm_yr, "peak_swe": float(peak_swe), "drift_m": drift, "aq_mean_m": float(aq.mean())}
            if not guard_runoff_ratio(sim_mm_yr, self.precip_mm_yr): return {"objective": FAIL_PENALTY, "reason": "runoff_ratio", **base}
            if not guard_snow(float(peak_swe), self.peak_swe_apriori): return {"objective": FAIL_PENALTY, "reason": "snow_guard", **base}
            if abs(drift) > DRIFT_MAX_M: return {"objective": FAIL_PENALTY, "reason": "storage_drift", **base}
            if not guard_lowflow(min7_median(sim), min7_median(self.obs)): return {"objective": FAIL_PENALTY, "reason": "lowflow_dry", **base}
            swe = P.swe_signatures(swe_hru)
            try: aso_eval = P.aso_band_evaluation(swe_hru, self.aso_df, self.hru_area, self.hru_band, self.band_order)
            except Exception as e: aso_eval = {"error": f"{type(e).__name__}: {e}"}
            res = P.signature_objective_plus(sim, self.obs, self.tau_target, weights=W_A, area_m2=self.area, swe=swe)
            wsum = sum(W_A[k] for k in res.parts); rl = sig_rising_limb(sim, self.obs); parts = dict(res.parts); parts["rising_limb"] = rl
            J = (res.objective*wsum + W_RISE*rl)/(wsum + W_RISE)
            return {"objective": float(J), "parts": parts, "diagnostics": res.diagnostics, "params": {k: v[1] for k, v in params.items()}, "eval_s": time.time()-t0, "aso_eval": aso_eval, **base}
        except subprocess.TimeoutExpired: return {"objective": FAIL_PENALTY, "reason": "timeout"}
        except Exception as e: return {"objective": FAIL_PENALTY, "reason": f"exc:{type(e).__name__}:{e}"}
        finally: shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--popsize", type=int, default=5); ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--time-one", action="store_true"); a = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); outdir = OUT_ROOT/"runs"/f"{DOM}_stageA_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    runner = RunnerA(work_root=str(outdir/"work"), **CFG[DOM]); x0 = seed_vector(runner, dict(SEED_A))
    if a.time_one:
        r = runner.evaluate_full(x0); print(json.dumps({k: r.get(k) for k in ("objective", "reason", "parts", "eval_s", "drift_m", "aq_mean_m", "runoff_mm", "peak_swe")}, indent=1, default=str)); return
    (outdir/"manifest.json").write_text(json.dumps(dict(domain=DOM, ts=ts, git=git_sha(), cfg=CFG[DOM], free=[s.name for s in runner.specs], bounds=runner.bounds, frozen=FROZEN, aq_init_m=AQ_INIT_M,
                                                         weights=W_A, rising_limb_weight=W_RISE, drift_max_m=DRIFT_MAX_M, seed=SEED_A, de=vars(a)), indent=2, default=str))
    print(f"[{DOM} stageA] {len(runner.specs)} free params popsize={a.popsize} maxiter={a.maxiter} workers={a.workers} -> {outdir}", flush=True); hist = []
    def cb(*aa, **kw):
        try:
            ir = kw.get("intermediate_result"); xx, fun = (np.asarray(ir.x), float(ir.fun)) if ir is not None else (np.asarray(aa[0]), None)
            hist.append(dict(t=time.time(), x=[float(v) for v in xx], objective=fun)); (outdir/"history.json").write_text(json.dumps(hist, indent=2)); print(f"  gen {len(hist):2d}", flush=True)
        except Exception as e: print("  [cb warning]", e, flush=True)
    res = differential_evolution(runner.evaluate, bounds=runner.bounds, workers=a.workers, popsize=a.popsize, maxiter=a.maxiter, seed=a.seed, updating="deferred", polish=False, tol=0.005, callback=cb, disp=False, x0=x0)
    best = runner.evaluate_full(res.x); (outdir/"result.json").write_text(json.dumps(dict(x=list(res.x), objective=float(res.fun), params=best.get("params"), parts=best.get("parts"), nfev=int(res.nfev)), indent=2, default=str)); print("DONE", res.fun)

if __name__ == "__main__":
    main()
