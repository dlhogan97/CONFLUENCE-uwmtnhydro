#!/usr/bin/env python3
"""
run_calibration_control.py -- the NO-STORAGE CONTROL for the memory experiments (both basins), 2026-10-20.

The control is fitted on within-year melt-season behaviour only; the storage treatments (East deep store, Tuolumne perennial snow) are added
afterwards and judged on the lag-1 autocorrelation of low-flow residuals (low flow regressed on same-year peak). Differences from run_calibration_v2 /
run_calibration_stageA:
  * Build: /home/dlhogan/summa_deepstore (wrapper ~/bin/summa_ds), so control and treatments share one executable. Leak off.
  * Canopy gaps: forest HRUs get canopyGapFrac = 1 - tree fraction (fixed prior, deepstore_build.gap_prior). Without it, forest HRUs (VAI 5.5-6.3
    under Beer's law) transmitted ~2% of SW and melted out 40-90 d after open HRUs at the same elevation.
  * Albedo: the varDecay FLOOR (albedoMinVisible, both basins [0.20, 0.60]; a spring floor above ~0.6 is implausible) AND albedoDecayRate are both
    calibrated. The decay rate is bounded to physically reasonable values, [3e5, 3e6] s (melting-surface e-folding ~1.5-15 d; SUMMA default 1e6 ~5 d):
    the previous calibrated rates (Tuolumne 6.7e4, East 1.65e5 s -> 9-20 h) were compensating for a floor stuck at a bright value, and with both
    free but the rate unbounded they would trade off against each other. albedoMinNearIR stays 0.30 in both basins.
  * Objective: signature_objective_plus.signature_objective_control -- melt-window terms only; lowflow/recession/absolute peak are reported.
  * East keeps the stage-A design: aquifer + routing frozen at the fast (~62 d) calibrated values, aquifer starts at equilibrium, storage-drift guard.
    A slow aquifer is itself a multi-year store and would contaminate the control (r2's released ~145 mm/yr regardless of year).
  * Tuolumne: kAnisotropic and routing frozen at the r2 best (they act mainly on the late season, which is no longer scored).
Window WY2009-2020 (spin-up from 2005-10-01); WY2021-2025 held out.

usage: python run_calibration_control.py East_River     --workers 10 --popsize 5 --maxiter 20
       python run_calibration_control.py Tuolumne_River --workers 13 --popsize 5 --maxiter 20
       python run_calibration_control.py <domain> --time-one          # one evaluation of the seed
"""
from __future__ import annotations
import argparse, json, os, subprocess, time, uuid, shutil
from datetime import datetime
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc
from scipy.optimize import differential_evolution
from calibration_runner import ParamSpec, FAIL_PENALTY, TAU_TARGET
from signature_objective import guard_runoff_ratio, guard_snow, guard_lowflow, min7_median
import signature_objective_plus as P
import deepstore_build as DB
from run_calibration import OUT_ROOT, git_sha, seed_vector
from run_calibration_v2 import RunnerV2, CFG

MELT = {"East_River": (3, 4, 5, 6, 7), "Tuolumne_River": (4, 5, 6, 7)}
EXCLUDE = {"East_River": (), "Tuolumne_River": (12, 1, 2, 3)}       # forcing-limited months, now excluded from EVERY term
SPECS = {
    "East_River": [ParamSpec("frozenPrecipMultip_low", "elev", 1.00, 1.20), ParamSpec("tempOffset", "forcing", -4.0, 2.0),
                   ParamSpec("k_soil", "mult", 0.1, 10.0, log=True), ParamSpec("albedoMinVisible", "abs", 0.20, 0.60),
                   ParamSpec("albedoDecayRate", "abs", 3e5, 3e6, log=True)],
    "Tuolumne_River": [ParamSpec("frozenPrecipMultip_low", "elev", 0.85, 1.20), ParamSpec("frozenPrecipMultip_delta", "elev", 0.0, 0.60),
                       ParamSpec("tempLapse", "forcing", -10.0, -2.0), ParamSpec("tempOffset", "forcing", -3.0, 3.0),
                       ParamSpec("swScale", "forcing", 1.00, 1.25), ParamSpec("albedoMinVisible", "abs", 0.20, 0.60),
                       ParamSpec("k_soil", "mult", 0.1, 10.0, log=True), ParamSpec("qSurfScale", "abs", 1.0, 50.0, log=True),
                       ParamSpec("albedoDecayRate", "abs", 3e5, 3e6, log=True)],
}
# frozen values: East = stage-A design (run East_River_stageA_20261019_142114); Tuolumne = r2 best (Tuolumne_River_r2_20261015_110603)
FROZEN = {
    "East_River": dict(frozenPrecipMultip_delta=0.02754, tempLapse=-2.607, swScale=0.915, lwOffset=-5.0,
                       aquiferBaseflowRate=9.43986e-06, aquiferScaleFactor=2.65777, aquiferBaseflowExp=2.056814, routingGammaScale=51629.17),
    "Tuolumne_River": dict(kAnisotropic=6.104, routingGammaScale=33160.0),
}
FORCING_KEYS = {"tempLapse", "tempOffset", "swScale", "lwOffset"}
SEED = {   # albedoDecayRate seeded at the SUMMA default 1e6 s in both basins
    # East: best trial (J 0.828) of the superseded control run East_River_control_20261020_131053 (decay rate frozen at 1.65e5 there)
    "East_River": dict(frozenPrecipMultip_low=1.09305, tempOffset=-1.88606, k_soil=0.277456, albedoMinVisible=0.536044, albedoDecayRate=1.0e6),
    # Tuolumne: r2 best, albedoMinVisible 0.75 clipped to the 0.60 ceiling
    "Tuolumne_River": dict(frozenPrecipMultip_low=0.9651, frozenPrecipMultip_delta=0.03934, tempLapse=-4.785, tempOffset=1.798, swScale=1.025,
                           albedoMinVisible=0.60, k_soil=2.449, qSurfScale=8.216, albedoDecayRate=1.0e6),
}
AQ_INIT_M = {"East_River": 0.08}
DRIFT_MAX_M = 0.15


class RunnerControl(RunnerV2):
    def __init__(self, domain, **cfg):
        super().__init__(domain, **cfg); self.specs = SPECS[domain]; self.frozen = FROZEN[domain]
        with xr.open_dataset(self.settings/"trialParams.nc") as tp:
            self.base_vals = {s.name: tp[s.name].values.copy() for s in self.specs if s.kind == "mult" and s.name in tp}
        self.apriori = dict(SEED[domain])
        self.gap = DB.gap_prior(domain, self.hruid)

    def _vector_to_params(self, x):
        p = super()._vector_to_params(x)
        for k, v in self.frozen.items():
            if k in FORCING_KEYS: p[k] = ("forcing", v)
            elif k == "frozenPrecipMultip_delta": p[k] = ("elev", v)
        return p

    def _stage(self, work):
        super()._stage(work); DB.declare_params(work)

    def _apply_params(self, work, params):
        super()._apply_params(work, params)            # mult/abs params already in trialParams + the frozen-precip ramp
        with nc.Dataset(work/"trialParams.nc", "a") as ds:
            for k, v in self.frozen.items():
                if k in ds.variables: ds[k][:] = v
        n = len(self.hruid)
        DB.set_trial_param(work, "canopyGapFrac", self.gap)
        DB.set_trial_param(work, "albedoMinVisible", [params["albedoMinVisible"][1]]*n)

    def _wet_soil(self, work):
        super()._wet_soil(work)
        if self.domain in AQ_INIT_M:
            with nc.Dataset(work/"coldState.nc", "a") as ds: ds["scalarAquiferStorage"][:] = AQ_INIT_M[self.domain]

    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId                | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\naverageRoutedRunoff  | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
                                              "scalarSWE            | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\nscalarAquiferStorage | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n")

    def _log_trial(self, x, r):
        logdir = self.work_root/"logs"; logdir.mkdir(exist_ok=True)
        rec = {"t": time.time(), "domain": self.domain, "x": [float(v) for v in np.asarray(x)], "objective": r["objective"], "reason": r.get("reason", "ok"),
               "parts": r.get("parts", {}), "diagnostics": r.get("diagnostics", {}), "params": r.get("params", {}), "runoff_mm": r.get("runoff_mm"),
               "peak_swe": r.get("peak_swe"), "aso_eval": r.get("aso_eval"), "drift_m": r.get("drift_m"), "eval_s": r.get("eval_s")}
        with open(logdir/f"trials_{os.getpid()}.jsonl", "a") as fh: fh.write(json.dumps(rec, default=str)+"\n")

    def run_summa(self, x, work, prefix="cal"):
        self._stage(work); params = self._vector_to_params(np.asarray(x)); self._apply_params(work, params); self._wet_soil(work)
        fdir = self._apply_forcing(work, params); self._write_filemanager(work, prefix, forcing=fdir)
        proc = subprocess.run([DB.SUMMA_DS, "-m", str(work/"fileManager.txt")], capture_output=True, text=True, timeout=6000, env={**os.environ, "OMP_NUM_THREADS": "1"})
        return params, proc

    def evaluate_full(self, x):
        work = self.work_root/f"w_{os.getpid()}_{uuid.uuid4().hex[:8]}"; t0 = time.time()
        try:
            params, proc = self.run_summa(x, work); out = work/"out"/"cal_timestep.nc"; pv = {k: v[1] for k, v in params.items()}
            if proc.returncode != 0 or not out.exists(): return {"objective": FAIL_PENALTY, "reason": "summa_failed", "params": pv}
            ds = xr.open_dataset(out); t = pd.DatetimeIndex(ds["time"].values); q_all = pd.Series(ds["averageRoutedRunoff"].isel(gru=0).values, index=t)
            sim = (q_all*self.area).resample("D").mean()[self.analysis_start:self.sim_end]
            swe_hru = pd.DataFrame(ds["scalarSWE"].values, index=t, columns=self.hruid).resample("D").mean()[self.analysis_start:self.sim_end]
            aq = pd.Series((ds["scalarAquiferStorage_mean"].values*self.hru_wt[None, :]).sum(1), index=t).resample("D").mean()[self.analysis_start:self.sim_end]
            wyi = swe_hru.index.year + (swe_hru.index.month >= 10).astype(int)
            peak_swe = pd.Series((swe_hru.values*self.hru_wt[None, :]).sum(1), index=swe_hru.index).groupby(wyi).max().mean()
            sim_mm_yr = (q_all*86400*1000).resample("D").mean()[self.analysis_start:self.sim_end].sum()/(len(sim)/365.25); ds.close()
            ann = aq.groupby(aq.index.year + (aq.index.month >= 10)).mean()
            drift = float(np.polyfit(np.arange(len(ann)), ann.values, 1)[0]*(len(ann)-1)) if len(ann) >= 4 else 0.0
            base = {"params": pv, "runoff_mm": sim_mm_yr, "peak_swe": float(peak_swe), "drift_m": drift, "eval_s": time.time()-t0}
            if not guard_runoff_ratio(sim_mm_yr, self.precip_mm_yr): return {"objective": FAIL_PENALTY, "reason": "runoff_ratio", **base}
            if not guard_snow(float(peak_swe), self.peak_swe_apriori): return {"objective": FAIL_PENALTY, "reason": "snow_guard", **base}
            if self.domain in AQ_INIT_M and abs(drift) > DRIFT_MAX_M: return {"objective": FAIL_PENALTY, "reason": "storage_drift", **base}
            if not guard_lowflow(min7_median(sim), min7_median(self.obs)): return {"objective": FAIL_PENALTY, "reason": "lowflow_dry", **base}
            swe = P.swe_signatures(swe_hru) if self.use_swe else None
            try: aso_eval = P.aso_band_evaluation(swe_hru, self.aso_df, self.hru_area, self.hru_band, self.band_order)
            except Exception as e: aso_eval = {"error": f"{type(e).__name__}: {e}"}
            res = P.signature_objective_control(sim, self.obs, MELT[self.domain], area_m2=self.area, swe=swe, exclude_months=EXCLUDE[self.domain],
                                                tau_target_days=TAU_TARGET[self.domain])
            return {"objective": res.objective, "parts": res.parts, "diagnostics": res.diagnostics, "aso_eval": aso_eval, **base}
        except subprocess.TimeoutExpired: return {"objective": FAIL_PENALTY, "reason": "timeout"}
        except Exception as e: return {"objective": FAIL_PENALTY, "reason": f"exc:{type(e).__name__}:{e}"}
        finally: shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("domain", choices=list(SPECS)); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--popsize", type=int, default=5); ap.add_argument("--maxiter", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-one", action="store_true"); a = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); outdir = OUT_ROOT/"runs"/f"{a.domain}_control_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    runner = RunnerControl(a.domain, work_root=str(outdir/"work"), **CFG[a.domain]); x0 = seed_vector(runner, dict(SEED[a.domain]))
    if a.time_one:
        r = runner.evaluate_full(x0); print(json.dumps({k: r.get(k) for k in ("objective", "reason", "parts", "diagnostics", "eval_s", "drift_m", "runoff_mm", "peak_swe", "aso_eval")},
                                                       indent=1, default=str)); return
    (outdir/"manifest.json").write_text(json.dumps(dict(domain=a.domain, ts=ts, git=git_sha(), cfg=CFG[a.domain], exe=DB.SUMMA_DS, free=[s.name for s in runner.specs],
        bounds=runner.bounds, frozen=FROZEN[a.domain], seed=SEED[a.domain], melt_months=MELT[a.domain], exclude_months=EXCLUDE[a.domain],
        canopy_gap=dict(zip(map(int, runner.hruid), map(float, runner.gap))), weights=P.DEFAULT_WEIGHTS_CONTROL, de=vars(a),
        objective="signature_objective_control"), indent=2, default=str))
    print(f"[{a.domain} control] {len(runner.specs)} free params popsize={a.popsize} maxiter={a.maxiter} workers={a.workers} -> {outdir}", flush=True); hist = []
    def cb(*aa, **kw):
        try:
            ir = kw.get("intermediate_result"); xx, fun = (np.asarray(ir.x), float(ir.fun)) if ir is not None else (np.asarray(aa[0]), None)
            hist.append(dict(t=time.time(), x=[float(v) for v in xx], objective=fun)); (outdir/"history.json").write_text(json.dumps(hist, indent=2))
            print(f"  gen {len(hist):2d}: best objective={fun}", flush=True)
        except Exception as e: print("  [cb warning]", e, flush=True)
    res = differential_evolution(runner.evaluate, bounds=runner.bounds, workers=a.workers, popsize=a.popsize, maxiter=a.maxiter, seed=a.seed,
                                 updating="deferred", polish=False, tol=0.005, callback=cb, disp=False, x0=x0)
    best = runner.evaluate_full(res.x)
    (outdir/"result.json").write_text(json.dumps(dict(x=list(res.x), objective=float(res.fun), params=best.get("params"), parts=best.get("parts"),
        diagnostics=best.get("diagnostics"), nfev=int(res.nfev)), indent=2, default=str)); print("DONE", res.fun)


if __name__ == "__main__":
    main()
