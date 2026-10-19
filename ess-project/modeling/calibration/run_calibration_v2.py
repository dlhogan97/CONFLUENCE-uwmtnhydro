#!/usr/bin/env python3
"""
run_calibration_v2.py -- recalibration of the snow/soil/aquifer/routing parameters on the v2 signature objective
(signature_objective_plus.py). Leaves calibration_runner.py / run_calibration.py untouched.

East  : v2 objective incl. the soft SNOTEL (Butte HRU 13) SWE terms; ASO stays evaluation-only.
        window WY2009-2020 (12 yr; spin-up WY2006-08), same as Tuolumne.
Tuolumne : v2 streamflow terms, scored on the REAL-GAUGE era only (WY2009-2020 window). The Hetch Hetchy reconstruction that
        covers the earlier years has a noisy low-flow level (see ess-project/utils/hh_lowflow_repair.py and the 2026-10-14
        analysis), so it is not used to fit anything. No SWE terms (no Tuolumne SNOTEL obs wired in); Dec-Mar months stay
        excluded from monthly_volume as before (PRISM winter precipitation deficit).
Both  : the temperature forcing has TWO knobs -- tempLapse (rotation about the lowest HRU; bounded -10..-2 C/km, deliberately
        NOT tied to a physical lapse rate) and tempOffset (uniform) -- plus albedoDecayRate, k_soil and frozenPrecipMultip ramp.
No snowpatch bins and no deep store here: this is the "no storage" control (M0 for East).

Usage:  python run_calibration_v2.py East_River --workers 10 --popsize 6 --maxiter 25
        python run_calibration_v2.py Tuolumne_River --time-one     # one evaluation of the seed, report wall time + parts
"""
from __future__ import annotations
import argparse, json, os, subprocess, time, uuid, shutil
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc
from scipy.optimize import differential_evolution

from calibration_runner import (CalibrationRunner, _esat_kpa, _lw_dilley_obrien, ParamSpec, _routing_scale_bounds, SUMMA_EXE, FAIL_PENALTY,
                                TAU_TARGET, MONTHLY_EXCLUDE, LOWFLOW_OVER_WEIGHT)
from signature_objective import guard_runoff_ratio, guard_snow, guard_lowflow, min7_median
import signature_objective_plus as P
from run_calibration import BASIN_CFG, OUT_ROOT, git_sha, seed_vector, save_generation_output

CFG = {
    # SAME 12-water-year scoring window for both basins (user, 2026-10-14): WY2009-2020, 3-yr spin-up WY2006-2008.
    # Everything after (WY2021-2025) is held out for evaluation.
    "East_River": dict(BASIN_CFG["East_River"], sim_start="2005-10-01", sim_end="2020-09-30", analysis_start="2008-10-01"),
    "Tuolumne_River": dict(sim_start="2005-10-01", sim_end="2020-09-30", analysis_start="2008-10-01",
                           obs_csv=BASIN_CFG["Tuolumne_River"]["obs_csv"], precip_mm_yr=1143.0, peak_swe_apriori=772.0),
}
SPECS = {
    "East_River": [
        ParamSpec("frozenPrecipMultip_low", "elev", 1.00, 1.20),
        ParamSpec("frozenPrecipMultip_delta", "elev", 0.0, 0.40),
        ParamSpec("tempLapse", "forcing", -10.0, -2.0),
        ParamSpec("tempOffset", "forcing", -4.0, 2.0),
        # soft forcing corrections from the ARM SAIL S3 radiometers (2856 m, WY2022-23): forcing SW is +20% (+45 W/m2) high in Mar-Jun, LW ~+10 W/m2 high.
        # Bounds run from NO correction to ~1.5x the point-measurement bias; the optimiser decides how much to use.
        ParamSpec("swScale", "forcing", 0.75, 1.00),
        ParamSpec("lwOffset", "forcing", -20.0, 0.0),
        ParamSpec("albedoDecayRate", "abs", 1e5, 5e6, log=True),
        ParamSpec("k_soil", "mult", 0.1, 10.0, log=True),
        ParamSpec("aquiferBaseflowRate", "abs", 1e-9, 1e-4, log=True),
        ParamSpec("aquiferScaleFactor", "abs", 0.5, 8.0),
        ParamSpec("aquiferBaseflowExp", "abs", 1.0, 6.0),
        ParamSpec("routingGammaScale", "abs", *_routing_scale_bounds(16, 36)),
    ],
    "Tuolumne_River": [
        ParamSpec("frozenPrecipMultip_low", "elev", 0.85, 1.20),
        ParamSpec("frozenPrecipMultip_delta", "elev", 0.0, 0.60),
        ParamSpec("tempLapse", "forcing", -10.0, -2.0),
        ParamSpec("tempOffset", "forcing", -3.0, 3.0),
        # soft forcing correction from CDEC FLV (Vogelsang, 3077 m, 2021-25): forcing SW is -11% (-34 W/m2) low in Mar-Jun, -14% in winter.
        ParamSpec("swScale", "forcing", 1.00, 1.25),
        ParamSpec("albedoDecayRate", "abs", 5e4, 5e6, log=True),
        ParamSpec("k_soil", "mult", 0.1, 10.0, log=True),
        ParamSpec("qSurfScale", "abs", 1.0, 50.0, log=True),
        ParamSpec("kAnisotropic", "abs", 0.01, 10.0, log=True),
        ParamSpec("routingGammaScale", "abs", *_routing_scale_bounds(8, 36)),
    ],
}
# warm starts = the current bests (tempLapse seeded at the measured lapse -> identity rotation; Tuolumne offset +1.04 K as calibrated)
SEED = {
    "East_River": dict(frozenPrecipMultip_low=1.0630, frozenPrecipMultip_delta=0.02754, tempLapse=-2.607, tempOffset=-1.3436, swScale=1.0, lwOffset=0.0,
                       albedoDecayRate=409426.0, k_soil=2.4941, aquiferBaseflowRate=9.43986e-06, aquiferScaleFactor=2.65777,
                       aquiferBaseflowExp=2.0568, routingGammaScale=51629.2),
    "Tuolumne_River": dict(frozenPrecipMultip_low=0.8793, frozenPrecipMultip_delta=0.06387, tempLapse=-3.289, tempOffset=1.0397, swScale=1.0,
                           albedoDecayRate=126754.0, k_soil=3.008, qSurfScale=5.7046, kAnisotropic=5.1999, routingGammaScale=24863.6),
}
WEIGHTS = {"East_River": None, "Tuolumne_River": None}      # v2 defaults (SWE terms only exist for East)


class RunnerV2(CalibrationRunner):
    def __init__(self, domain, **cfg):
        super().__init__(domain=domain, **cfg)
        self.specs = SPECS[domain]
        with xr.open_dataset(self.settings/"attributes.nc") as a: self.hruid = a["hruId"].values.astype(int)
        with xr.open_dataset(self.settings/"trialParams.nc") as tp:     # re-derive a-priori / base fields for the v2 spec
            self.base_vals = {s.name: tp[s.name].values.copy() for s in self.specs if s.kind == "mult" and s.name in tp}
            self.apriori = {s.name: float(np.ravel(tp[s.name].values)[0]) for s in self.specs if s.name in tp}
        for s in self.specs:
            if s.kind == "forcing" and s.name not in self.apriori:
                self.apriori[s.name] = {"tempLapse": self.lapse_ref*1000.0, "tempOffset": 0.0, "swScale": 1.0, "lwOffset": 0.0}.get(s.name, 0.0)
        self.use_swe = domain == "East_River"
        # ---- band-resolved ASO evaluation (evaluation only) ----
        obsdir = Path(__file__).parent/"observations"
        self.hru_area = pd.Series(self.area*self.hru_wt, index=self.hruid)
        if domain == "East_River":
            self.aso_df = pd.read_csv(obsdir/"east_aso_hru_swe.csv", parse_dates=["date"])
            E_ = pd.Series(self.elev, index=self.hruid); self.band_order = ["~2700", "~3000", "~3350", "~3700"]
            self.hru_band = pd.Series(pd.cut(E_, [0, 2800, 3200, 3500, 4000], labels=self.band_order).astype(str), index=self.hruid)
        else:
            import geopandas as gpd
            g_ = gpd.read_file("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/"
                               "Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp")
            self.aso_df = pd.read_csv(obsdir/"tuolumne_aso_hru_swe.csv", parse_dates=["date"])
            self.band_order = [f"b{k}" for k in range(2, 9)]; self.hru_band = pd.Series({int(h): f"b{int(c)}" for h, c in zip(g_.HRU_ID, g_.elevClass)})

    # forcing builder: base behaviour (+ lapse rotation, uniform offset, precip ramp) PLUS the soft SW multiplier and LW offset
    def _apply_forcing(self, work, params):
        lapse = params.get("tempLapse", (None, None))[1]; offset = params.get("tempOffset", (None, None))[1]
        pramp = params.get("precipRampHigh", (None, None))[1]; sws = params.get("swScale", (None, None))[1]; lwo = params.get("lwOffset", (None, None))[1]
        if all(v is None for v in (lapse, offset, pramp, sws, lwo)): return self.forcing
        dT = None
        if lapse is not None or offset is not None:
            dT = np.zeros_like(self.elev, dtype=float)
            if lapse is not None: dT += (lapse/1000.0 - self.lapse_ref)*(self.elev - self.elev_ref)
            if offset is not None: dT += float(offset)
        span = self.elev.max() - self.elev.min()
        pm = (1.0 + pramp*(self.elev - self.elev.min())/span) if pramp is not None else None
        fdir = work/"forcing"; fdir.mkdir(parents=True, exist_ok=True)
        y0, m0 = int(self.sim_start[:4]), int(self.sim_start[5:7]); y1, m1 = int(self.sim_end[:4]), int(self.sim_end[5:7])
        for f in sorted(self.forcing.glob("*.nc")):
            stem = f.name.split("_")[-1].replace(".nc", "")
            try: yy, mm = int(stem[:4]), int(stem[4:6])
            except ValueError: (fdir/f.name).symlink_to(f); continue
            if not ((y0, m0) <= (yy, mm) <= (y1, m1)): (fdir/f.name).symlink_to(f); continue
            with xr.open_dataset(f) as _d: ds = _d.load().copy()
            if dT is not None:
                T0 = ds["airtemp"].values; q0 = ds["spechum"].values; pp = ds["airpres"].values; T1 = T0 + dT[None, :]; q1 = q0*(_esat_kpa(T1)/_esat_kpa(T0))
                ds["airtemp"].values[:] = T1; ds["spechum"].values[:] = q1
                ds["LWRadAtm"].values[:] = ds["LWRadAtm"].values + _lw_dilley_obrien(T1, pp, q1) - _lw_dilley_obrien(T0, pp, q0)
            if lwo is not None: ds["LWRadAtm"].values[:] = ds["LWRadAtm"].values + float(lwo)
            if sws is not None: ds["SWRadAtm"].values[:] = ds["SWRadAtm"].values*float(sws)
            if pm is not None: ds["pptrate"].values[:] = ds["pptrate"].values*pm[None, :]
            ds.to_netcdf(fdir/f.name); ds.close()
        return fdir

    def _log_trial(self, x, r):
        logdir = self.work_root/"logs"; logdir.mkdir(exist_ok=True)
        rec = {"t": time.time(), "domain": self.domain, "x": [float(v) for v in np.asarray(x)], "objective": r["objective"], "reason": r.get("reason", "ok"),
               "parts": r.get("parts", {}), "diagnostics": r.get("diagnostics", {}), "params": r.get("params", {}), "runoff_mm": r.get("runoff_mm"),
               "peak_swe": r.get("peak_swe"), "aso_eval": r.get("aso_eval")}
        with open(logdir/f"trials_{os.getpid()}.jsonl", "a") as fh: fh.write(json.dumps(rec, default=str)+"\n")

    def evaluate_full(self, x):
        work = self.work_root/f"w_{os.getpid()}_{uuid.uuid4().hex[:8]}"; prefix = "cal"
        try:
            self._stage(work); params = self._vector_to_params(np.asarray(x)); self._apply_params(work, params); self._wet_soil(work)
            fdir = self._apply_forcing(work, params); self._write_filemanager(work, prefix, forcing=fdir)
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            t0 = time.time()
            proc = subprocess.run([SUMMA_EXE, "-m", str(work/"fileManager.txt")], capture_output=True, text=True, timeout=4800, env=env)
            out = work/"out"/f"{prefix}_timestep.nc"
            if proc.returncode != 0 or not out.exists():
                return {"objective": FAIL_PENALTY, "reason": "summa_failed", "params": {k: v[1] for k, v in params.items()}}
            ds = xr.open_dataset(out); t = pd.DatetimeIndex(ds["time"].values)
            q_all = pd.Series(ds["averageRoutedRunoff"].isel(gru=0).values, index=t)
            sim = (q_all*self.area).resample("D").mean()[self.analysis_start:self.sim_end]
            swe_hru = pd.DataFrame(ds["scalarSWE"].values, index=t, columns=self.hruid).resample("D").mean()[self.analysis_start:self.sim_end]
            swe_basin = (swe_hru.values*self.hru_wt[None, :]).sum(1)
            peak_swe = pd.Series(swe_basin, index=swe_hru.index).groupby(swe_hru.index.year+(swe_hru.index.month >= 10).astype(int)).max().mean()
            sim_mm_yr = (q_all*86400*1000).resample("D").mean()[self.analysis_start:self.sim_end].sum()/(len(sim)/365.25)
            ds.close()
            if not guard_runoff_ratio(sim_mm_yr, self.precip_mm_yr): return {"objective": FAIL_PENALTY, "reason": "runoff_ratio", "runoff_mm": sim_mm_yr}
            if not guard_snow(float(peak_swe), self.peak_swe_apriori): return {"objective": FAIL_PENALTY, "reason": "snow_guard", "peak_swe": float(peak_swe)}
            if not guard_lowflow(min7_median(sim), min7_median(self.obs)): return {"objective": FAIL_PENALTY, "reason": "lowflow_dry"}
            swe = P.swe_signatures(swe_hru) if self.use_swe else None
            try: aso_eval = P.aso_band_evaluation(swe_hru, self.aso_df, self.hru_area, self.hru_band, self.band_order)
            except Exception as e: aso_eval = {"error": f"{type(e).__name__}: {e}"}
            res = P.signature_objective_plus(sim, self.obs, self.tau_target, weights=WEIGHTS[self.domain], area_m2=self.area, swe=swe,
                                             lowflow_over_weight=self.lowflow_over_weight, exclude_months=self.monthly_exclude)
            return {"objective": res.objective, "parts": res.parts, "diagnostics": res.diagnostics,
                    "params": {k: v[1] for k, v in params.items()}, "runoff_mm": sim_mm_yr, "peak_swe": float(peak_swe), "eval_s": time.time()-t0, "aso_eval": aso_eval}
        except subprocess.TimeoutExpired: return {"objective": FAIL_PENALTY, "reason": "timeout"}
        except Exception as e: return {"objective": FAIL_PENALTY, "reason": f"exc:{type(e).__name__}:{e}"}
        finally: shutil.rmtree(work, ignore_errors=True)


def best_trial(rundir):
    import glob
    rec = [json.loads(l) for f in glob.glob(str(Path(rundir)/"work/logs/trials_*.jsonl")) for l in open(f)]
    return sorted([r for r in rec if r["reason"] == "ok"], key=lambda r: r["objective"])


# energy / snow parameters trade off against the new soft shortwave and longwave corrections, which round 1 did not have; their round-1 optima are
# conditional on an uncorrected forcing, so they keep their full bounds. Only the water-balance parameters are narrowed.
KEEP_FULL = {"frozenPrecipMultip_low", "frozenPrecipMultip_delta", "tempLapse", "tempOffset", "swScale", "lwOffset", "albedoDecayRate"}


def narrow_specs(specs, rundir, top_n=10, widen=0.25, edge=0.05):
    """Round-2 bounds from a finished/paused round-1 run: span of the top_n trials per parameter (optimised coordinate), widened by `widen` of that
    span each side, never narrower than 10% of the original range, clipped to the original bounds -- and a side that the cluster sits on (within
    `edge` of the original range) keeps its ORIGINAL bound, so a pegged parameter is not trapped. Parameters absent from round 1 keep their spec."""
    top = best_trial(rundir)[:top_n]; out = []
    for sp in specs:
        vals = [t["params"][sp.name] for t in top if sp.name in t["params"]]
        if sp.name in KEEP_FULL or len(vals) < 5: out.append(sp); continue
        v = np.log10(vals) if sp.log else np.array(vals); lo0, hi0 = sp.bounds; rng = hi0-lo0
        lo, hi = float(v.min()), float(v.max()); span = max(hi-lo, 0.10*rng); mid = 0.5*(lo+hi)
        lo, hi = mid-0.5*span, mid+0.5*span; lo, hi = lo-widen*span, hi+widen*span
        if float(v.min()) <= lo0+edge*rng: lo = lo0
        if float(v.max()) >= hi0-edge*rng: hi = hi0
        lo, hi = max(lo, lo0), min(hi, hi0)
        nlo, nhi = (10**lo, 10**hi) if sp.log else (lo, hi)
        out.append(ParamSpec(sp.name, sp.kind, nlo, nhi, sp.log))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("domain", choices=list(SPECS)); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--popsize", type=int, default=6); ap.add_argument("--maxiter", type=int, default=25); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-one", action="store_true"); ap.add_argument("--narrow-from", default=None, help="round-1 run dir: narrow bounds around its top trials and seed from its best")
    ap.add_argument("--tag", default="v2"); a = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); outdir = OUT_ROOT/"runs"/f"{a.domain}_{a.tag}_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    cfg = CFG[a.domain]; runner = RunnerV2(a.domain, work_root=str(outdir/"work"), **cfg)
    seed = dict(SEED[a.domain])
    if a.narrow_from:
        runner.specs = narrow_specs(runner.specs, a.narrow_from); b = best_trial(a.narrow_from)[0]
        seed.update({k: v for k, v in b["params"].items() if k in {s_.name for s_ in runner.specs}})
        print("round-2 bounds:", {s_.name: (float(f"{s_.lo:.4g}"), float(f"{s_.hi:.4g}")) for s_ in runner.specs}, flush=True)
        print("seed (round-1 best, objective %.3f):" % b["objective"], {k: float(f"{v:.4g}") for k, v in seed.items()}, flush=True)
    x0 = seed_vector(runner, seed)
    if a.time_one:
        t0 = time.time(); r = runner.evaluate_full(x0)
        print(json.dumps({k: r.get(k) for k in ("objective", "reason", "parts", "diagnostics", "eval_s", "runoff_mm", "peak_swe")}, indent=1, default=str))
        print("wall s", round(time.time()-t0, 1)); return
    (outdir/"manifest.json").write_text(json.dumps(dict(domain=a.domain, ts=ts, git=git_sha(), cfg=cfg, params=[s.name for s in runner.specs],
        bounds=runner.bounds, seed=seed, de=vars(a), objective="v2 (signature_objective_plus)"), indent=2, default=str))
    print(f"[{a.domain} v2] {len(runner.specs)} params popsize={a.popsize} maxiter={a.maxiter} workers={a.workers} -> {outdir}", flush=True)
    hist = []
    def cb(*aa, **kw):
        try:
            ir = kw.get("intermediate_result"); x, fun = (np.asarray(ir.x), float(ir.fun)) if ir is not None else (np.asarray(aa[0]), None)
            hist.append(dict(t=time.time(), x=[float(v) for v in x], objective=fun)); (outdir/"history.json").write_text(json.dumps(hist, indent=2))
            print(f"  gen {len(hist):2d}: best objective={fun}", flush=True)
        except Exception as e: print("  [cb warning]", e, flush=True)
    res = differential_evolution(runner.evaluate, bounds=runner.bounds, workers=a.workers, popsize=a.popsize, maxiter=a.maxiter, seed=a.seed,
                                 updating="deferred", polish=False, tol=0.01, callback=cb, disp=False, x0=x0)
    best = runner.evaluate_full(res.x)
    (outdir/"result.json").write_text(json.dumps(dict(x=list(res.x), objective=float(res.fun), physical_params=best.get("params"), parts=best.get("parts"),
        diagnostics=best.get("diagnostics"), nfev=int(res.nfev)), indent=2, default=str)); print("DONE", res.fun)


if __name__ == "__main__":
    main()
