#!/usr/bin/env python3
"""
run_calibration_melt.py -- Tuolumne SNOWMELT calibration against the Dana Meadows pillow (2026-10-23).

Design (user, 2026-10-23): snow volume and peak are taken as right; the spring energy balance is not. Calibrate ONLY melt timing and melt rate
of the elevation bands that bracket the Dana Meadows pillow (2975 m): HRUs 8-12 (3060 m) and 13-17 (2747 m).
  * Air temperature: universal -2.0 C, RH preserved, Dilley-O'Brien longwave increment applied (no station / elevation / aspect structure).
  * Free (global, same value on every HRU): swScale, lwOffset (additive W/m2 on LWRadAtm), albedoMinVisible, albedoMinNearIR, albedoDecayRate.
  * Frozen: precipitation (stage-1 values), runoff generation, canopy gaps.
  * Only the 10 bracketing HRUs are simulated (HRUs are independent for SWE), which makes a trial ~3x cheaper than the 28-HRU model.
  * Model curve at the pillow = elevation-weighted mean of the two band means (each averaged across aspect).
  * Objective = mean over water years of |model - pillow| for three melt-curve times, each scaled by 7 d (so an objective <= 1 means within 7 days):
        t50   day SWE falls to 50% of that year's peak        (timing)
        tout  day SWE falls below 25.4 mm                      (timing)
        d8020 days between 80% and 20% of peak                 (rate)
    Peak timing and peak volume are guards, not objective terms: |peak-date error| <= 14 d, peak ratio within 0.7-1.4.
usage: python run_calibration_melt.py --time-one | --workers 20 --popsize 8 --maxiter 25
"""
import argparse, json, os, subprocess, time, uuid, shutil
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc
from scipy.optimize import differential_evolution
from calibration_runner import ParamSpec, FAIL_PENALTY, _esat_kpa, _lw_dilley_obrien
import deepstore_build as DB
from run_calibration import OUT_ROOT, git_sha, seed_vector
import run_calibration_snow as S

DOM = "Tuolumne_River"
CFG = S.CFG                                    # sim 2011-10-01 .. 2020-09-30, analysis from WY2013
UP, LO = [8, 9, 10, 11, 12], [13, 14, 15, 16, 17]   # hruIds: 3060 m band, 2747 m band
Z_UP, Z_LO, Z_DAN = 3060.0, 2747.0, 2975.0
W_UP = (Z_DAN - Z_LO)/(Z_UP - Z_LO)
DT_C = -2.0
PILLOW = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/observations/formatted/snotel_swe.csv"   # pre-cleaned Dana Meadows daily SWE (mm)
SCALE_D, MELTOUT_MM, MIN_PEAK_MM = 7.0, 25.4, 100.0
PEAK_DATE_MAX_D, PEAK_RATIO = 14.0, (0.7, 1.4)
SPECS = [ParamSpec("swScale", "forcing", 0.85, 1.30), ParamSpec("lwOffset", "forcing", -25.0, 25.0), ParamSpec("albMinVis", "snow", 0.20, 0.60),
         ParamSpec("albMinNIR", "snow", 0.15, 0.45), ParamSpec("albedoDecayRate", "snow", 3e5, 3e6, log=True)]
SEED = dict(swScale=1.0505, lwOffset=0.0, albMinVis=0.57, albMinNIR=0.30, albedoDecayRate=1.0e6)
PRECIP = dict(frozenPrecipMultip_low=0.851, frozenPrecipMultip_delta=0.087)


def melt_times(s):
    """s: daily SWE of one water year (mm, DatetimeIndex). dict of day-of-WY times, or None if the year has no real pack."""
    s = s.rolling(7, center=True, min_periods=4).mean()
    if s.isna().all() or s.max() < MIN_PEAK_MM: return None
    pk = s.idxmax(); post = s.loc[pk:]; mx = s.max(); first = lambda thr: post[post < thr]
    f50, f80, f20, fo = first(.5*mx), first(.8*mx), first(.2*mx), first(MELTOUT_MM)
    if min(len(f50), len(f80), len(f20), len(fo)) == 0: return None
    oct1 = pd.Timestamp(pk.year - (pk.month < 10), 10, 1); dd = lambda t: (t - oct1).days
    return dict(peak=dd(pk), peak_mm=float(mx), t50=dd(f50.index[0]), tout=dd(fo.index[0]), d8020=dd(f20.index[0]) - dd(f80.index[0]))


def wy_of(idx): return idx.year + (idx.month >= 10)


def melt_table(swe):
    out = {}
    for y, g in swe.groupby(wy_of(swe.index)):
        if y >= int(CFG["analysis_start"][:4]) + 1:          # WY2013 on; WY2012 is spin-up
            m = melt_times(g)
            if m: out[y] = m
    return pd.DataFrame(out).T


def melt_objective(model_swe, pillow_swe):
    a, b = melt_table(model_swe), melt_table(pillow_swe); yrs = a.index.intersection(b.index)
    if len(yrs) < 4: return None
    e = (a.loc[yrs] - b.loc[yrs]).astype(float)
    parts = {k: float(e[k].abs().mean()/SCALE_D) for k in ("t50", "tout", "d8020")}
    diag = dict(years=[int(y) for y in yrs], bias_d={k: float(e[k].mean()) for k in ("t50", "tout", "d8020", "peak")}, mae_d={k: float(e[k].abs().mean()) for k in ("t50", "tout", "d8020", "peak")},
                peak_ratio=float((a.loc[yrs].peak_mm/b.loc[yrs].peak_mm).mean()))
    return parts, diag


class RunnerMelt(S.RunnerSnow):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.specs = SPECS; ids = self.hruid.astype(int)
        self.idx = [int(np.where(ids == h)[0][0]) for h in UP + LO]; self.pillow = pd.read_csv(PILLOW, index_col=0, parse_dates=True).value.asfreq("D")

    def _vector_to_params(self, x):
        p = {s.name: (s.kind, s.to_physical(xi)) for s, xi in zip(self.specs, x)}
        for k, v in PRECIP.items(): p[k] = ("elev", v)
        return p

    def _apply_params(self, work, params):
        super(S.C.RunnerControl, self)._apply_params(work, params)          # base: frozen-precip elevation ramp
        n = len(self.hruid)
        with nc.Dataset(work/"trialParams.nc", "a") as ds:
            ds["k_soil"][:] = self.k_soil_base*S.RUNOFF_FIXED["k_soil_mult"]
            for k in ("qSurfScale", "kAnisotropic", "routingGammaScale"): ds[k][:] = S.RUNOFF_FIXED[k]
            ds["albedoDecayRate"][:] = params["albedoDecayRate"][1]
        DB.set_trial_param(work, "canopyGapFrac", self.gap)
        DB.set_trial_param(work, "albedoMinVisible", [params["albMinVis"][1]]*n)
        DB.set_trial_param(work, "albedoMinNearIR", [params["albMinNIR"][1]]*n)

    def _subset(self, work):
        """Keep only the bracketing HRUs in attributes / coldState / trialParams (HRUs are independent for SWE; lateral routing is cut).
        netCDF4 copy, not xarray: xarray drops dimensions no variable uses (midSoil), which SUMMA needs."""
        for fn in ("attributes.nc", "coldState.nc", "trialParams.nc"):
            src = work/fn; tmp = work/(fn + ".sub")
            with nc.Dataset(src) as a, nc.Dataset(tmp, "w", format="NETCDF4") as b:
                b.setncatts({k: a.getncattr(k) for k in a.ncattrs()})
                for n, d in a.dimensions.items(): b.createDimension(n, len(self.idx) if n == "hru" else (None if d.isunlimited() else len(d)))
                for n, v in a.variables.items():
                    fv = v._FillValue if "_FillValue" in v.ncattrs() else None
                    o = b.createVariable(n, v.dtype, v.dimensions, fill_value=fv)
                    o.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
                    ax = v.dimensions.index("hru") if "hru" in v.dimensions else None
                    o[:] = v[:] if ax is None else np.take(v[:], self.idx, axis=ax)
                if fn == "attributes.nc" and "downHRUindex" in b.variables: b["downHRUindex"][:] = 0
            tmp.replace(src)

    def _apply_forcing(self, work, params):
        sws, lwo = params["swScale"][1], params["lwOffset"][1]
        fdir = work/"forcing"; fdir.mkdir(parents=True, exist_ok=True)
        y0, m0 = int(self.sim_start[:4]), int(self.sim_start[5:7]); y1, m1 = int(self.sim_end[:4]), int(self.sim_end[5:7])
        for f in sorted(self.forcing.glob("*.nc")):
            stem = f.name.split("_")[-1].replace(".nc", "")
            try: yy, mm = int(stem[:4]), int(stem[4:6])
            except ValueError: continue
            if not ((y0, m0) <= (yy, mm) <= (y1, m1)): continue
            with xr.open_dataset(f) as _d: ds = _d.isel(hru=self.idx).load().copy()
            T0 = ds["airtemp"].values.copy(); q0 = ds["spechum"].values.copy(); pp = ds["airpres"].values
            T1 = T0 + DT_C; q1 = q0*(_esat_kpa(T1)/_esat_kpa(T0))                       # preserve relative humidity
            q1 = np.minimum(q1, 0.622*_esat_kpa(T1)*1000.0/(pp - 0.378*_esat_kpa(T1)*1000.0))
            ds["airtemp"].values[:] = T1; ds["spechum"].values[:] = q1
            ds["LWRadAtm"].values[:] = ds["LWRadAtm"].values + _lw_dilley_obrien(T1, pp, q1) - _lw_dilley_obrien(T0, pp, q0) + lwo
            ds["SWRadAtm"].values[:] = ds["SWRadAtm"].values*sws
            ds.to_netcdf(fdir/f.name); ds.close()
        files = sorted(fdir.glob("*.nc")); (work/"forcingFileList.txt").write_text("".join(f"{f.name}\n" for f in files))
        return fdir

    def _write_filemanager(self, work, prefix, forcing=None):
        super(S.C.RunnerControl, self)._write_filemanager(work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\nscalarSWE | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n")

    def run_summa(self, x, work, prefix="cal"):
        self._stage(work); params = self._vector_to_params(np.asarray(x)); self._apply_params(work, params); self._wet_soil(work); self._subset(work)
        fdir = self._apply_forcing(work, params); self._write_filemanager(work, prefix, forcing=fdir)
        proc = subprocess.run([DB.SUMMA_DS, "-m", str(work/"fileManager.txt")], capture_output=True, text=True, timeout=6000, env={**os.environ, "OMP_NUM_THREADS": "1"})
        return params, proc

    def model_curves(self, out):
        d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
        z = np.array(d["scalarSWE_mean"][:]).astype(float); z[np.abs(z) > 1e8] = np.nan; d.close()
        df = pd.DataFrame(z, index=t, columns=UP + LO); up, lo = df[UP].mean(1), df[LO].mean(1)
        return W_UP*up + (1 - W_UP)*lo, up, lo, df

    def evaluate_full(self, x):
        work = self.work_root/f"w_{os.getpid()}_{uuid.uuid4().hex[:8]}"; t0 = time.time()
        try:
            params, proc = self.run_summa(x, work); pv = {k: v[1] for k, v in params.items()}; out = work/"out"/"cal_day.nc"
            if proc.returncode != 0 or not out.exists(): return {"objective": FAIL_PENALTY, "reason": "summa_failed", "params": pv, "stderr": proc.stderr[-300:]}
            mod, up, lo, _ = self.model_curves(out); res = melt_objective(mod, self.pillow.reindex(mod.index))
            base = {"params": pv, "eval_s": time.time() - t0}
            if res is None: return {"objective": FAIL_PENALTY + 5, "reason": "no_melt_years", **base}
            parts, diag = res
            viol = {"peak_date": max(diag["mae_d"]["peak"]/PEAK_DATE_MAX_D - 1.0, 0.0), "peak_ratio": max(PEAK_RATIO[0]/max(diag["peak_ratio"], 1e-6) - 1.0, diag["peak_ratio"]/PEAK_RATIO[1] - 1.0, 0.0)}
            bad = {k: v for k, v in viol.items() if v > 0}
            obj = float(np.mean(list(parts.values())))
            if bad: return {"objective": FAIL_PENALTY + sum(bad.values()), "reason": max(bad, key=bad.get), "violations": bad, "parts": parts, "diagnostics": diag, **base}
            return {"objective": obj, "parts": parts, "diagnostics": diag, **base}
        except subprocess.TimeoutExpired: return {"objective": FAIL_PENALTY, "reason": "timeout"}
        except Exception as e: return {"objective": FAIL_PENALTY, "reason": f"exc:{type(e).__name__}:{e}"}
        finally: shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=25); ap.add_argument("--seed", type=int, default=5); ap.add_argument("--time-one", action="store_true")
    a = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); outdir = OUT_ROOT/"runs"/f"{DOM}_melt_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    runner = RunnerMelt(work_root=str(outdir/"work"), **CFG); x0 = seed_vector(runner, dict(SEED))
    if a.time_one:
        r = runner.evaluate_full(x0); print(json.dumps({k: r.get(k) for k in ("objective", "reason", "parts", "diagnostics", "violations", "eval_s", "stderr")}, indent=1, default=str)); return
    (outdir/"manifest.json").write_text(json.dumps(dict(domain=DOM, ts=ts, git=git_sha(), cfg=CFG, exe=DB.SUMMA_DS, free=[s.name for s in SPECS], bounds=runner.bounds, seed=SEED,
        dT=DT_C, precip=PRECIP, bands=dict(up=UP, lo=LO, w_up=W_UP), pillow=PILLOW, de=vars(a), objective="melt_timing_rate_dana"), indent=2, default=str))
    print(f"[{DOM} melt] {len(SPECS)} free params popsize={a.popsize} maxiter={a.maxiter} workers={a.workers} -> {outdir}", flush=True); hist = []
    def cb(intermediate_result):
        hist.append(dict(t=time.time(), x=[float(v) for v in intermediate_result.x], objective=float(intermediate_result.fun)))
        (outdir/"history.json").write_text(json.dumps(hist, indent=2)); print(f"  gen {len(hist):2d}: best objective={intermediate_result.fun:.4f}", flush=True)
    res = differential_evolution(runner.evaluate, bounds=runner.bounds, workers=a.workers, popsize=a.popsize, maxiter=a.maxiter, seed=a.seed,
                                 updating="deferred", polish=False, tol=0.002, callback=cb, disp=False, x0=x0)
    (outdir/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev)), indent=2)); print("DONE", res.fun, flush=True)


if __name__ == "__main__":
    main()
