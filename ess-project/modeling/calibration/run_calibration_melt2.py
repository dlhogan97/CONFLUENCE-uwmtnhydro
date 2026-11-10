#!/usr/bin/env python3
"""
run_calibration_melt2.py -- Tuolumne snowmelt calibration on BOTH cleaned pillows in a handful of well-aligned years (2026-10-27).

Why (user, 2026-10-27): the Dana-only fit on WY2013-2020 was driven by bad pillow years and fit one elevation. Here:
  * pillows: cleaned Dana Meadows (2975 m) and cleaned Tuolumne Meadows (2591 m); years chosen ONLY on pillow agreement
    (shape correlation, peak ratio, plausible melt-out lag, no interpolated stretches, clean summer baseline), before any model fit:
    WY1996, 1998, 2000, 2001, 2024.  The old WY2013-2020 set is the validation period (not used here).
  * model: HRUs 8-22 (bands 3060 / 2747 / 2415 m, 5 aspects each) simulated one water year at a time from Oct 1; band means averaged across aspect;
    Dana curve = elevation-weighted mean of the 3060 and 2747 bands, Tuolumne Meadows curve = weighted mean of the 2747 and 2415 bands.
  * forcing: universal -2.0 C at the reference elevation plus a free elevation tilt (C/km), RH preserved, Dilley-O'Brien longwave increment.
  * free (global): swScale, lwOffset, albedoMinVisible, albedoMinNearIR, albedoDecayRate, tilt.  Frozen: precipitation, runoff generation, canopy gaps.
  * objective = mean over (year, pillow) of |model - pillow| in days for t50 (50% of peak), t10 (10% of peak = melt-out, baseline-free) and d8020 (80%->20% duration), over 7 d.
    Guard only on peak volume (model/pillow within 0.6-1.6); no peak-date guard.
usage: python run_calibration_melt2.py --time-one | --workers 20 --popsize 6 --maxiter 20
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
import run_calibration_melt as M

DOM = "Tuolumne_River"
CFG = dict(S.CFG, sim_start="1995-10-01", analysis_start="1995-10-01", sim_end="2024-09-30")   # CFG only supplies obs/aso bookkeeping; years are run separately
YEARS = [1996, 1998, 2000, 2001, 2024]
HRUS = list(range(8, 23)); BANDS = {3060: [8, 9, 10, 11, 12], 2747: [13, 14, 15, 16, 17], 2415: [18, 19, 20, 21, 22]}
Z_REF = 2700.0; DT_C = -2.0
PIL = {"Dana": (2975.0, "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/observations/formatted/snotel_swe.csv", (3060, 2747)),
       "TUM": (2591.0, "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_lumped/observations/formatted/snotel_swe.csv", (2747, 2415))}
SCALE_D, MIN_PEAK_MM, PEAK_RATIO = 7.0, 100.0, (0.6, 1.6)
SPECS = [ParamSpec("swScale", "forcing", 0.85, 1.30), ParamSpec("lwOffset", "forcing", -25.0, 25.0), ParamSpec("albMinVis", "snow", 0.20, 0.60),
         ParamSpec("albMinNIR", "snow", 0.15, 0.45), ParamSpec("albedoDecayRate", "snow", 3e5, 3e6, log=True), ParamSpec("tilt", "forcing", -3.0, 3.0)]
SEED = dict(swScale=1.265, lwOffset=0.44, albMinVis=0.463, albMinNIR=0.345, albedoDecayRate=1.29e6, tilt=0.0)      # best of the Dana-only run, no tilt
PRECIP = M.PRECIP


def melt_times(s):
    """s: daily SWE (mm) of one water year. day-of-WY times, or None if no real pack."""
    s = s.rolling(7, center=True, min_periods=4).mean()
    if s.isna().mean() > 0.1 or s.max() < MIN_PEAK_MM: return None
    mx = s.max(); pk = s.idxmax(); post = s.loc[pk:]; oct1 = pd.Timestamp(pk.year - (pk.month < 10), 10, 1); dd = lambda t: (t - oct1).days
    f = lambda thr: post[post < thr]
    f50, f80, f20, f10 = f(.5*mx), f(.8*mx), f(.2*mx), f(.1*mx)
    if min(len(f50), len(f80), len(f20), len(f10)) == 0: return None
    return dict(peak_mm=float(mx), peak=dd(pk), t50=dd(f50.index[0]), t10=dd(f10.index[0]), d8020=dd(f20.index[0]) - dd(f80.index[0]))


class RunnerMelt2(M.RunnerMelt):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.specs = SPECS; ids = self.hruid.astype(int)
        self.idx = [int(np.where(ids == h)[0][0]) for h in HRUS]; self.z = self.elev[self.idx]
        self.pil = {k: pd.read_csv(v[1], index_col=0, parse_dates=True).value.asfreq("D") for k, v in PIL.items()}
        self.run_start, self.run_end = None, None

    def _apply_forcing(self, work, params):
        sws, lwo, tilt = params["swScale"][1], params["lwOffset"][1], params["tilt"][1]
        dT = (DT_C + tilt*(self.z - Z_REF)/1000.0)[None, :]
        fdir = work/"forcing"; fdir.mkdir(parents=True, exist_ok=True)
        y0, m0, y1, m1 = int(self.run_start[:4]), int(self.run_start[5:7]), int(self.run_end[:4]), int(self.run_end[5:7])
        for f in sorted(self.forcing.glob("*.nc")):
            stem = f.name.split("_")[-1].replace(".nc", "")
            try: yy, mm = int(stem[:4]), int(stem[4:6])
            except ValueError: continue
            if not ((y0, m0) <= (yy, mm) <= (y1, m1)): continue
            with xr.open_dataset(f) as _d: ds = _d.isel(hru=self.idx).load().copy()
            T0 = ds["airtemp"].values.copy(); q0 = ds["spechum"].values.copy(); pp = ds["airpres"].values
            T1 = T0 + dT; q1 = q0*(_esat_kpa(T1)/_esat_kpa(T0))
            q1 = np.minimum(q1, 0.622*_esat_kpa(T1)*1000.0/(pp - 0.378*_esat_kpa(T1)*1000.0))
            ds["airtemp"].values[:] = T1; ds["spechum"].values[:] = q1
            ds["LWRadAtm"].values[:] = ds["LWRadAtm"].values + _lw_dilley_obrien(T1, pp, q1) - _lw_dilley_obrien(T0, pp, q0) + lwo
            ds["SWRadAtm"].values[:] = ds["SWRadAtm"].values*sws
            ds.to_netcdf(fdir/f.name); ds.close()
        (work/"forcingFileList.txt").write_text("".join(f"{f.name}\n" for f in sorted(fdir.glob("*.nc"))))
        return fdir

    def _write_filemanager(self, work, prefix, forcing=None):
        self.sim_start, self.sim_end = self.run_start, self.run_end                 # base class reads these
        M.RunnerMelt._write_filemanager(self, work, prefix, forcing)

    def run_year(self, x, y, work):
        self.run_start, self.run_end = f"{y-1}-10-01", f"{y}-09-30"
        params, proc = self.run_summa(x, work); out = work/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): return params, None, proc
        d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
        z = np.array(d["scalarSWE_mean"][:]).astype(float); z[np.abs(z) > 1e8] = np.nan; d.close()
        return params, pd.DataFrame(z, index=t, columns=HRUS), proc

    @staticmethod
    def pillow_curve(df, z, bands):
        zu, zl = bands; w = (z - zl)/(zu - zl)
        return w*df[BANDS[zu]].mean(1) + (1 - w)*df[BANDS[zl]].mean(1)

    def evaluate_full(self, x):
        t0 = time.time(); errs, per, ratios = [], [], {k: [] for k in PIL}
        try:
            for y in YEARS:
                work = self.work_root/f"w_{os.getpid()}_{uuid.uuid4().hex[:8]}"
                try: params, df, proc = self.run_year(x, y, work)
                finally: shutil.rmtree(work, ignore_errors=True)
                if df is None: return {"objective": FAIL_PENALTY, "reason": "summa_failed", "stderr": proc.stderr[-300:]}
                for k, (zp, _, bands) in PIL.items():
                    mod = self.pillow_curve(df, zp, bands); pil = self.pil[k].reindex(mod.index); a, b = melt_times(mod), melt_times(pil)
                    if a is None or b is None: continue
                    e = {m: a[m] - b[m] for m in ("t50", "t10", "d8020", "peak")}; per.append(dict(wy=y, pillow=k, **e, ratio=a["peak_mm"]/b["peak_mm"])); ratios[k].append(a["peak_mm"]/b["peak_mm"])
                    errs.append([abs(e["t50"]), abs(e["t10"]), abs(e["d8020"])])
            pv = {k: v[1] for k, v in params.items()}; base = {"params": pv, "eval_s": time.time() - t0}
            if len(errs) < 6: return {"objective": FAIL_PENALTY + 5, "reason": "too_few_pairs", **base}
            E = np.array(errs); parts = dict(zip(("t50", "t10", "d8020"), (E.mean(0)/SCALE_D).tolist()))
            diag = dict(per_year=per, mae_d=dict(zip(("t50", "t10", "d8020"), E.mean(0).tolist())), bias_d={m: float(np.mean([p[m] for p in per])) for m in ("t50", "t10", "d8020", "peak")},
                        peak_ratio={k: float(np.mean(v)) for k, v in ratios.items() if v})
            viol = {f"peak_ratio_{k}": max(PEAK_RATIO[0]/max(r, 1e-6) - 1.0, r/PEAK_RATIO[1] - 1.0, 0.0) for k, r in diag["peak_ratio"].items()}
            bad = {k: v for k, v in viol.items() if v > 0}; obj = float(np.mean(list(parts.values())))
            if bad: return {"objective": FAIL_PENALTY + sum(bad.values()), "reason": max(bad, key=bad.get), "violations": bad, "parts": parts, "diagnostics": diag, **base}
            return {"objective": obj, "parts": parts, "diagnostics": diag, **base}
        except subprocess.TimeoutExpired: return {"objective": FAIL_PENALTY, "reason": "timeout"}
        except Exception as e: return {"objective": FAIL_PENALTY, "reason": f"exc:{type(e).__name__}:{e}"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--popsize", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=20); ap.add_argument("--seed", type=int, default=9); ap.add_argument("--time-one", action="store_true")
    a = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); outdir = OUT_ROOT/"runs"/f"{DOM}_melt2_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    runner = RunnerMelt2(work_root=str(outdir/"work"), **CFG); x0 = seed_vector(runner, dict(SEED))
    if a.time_one:
        r = runner.evaluate_full(x0); print(json.dumps({k: r.get(k) for k in ("objective", "reason", "parts", "violations", "eval_s", "stderr")}, indent=1, default=str))
        if r.get("diagnostics"): print(pd.DataFrame(r["diagnostics"]["per_year"]).round(1).to_string(index=False)); print(r["diagnostics"]["mae_d"], r["diagnostics"]["bias_d"], r["diagnostics"]["peak_ratio"])
        return
    (outdir/"manifest.json").write_text(json.dumps(dict(domain=DOM, ts=ts, git=git_sha(), exe=DB.SUMMA_DS, free=[s.name for s in SPECS], bounds=runner.bounds, seed=SEED, dT=DT_C, z_ref=Z_REF, years=YEARS,
        precip=PRECIP, hrus=HRUS, pillows={k: v[:2] for k, v in PIL.items()}, de=vars(a), objective="melt_timing_two_pillows_selected_years"), indent=2, default=str))
    print(f"[{DOM} melt2] {len(SPECS)} free params popsize={a.popsize} maxiter={a.maxiter} workers={a.workers} -> {outdir}", flush=True); hist = []
    def cb(intermediate_result):
        hist.append(dict(t=time.time(), x=[float(v) for v in intermediate_result.x], objective=float(intermediate_result.fun)))
        (outdir/"history.json").write_text(json.dumps(hist, indent=2)); print(f"  gen {len(hist):2d}: best objective={intermediate_result.fun:.4f}", flush=True)
    res = differential_evolution(runner.evaluate, bounds=runner.bounds, workers=a.workers, popsize=a.popsize, maxiter=a.maxiter, seed=a.seed, updating="deferred", polish=False, tol=0.002, callback=cb, disp=False, x0=x0)
    (outdir/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev)), indent=2)); print("DONE", res.fun, flush=True)


if __name__ == "__main__":
    main()
