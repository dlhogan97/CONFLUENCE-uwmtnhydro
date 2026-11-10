#!/usr/bin/env python3
"""
run_calibration_snow.py -- STAGE 1 of the Tuolumne control: calibrate the SNOWPACK against ASO by HRU (2026-10-21).

Why: the control calibrated on streamflow alone (run_calibration_control.py) put the forcing ~4 C below the CDEC stations,
melted the alpine out far too early (ASO keeps 74-103% of early-spring SWE into May-Jul above 3300 m; the control kept 21-45%)
and sublimated ~16% of basin snowfall. Streamflow cannot tell one HRU from another, so the snow parameters are fitted here
against ASO per HRU, with the runoff parameters FIXED; stage 2 then fits runoff generation against streamflow with the snow fixed.

Forcing corrections (applied per HRU, per calendar month), seeded from stations -- observations/tuolumne_forcing_priors.json,
built by ess-project/utils/tuo_station_priors.py from 9 CDEC stations in / within ~20 km of the basin:
  temperature    dT_i,m = a_m + tA_dev + (b + tB_dev)*(z_i - Z_REF)/1000 + tNorth*northness_i
                 a_m, b = station prior (b = -4.06 C/km); ONLY tA_dev, tB_dev, tNorth are calibrated, inside narrow bounds.
                 northness = cos(aspect)*sin(slope). Replaces tempLapse/tempOffset.
  humidity       VP_CORRECTION = False (user, 2026-10-22): relative humidity is PRESERVED when temperature changes (spechum x esat(T1)/esat(T0)),
                 the convention of every earlier calibration. The station vapour-pressure correction (e_base x exp(alpha_m + beta*dz), up to ~2x
                 in May-Sep) is kept as an option: it cut sublimation from ~16% to ~8% of snowfall but removed ~28 W/m2 of latent cooling from the
                 alpine snowpack in May and added Dilley-O'Brien longwave, so net melt energy did not fall (stage-1 run Tuolumne_River_snow_20261021_120027).
  longwave       receives the Dilley-O'Brien INCREMENT from the T and q changes (never replaced).
  shortwave      x SW_SCALE fixed (control best; the FLV radiometer suggests ~1.12 -- left as a stated choice).
Snow parameters:
  albedoMinVisible linear in elevation between albMinVis_lo (lowest HRU) and albMinVis_hi (highest HRU), each [0.20, 0.60];
  albedoMinNearIR  one basin-wide value albMinNIR [0.15, 0.45] (fixed at 0.30 in the first stage-1 run);
  albedoDecayRate fixed at the SUMMA default 1e6 s. frozenPrecipMultip_low/_delta: the existing elevation ramp.
Objective (ASO per HRU, observations/tuolumne_aso_hru_swe.csv, flights WY2013-2020, coverage >= 0.8):
  swe_amount   per flight: area-weighted |model - ASO| / area-weighted ASO; mean over flights; / 0.25
  melt_order   per HRU: (mean SWE on late flights, May 20-Jul) / (mean SWE on early flights, Mar-Apr), same years; area-weighted
               |model - ASO| over HRUs with >= 100 mm early ASO; / 0.15
  total_volume streamflow, Apr-Nov (as the control), weight 0.5 -- a water-balance guard only.
Window: simulation from WY2012 (one snow spin-up year), scored WY2013-2020.

usage: python run_calibration_snow.py --workers 7 --popsize 5 --maxiter 15
       python run_calibration_snow.py --time-one          # one evaluation of the seed
"""
from __future__ import annotations
import argparse, json, os, subprocess, time, uuid, shutil
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc
from scipy.optimize import differential_evolution
from calibration_runner import ParamSpec, FAIL_PENALTY, _esat_kpa, _lw_dilley_obrien
from signature_objective import guard_runoff_ratio, guard_snow
import signature_objective_plus as P
import deepstore_build as DB
from run_calibration import OUT_ROOT, git_sha, seed_vector
import run_calibration_control as C

DOM = "Tuolumne_River"
OBSD = Path(__file__).parent/"observations"
PRI = json.loads((OBSD/"tuolumne_forcing_priors.json").read_text())
CFG = dict(C.CFG[DOM], sim_start="2011-10-01", analysis_start="2012-10-01", sim_end="2020-09-30")
SPECS = [ParamSpec("tA_dev", "forcing", -1.5, 1.5), ParamSpec("tB_dev", "forcing", -2.0, 2.0), ParamSpec("tNorth", "forcing", -2.0, 2.0),
         ParamSpec("albMinVis_lo", "snow", 0.20, 0.60), ParamSpec("albMinVis_hi", "snow", 0.20, 0.60), ParamSpec("albMinNIR", "snow", 0.15, 0.45),
         ParamSpec("frozenPrecipMultip_low", "elev", 0.85, 1.20), ParamSpec("frozenPrecipMultip_delta", "elev", 0.0, 0.60)]
# seed = best trial of the first stage-1 run (Tuolumne_River_snow_20261021_120027, humidity correction ON, objective 1.150); albMinNIR at the SUMMA default 0.30
SEED = dict(tA_dev=0.247, tB_dev=-1.6, tNorth=-0.92, albMinVis_lo=0.592, albMinVis_hi=0.539, albMinNIR=0.30, frozenPrecipMultip_low=0.851, frozenPrecipMultip_delta=0.087)
SW_SCALE = 1.0505
VP_CORRECTION = False
UNIFORM_DT = float(os.environ["SNOW_UNIFORM_DT"]) if os.environ.get("SNOW_UNIFORM_DT") else None   # e.g. -3 -> universal -3 C (LW increment follows)
LW_INCREMENT = os.environ.get("SNOW_LW_INCREMENT", "1") != "0"      # add the Dilley-O'Brien LW change for dT (the fix of 2026-10-23); SNOW_LW_INCREMENT=0 reproduces the old zero-increment behaviour for A/B tests
ALBEDO_DECAY = 1.0e6
# runoff generation held at the control best until stage 2 (k_soil is a multiplier on the a-priori field)
RUNOFF_FIXED = dict(k_soil_mult=0.1999, qSurfScale=7.394, kAnisotropic=6.104, routingGammaScale=33160.0)
ASO_MIN_COVER, EARLY_MIN_MM = 0.8, 100.0
W = dict(swe_amount=1.0, melt_order=1.0, total_volume=0.5)


def aso_objective(swe, aso, w_hru):
    """swe: daily model SWE (mm), columns = hruId. aso: date/hru/aso_mm rows. w_hru: area weights indexed by hruId."""
    a = aso[aso.date.isin(swe.index)].copy()
    a["mdl"] = [swe.at[d, h] for d, h in zip(a.date, a.hru)]; a["w"] = a.hru.map(w_hru)
    per = a.groupby("date").apply(lambda g: np.sum(g.w*(g.mdl - g.aso_mm).abs())/max(np.sum(g.w*g.aso_mm), 1.0))
    early = a[a.date.dt.month.isin([3, 4])]; late = a[((a.date.dt.month == 5) & (a.date.dt.day >= 20)) | a.date.dt.month.isin([6, 7])]
    yrs = sorted(set(early.date.dt.year) & set(late.date.dt.year))
    e = early[early.date.dt.year.isin(yrs)].groupby("hru")[["aso_mm", "mdl"]].mean(); l = late[late.date.dt.year.isin(yrs)].groupby("hru")[["aso_mm", "mdl"]].mean()
    ok = e.index[e.aso_mm >= EARLY_MIN_MM]
    fo, fm = (l.aso_mm/e.aso_mm)[ok].clip(upper=1.5), (l.mdl/e.mdl.clip(lower=1.0))[ok].clip(upper=1.5)
    wo = w_hru[ok]/w_hru[ok].sum()
    parts = dict(swe_amount=float(np.clip(per.mean()/0.25, 0, 3)), melt_order=float(np.clip(np.sum(wo*(fm - fo).abs())/0.15, 0, 3)))
    diag = dict(swe_nmae=float(per.mean()), melt_order_mae=float(np.sum(wo*(fm - fo).abs())), n_flights=int(a.date.nunique()), melt_years=yrs,
                frac_left_aso={int(h): round(float(v), 2) for h, v in fo.items()}, frac_left_mdl={int(h): round(float(v), 2) for h, v in fm.items()})
    return parts, diag


class RunnerSnow(C.RunnerControl):
    def __init__(self, **cfg):
        super().__init__(DOM, **cfg)
        self.specs = SPECS
        with xr.open_dataset(self.settings/"trialParams.nc") as tp: self.k_soil_base = tp["k_soil"].values.copy()
        with xr.open_dataset(self.settings/"attributes.nc") as at:
            self.northness = np.cos(np.deg2rad(at["aspect"].values))*np.sin(np.arctan(np.abs(at["tan_slope"].values)))
        self.dz_km = (self.elev - PRI["z_ref"])/1000.0
        self.ezn = (self.elev - self.elev.min())/(self.elev.max() - self.elev.min())
        aso = pd.read_csv(OBSD/"tuolumne_aso_hru_swe.csv", parse_dates=["date"])
        self.aso = aso[(aso.covf >= ASO_MIN_COVER) & (aso.date >= self.analysis_start) & (aso.date <= self.sim_end)][["date", "hru", "aso_mm"]]
        self.w_hru = pd.Series(self.hru_wt, index=self.hruid.astype(int))

    def _vector_to_params(self, x):
        p = {s.name: (s.kind, s.to_physical(xi)) for s, xi in zip(self.specs, x)}
        p["swScale"] = ("forcing", SW_SCALE)
        return p

    def _apply_params(self, work, params):
        super(C.RunnerControl, self)._apply_params(work, params)          # base: frozen-precip elevation ramp
        n = len(self.hruid); lo, hi = params["albMinVis_lo"][1], params["albMinVis_hi"][1]
        with nc.Dataset(work/"trialParams.nc", "a") as ds:
            ds["k_soil"][:] = self.k_soil_base*RUNOFF_FIXED["k_soil_mult"]
            for k in ("qSurfScale", "kAnisotropic", "routingGammaScale"): ds[k][:] = RUNOFF_FIXED[k]
            ds["albedoDecayRate"][:] = ALBEDO_DECAY
        DB.set_trial_param(work, "canopyGapFrac", self.gap)
        DB.set_trial_param(work, "albedoMinVisible", lo + (hi - lo)*self.ezn)
        DB.set_trial_param(work, "albedoMinNearIR", [params["albMinNIR"][1]]*n)

    def dT_month(self, params):
        """(12, nHRU) temperature correction, C."""
        if UNIFORM_DT is not None: return np.full((12, len(self.hruid)), UNIFORM_DT)      # test: one universal offset, no station/elevation/aspect structure
        t = PRI["temp"]; da, db, c = params["tA_dev"][1], params["tB_dev"][1], params["tNorth"][1]
        return np.array(t["a_month"])[:, None] + da + (t["b_per_km"] + db)*self.dz_km[None, :] + c*self.northness[None, :]

    def vp_month(self):
        v = PRI["vp"]
        return np.exp(np.array(v["ln_alpha_month"])[:, None] + v["beta_per_km"]*self.dz_km[None, :])

    def _apply_forcing(self, work, params):
        dTm, rvm = self.dT_month(params), self.vp_month(); sws = params["swScale"][1]
        fdir = work/"forcing"; fdir.mkdir(parents=True, exist_ok=True)
        y0, m0 = int(self.sim_start[:4]), int(self.sim_start[5:7]); y1, m1 = int(self.sim_end[:4]), int(self.sim_end[5:7])
        for f in sorted(self.forcing.glob("*.nc")):
            stem = f.name.split("_")[-1].replace(".nc", "")
            try: yy, mm = int(stem[:4]), int(stem[4:6])
            except ValueError: (fdir/f.name).symlink_to(f); continue
            if not ((y0, m0) <= (yy, mm) <= (y1, m1)): (fdir/f.name).symlink_to(f); continue
            with xr.open_dataset(f) as _d: ds = _d.load().copy()
            T0 = ds["airtemp"].values.copy(); q0 = ds["spechum"].values.copy(); pp = ds["airpres"].values
            T1 = T0 + dTm[mm - 1][None, :]
            if VP_CORRECTION: q1 = q0*rvm[mm - 1][None, :]                 # station-corrected vapour pressure (e ~ q at fixed p)
            else: q1 = q0*(_esat_kpa(T1)/_esat_kpa(T0))                    # preserve relative humidity
            qsat = 0.622*_esat_kpa(T1)*1000.0/(pp - 0.378*_esat_kpa(T1)*1000.0)
            q1 = np.minimum(q1, qsat)
            ds["airtemp"].values[:] = T1; ds["spechum"].values[:] = q1
            if LW_INCREMENT: ds["LWRadAtm"].values[:] = ds["LWRadAtm"].values + _lw_dilley_obrien(T1, pp, q1) - _lw_dilley_obrien(T0, pp, q0)
            ds["SWRadAtm"].values[:] = ds["SWRadAtm"].values*sws
            ds.to_netcdf(fdir/f.name); ds.close()
        return fdir

    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n" + "".join(
            f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in ("averageRoutedRunoff", "scalarSWE", "scalarSnowfall", "scalarSnowSublimation",
                                                                  "scalarCanopySublimation")))

    def evaluate_full(self, x):
        work = self.work_root/f"w_{os.getpid()}_{uuid.uuid4().hex[:8]}"; t0 = time.time()
        try:
            params, proc = self.run_summa(x, work); pv = {k: v[1] for k, v in params.items()}
            out = work/"out"/"cal_day.nc"
            if proc.returncode != 0 or not out.exists(): return {"objective": FAIL_PENALTY, "reason": "summa_failed", "params": pv, "stderr": proc.stderr[-300:]}
            d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
            V = lambda v: pd.DataFrame(np.array(d[v][:]), index=t, columns=self.hruid.astype(int)).where(lambda z: np.abs(z) < 1e8)
            swe = V("scalarSWE_mean"); q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:, 0]), index=t)
            sel = (t >= self.analysis_start) & (t <= self.sim_end); wy = (t.year + (t.month >= 10))[sel]
            ann = lambda v: (V(v)[sel]*86400).groupby(wy).sum().mean()
            sf, sg, sc = ann("scalarSnowfall_mean"), -ann("scalarSnowSublimation_mean"), -ann("scalarCanopySublimation_mean"); d.close()
            sim = (q*self.area)[sel]; sim_mm_yr = (q[sel]*86400*1000).sum()/(sel.sum()/365.25)
            peak_swe = float((swe[sel]*self.hru_wt[None, :]).sum(1).groupby(wy).max().mean())
            base = {"params": pv, "runoff_mm": float(sim_mm_yr), "peak_swe": peak_swe, "eval_s": time.time()-t0}
            if not guard_runoff_ratio(sim_mm_yr, self.precip_mm_yr): return {"objective": FAIL_PENALTY, "reason": "runoff_ratio", **base}
            if not guard_snow(peak_swe, self.peak_swe_apriori): return {"objective": FAIL_PENALTY, "reason": "snow_guard", **base}
            parts, diag = aso_objective(swe, self.aso, self.w_hru)
            obs = self.obs[self.analysis_start:self.sim_end]; j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna()
            keep = ~j.index.month.isin(C.EXCLUDE[DOM]); parts["total_volume"] = P.sig_total_volume(j.s[keep], j.o[keep])
            obj = sum(W[k]*parts[k] for k in parts)/sum(W.values())
            w = self.hru_wt
            diag.update(subl_ground_pct=float(100*np.sum(w*sg)/np.sum(w*sf)), subl_canopy_pct=float(100*np.sum(w*sc)/np.sum(w*sf)),
                        KGE=float(1 - np.sqrt((np.corrcoef(j.s, j.o)[0, 1] - 1)**2 + (j.s.std()/j.o.std() - 1)**2 + (j.s.mean()/j.o.mean() - 1)**2)))
            return {"objective": float(obj), "parts": parts, "diagnostics": diag, **base}
        except subprocess.TimeoutExpired: return {"objective": FAIL_PENALTY, "reason": "timeout"}
        except Exception as e: return {"objective": FAIL_PENALTY, "reason": f"exc:{type(e).__name__}:{e}"}
        finally: shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=7); ap.add_argument("--popsize", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=15); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--time-one", action="store_true")
    a = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); outdir = OUT_ROOT/"runs"/f"{DOM}_snow_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    runner = RunnerSnow(work_root=str(outdir/"work"), **CFG); x0 = seed_vector(runner, dict(SEED))
    if a.time_one:
        r = runner.evaluate_full(x0); runner._log_trial(x0, r)
        print(json.dumps({k: r.get(k) for k in ("objective", "reason", "parts", "diagnostics", "eval_s", "runoff_mm", "peak_swe", "stderr")}, indent=1, default=str)); return
    (outdir/"manifest.json").write_text(json.dumps(dict(domain=DOM, ts=ts, git=git_sha(), cfg=CFG, exe=DB.SUMMA_DS, free=[s.name for s in SPECS], bounds=runner.bounds,
        seed=SEED, sw_scale=SW_SCALE, vp_correction=VP_CORRECTION, albedo_decay=ALBEDO_DECAY, runoff_fixed=RUNOFF_FIXED, priors=PRI, weights=W, de=vars(a), objective="aso_objective"), indent=2, default=str))
    print(f"[{DOM} snow] {len(SPECS)} free params popsize={a.popsize} maxiter={a.maxiter} workers={a.workers} -> {outdir}", flush=True); hist = []
    def cb(intermediate_result):
        hist.append(dict(t=time.time(), x=[float(v) for v in intermediate_result.x], objective=float(intermediate_result.fun)))
        (outdir/"history.json").write_text(json.dumps(hist, indent=2)); print(f"  gen {len(hist):2d}: best objective={intermediate_result.fun:.4f}", flush=True)
    res = differential_evolution(runner.evaluate, bounds=runner.bounds, workers=a.workers, popsize=a.popsize, maxiter=a.maxiter, seed=a.seed,
                                 updating="deferred", polish=False, tol=0.005, callback=cb, disp=False, x0=x0)
    (outdir/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev)), indent=2))
    print("DONE", res.fun, flush=True)


if __name__ == "__main__":
    main()
