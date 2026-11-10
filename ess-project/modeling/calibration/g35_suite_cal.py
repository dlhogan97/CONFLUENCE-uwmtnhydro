"""
Tuolumne G35 calibration suite (2026-10-28): spring water through realistic low-elevation precipitation and downslope soil drainage.
Builds on g35_soil_tests.py round 2 (tm_lat_fc best KGE 0.66; lateral drainage closed most of the Mar-May deficit; tempCritRain +2 C did nothing;
draining the low bands straight to the stream overshot and removes the high-to-low transfer, so it is NOT used here).
Fixed: G35 albedo ramps, forcing (-2 C + its LW increment, swScale / lwOffset), frozenPrecipMultip ramp, every ET parameter, thinned soil
(depth 0.5 limited by roots), downHRUindex chain intact (water reaches the stream only through the outlet HRUs).
Free (6):
  log_kAnisotropic   [0, 1]          kAnisotropic 1-10 (documented range 0.1-10; round 2 used 10)
  fieldCapacity      [0.10, 0.20]    lateral drainage starts above this moisture
  zScale_TOPMODEL    [1.0, 2.5]      transmissivity exponent (1 = linear in drainable storage)
  log_k_macropore    [-4, -2]        1e-4 - 1e-2 m/s
  lowP_boost         [0, 0.7]        precipitation (rain and snow) x (1 + boost * w), w = 1 at the lowest HRU (1513 m) -> 0 at 2750 m.
                                     Hetch Hetchy (HTH, 1180 m, CDEC) records 855 mm/yr WY2013-2020 before undercatch; the forcing gives 523 mm at
                                     1513-1702 m and 624 mm at 2060 m, so 0.7 brings the lowest band to ~890 mm (about the gauge) and no further.
  log_routingGammaScale [3.7, 5.3]   hillslope routing gamma scale 5e3 - 2e5 s (now 24863 s, shape 2.5)
Objective (equal weights, each clipped at 3; NO low-flow or memory terms), WY2013-2020:
  peak_timing, peak_volume, falling_limb, com_timing      signature_objective_routing
  monthly_bias        sum|monthly-mean bias| Dec-Jul / sum(obs) / 25%  (Dec-Mar now scored: the precip boost is the fix for that deficit)
  com_timing here uses 14 d, not 7: at 10% / 7 d both terms sat at the clip (3) on the round-2 seed and gave the optimizer no gradient
  spring_volume       |Mar-May volume sim/obs - 1| / 10%
  total_volume        |water-year volume sim/obs - 1| / 5%  (keeps the soil from buying spring flow by cutting ET)
  swe_amount, melt_order   ASO terms (run_calibration_snow.aso_objective): snowpack amount and elevation structure as calibrated
Guards: KGE >= 0.5; ASO SWE nMAE <= 1.15 x G35 (0.483). Reported only: lateral inflow to HRUs below 2500 m as % of their rain + melt, rain fraction
below 2500 m, low-band precipitation, ET, runoff split, low-flow memory lag-1.
Every trial -> OUT/suite_cal/trials.jsonl.
usage: python g35_suite_cal.py [--workers 24] [--popsize 4] [--maxiter 14]
"""
import os, json, uuid, time, shutil, argparse, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from scipy.optimize import differential_evolution
import deepstore_build as DB
import run_calibration_snow as S
import signature_objective_plus as P
import full_basin_albedo as F
import g35_soil_tests as ST

OUT = F.OUT/"suite_cal"; OUT.mkdir(exist_ok=True)
BOUNDS = [("log_kAnisotropic", 0.0, 1.0), ("fieldCapacity", 0.10, 0.20), ("zScale_TOPMODEL", 1.0, 2.5), ("log_k_macropore", -4.0, -2.0),
          ("lowP_boost", 0.0, 0.7), ("log_routingGammaScale", 3.7, 5.3)]
SEED = [1.0, 0.12, 1.781, -3.0, 0.0, np.log10(24863.0)]                 # round-2 tm_lat_fc
Z_LO, Z_HI = 1513.0, 2750.0
KGE_MIN, ASO_MAX, FAIL = 0.5, 1.15*0.483, 10.0
HRUV = ["scalarRainPlusMelt", "scalarRainfall", "scalarSnowfall", "scalarTotalET", "scalarSWE", "mLayerColumnInflow"]
BASV = ["averageRoutedRunoff", "basin__SurfaceRunoff"]


def physical(x):
    p = dict(zip([b[0] for b in BOUNDS], map(float, x)))
    return dict(kAnisotropic=10**p["log_kAnisotropic"], fieldCapacity=p["fieldCapacity"], zScale_TOPMODEL=p["zScale_TOPMODEL"],
                k_macropore=10**p["log_k_macropore"], lowP_boost=p["lowP_boost"], routingGammaScale=10**p["log_routingGammaScale"], depth=0.5)


class Runner(ST.Runner):
    def _apply_params(self, work, params):
        super()._apply_params(work, params)
        with nc.Dataset(work/"trialParams.nc", "a") as ds: ds["routingGammaScale"][:] = self.soil["routingGammaScale"]
    def _apply_forcing(self, work, params):
        fdir = super()._apply_forcing(work, params); b = self.soil["lowP_boost"]
        if b > 0:
            m = 1.0 + b*np.clip((Z_HI - self.elev)/(Z_HI - Z_LO), 0.0, 1.0)
            for f in sorted(fdir.glob("*.nc")):
                with nc.Dataset(f, "a") as d: d["pptrate"][:] = np.array(d["pptrate"][:])*m[None, :]
        return fdir
    def _write_filemanager(self, work, prefix, forcing=None):
        F.Runner._write_filemanager(self, work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n" + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in BASV + HRUV))


def score(out, r, soil):
    d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    ids = r.hruid.astype(int); H = {}
    for v in HRUV:
        a = np.array(d[f"{v}_mean"][:]).astype(float); a[np.abs(a) > 1e8] = np.nan
        H[v] = pd.DataFrame(np.nansum(a, axis=1) if a.ndim == 3 else a, index=t, columns=ids)     # layered (time, midSoil, hru) -> column total
    B = {v: pd.Series(np.array(d[f"{v}_mean"][:]).astype(float).ravel()[:len(t)], index=t) for v in BASV}; d.close()
    sel = (t >= r.analysis_start) & (t <= r.sim_end); q = (B["averageRoutedRunoff"]*r.area)[sel]; obs = r.obs[r.analysis_start:r.sim_end]
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1).dropna()
    rt = P.signature_objective_routing(j.s, j.o, bias_months=(12, 1, 2, 3, 4, 5, 6, 7)); parts = dict(rt.parts)
    mm = lambda m: j[j.index.month.isin(m)].sum()
    m = j.index.month.isin([12, 1, 2, 3, 4, 5, 6, 7]); g = j[m].groupby(j[m].index.month).mean()
    parts["monthly_bias"] = float(np.clip((g.s - g.o).abs().sum()/g.o.sum()/0.25, 0, 3))
    cs, co = P._com_doy(j.s), P._com_doy(j.o); cc = cs.index.intersection(co.index); parts["com_timing"] = float(np.clip((cs[cc] - co[cc]).abs().mean()/14.0, 0, 3))
    sp = mm([3, 4, 5]); parts["spring_volume"] = float(np.clip(abs(sp.s/sp.o - 1)/0.10, 0, 3)); parts["total_volume"] = P.sig_total_volume(j.s, j.o)
    ap, ad = S.aso_objective(H["scalarSWE"], r.aso, r.w_hru); parts["swe_amount"], parts["melt_order"] = ap["swe_amount"], ap["melt_order"]
    obj = float(np.mean(list(parts.values()))); kge = rt.diagnostics["KGE"]
    w = r.hru_wt; low = r.elev < 2500.0; ar = np.array(r.hru_area) if hasattr(r, "hru_area") else w*r.area; yrs = sel.sum()/365.25
    tot = lambda v, m, k=1000: float((H[v][sel].values[:, m]*86400*k).sum(0).dot(w[m])/w[m].sum()/yrs)   # k: m/s -> mm (rain/snow are already mm/s)
    lat_in = float((H["mLayerColumnInflow"][sel].values[:, low]*86400).sum(0).sum()/ar[low].sum()*1000/yrs)
    rpm, rain, snow = tot("scalarRainPlusMelt", low), tot("scalarRainfall", low, 1), tot("scalarSnowfall", low, 1)
    diag = dict(KGE=kge, NSE=rt.diagnostics["NSE"], aso_swe_nmae=ad["swe_nmae"], MarMay_mm=float(sp.s/r.area*86400*1000/yrs), annual_mm=float(j.s.sum()/r.area*86400*1000/yrs),
                ET_mm=float(-(H["scalarTotalET"][sel]*w).sum(1).sum()*86400/yrs), surface_share=float((B["basin__SurfaceRunoff"]*r.area)[sel].sum()/q.sum()),
                lowband_lat_in_pct_of_input=100*lat_in/max(rpm, 1e-9), lowband_rain_frac=rain/max(rain + snow, 1e-9), lowband_precip_mm=rain + snow,
                resid_lag1_sim=rt.diagnostics["resid_lag1_sim"], com_bias_d=rt.diagnostics["com_bias_d"], total_vol_bias_pct=rt.diagnostics["total_vol_bias_pct"])
    if kge < KGE_MIN: obj = FAIL + (KGE_MIN - kge)*10
    elif ad["swe_nmae"] > ASO_MAX: obj = FAIL + (ad["swe_nmae"] - ASO_MAX)*10
    return obj, parts, diag


def evaluate(x):
    soil = physical(x); t0 = time.time(); r = Runner(work_root=str(OUT/f"w_{uuid.uuid4().hex[:8]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.soil = soil
    w = Path(r.work_root)/"keep"
    try:
        _, proc = r.run_summa(np.zeros(len(r.specs)), w); out = w/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): obj, rec = FAIL, dict(reason="summa_failed", stderr=proc.stderr[-300:])
        else: obj, parts, diag = score(out, S.RunnerSnow(work_root=str(OUT/"score"), **S.CFG), soil); rec = dict(parts=parts, diagnostics=diag)
    except Exception as e:
        obj, rec = FAIL, dict(reason=f"{type(e).__name__}:{e}")
    finally:
        shutil.rmtree(r.work_root, ignore_errors=True)
    with open(OUT/"trials.jsonl", "a") as fh: fh.write(json.dumps(dict(x=list(map(float, x)), params=soil, objective=obj, eval_s=time.time() - t0, **rec), default=float) + "\n")
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--popsize", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=14); ap.add_argument("--seed", type=int, default=5); a = ap.parse_args(); hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUT/"history.json").write_text(json.dumps(hist))
        print(f"[g35 suite] gen {len(hist):2d}: best {hist[-1]:.4f}  x {np.round(intermediate_result.x, 3).tolist()}", flush=True)
    res = differential_evolution(evaluate, bounds=[(lo, hi) for _, lo, hi in BOUNDS], popsize=a.popsize, init="latinhypercube", x0=SEED, maxiter=a.maxiter,
                                 workers=a.workers, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUT/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), params=physical(res.x), objective=float(res.fun), nfev=int(res.nfev))))
    print(f"[g35 suite] DONE {res.fun:.4f} nfev {res.nfev}", flush=True)


if __name__ == "__main__":
    main()
