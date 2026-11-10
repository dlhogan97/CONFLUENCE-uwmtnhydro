"""
Tuolumne G35 calibration, round 3 (2026-11-02): seeded from the elevation-split soil test (g35_et_tests.py p100_split), under the user's rules of 2026-10-30.
What rounds 2 and the ET tests taught: ET (~240 mm) is limited by fast soil drainage, not by water, roots or LAI; precipitation at its 0.90 floor broke melt order;
slow macropore flow above ~2600 m buys ET while fast flow below keeps spring runoff (p100_split: ET 340 mm, ASO and melt order inside limits, KGE 0.55 uncalibrated).
Free (10):
  precip_scale [0.95, 1.10]   lowP_boost [0, 0.7]   tempCritRain [270.65, 273.15] K (rain ABOVE it)
  split_m [2300, 2900]        elevation dividing fast (below) from slow (above) macropore flow
  log_kmac_lo [-4, -2]        log_kmac_hi [-4.5, -3]   k_macropore below / above split, m/s
  log_kAnisotropic [0, 1]     zScale_TOPMODEL [1, 2.5]   fieldCapacity [0.10, 0.14] (above ~0.16 late-season flow stops)   log_routingGammaScale [3.7, 5.3]
Objective (equal weights, WY2013-2020): peak_timing, peak_volume, falling_limb, com_timing, monthly_bias (Apr-Oct)   [signature_objective_routing]
  No Dec-Mar term (winter error is forcing), no recession term (reported only), no low-flow level or memory term.
Rejection (objective = 10 + summed scaled violations, so the search can climb toward feasibility):
  KGE on Apr-Nov days < 0.75;  ET outside 300-380 mm/yr (user: ET ~100 mm above the ~240 mm of round 2; sublimation not counted);
  ASO SWE nMAE > 1.15 x G35 (0.556);  ASO melt-order error > 1.15 x G35 (0.520).
Every trial -> OUT/suite3_cal/trials.jsonl.
usage: python g35_suite3_cal.py [--workers 20] [--popsize 2] [--maxiter 14]
"""
import json, uuid, time, shutil, argparse, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from scipy.optimize import differential_evolution
import deepstore_build as DB
import g35_suite_cal as G, g35_suite2_cal as C2
import signature_objective_plus as P
from east_ctl_treat_cal_v2 import recession_taus
S, F = C2.S, C2.F

OUT = F.OUT/"suite3_cal"; OUT.mkdir(exist_ok=True)
BOUNDS = [("precip_scale", 0.95, 1.10), ("lowP_boost", 0.0, 0.7), ("tempCritRain", 270.65, 273.15), ("split_m", 2300.0, 2900.0), ("log_kmac_lo", -4.0, -2.0),
          ("log_kmac_hi", -4.5, -3.0), ("log_kAnisotropic", 0.0, 1.0), ("zScale_TOPMODEL", 1.0, 2.5), ("fieldCapacity", 0.10, 0.14), ("log_routingGammaScale", 3.7, 5.3)]
SEED = [1.00, 0.563, 271.895, 2600.0, np.log10(3.62e-3), -4.0, np.log10(6.052), 1.726, 0.108, np.log10(22155.7)]       # p100_split
KGE_MIN, ET_LO, ET_HI, ASO_MAX, MO_MAX, FAIL = 0.75, 300.0, 380.0, 1.15*0.483, 1.15*0.452, 10.0


def physical(x):
    p = dict(zip([b[0] for b in BOUNDS], map(float, x)))
    return dict(precip_scale=p["precip_scale"], lowP_boost=p["lowP_boost"], tempCritRain=p["tempCritRain"], split_m=p["split_m"], k_macropore=10**p["log_kmac_lo"],
                k_macropore_hi=10**p["log_kmac_hi"], kAnisotropic=10**p["log_kAnisotropic"], zScale_TOPMODEL=p["zScale_TOPMODEL"], fieldCapacity=p["fieldCapacity"],
                routingGammaScale=10**p["log_routingGammaScale"], depth=0.5)


class Runner(C2.Runner):
    def _apply_params(self, work, params):
        super()._apply_params(work, params); s = self.soil
        DB.set_trial_param(work, "k_macropore", np.where(self.elev < s["split_m"], s["k_macropore"], s["k_macropore_hi"]))


def score(out, r, soil):
    _, p1, diag = G.score(out, r, soil)
    d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).astype(float).ravel()[:len(t)], index=t)*r.area; d.close()
    j = pd.concat([q.rename("s"), r.obs.rename("o")], axis=1)[r.analysis_start:r.sim_end].dropna()
    rt = P.signature_objective_routing(j.s, j.o); parts = {k: float(rt.parts[k]) for k in ("peak_timing", "peak_volume", "falling_limb", "com_timing", "monthly_bias")}
    a = j[j.index.month.isin(range(4, 12))]; c = a.s.corr(a.o); kge = float(1-np.sqrt((c-1)**2+(a.s.std()/a.o.std()-1)**2+(a.s.mean()/a.o.mean()-1)**2))
    y0, y1 = int(r.analysis_start[:4])+1, int(r.sim_end[:4]); mo = j.index.month
    pct = lambda m: float(100*(j.s[np.isin(mo, m)].sum()/j.o[np.isin(mo, m)].sum()-1))
    low7 = lambda s: pd.Series({y: s[f"{y}-08-01":f"{y}-11-30"].rolling(7, center=True, min_periods=7).mean().min() for y in range(y0, y1)})
    diag.update(KGE_AprNov=kge, NSE_AprNov=float(1-((a.s-a.o)**2).sum()/((a.o-a.o.mean())**2).sum()), aso_melt_order=p1["melt_order"], recess_sim_d=float(recession_taus(j.s, y0, y1).median()),
                recess_obs_d=float(recession_taus(j.o, y0, y1).median()), AprNov_bias=pct(range(4, 12)), AprMay_bias=pct([4, 5]), JunJul_bias=pct([6, 7]), AugSep_bias=pct([8, 9]),
                OctNov_bias=pct([10, 11]), min7_ratio=float((low7(j.s)/low7(j.o)).median()))
    obj = float(np.mean(list(parts.values())))
    v = (max(0.0, KGE_MIN-kge)*10 + max(0.0, ET_LO-diag["ET_mm"], diag["ET_mm"]-ET_HI)/20.0 + max(0.0, diag["aso_swe_nmae"]-ASO_MAX)*10 + max(0.0, p1["melt_order"]-MO_MAX)*10)
    if v > 0: obj = FAIL + v
    return obj, parts, diag


def evaluate(x):
    soil = physical(x); t0 = time.time(); r = Runner(work_root=str(OUT/f"w_{uuid.uuid4().hex[:8]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.soil = soil
    w = Path(r.work_root)/"keep"
    try:
        _, proc = r.run_summa(np.zeros(len(r.specs)), w); out = w/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): obj, rec = FAIL+50, dict(reason="summa_failed", stderr=proc.stderr[-300:])
        else: obj, parts, diag = score(out, S.RunnerSnow(work_root=str(OUT/"score"), **S.CFG), soil); rec = dict(parts=parts, diagnostics=diag)
    except Exception as e:
        obj, rec = FAIL+50, dict(reason=f"{type(e).__name__}:{e}")
    finally:
        shutil.rmtree(r.work_root, ignore_errors=True)
    with open(OUT/"trials.jsonl", "a") as fh: fh.write(json.dumps(dict(x=list(map(float, x)), params=soil, objective=obj, eval_s=time.time()-t0, **rec), default=float)+"\n")
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--popsize", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=14); ap.add_argument("--seed", type=int, default=11); a = ap.parse_args(); hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUT/"history.json").write_text(json.dumps(hist))
        print(f"[g35 suite3] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
    res = differential_evolution(evaluate, bounds=[(lo, hi) for _, lo, hi in BOUNDS], popsize=a.popsize, init="latinhypercube", x0=SEED, maxiter=a.maxiter,
                                 workers=a.workers, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUT/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), params=physical(res.x), objective=float(res.fun), nfev=int(res.nfev))))
    print(f"[g35 suite3] DONE {res.fun:.4f} nfev {res.nfev}", flush=True)


if __name__ == "__main__":
    main()
