"""
Tuolumne G35 calibration, round 6 (2026-11-06), SUMMA. Shorter run, set with the user 2026-11-06.
New fixed physics (from the snowpack/soil energy work of 11-05/06):
  - cloud-based incoming LW, strength 1 (g35_cloud_lw.py): storm hours overcast (eps -> ~1), clear sky unchanged. On the round-5 best it gave KGE_AprJul 0.80
    and removed the Jun-Jul excess, but cut Apr 1 SWE 4-17 % (ASO/order failed), so precip_scale may now rise to 1.30
  - soil thermal conductivity: thCondSoil mixConstit, thCond_soil 0.5 (moist bulk ~0.45 W/m/K vs ~1.7 with funcSoilWet); on round 5 alone it gave obj 3.59 -> 3.04
  - ground heat: bcLowrTdyn presTemp with lowerBoundTemp FREE [273.16, 277.15] K (user: these soils should essentially never freeze)
Objective (user: monthly terms + Apr-Jul runoff are primary; recession is nice but the snowpatch should be what reduces late-season bias):
  (1 - KGE on Apr-Jul days)/0.1 + mean(|Apr|, |May|, |Jun|, |Jul| monthly volume bias)/15 %. No centroid term.
Limits: as round 4 (ASO SWE nMAE <= 0.556, band melt order <= 0.10, ET + sublimation 350-450 mm) PLUS Aug-Sep volume bias in [-60, +10] %: the control must not
  be wet in late summer, or the snowpatch cannot show a bias reduction. Dec-Mar is excluded (forcing).
Free (8): precip_scale [0.95, 1.30]  lowP_boost [0, 0.7]  tempCritRain [268.5, 271.0]  split_m [2300, 2900]  log_kmac_lo [-4, -2]  log_kmac_hi [-4.5, -2]
          fieldCapacity [0.06, 0.14]  lowerBoundTemp [273.16, 277.15]
Fixed at the round-5 best: log_kAnisotropic, zScale_TOPMODEL, log_routingGammaScale, Fcapil, k_snow, mw_exp.
Every trial -> OUT/suite6_cal/trials.jsonl
usage: python g35_suite6_cal.py [--workers 16] [--popsize 2] [--maxiter 10]      |      python g35_suite6_cal.py --score <cal_day.nc>
"""
import json, uuid, time, shutil, argparse, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from scipy.optimize import differential_evolution
import g35_suite5_cal as C5
import g35_cloud_lw as CL
import deepstore_build as DB
C4, C3, S, F = C5.C4, C5.C3, C5.S, C5.F

OUT = F.OUT/"suite6_cal"; OUT.mkdir(exist_ok=True)
R5 = json.loads((C5.OUT/"best"/"best_obj"/"trial.json").read_text())["x"]             # 13: 10 round-4 params + Fcapil, log_k_snow, mw_exp
N5 = [b[0] for b in C5.BOUNDS]
BOUNDS = [("precip_scale", 0.95, 1.30), ("lowP_boost", 0.0, 0.7), ("tempCritRain", 268.5, 271.0), ("split_m", 2300.0, 2900.0), ("log_kmac_lo", -4.0, -2.0),
          ("log_kmac_hi", -4.5, -2.0), ("fieldCapacity", 0.06, 0.14), ("lowerBoundTemp", 273.16, 277.15)]
NAMES = [b[0] for b in BOUNDS]
SEED = [R5[N5.index(n)] for n in NAMES[:-1]] + [274.15]
LW_STRENGTH, THCOND = 1.0, 0.5
ASO_MAX, ORDER_MAX, VAP_LO, VAP_HI, AUGSEP_LO, AUGSEP_HI, FAIL = 0.556, 0.10, 350.0, 450.0, -60.0, 10.0, 10.0


def full_x(x):
    """8 free values -> the 13-vector of round 5 (fixed entries from the round-5 best)."""
    y = list(R5)
    for n, v in zip(NAMES[:-1], x[:-1]): y[N5.index(n)] = float(v)
    return y


def physical(x):
    p = C5.physical(full_x(x)); p["lowerBoundTemp"] = float(x[-1]); return p


class Runner(C5.Runner):
    def _subset(self, work):
        super()._subset(work)
        with nc.Dataset(work/"trialParams.nc") as t: n = len(np.array(t["hruId"][:]).ravel())
        DB.set_trial_param(work, "thCond_soil", np.full(n, THCOND)); DB.set_trial_param(work, "lowerBoundTemp", np.full(n, self.soil["lowerBoundTemp"]))

    def _apply_forcing(self, work, params):
        fdir = super()._apply_forcing(work, params); CL.apply(fdir, self.forcing, LW_STRENGTH); return fdir

    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing); md = work/"modelDecisions.txt"; txt = md.read_text()
        assert "bcLowrTdyn     zeroFlux" in txt and "thCondSoil     funcSoilWet" in txt, "unexpected modelDecisions lines"
        md.write_text(txt.replace("bcLowrTdyn     zeroFlux", "bcLowrTdyn     presTemp").replace("thCondSoil     funcSoilWet", "thCondSoil     mixConstit "))


def score(out, obs, a0="2012-10-01", a1="2020-09-30"):
    _, _, diag = C4.score(out, obs, a0, a1)
    d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).astype(float).ravel()[:len(t)], index=t)*86400*1000; d.close()
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1)[a0:a1].dropna(); m = j.index.month
    for mm, lab in zip(range(4, 8), ("Apr", "May", "Jun", "Jul")): diag[f"{lab}_bias"] = float(100*(j.s[m == mm].sum()/j.o[m == mm].sum() - 1))
    parts = dict(kge=(1 - diag["KGE_AprJul"])/0.1, monthly=float(np.mean([abs(diag[f"{l}_bias"]) for l in ("Apr", "May", "Jun", "Jul")]))/15)
    obj = float(sum(parts.values()))
    v = (max(0.0, diag["aso_swe_nmae"] - ASO_MAX)/0.05 + max(0.0, diag["band_order_mae"] - ORDER_MAX)/0.02
         + max(0.0, VAP_LO - diag["vapour_mm"], diag["vapour_mm"] - VAP_HI)/10 + max(0.0, AUGSEP_LO - diag["AugSep_bias"], diag["AugSep_bias"] - AUGSEP_HI)/5)
    if v > 0: obj = FAIL + v
    return obj, parts, diag


def evaluate(x):
    soil = physical(x); t0 = time.time(); r = Runner(work_root=str(OUT/f"w_{uuid.uuid4().hex[:8]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.soil = soil
    w = Path(r.work_root)/"keep"
    try:
        _, proc = r.run_summa(np.zeros(len(r.specs)), w); out = w/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): obj, rec = FAIL + 50, dict(reason="summa_failed", stderr=proc.stderr[-300:])
        else: obj, parts, diag = score(out, C4.obs_mm()); diag.pop("band_order", None); rec = dict(parts=parts, diagnostics=diag)
    except Exception as e:
        obj, rec = FAIL + 50, dict(reason=f"{type(e).__name__}:{e}")
    finally:
        shutil.rmtree(r.work_root, ignore_errors=True)
    with open(OUT/"trials.jsonl", "a") as fh: fh.write(json.dumps(dict(x=list(map(float, x)), params=soil, objective=obj, eval_s=time.time() - t0, **rec), default=float) + "\n")
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--popsize", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=10); ap.add_argument("--seed", type=int, default=16); ap.add_argument("--score"); a = ap.parse_args()
    if a.score:
        obj, parts, diag = score(a.score, C4.obs_mm()); diag.pop("band_order", None); print(obj, parts); print(json.dumps(diag, indent=1, default=float)); return
    C4.obs_mm(); assert CL.CACHE.exists(), "clear-sky SW cache missing: run g35_cloud_lw.clear_sw(<raw forcing dir>) first"
    hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUT/"history.json").write_text(json.dumps(hist))
        print(f"[g35 suite6] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
    res = differential_evolution(evaluate, bounds=[(lo, hi) for _, lo, hi in BOUNDS], popsize=a.popsize, init="latinhypercube", x0=SEED, maxiter=a.maxiter,
                                 workers=a.workers, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUT/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), params=physical(res.x), objective=float(res.fun), nfev=int(res.nfev))))
    print(f"[g35 suite6] DONE {res.fun:.4f} nfev {res.nfev}", flush=True)


if __name__ == "__main__":
    main()
