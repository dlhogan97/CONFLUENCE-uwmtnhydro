"""
Tuolumne G35 calibration, round 7 (2026-11-06), SUMMA. Set with the user after the timing-lever runs (g35_lever_tests.py / g35_lever_table.py).
Fixed physics (= lever "base"): Beer's law + canopyGapFrac prior; LAI_method monTable with the evergreen-needleleaf LAIM restored to the near-flat 5.1-6.4
(g35_lai_fix.py); thCondSoil mixConstit, thCond_soil 0.5; bcLowrTdyn presTemp, lowerBoundTemp 274.15 K. NO cloud-LW correction. Lateral routing (downHRUindex)
unchanged.
Why: the year-round canopy alone removed most of the May-Jul excess (KGE_AprJul 0.71 -> 0.77, Jul +24 -> +8 %); lower Fcapil was the cleanest timing lever
(KGE 0.80, Apr -16 -> -8 %); faster low-elevation soil moved water into Nov-Mar and dried Aug-Sep. The canopy costs ~30 mm of snow (ET+sublimation up), so
precip_scale may rise.
Objective (user: monthly terms + Apr-Jul runoff are primary): (1 - KGE_AprJul)/0.1 + mean(|Apr|,|May|,|Jun|,|Jul| volume bias)/15 %.
Limits: user 2026-11-06 "the ASO match being off is okay" -> ASO SWE nMAE <= 0.65 (was 0.556) and band melt order <= 0.14 (was 0.10), so the lever base, and fcapil (0.134),
  (0.585 / 0.127) passes. ET + sublimation 350-450 mm. Aug-Sep bias in [-60, +10] % (may dry out for the snowpatch, may not collapse).
Free (10): precip_scale [0.95, 1.30] lowP_boost [0, 0.7] tempCritRain [268.5, 271.0] split_m [2300, 2900] log_kmac_lo [-4.0, -2.0] log_kmac_hi [-4.5, -2.0]
           fieldCapacity [0.06, 0.14] Fcapil [0.005, 0.04] log_k_snow [-2.3, -1.3] mw_exp [1.5, 3.5]
Fixed at the round-5 best: log_kAnisotropic, zScale_TOPMODEL, log_routingGammaScale.   Seed: round-5 best with Fcapil 0.01 (lever "fcapil").
Every trial -> OUT/suite7_cal/trials.jsonl
usage: python g35_suite7_cal.py [--workers 22] [--popsize 2] [--maxiter 10]      |      python g35_suite7_cal.py --score <cal_day.nc>
"""
import json, uuid, time, shutil, argparse, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from scipy.optimize import differential_evolution
import g35_suite5_cal as C5
import g35_lever_tests as L
C4, S, F = C5.C4, C5.S, C5.F

OUT = F.OUT/"suite7_cal"; OUT.mkdir(exist_ok=True)
X5 = L.X5; N5 = L.N5
BOUNDS = [("precip_scale", 0.95, 1.30), ("lowP_boost", 0.0, 0.7), ("tempCritRain", 268.5, 271.0), ("split_m", 2300.0, 2900.0), ("log_kmac_lo", -4.0, -2.0),
          ("log_kmac_hi", -4.5, -2.0), ("fieldCapacity", 0.06, 0.14), ("Fcapil", 0.005, 0.04), ("log_k_snow", -2.3, -1.3), ("mw_exp", 1.5, 3.5)]
NAMES = [b[0] for b in BOUNDS]
SEED = [0.01 if n == "Fcapil" else X5[N5.index(n)] for n in NAMES]
ASO_MAX, ORDER_MAX, VAP_LO, VAP_HI, AUGSEP_LO, AUGSEP_HI, FAIL = 0.65, 0.14, 350.0, 450.0, -60.0, 10.0, 10.0
Runner = L.runner("base")


def full_x(x):
    y = list(X5)
    for n, v in zip(NAMES, x): y[N5.index(n)] = float(v)
    return y


def physical(x): return C5.physical(full_x(x))


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
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=22); ap.add_argument("--popsize", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=10); ap.add_argument("--seed", type=int, default=17); ap.add_argument("--score"); a = ap.parse_args()
    if a.score:
        obj, parts, diag = score(a.score, C4.obs_mm()); diag.pop("band_order", None); print(obj, parts); print(json.dumps(diag, indent=1, default=float)); return
    C4.obs_mm(); hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUT/"history.json").write_text(json.dumps(hist))
        print(f"[g35 suite7] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
    res = differential_evolution(evaluate, bounds=[(lo, hi) for _, lo, hi in BOUNDS], popsize=a.popsize, init="latinhypercube", x0=SEED, maxiter=a.maxiter,
                                 workers=a.workers, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUT/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), params=physical(res.x), objective=float(res.fun), nfev=int(res.nfev))))
    print(f"[g35 suite7] DONE {res.fun:.4f} nfev {res.nfev}", flush=True)


if __name__ == "__main__":
    main()
