"""
Tuolumne G35 calibration, round 4 (2026-11-03), SUMMA. Same objective and limits as the emulator round 4 (tuo_emulator_r4.py), set with the user 2026-11-03:
  objective  Apr-Jul ONLY: (1 - KGE on Apr-Jul days)/0.1 + mean |Apr-May|, |Jun-Jul| volume bias / 15 % + |Apr-Jul flow centroid| / 5 d.
             Aug-Oct is left for the snowpatch (reported only); Dec-Mar is forcing.
  limits     ASO SWE nMAE <= 0.556; elevation-band melt order (ASO late/early fraction per band, no Landsat) MAE <= 0.10;
             ET + sublimation 350-450 mm/yr (user: true total ~400; round 3 = ET 307 + sublimation 93). objective = 10 + scaled violations if any fail.
What changed from round 3 (g35_suite3_cal.py): precip_scale may rise to 1.20 (round 3 sat on its 0.95 floor; the emulator needs ~10% more water);
k_macropore above the split may reach 1e-2 and fieldCapacity 0.06 (emulator: little storage above field capacity, fast drainage); KGE Apr-Nov guard and the
per-HRU melt-order guard replaced by the Apr-Jul objective and the band melt order.
Free (10): precip_scale [0.95, 1.20]  lowP_boost [0, 0.7]  tempCritRain [270.65, 273.15] K  split_m [2300, 2900]  log_kmac_lo [-4, -2]  log_kmac_hi [-4.5, -2]
           log_kAnisotropic [0, 1.2]  zScale_TOPMODEL [1, 2.5]  fieldCapacity [0.06, 0.14]  log_routingGammaScale [3.7, 5.3]
Seeded from the round-3 feasible trial. Every trial -> OUT/suite4_cal/trials.jsonl
usage: python g35_suite4_cal.py [--workers 20] [--popsize 2] [--maxiter 14]      |      python g35_suite4_cal.py --score <cal_day.nc>   (test the scorer)
"""
import json, uuid, time, shutil, argparse, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from scipy.optimize import differential_evolution
import g35_suite3_cal as C3
import tuo_snow_emulator as SE
S, F = C3.S, C3.F

OUT = F.OUT/"suite4_cal"; OUT.mkdir(exist_ok=True)
BOUNDS = [("precip_scale", 0.95, 1.20), ("lowP_boost", 0.0, 0.7), ("tempCritRain", 270.65, 273.15), ("split_m", 2300.0, 2900.0), ("log_kmac_lo", -4.0, -2.0),
          ("log_kmac_hi", -4.5, -2.0), ("log_kAnisotropic", 0.0, 1.2), ("zScale_TOPMODEL", 1.0, 2.5), ("fieldCapacity", 0.06, 0.14), ("log_routingGammaScale", 3.7, 5.3)]
SEED = json.loads((C3.OUT/"wb"/"trial.json").read_text())["x"]                     # round-3 feasible trial, same parameter order
ASO_MAX, ORDER_MAX, VAP_LO, VAP_HI, FAIL = 0.556, 0.10, 350.0, 450.0, 10.0
EXTRA = ["scalarSnowSublimation", "scalarCanopySublimation"]
H = SE.hrus()
physical = C3.physical


class Runner(C3.Runner):
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing); oc = work/"outputControl.txt"; have = oc.read_text()
        oc.write_text(have + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in EXTRA if f"{v} |" not in have))


def score(out, obs, a0="2012-10-01", a1="2020-09-30"):
    d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    ids = np.array(d["hruId"][:]).astype(int); V = lambda v, f=1.0: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float)*f, index=t, columns=ids)
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).astype(float).ravel()[:len(t)], index=t)*86400*1000            # mm/d
    swe, et, sub = V("scalarSWE"), V("scalarTotalET", -86400), V("scalarSnowSublimation", -86400) + V("scalarCanopySublimation", -86400); d.close()
    w = H.w.reindex(ids).values; yrs = (pd.Timestamp(a1) - pd.Timestamp(a0)).days/365.25
    et_mm, sub_mm = float((et[a0:a1]*w).sum().sum()/yrs), float((sub[a0:a1]*w).sum().sum()/yrs)
    sc = SE.AsoScorer(swe.index, SE.aso_table(H, a0, a1), H); nmae, mo, fm = sc(swe[H.index].values)
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1)[a0:a1].dropna(); m = j.index.month
    pct = lambda ms: float(100*(j.s[np.isin(m, ms)].sum()/j.o[np.isin(m, ms)].sum() - 1))
    a = j[np.isin(m, range(4, 8))]; c = np.corrcoef(a.s, a.o)[0, 1]
    kge = float(1 - np.sqrt((c - 1)**2 + (a.s.std()/a.o.std() - 1)**2 + (a.s.mean()/a.o.mean() - 1)**2))
    dd = np.asarray((a.index - pd.to_datetime(a.index.year.astype(str) + "-04-01")).days)
    cen = lambda k: pd.Series(dd*a[k].values).groupby(a.index.year).sum()/a[k].groupby(a.index.year).sum()
    b = j[np.isin(m, range(4, 12))]; cb = np.corrcoef(b.s, b.o)[0, 1]
    diag = dict(KGE_AprJul=kge, NSE_AprJul=float(1 - ((a.s - a.o)**2).sum()/((a.o - a.o.mean())**2).sum()), centroid_AprJul_d=float((cen("s") - cen("o")).mean()),
                AprMay_bias=pct([4, 5]), JunJul_bias=pct([6, 7]), AugSep_bias=pct([8, 9]), OctNov_bias=pct([10, 11]),
                KGE_AprNov=float(1 - np.sqrt((cb - 1)**2 + (b.s.std()/b.o.std() - 1)**2 + (b.s.mean()/b.o.mean() - 1)**2)),
                annual_mm=float(j.s.sum()/yrs), annual_obs_mm=float(j.o.sum()/yrs), ET_mm=et_mm, subl_mm=sub_mm, vapour_mm=et_mm + sub_mm,
                aso_swe_nmae=nmae, band_order_mae=mo, band_order={int(e): [round(float(x), 2), round(float(y), 2)] for e, x, y in zip(sc.elev, sc.fo, fm)})
    parts = dict(kge=(1 - kge)/0.1, volume=(abs(diag["AprMay_bias"]) + abs(diag["JunJul_bias"]))/2/15, centroid=abs(diag["centroid_AprJul_d"])/5)
    obj = float(sum(parts.values()))
    v = max(0.0, nmae - ASO_MAX)/0.05 + max(0.0, mo - ORDER_MAX)/0.02 + max(0.0, VAP_LO - diag["vapour_mm"], diag["vapour_mm"] - VAP_HI)/10
    if v > 0: obj = FAIL + v
    return obj, parts, diag


def obs_mm():
    f = OUT/"obs_mm_day.csv"
    if not f.exists():
        r = Runner(work_root=str(OUT/"_obs"), **S.CFG); (r.obs/r.area*86400*1000).rename("obs").to_csv(f); shutil.rmtree(OUT/"_obs", ignore_errors=True)
    return pd.read_csv(f, index_col=0, parse_dates=True)["obs"]


def evaluate(x):
    soil = physical(x); t0 = time.time(); r = Runner(work_root=str(OUT/f"w_{uuid.uuid4().hex[:8]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.soil = soil
    w = Path(r.work_root)/"keep"
    try:
        _, proc = r.run_summa(np.zeros(len(r.specs)), w); out = w/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): obj, rec = FAIL + 50, dict(reason="summa_failed", stderr=proc.stderr[-300:])
        else: obj, parts, diag = score(out, obs_mm()); rec = dict(parts=parts, diagnostics=diag)
    except Exception as e:
        obj, rec = FAIL + 50, dict(reason=f"{type(e).__name__}:{e}")
    finally:
        shutil.rmtree(r.work_root, ignore_errors=True)
    with open(OUT/"trials.jsonl", "a") as fh: fh.write(json.dumps(dict(x=list(map(float, x)), params=soil, objective=obj, eval_s=time.time() - t0, **rec), default=float) + "\n")
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--popsize", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=14); ap.add_argument("--seed", type=int, default=12); ap.add_argument("--score"); a = ap.parse_args()
    if a.score:
        obj, parts, diag = score(a.score, obs_mm()); print(obj, parts); print(json.dumps(diag, indent=1, default=float)); return
    obs_mm(); hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUT/"history.json").write_text(json.dumps(hist))
        print(f"[g35 suite4] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
    res = differential_evolution(evaluate, bounds=[(lo, hi) for _, lo, hi in BOUNDS], popsize=a.popsize, init="latinhypercube", x0=SEED, maxiter=a.maxiter,
                                 workers=a.workers, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUT/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), params=physical(res.x), objective=float(res.fun), nfev=int(res.nfev))))
    print(f"[g35 suite4] DONE {res.fun:.4f} nfev {res.nfev}", flush=True)


if __name__ == "__main__":
    main()
