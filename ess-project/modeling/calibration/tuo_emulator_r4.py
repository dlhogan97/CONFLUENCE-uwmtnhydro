"""
Tuolumne round 4 in the emulator (2026-11-03): snow (tuo_snow_emulator, 9) + below-snow (tuo_below_emulator, 8) + precipitation scale P fitted together.
Set with the user 2026-11-03:
  objective  Apr-Jul ONLY: KGE on Apr-Jul days, Apr-May and Jun-Jul volume bias, Apr-Jul flow centroid (days). Aug-Oct is left for the snowpatch and only
             reported; Dec-Mar is forcing.
  limits     ASO SWE nMAE <= 0.556; elevation-band melt order (ASO, no Landsat) MAE <= 0.10; ET + sublimation 350-450 mm/yr (user: true total ~400;
             SUMMA round 3 = ET 307 + sublimation 93)
  P          scales snowfall and rain [0.95, 1.20]; sublimation keeps SUMMA's per-HRU monthly rate (the dry-air forcing issue is left alone)
Seeds run in parallel. -> OUT/suite3_cal/emulator/r4.json, r4_q.csv
usage: python tuo_emulator_r4.py [--seeds 3] [--maxiter 150]
"""
import json, argparse, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import differential_evolution
import tuo_emulator_fit as F, tuo_snow_emulator as SE, tuo_below_emulator as BE

P_BOUNDS = (0.95, 1.20)
ASO_MAX, ORDER_MAX, VAP_LO, VAP_HI = 0.556, 0.10, 350.0, 450.0
NS, NB = len(SE.PARAMS), len(BE.PARAMS)
OBS = F.obs_flow()


def unpack(u):
    return SE.params_of(u[:NS]), BE.params_of(u[NS:NS+NB]), P_BOUNDS[0] + u[-1]*(P_BOUNDS[1] - P_BOUNDS[0])


def apr_jul(s, o):
    j = pd.concat([s.rename("s"), o.rename("o")], axis=1)[F.A0:F.A1].dropna(); a = j[j.index.month.isin(range(4, 8))]
    d = np.asarray((a.index - pd.to_datetime(a.index.year.astype(str) + "-04-01")).days)
    cen = lambda c: pd.Series(d*a[c].values).groupby(a.index.year).sum()/a[c].groupby(a.index.year).sum()
    return dict(KGE_AprJul=F.kge(a.s, a.o), NSE_AprJul=F.nse(a.s, a.o), centroid_AprJul_d=float((cen("s") - cen("o")).mean()))


def run(u):
    ps, pb, P = unpack(u); Xs = {**F.X, "snow": F.X["snow"]*P, "rain": F.X["rain"]*P}
    swe, swi = SE.simulate(ps, Xs, F.H, F.SRATE); nmae, mo, fm = F.SCORER(swe.values)
    q, et = BE.simulate(pb, swi, swe > 1.0, F.X["SW"], F.X["T"], F.H, F.SPLIT, F.TOPO)
    w = F.H.w.values; sl = slice(F.A0, F.A1); yrs = (pd.Timestamp(F.A1) - pd.Timestamp(F.A0)).days/365.25
    inp = ((Xs["snow"][sl] + Xs["rain"][sl].clip(lower=0))*w).sum().sum(); dswe = ((swe.loc[F.A1] - swe.loc[F.A0])*w).sum()
    subl = float((inp - (swi[sl]*w).sum().sum() - dswe)/yrs); etm = float((et[sl]*w).sum().sum()/yrs)
    d = dict(P=P, aso_nmae=nmae, band_order_mae=mo, band_order={int(e): [round(float(a), 2), round(float(m), 2)] for e, a, m in zip(F.SCORER.elev, F.SCORER.fo, fm)},
             ET=etm, subl=subl, vapour=etm + subl, **apr_jul(q, OBS), **F.flow_diag(q, OBS))
    return d, q


def objective(u):
    d, _ = run(u)
    j = (1 - d["KGE_AprJul"])/0.1 + (abs(d["AprMay"]) + abs(d["JunJul"]))/2/15 + abs(d["centroid_AprJul_d"])/5
    v = max(d["aso_nmae"] - ASO_MAX, 0)/0.05 + max(d["band_order_mae"] - ORDER_MAX, 0)/0.02 + max(VAP_LO - d["vapour"], 0, d["vapour"] - VAP_HI)/10
    return j + 10*v


def one(args):
    seed, maxiter = args
    r = differential_evolution(objective, [(0, 1)]*(NS + NB + 1), seed=seed, maxiter=maxiter, popsize=12, tol=1e-7, polish=False)
    return seed, float(r.fun), list(map(float, r.x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--maxiter", type=int, default=150); a = ap.parse_args()
    with ProcessPoolExecutor(a.seeds) as ex: res = list(ex.map(one, [(s, a.maxiter) for s in range(1, a.seeds + 1)]))
    for s, f, _ in res: print(f"seed {s}: {f:.4f}", flush=True)
    s, f, x = min(res, key=lambda r: r[1]); d, q = run(np.array(x)); ps, pb, P = unpack(np.array(x))
    q.rename("q_mm_d").to_csv(F.OUT/"r4_q.csv")
    ref = F.flow_diag(F.X["q_routed"], OBS); ref.update(apr_jul(F.X["q_routed"], OBS))
    out = dict(objective=f, seed=s, snow=ps, below=pb, **d, seeds={r[0]: r[1] for r in res}, summa_round3=ref)
    (F.OUT/"r4.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
