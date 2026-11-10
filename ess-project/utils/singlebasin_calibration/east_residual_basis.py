"""
Does the East memory result depend on what the fall minimum is adjusted for? (2026-10-28)
ln(Aug 1-Nov 30 min 7-d mean) regressed on, per water year:
  peak      ln(Apr-Jul max 7-d mean)            [the standard metric]
  AprJul    ln(Apr-Jul mean flow) = runoff volume
  WY        ln(water-year mean flow)
Spearman lag-1/lag-2 of the residual, pairs matched by year, two-sided permutation p; WY1987-2024 and WY2009-2024.
usage: STAGEB_T0=1981-10-01 STAGEB_END=2025-09-30 python east_residual_basis.py
"""
import os, json, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd
import emu_core as E, east_ctl_treat_cal as C, east_threshold_cal as T
from east_model_inventory_lite import load_main


def drivers(q, y0, y1):
    lf, pk, aj, wy = {}, {}, {}, {}
    for y in range(y0, y1 + 1):
        a = q[f"{y}-08-01":f"{y}-11-30"]
        if a.isna().mean() <= 0.1 and len(a) > 100:
            m = a.rolling(7, center=True, min_periods=7).mean().min()
            if np.isfinite(m) and m > 0.01: lf[y] = m
        b = q[f"{y}-04-01":f"{y}-07-31"]
        if b.isna().mean() <= 0.05: pk[y] = b.interpolate(limit=5).rolling(7, center=True, min_periods=7).mean().max(); aj[y] = b.mean()
        w = q[f"{y-1}-10-01":f"{y}-09-30"]
        if w.isna().mean() <= 0.05: wy[y] = w.mean()
    return np.log(pd.DataFrame(dict(lf=lf, peak=pk, AprJul=aj, WY=wy))).dropna()


def lag_p(r, k, n=2000, seed=0):
    f = lambda s: E.lag_r(s, k); est = f(r); rng = np.random.default_rng(seed + k)
    null = [f(pd.Series(rng.permutation(r.values), index=r.index)) for _ in range(n)]
    return est, (np.sum(np.abs(null) >= abs(est)) + 1)/(n + 1)


rows = []
for name, q in load_main().items():
    for (y0, y1) in ((1987, 2024), (2009, 2024)):
        d = drivers(q, y0, y1)
        for basis in ("peak", "AprJul", "WY"):
            r = pd.Series(d.lf - np.polyval(np.polyfit(d[basis], d.lf, 1), d[basis]), index=d.index)
            (a1, p1), (a2, p2) = lag_p(r, 1), lag_p(r, 2)
            rows.append(dict(model=name, window=f"{y0}-{y1}", basis=basis, rho1=a1, p1=p1, rho2=a2, p2=p2,
                             r2_driver=np.corrcoef(d[basis], d.lf)[0, 1]**2))
R = pd.DataFrame(rows); R.to_csv(C.X.OUT/"east_residual_basis.csv", index=False)
fmt = lambda a, p: f"{a:+.2f}{'*' if p < 0.05 else ('.' if p < 0.10 else ' ')}"
for w in ("1987-2024", "2009-2024"):
    sub = R[R.window == w]
    tab = sub.assign(v=[f"{fmt(a, p)} / {fmt(b, q)}" for a, p, b, q in zip(sub.rho1, sub.p1, sub.rho2, sub.p2)]).pivot(index="model", columns="basis", values="v")
    print(f"\nWY{w}: rho1 / rho2 of the residual  (* p<0.05, . p<0.10)"); print(tab[["peak", "AprJul", "WY"]].reindex(list(load_main().keys())).to_string())
print("\nshare of fall-min variance explained by the driver (obs, WY1987-2024):", R[(R.model == "observed") & (R.window == "1987-2024")].set_index("basis").r2_driver.round(2).to_dict())
