"""
Minimal-calibration route to a shallow layer that drains before fall (2026-10-27): LINEAR shallow bucket with timescale tau_s (no free shape parameters),
linear leak f*S/tau_s to a linear deep store (tau_d = 2 yr), f from the Dec-Feb volume balance, routing scale fixed at the routing control (48921 s).
Sweep tau_s; the choice is the tau_s whose TOTAL-flow late-summer recession (lower envelope, Aug 1 - Sep 30, Theil-Sen; same metric as the calibration)
matches the observed median (68 d). Reported for each tau_s: recession, hydrograph skill, fall-minimum bias, deep share of the fall minimum,
low-flow memory (rho1 on the same-year observed-definition residual, two-sided permutation p) and prior-storage link; plus each tau_s with the store OFF.
usage: python east_shallow_tau_sweep.py
"""
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from multiprocessing import Pool
import east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V

E, st, W, P = C.E, C.st, C.W, C.P
TAUS = [10, 15, 20, 27, 35, 45, 60]; LSCALE = np.log10(48921.0)
LF_OBS = np.exp(E.series(E.obs, 1987, 2024)[0]).mean()


def r1(q, y0, y1, rng):
    lf, pk = E.series(q, y0, y1); d = pd.concat([lf, pk], axis=1).dropna(); d = d.loc[[y for y in d.index if y0 <= y <= y1]]
    r = d.iloc[:, 0].values - np.polyval(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1), d.iloc[:, 1]); est = spearmanr(r[:-1], r[1:])[0]
    pn = np.array([(lambda s: spearmanr(s[:-1], s[1:])[0])(rng.permutation(r)) for _ in range(3000)]); return est, (np.sum(np.abs(pn) >= abs(est)) + 1)/3001, pd.Series(r, index=d.index)


def point(tau_s):
    rng = np.random.default_rng(int(tau_s)); p = dict(lrate=np.log10(1.0/(tau_s*86400.0)), Sf=1.0, exp=1.0, lscale=LSCALE); out = []
    f = C.f_balance(p)
    for lab, ff in (("store", f), ("store OFF", 0.0)):
        Qs, Qd, Sd = C.simulate(p, ff); q = C.flow(Qs, Qd, p); qb = C.B.daily_Q(st, Qs, np.zeros_like(Qd), 10**LSCALE)
        r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], E.obs["2008-10-01":"2020-09-30"])
        j = pd.concat([q.rename("s"), E.obs.rename("o")], axis=1)["2008-10-01":"2024-09-30"].dropna(); tm = V.recession_taus(q)
        sh = []
        for y in range(1987, 2025):
            a = q.loc[f"{y}-08-01":f"{y}-11-30"]; dmin = a.rolling(7, center=True, min_periods=7).mean().idxmin(); sl = slice(dmin - pd.Timedelta(days=3), dmin + pd.Timedelta(days=3)); sh.append(1 - qb[sl].mean()/q[sl].mean())
        a87, a09 = r1(q, 1987, 2024, rng), r1(q, 2009, 2024, rng)
        S = pd.Series(Sd, index=st["tt"]).resample("D").mean(); lk = spearmanr(a87[2], pd.Series({y: S.get(pd.Timestamp(f"{y-1}-09-30"), np.nan) for y in a87[2].index}))[0] if ff > 0 else np.nan
        rec_term = float(np.median(np.abs(np.log(pd.concat([tm.rename("m"), V.TAU_OBS.rename("o")], axis=1).fillna(5000).clip(upper=5000).pipe(lambda z: z.m/z.o))))/np.log(1.25))
        out.append(dict(tau_s=tau_s, run=lab, f=ff, recess_tau=tm.median(), recession_term=rec_term, routing_obj=r.objective, peak_volume=r.parts["peak_volume"], KGE=r.diagnostics["KGE"],
                        logKGE=E.kge(np.log(j.o.clip(lower=.05)), np.log(j.s.clip(lower=.05))), DJF=C.djf(q), fallmin_bias_pct=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/LF_OBS - 1),
                        deep_share_fallmin=float(np.mean(sh)), rho1_87=a87[0], p87=a87[1], rho1_09=a09[0], p09=a09[1], storage_link=lk))
    return out


if __name__ == "__main__":
    with Pool(len(TAUS)) as pool: rows = [r for o in pool.map(point, TAUS) for r in o]
    T = pd.DataFrame(rows); T.to_csv(C.X.OUT/"east_shallow_tau_sweep.csv", index=False); pd.set_option("display.width", 260)
    print(T.round(3).to_string(index=False)); print(f"observed: late-summer recession tau median {V.TAU_OBS.median():.0f} d; rho1 0.38 (p 0.02) WY1987-2024, 0.54 (p 0.035) WY2009-2024")
