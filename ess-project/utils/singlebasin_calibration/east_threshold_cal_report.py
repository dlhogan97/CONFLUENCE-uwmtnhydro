"""
Diagnostics of the east_threshold_cal.py bests (2026-10-27): the quantities kept OUT of the objective, for control and treatment.
  recession tau, KGE / logKGE, fall-minimum bias, rho1 of the peak-residual fall minimum (two-sided permutation p, WY1987-2024 and WY2009-2024),
  link between prior end-of-September storage and the residual (deep store for the treatment; mean shallow storage for both), deep share of
  the fall minimum. Plus the hydrograph figure of both bests against observations.
usage: python east_threshold_cal_report.py
"""
import json, logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import east_threshold_cal as T, east_shallow_tau_sweep as TS, east_ctl_treat_cal_v2 as V

E, st, W, P, C = T.E, T.st, T.W, T.P, T.C
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"


def best(model):
    tr = [json.loads(l) for l in open(T.OUTD/f"trials_{model}.jsonl")]; return min([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])


def shallow_storage(p, tl, deep):
    """mean HRU shallow storage, end of each September (re-simulated, kernel keeps only final S, so step year by year)."""
    sth, ts = p["S_th"], 10**p["log_tau_s"]; S = np.full(T.NH, sth); sd = 0.0; out = {}
    tt = pd.DatetimeIndex(st["tt"]); Rch = st["Rch"]
    for _ in range(3): S, sd = T.kernel(Rch, W, sth, ts*86400.0, tl*86400.0, T.TAU_D, deep, S, sd, np.zeros_like(Rch), np.zeros(len(tt)), np.zeros(len(tt)))
    S = np.full(T.NH, sth); sd = 0.0
    for _ in range(2): S, sd = T.kernel(Rch, W, sth, ts*86400.0, tl*86400.0, T.TAU_D, deep, S, sd, np.zeros_like(Rch), np.zeros(len(tt)), np.zeros(len(tt)))
    yrs = sorted(set(tt.year)); b = 0
    for y in yrs:
        e = np.searchsorted(tt, pd.Timestamp(f"{y}-10-01"))
        if e <= b: continue
        n = e - b; S, sd = T.kernel(Rch[b:e], W, sth, ts*86400.0, tl*86400.0, T.TAU_D, deep, S, sd, np.zeros((n, T.NH)), np.zeros(n), np.zeros(n)); out[y] = float(np.dot(W, S)); b = e
    return pd.Series(out)


rows, Q = [], {}
for model in ("control", "treatment"):
    b = best(model); p = b["params"]; rng = np.random.default_rng(7)
    q, tl, Qs, Qd, Sd = T.run_model(p, model); Q[model] = (q, p, tl)
    j = pd.concat([q.rename("s"), E.obs.rename("o")], axis=1)["2008-10-01":"2024-09-30"].dropna(); tm = V.recession_taus(q)
    a87, a09 = TS.r1(q, 1987, 2024, rng), TS.r1(q, 2009, 2024, rng)
    sh = []
    if model == "treatment":
        qb = C.B.daily_Q(st, Qs, np.zeros_like(Qd), 10**p["lscale"])
        for y in range(1987, 2025):
            a = q.loc[f"{y}-08-01":f"{y}-11-30"]; dmin = a.rolling(7, center=True, min_periods=7).mean().idxmin(); sl = slice(dmin - pd.Timedelta(days=3), dmin + pd.Timedelta(days=3)); sh.append(1 - qb[sl].mean()/q[sl].mean())
        Sdd = pd.Series(Sd, index=st["tt"]).resample("D").mean(); lk_deep = spearmanr(a87[2], pd.Series({y: Sdd.get(pd.Timestamp(f"{y-1}-09-30"), np.nan) for y in a87[2].index}))[0]
    else: lk_deep = np.nan
    ss = shallow_storage(p, tl, model == "treatment"); lk_sh = spearmanr(a87[2], ss.reindex(a87[2].index - 1).values)[0]
    rows.append(dict(model=model, objective=b["objective"], tau_s_d=10**p["log_tau_s"], S_th_m=p["S_th"], tau_l_d=tl, routing_scale_s=10**p["lscale"],
                     recess_tau_d=tm.median(), KGE=b["KGE"], logKGE=E.kge(np.log(j.o.clip(lower=.05)), np.log(j.s.clip(lower=.05))), DJF=b["DJF_ratio"],
                     fallmin_bias_pct=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/TS.LF_OBS - 1), rho1_87=a87[0], p87=a87[1], rho1_09=a09[0], p09=a09[1],
                     link_deep=lk_deep, link_shallow=lk_sh, deep_share_fallmin=float(np.mean(sh)) if sh else 0.0, **{f"part_{k}": v for k, v in b["parts"].items()}))
R = pd.DataFrame(rows).set_index("model").T; pd.set_option("display.width", 200); print(R.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"observed: recession {V.TAU_OBS.median():.0f} d; rho1 0.38 (p 0.02) WY1987-2024, 0.54 (p 0.035) WY2009-2024")
R.to_csv(T.OUTD/"report.csv")

fig, ax = plt.subplots(3, 1, figsize=(15, 12))
for a, (t0, t1, ttl) in zip(ax, [("2008-10-01", "2024-09-30", "WY2009-2024"), ("2011-06-01", "2013-06-30", "WY2012-2013 low-flow seasons"), ("2019-06-01", "2019-12-31", "2019 summer-fall recession")]):
    a.plot(E.obs[t0:t1].index, E.obs[t0:t1], color="black", lw=2.2, label="observed", zorder=5)
    for m, c in (("control", "#eb6834"), ("treatment", "#2a78d6")):
        q, p, tl = Q[m]; a.plot(q[t0:t1].index, q[t0:t1], color=c, lw=1.3, label=f"{m}: tau_s {10**p['log_tau_s']:.0f} d, S_th {p['S_th']:.3f} m, tau_l {tl:.0f} d" + (" -> deep 2 yr" if m == "treatment" else " -> stream"))
    a.set_yscale("log"); a.set_ylim(0.05, 150); a.set_ylabel("discharge (m3/s)"); a.set_title(ttl, loc="left"); a.grid(alpha=.3, which="both")
ax[0].legend(fontsize=9); fig.tight_layout(); fig.savefig(FIG + "east_threshold_cal_hydrographs.png", dpi=120); print("figure written")
