"""
Is the recession-vs-memory conflict structural? (2026-10-27) Linear shallow bucket fixed at tau_s = 35 d (the sweep value that kept memory),
linear leak f*S/tau_s, deep store  Q_d = (S/tau_d)*(S/S_ref)^(n-1)  with S_ref = mean leak inflow * tau_d; f from the Dec-Feb balance; routing 48921 s.
Sweep tau_d (0.25-2 yr) and n (1-3). Same diagnostics as east_shallow_tau_sweep.py. Also writes the streamflow figure for tau_s = 35 d, tau_d = 2 yr, n = 1.
usage: python east_deep_shape_sweep.py
"""
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import spearmanr
from multiprocessing import Pool
import east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V, east_shallow_tau_sweep as TS

E, st, W, P = C.E, C.st, C.W, C.P
NT, NH = st["Rch"].shape; YR = C.YR; TAU_S = 35.0; RATE = 1.0/(TAU_S*86400.0); LSCALE = TS.LSCALE; SCALE = 10**LSCALE
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"


def simulate(f, tau_d, n, passes=4):
    def run(ibar):
        Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT); S = st["S0"].copy(); ts = tau_d*YR
        sfd, rated, sdi = (ibar*ts, ibar, ibar*ts) if f > 0 else (1.0, 1.0/ts, 0.0)
        for _ in range(passes): S, sdi = E.run_buckets(st["Rch"], st["A"], 1.0, RATE, 1.0, float(f), 1.0, float(sfd), float(n), float(rated), S, float(sdi), 3600.0, Qs, Qd, Sd)
        return Qs, Qd, Sd
    out = run(1e-9 if f == 0 else f*1e-9)
    if f > 0: out = run(float(f*(out[0]*W).sum(1).mean()))                     # mean leak inflow = f * mean bucket outflow (linear leak, linear bucket)
    return out


def flow(Qs, Qd): return C.B.daily_Q(st, Qs, Qd, SCALE)


def f_bal(tau_d, n):
    g = lambda lf: C.djf(flow(*simulate(10**lf, tau_d, n, passes=3)[:2])) - 1.0
    lo, hi = -3.0, np.log10(5.0); glo, ghi = g(lo), g(hi)
    if glo >= 0: return 10**lo
    if ghi < 0: return np.nan
    return float(10**brentq(g, lo, hi, xtol=0.02, rtol=1e-3, maxiter=12))


def point(args):
    tau_d, n = args; rng = np.random.default_rng(int(tau_d*100 + n)); f = f_bal(tau_d, n)
    if not np.isfinite(f): return dict(tau_d=tau_d, n=n, f=np.nan)
    Qs, Qd, Sd = simulate(f, tau_d, n); q = flow(Qs, Qd); qb = C.B.daily_Q(st, Qs, np.zeros_like(Qd), SCALE)
    r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], E.obs["2008-10-01":"2020-09-30"]); tm = V.recession_taus(q)
    j = pd.concat([q.rename("s"), E.obs.rename("o")], axis=1)["2008-10-01":"2024-09-30"].dropna()
    sh = []
    for y in range(1987, 2025):
        a = q.loc[f"{y}-08-01":f"{y}-11-30"]; dmin = a.rolling(7, center=True, min_periods=7).mean().idxmin(); sl = slice(dmin - pd.Timedelta(days=3), dmin + pd.Timedelta(days=3)); sh.append(1 - qb[sl].mean()/q[sl].mean())
    a87, a09 = TS.r1(q, 1987, 2024, rng), TS.r1(q, 2009, 2024, rng)
    S = pd.Series(Sd, index=st["tt"]).resample("D").mean(); lk = spearmanr(a87[2], pd.Series({y: S.get(pd.Timestamp(f"{y-1}-09-30"), np.nan) for y in a87[2].index}))[0]
    qd = C.B.daily_Q(st, np.zeros_like(Qs), Qd, SCALE); deep_aug_sep = float(np.median([np.log(qd[f"{y}-08-01":f"{y}-08-07"].mean()/qd[f"{y}-09-24":f"{y}-09-30"].mean()) for y in range(1987, 2025)]))
    return dict(tau_d=tau_d, n=n, f=f, recess_tau=tm.median(), deep_drawdown_AugSep_pct=100*(1 - np.exp(-deep_aug_sep)), routing_obj=r.objective, KGE=r.diagnostics["KGE"],
                logKGE=E.kge(np.log(j.o.clip(lower=.05)), np.log(j.s.clip(lower=.05))), fallmin_bias_pct=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/TS.LF_OBS - 1),
                deep_share_fallmin=float(np.mean(sh)), rho1_87=a87[0], p87=a87[1], rho1_09=a09[0], p09=a09[1], storage_link=lk)


def figure():
    f = f_bal(2.0, 1.0); Qs, Qd, _ = simulate(f, 2.0, 1.0); q = flow(Qs, Qd); Qs0, Qd0, _ = simulate(0.0, 2.0, 1.0); q0 = flow(Qs0, Qd0)
    S = {"Observed": (E.obs, "black", "-", 2.2), f"Linear shallow 35 d + deep store 2 yr (f = {f:.2f})": (q, "#1f5fa8", "-", 1.3), "Same, deep store OFF": (q0, "#c0571b", "--", 1.1)}
    fig, ax = plt.subplots(4, 1, figsize=(15, 15))
    for a, (t0, t1, ttl) in zip(ax, [("2008-10-01", "2024-09-30", "WY2009-2024"), ("2017-10-01", "2020-09-30", "WY2018-2020"), ("2011-06-01", "2013-06-30", "WY2012-2013 low-flow seasons"),
                                     ("2019-06-01", "2019-12-31", "2019 summer-fall recession")]):
        for k, (s, c, ls, lw) in S.items(): a.plot(s[t0:t1].index, s[t0:t1], ls, color=c, lw=lw, label=k, zorder=5 if k == "Observed" else 2)
        a.set_yscale("log"); a.set_ylim(0.05 if "2009" in ttl else 0.05, 150); a.set_ylabel("discharge (m3/s)"); a.set_title(ttl, loc="left"); a.grid(alpha=.3, which="both")
    ax[0].legend(fontsize=9); fig.tight_layout(); fig.savefig(FIG + "east_shallow35_deep2yr_hydrograph.png", dpi=120)


if __name__ == "__main__":
    grid = [(t, n) for t in (0.25, 0.5, 1.0, 2.0) for n in (1.0, 2.0, 3.0)]
    with Pool(12) as pool: rows = pool.map(point, grid)
    T = pd.DataFrame(rows); T.to_csv(C.X.OUT/"east_deep_shape_sweep_taus35.csv", index=False); pd.set_option("display.width", 260)
    print(T.round(3).to_string(index=False)); print(f"observed: late-summer recession {V.TAU_OBS.median():.0f} d; rho1 0.38 / 0.54")
    figure(); print("figure written")
