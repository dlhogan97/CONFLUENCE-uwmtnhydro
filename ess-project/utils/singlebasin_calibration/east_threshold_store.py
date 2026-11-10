"""
Threshold shallow store (2026-10-27): the shallow layer drains to the STREAM only above a threshold; it leaks to the deep store at all storage levels.
  per HRU:   Q_i    = max(S_i - S_th, 0)/tau_s        (to stream; stops once storage falls to S_th)
             leak_i = S_i/tau_l                        (to the basin deep store, always)
  basin:     Q_d    = S_d/tau_d,  tau_d = 2 yr         (linear deep store)
tau_l is set by the Dec-Feb volume balance (as f was); tau_s and S_th are swept. Explicit hourly update (dt/tau <= 0.002). Routing 48921 s.
Diagnostics as in east_shallow_tau_sweep.py, plus the hydrograph of the selected case.
usage: python east_threshold_store.py
"""
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, numba as nb, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import spearmanr
from multiprocessing import Pool
import east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V, east_shallow_tau_sweep as TS

E, st, W, P = C.E, C.st, C.W, C.P
NT, NH = st["Rch"].shape; YR = C.YR; DT = 3600.0; TAU_D = 2.0*YR; SCALE = 10**TS.LSCALE
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"


@nb.njit(cache=True)
def kernel(Rch, w, sth, tau_s, tau_l, tau_d, S, sd, Qs, Qd, Sd):
    nt, nh = Rch.shape
    for t in range(nt):
        leak = 0.0
        for i in range(nh):
            s = S[i] + Rch[t, i]*DT
            if s < 0.0: s = 0.0
            q = max(s - sth, 0.0)/tau_s*DT; l = s/tau_l*DT
            if q + l > s: f = s/(q + l); q *= f; l *= f
            S[i] = s - q - l; Qs[t, i] = q/DT; leak += w[i]*l
        sd = sd + leak; qd = sd/tau_d*DT; sd -= qd; Qd[t] = qd/DT; Sd[t] = sd
    return S, sd


def simulate(sth, tau_s_d, tau_l_d, passes=4):
    Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT); S = np.full(NH, sth); sd = 0.0
    if not np.isfinite(tau_l_d): tau_l = 1e30
    else: tau_l = tau_l_d*86400.0
    for _ in range(passes): S, sd = kernel(st["Rch"], W, sth, tau_s_d*86400.0, tau_l, TAU_D, S, sd, Qs, Qd, Sd)
    return Qs, Qd, Sd


def flow(Qs, Qd): return C.B.daily_Q(st, Qs, Qd, SCALE)


def tau_l_balance(sth, tau_s):
    g = lambda lt: C.djf(flow(*simulate(sth, tau_s, 10**lt, passes=3)[:2])) - 1.0          # larger tau_l -> less leak -> less winter flow
    lo, hi = np.log10(5.0), np.log10(20*365.0); glo, ghi = g(lo), g(hi)
    if glo < 0 or ghi > 0: return np.nan, glo, ghi
    return float(10**brentq(g, lo, hi, xtol=0.02, rtol=1e-3, maxiter=14)), glo, ghi


def point(args):
    tau_s, sth = args; rng = np.random.default_rng(int(tau_s*1000 + sth*1e4)); tl, glo, ghi = tau_l_balance(sth, tau_s)
    if not np.isfinite(tl): return dict(tau_s=tau_s, S_th=sth, tau_l_d=np.nan, note=f"DJF unreachable ({glo:+.2f}..{ghi:+.2f})")
    Qs, Qd, Sd = simulate(sth, tau_s, tl); q = flow(Qs, Qd); qb = C.B.daily_Q(st, Qs, np.zeros_like(Qd), SCALE)
    r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], E.obs["2008-10-01":"2020-09-30"]); tm = V.recession_taus(q)
    j = pd.concat([q.rename("s"), E.obs.rename("o")], axis=1)["2008-10-01":"2024-09-30"].dropna(); sh = []
    for y in range(1987, 2025):
        a = q.loc[f"{y}-08-01":f"{y}-11-30"]; dmin = a.rolling(7, center=True, min_periods=7).mean().idxmin(); sl = slice(dmin - pd.Timedelta(days=3), dmin + pd.Timedelta(days=3)); sh.append(1 - qb[sl].mean()/q[sl].mean())
    a87, a09 = TS.r1(q, 1987, 2024, rng), TS.r1(q, 2009, 2024, rng)
    S = pd.Series(Sd, index=st["tt"]).resample("D").mean(); lk = spearmanr(a87[2], pd.Series({y: S.get(pd.Timestamp(f"{y-1}-09-30"), np.nan) for y in a87[2].index}))[0]
    leak_frac = float(1 - (Qs*W).sum(1).mean()/max(st["Rch"].clip(min=0).dot(W).mean(), 1e-12))
    return dict(tau_s=tau_s, S_th=sth, tau_l_d=tl, leak_share=leak_frac, recess_tau=tm.median(), routing_obj=r.objective, peak_volume=r.parts["peak_volume"], KGE=r.diagnostics["KGE"],
                logKGE=E.kge(np.log(j.o.clip(lower=.05)), np.log(j.s.clip(lower=.05))), fallmin_bias_pct=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/TS.LF_OBS - 1),
                deep_share_fallmin=float(np.mean(sh)), rho1_87=a87[0], p87=a87[1], rho1_09=a09[0], p09=a09[1], storage_link=lk)


def figure(tau_s, sth, tl, tag):
    Qs, Qd, _ = simulate(sth, tau_s, tl); q = flow(Qs, Qd); Qs0, Qd0, _ = simulate(sth, tau_s, tl); Qd0[:] = 0.0; q0 = flow(Qs0, Qd0)
    S = {"Observed": (E.obs, "black", "-", 2.2), f"Threshold shallow (tau_s {tau_s:g} d, S_th {sth:g} m, tau_l {tl:.0f} d) + deep 2 yr": (q, "#1f5fa8", "-", 1.3),
         "Same, deep-store outflow removed": (q0, "#c0571b", "--", 1.1)}
    fig, ax = plt.subplots(4, 1, figsize=(15, 15))
    for a, (t0, t1, ttl) in zip(ax, [("2008-10-01", "2024-09-30", "WY2009-2024"), ("2017-10-01", "2020-09-30", "WY2018-2020"), ("2011-06-01", "2013-06-30", "WY2012-2013 low-flow seasons"),
                                     ("2019-06-01", "2019-12-31", "2019 summer-fall recession")]):
        for k, (s, c, ls, lw) in S.items(): a.plot(s[t0:t1].index, s[t0:t1], ls, color=c, lw=lw, label=k, zorder=5 if k == "Observed" else 2)
        a.set_yscale("log"); a.set_ylim(0.05, 150); a.set_ylabel("discharge (m3/s)"); a.set_title(ttl, loc="left"); a.grid(alpha=.3, which="both")
    ax[0].legend(fontsize=9); fig.tight_layout(); fig.savefig(FIG + f"east_threshold_store_{tag}.png", dpi=120)


if __name__ == "__main__":
    grid = [(ts, sth) for ts in (20.0, 35.0, 50.0) for sth in (0.0, 0.025, 0.05, 0.1, 0.2)]
    with Pool(15) as pool: rows = pool.map(point, grid)
    T = pd.DataFrame(rows); T.to_csv(C.X.OUT/"east_threshold_store_sweep.csv", index=False); pd.set_option("display.width", 280)
    print(T.round(3).to_string(index=False)); print(f"observed: late-summer recession {V.TAU_OBS.median():.0f} d; rho1 0.38 (p 0.02) / 0.54 (p 0.035)")
    ok = T.dropna(subset=["rho1_87"]); ok = ok.assign(score=np.abs(np.log(ok.recess_tau/68.0))/np.log(1.25) + np.abs(ok.rho1_87 - 0.38)/0.1 + np.abs(ok.fallmin_bias_pct)/10)
    b = ok.sort_values("score").iloc[0]; print("selected for the figure:", b[["tau_s", "S_th", "tau_l_d"]].to_dict()); figure(b.tau_s, b.S_th, b.tau_l_d, "selected"); print("figure written")
