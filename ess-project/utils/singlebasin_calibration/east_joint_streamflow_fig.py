"""Streamflow comparison: observed vs control (no store, buckets refit) vs surface-runoff-capture deep store, both from east_memory_joint.py (same objective). 2026-11-02.
(a) daily flow Oct 1986 - Dec 2024 (the Aug-Nov 2024 minimum is part of the memory series), log  (b) WY2012-2013 low-flow seasons  (c) 2019 summer-fall recession  (d) mean daily flow by day of water year, log
(e) Aug-Nov 7-day minimum per water year  (f) lag 1-5 Spearman ACF of the peak-adjusted fall minimum (residual of ln min on ln Apr-Jul peak).
-> figures/draft/east_joint_streamflow.png"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import emu_core as E, east_booster as EB, east_memory_refit as RF, east_memory_search as M
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
R = pd.read_csv(M.OUTD/"joint"/"report.csv", index_col=0)
S = {"observed": (EB.obs, "black", 1.6), "control (no store)": (M.simulate("control_refit", json.loads(R.loc["control_refit", "params"]))[0], "#D55E00", 1.2),
     "deep store (surface-runoff capture)": (M.simulate("sr_capture_refit", json.loads(R.loc["sr_capture_refit", "params"]))[0], "#0072B2", 1.2)}
S = {k: (s["1986-10-01":"2024-12-31"], c, w) for k, (s, c, w) in S.items()}
fig = plt.figure(figsize=(14, 13)); g = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])
ax = [fig.add_subplot(g[0, :]), fig.add_subplot(g[1, :2]), fig.add_subplot(g[1, 2]), fig.add_subplot(g[2, 0]), fig.add_subplot(g[2, 1:]), fig.add_subplot(g[3, :])]
for k, (s, c, w) in S.items():
    ax[0].plot(s.index, s, color=c, lw=w*0.7)
    z = s["2011-06-01":"2013-07-01"]; ax[1].plot(z.index, z, color=c, lw=w)
    z = s["2019-06-01":"2019-12-31"]; ax[2].plot(z.index, z, color=c, lw=w)
    d = s.groupby((s.index - pd.DateOffset(months=9)).dayofyear).mean().iloc[:365]; ax[3].plot(d.index, d.values, color=c, lw=w)
lf = {k: np.exp(E.series(s, 1987, 2024)[0]) for k, (s, _, _) in S.items()}
for k, (s, c, w) in S.items(): ax[4].plot(lf[k].index, lf[k].values, "-o", color=c, lw=w, ms=4, mfc="white" if k != "observed" else c)
lags = np.arange(1, 6); bw = 0.27
for i, (k, (s, c, _)) in enumerate(S.items()):
    r = E.resid(*E.series(s, 1987, 2024)); rho = [E.lag_r(r, l) for l in lags]
    ax[5].bar(lags + (i-1)*bw, rho, bw, color=c, edgecolor="black", lw=0.5)
    for x, v in zip(lags + (i-1)*bw, rho): ax[5].text(x, v + (0.02 if v >= 0 else -0.06), f"{v:+.2f}", ha="center", fontsize=8)
n = len(E.resid(*E.series(S["observed"][0], 1987, 2024))); ax[5].axhspan(-1.96/np.sqrt(n), 1.96/np.sqrt(n), color="0.85", zorder=0); ax[5].axhline(0, color="black", lw=0.8)
ax[5].set_xticks(lags); ax[5].set_xlabel("lag (years)"); ax[5].set_ylabel("Spearman ρ"); ax[5].set_ylim(-0.75, 0.85)
for a in ax[:5]: a.set_yscale("log"); a.set_ylabel("flow (m$^3$/s)")
ax[3].set_xticks([1, 32, 62, 93, 124, 152, 183, 213, 244, 274, 305, 336]); ax[3].set_xticklabels(list("ONDJFMAMJJAS")); ax[4].set_ylabel("7-day min (m$^3$/s)"); ax[4].set_xlabel("year"); import matplotlib.dates as mdates; ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
T = ["a  daily flow, WY1987–2024", "b  WY2012–2013 low-flow seasons", "c  2019 summer–fall recession", "d  mean daily flow", "e  Aug–Nov 7-day minimum", "f  memory: lag autocorrelation of the peak-adjusted fall minimum (grey: ±1.96/√n)"]
for a, t in zip(ax, T):
    a.set_title(t, loc="left", fontsize=11); a.grid(alpha=0.2, lw=0.6); a.set_axisbelow(True); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
sc = lambda k: R.loc[k]
h = [plt.Line2D([], [], color=c, lw=2.2, label=k + ("" if k == "observed" else f"   eval KGE {sc(m).KGE_eval:.2f}, NSE {sc(m).NSE_eval:.2f}, winter recession {sc(m).winter_tau_d:.0f} d, ρ1 {sc(m).rho1_87:+.2f}"))
     for (k, (_, c, _)), m in zip(S.items(), [None, "control_refit", "sr_capture_refit"])]
fig.legend(handles=h, loc="upper center", ncol=1, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 1.035)); fig.tight_layout()
fig.savefig(FIG+"east_joint_streamflow.png", dpi=150, bbox_inches="tight", facecolor="white"); print("wrote", FIG+"east_joint_streamflow.png")
