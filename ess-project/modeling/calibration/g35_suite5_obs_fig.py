"""
Round-5 best (g35_suite5_rerun.py) against observed Tuolumne flow, WY2013-2020 (2026-11-05). Fig: (a) daily flow, log, full record; (b-i) one linear panel per
water year; (j) mean monthly flow; (k) daily sim vs obs, log-log, coloured by season.
-> figures/draft/tuo_suite5_vs_obs.png
usage: python g35_suite5_obs_fig.py
"""
import numpy as np, pandas as pd, netCDF4 as nc, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run_calibration_snow as S
import full_basin_albedo as F
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
r = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG); o = r.obs[r.analysis_start:r.sim_end].asfreq("D")
d = nc.Dataset(F.OUT/"suite5_cal/best/best_obj/cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
s = (pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).astype(float).ravel()[:len(t)], index=t)*r.area).resample("D").mean()[r.analysis_start:r.sim_end]; d.close()
OBS, SIM = "black", "#D55E00"

fig = plt.figure(figsize=(14, 13)); g = fig.add_gridspec(4, 4, height_ratios=[1.1, 1, 1, 1.15], hspace=0.45, wspace=0.28)
ax = fig.add_subplot(g[0, :]); ax.plot(o.index, o, color=OBS, lw=1.5, label="observed"); ax.plot(s.index, s, color=SIM, lw=1.0, label="round 5 best")
ax.set_yscale("log"); ax.set_ylim(0.05, 400); ax.set_ylabel("flow (m$^3$/s)"); ax.set_title("a  daily flow, log", loc="left")
for i, wy in enumerate(range(2013, 2021)):
    a = fig.add_subplot(g[1 + i//4, i % 4]); sl = slice(f"{wy-1}-10-01", f"{wy}-09-30"); oo, ss = o[sl], s[sl]
    a.plot(oo.index, oo, color=OBS, lw=1.4); a.plot(ss.index, ss, color=SIM, lw=1.0)
    k = 1 - np.sqrt((ss.corr(oo) - 1)**2 + (ss.std()/oo.std() - 1)**2 + (ss.mean()/oo.mean() - 1)**2)
    a.set_title(f"{'bcdefghi'[i]}  WY{wy}   KGE {k:.2f}, volume {100*(ss.sum()/oo.sum() - 1):+.0f}%", loc="left", fontsize=10)
    a.set_xticks(pd.to_datetime([f"{wy-1}-10-01", f"{wy}-01-01", f"{wy}-04-01", f"{wy}-07-01"])); a.set_xticklabels(["Oct", "Jan", "Apr", "Jul"])
    a.set_ylim(0, None); a.set_ylabel("flow (m$^3$/s)" if i % 4 == 0 else "")
    a.grid(alpha=.2, lw=.6); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
a1 = fig.add_subplot(g[3, :2]); order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]; cl = lambda x: x.groupby(x.index.month).mean().reindex(order).values
a1.plot(range(12), cl(o), "-o", color=OBS, lw=2, ms=4); a1.plot(range(12), cl(s), "-o", color=SIM, lw=1.6, ms=3.5)
a1.set_xticks(range(12)); a1.set_xticklabels(list("ONDJFMAMJJAS")); a1.set_ylabel("mean flow (m$^3$/s)"); a1.set_title("j  mean monthly flow, WY2013–2020", loc="left")
a2 = fig.add_subplot(g[3, 2:]); j = pd.concat([o.rename("o"), s.rename("s")], axis=1).dropna(); m = j.index.month
SEAS = {"Dec–Mar": ([12, 1, 2, 3], "#0072B2"), "Apr–Jul": ([4, 5, 6, 7], "#009E73"), "Aug–Nov": ([8, 9, 10, 11], "#E69F00")}
for lab, (ms, c) in SEAS.items(): k = np.isin(m, ms); a2.scatter(j.o[k], j.s[k], s=5, color=c, alpha=.5, lw=0, label=lab)
a2.plot([0.05, 400], [0.05, 400], color="0.4", lw=1, ls="--"); a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(0.05, 400); a2.set_ylim(0.05, 400)
a2.set_xlabel("observed (m$^3$/s)"); a2.set_ylabel("round 5 (m$^3$/s)"); a2.set_title("k  daily flow, model vs observed", loc="left")
a2.legend(frameon=False, markerscale=3, loc="upper left", fontsize=9)
for a in (ax, a1, a2): a.grid(alpha=.2, lw=.6); a.set_axisbelow(True); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
h = [plt.Line2D([], [], color=OBS, lw=2, label="observed"), plt.Line2D([], [], color=SIM, lw=2, label="round 5 best (objective 3.59)")]
fig.legend(handles=h, frameon=False, ncol=2, loc="lower center", fontsize=11, bbox_to_anchor=(0.5, 0.01))
fig.savefig(FIG + "tuo_suite5_vs_obs.png", dpi=150, bbox_inches="tight", facecolor="white"); print(FIG + "tuo_suite5_vs_obs.png")
