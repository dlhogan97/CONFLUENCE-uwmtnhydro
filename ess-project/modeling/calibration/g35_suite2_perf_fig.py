"""
Tuolumne model performance as it stands (2026-10-30): round-2 candidates (g35_suite2_rerun.py) vs observed, the G35 starting point and the paper baseline, WY2013-2020.
No run here has a snowpatch. Fig: (a) daily hydrograph, log (best-objective candidate only, to stay readable); (b) mean monthly flow; (c) mean daily flow Aug-Nov, log; (d) Aug-Nov 7-day minimum per water year.
-> figures/draft/tuo_suite2_performance.png, tuo_suite2_performance.csv
usage: python g35_suite2_perf_fig.py
"""
import numpy as np, pandas as pd, netCDF4 as nc, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run_calibration_snow as S
import full_basin_albedo as F
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
BEST = F.OUT/"suite2_cal"/"best"; BASE = "/scratch/dlhogan/ess-project-data/calibration/bestruns/Tuolumne_River_baseline_1981_2025/run"
r = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG); o = r.obs[r.analysis_start:r.sim_end].asfreq("D")


def read(path, var):
    d = nc.Dataset(path); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
    q = pd.Series(np.array(d[var][:]).astype(float).ravel()[:len(t)], index=t)*r.area; d.close(); return q.resample("D").mean()[r.analysis_start:r.sim_end]


RUNS = {"best objective (KGE 0.75)": (read(BEST/"best_obj/keep/out/cal_day.nc", "averageRoutedRunoff_mean"), "#D55E00"),
        "best KGE": (read(BEST/"best_kge/keep/out/cal_day.nc", "averageRoutedRunoff_mean"), "#0072B2"),
        "best inside ASO limits": (read(BEST/"best_order/keep/out/cal_day.nc", "averageRoutedRunoff_mean"), "#009E73"),
        "G35 (before soil/precip)": (read(F.OUT/"G35/cal_day.nc", "averageRoutedRunoff_mean"), "#999999"),
        "paper baseline": (read(BASE+"/out/baseline_timestep.nc", "averageRoutedRunoff"), "#CC79A7")}


def low7(q): return pd.Series({y: q[f"{y}-08-01":f"{y}-11-30"].rolling(7, center=True, min_periods=7).mean().min() for y in range(2013, 2020)})


def metrics(s):
    j = pd.concat([o.rename("o"), s.rename("s")], axis=1).dropna(); a = j.s.corr(j.o); kge = 1-np.sqrt((a-1)**2+(j.s.std()/j.o.std()-1)**2+(j.s.mean()/j.o.mean()-1)**2); e = 0.01*j.o.mean()
    pct = lambda m: 100*(j.s[m].sum()/j.o[m].sum()-1); mo = j.index.month
    l, lo = low7(s), low7(o)
    return dict(KGE=kge, NSE=1-((j.s-j.o)**2).sum()/((j.o-j.o.mean())**2).sum(), logNSE=1-((np.log(j.s+e)-np.log(j.o+e))**2).sum()/((np.log(j.o+e)-np.log(j.o+e).mean())**2).sum(),
                vol_bias=pct(mo > 0), MarMay_bias=pct(np.isin(mo, [3, 4, 5])), JunJul_bias=pct(np.isin(mo, [6, 7])), AugNov_bias=pct(np.isin(mo, [8, 9, 10, 11])), DJF_bias=pct(np.isin(mo, [12, 1, 2])), min7_median_ratio=(l/lo).median(), min7_rank_r=l.corr(lo, method="spearman"))


T = pd.DataFrame({k: metrics(v[0]) for k, v in RUNS.items()}).T; T.to_csv(FIG+"tuo_suite2_performance.csv"); pd.set_option("display.width", 220); print(T.round(2).to_string()); print("obs Aug-Nov 7d min", low7(o).round(3).to_dict())
fig = plt.figure(figsize=(13, 9)); g = fig.add_gridspec(2, 3, height_ratios=[1.1, 1]); ax0 = fig.add_subplot(g[0, :]); ax1, ax2, ax3 = (fig.add_subplot(g[1, i]) for i in range(3))
ax0.plot(o.index, o, color="black", lw=1.6, label="observed", zorder=5)
for k in list(RUNS)[:1]+list(RUNS)[3:]: ax0.plot(RUNS[k][0].index, RUNS[k][0], color=RUNS[k][1], lw=1.0)
ax0.set_yscale("log"); ax0.set_ylabel("flow (m$^3$/s)"); ax0.set_title("a  daily flow", loc="left")
order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]; cl = lambda s: s.groupby(s.index.month).mean().reindex(order).values
ax1.plot(range(12), cl(o), "-o", color="black", lw=2, ms=4, label="observed", zorder=5)
for k, (s, c) in RUNS.items(): ax1.plot(range(12), cl(s), color=c, lw=1.6)
ax1.set_xticks(range(12)); ax1.set_xticklabels(list("ONDJFMAMJJAS")); ax1.set_ylabel("mean flow (m$^3$/s)"); ax1.set_title("b  mean monthly flow", loc="left")
doy = lambda s: s[s.index.month.isin([8, 9, 10, 11])].groupby(lambda d: (d.month, d.day)).mean().values
ax2.plot(doy(o), color="black", lw=2, zorder=5)
for k, (s, c) in RUNS.items(): ax2.plot(doy(s), color=c, lw=1.4)
ax2.set_yscale("log"); ax2.set_xticks([0, 31, 61, 92]); ax2.set_xticklabels(["Aug", "Sep", "Oct", "Nov"]); ax2.set_ylabel("mean flow (m$^3$/s)"); ax2.set_title("c  low-flow season, mean by day", loc="left")
yy = low7(o).index; ax3.plot(yy, low7(o), "-o", color="black", lw=2, zorder=5)
for k, (s, c) in RUNS.items(): ax3.plot(yy, low7(s), "-o", ms=3.5, color=c, lw=1.4)
ax3.set_yscale("log"); ax3.set_xlabel("year"); ax3.set_ylabel("7-day minimum (m$^3$/s)"); ax3.set_title("d  Aug–Nov 7-day minimum", loc="left")
for a in (ax0, ax1, ax2, ax3):
    a.grid(alpha=.2, lw=.6); a.set_axisbelow(True); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
h = [plt.Line2D([], [], color="black", lw=2, label="observed")]+[plt.Line2D([], [], color=c, lw=2, label=k) for k, (_, c) in RUNS.items()]
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.legend(handles=h, frameon=False, ncol=6, loc="lower center", fontsize=10); fig.savefig(FIG+"tuo_suite2_performance.png", dpi=150, bbox_inches="tight", facecolor="white")
