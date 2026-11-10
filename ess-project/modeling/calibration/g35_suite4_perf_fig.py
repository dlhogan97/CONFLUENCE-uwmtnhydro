"""
Tuolumne model performance after round 4 (2026-11-04): round-4 SUMMA trials (g35_suite4_rerun.py) vs observed, the round-3 trial, the round-4 emulator
(tuo_emulator_r4.py) and the paper baseline, WY2013-2020. Same layout as g35_suite2_perf_fig.py plus a linear panel, since the round-4 objective is Apr-Jul.
No run here has a snowpatch. Fig: (a) daily flow, log; (b) daily flow, linear; (c) mean monthly flow; (d) mean daily flow Aug-Nov, log; (e) Aug-Nov 7-day minimum.
-> figures/draft/tuo_suite4_performance.png, tuo_suite4_performance.csv
usage: python g35_suite4_perf_fig.py [patch|r5|r5patch]   (patch: round-4 best + snowpatch, g35_suite4_snowpatch.py -> tuo_suite4_snowpatch.png;
                                                   r5: round-5 best, g35_suite5_rerun.py, vs round 4 -> tuo_suite5_performance.png;
                                                   r5patch: round 5 vs round 5 + snowpatch -> tuo_suite5_snowpatch.png)
"""
import sys, numpy as np, pandas as pd, netCDF4 as nc, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run_calibration_snow as S
import full_basin_albedo as F
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
BEST = F.OUT/"suite4_cal"/"best"; BASE = "/scratch/dlhogan/ess-project-data/calibration/bestruns/Tuolumne_River_baseline_1981_2025/run"
r = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG); o = r.obs[r.analysis_start:r.sim_end].asfreq("D")


def read(path, var):
    d = nc.Dataset(path); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
    q = pd.Series(np.array(d[var][:]).astype(float).ravel()[:len(t)], index=t)*r.area; d.close(); return q.resample("D").mean()[r.analysis_start:r.sim_end]


def emulator():
    q = pd.read_csv(F.OUT/"suite3_cal"/"emulator"/"r4_q.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return (q/1000*r.area/86400).resample("D").mean()[r.analysis_start:r.sim_end]


PATCH = "patch" in sys.argv[1:]                                                   # g35_suite4_snowpatch.py run in place of the best-KGE trial
R5P = "r5patch" in sys.argv[1:]                                                  # round 5 + snowpatch (g35_suite5_snowpatch.py)
R5 = "r5" in sys.argv[1:] or R5P
NAME = "tuo_suite4_snowpatch" if PATCH else "tuo_suite5_snowpatch" if R5P else "tuo_suite5_performance" if R5 else "tuo_suite4_performance"
SECOND = (("round 4 best + snowpatch", BEST/"best_obj_patch/cal_day_28.nc") if PATCH else ("round 4 best Apr–Jul KGE", BEST/"best_kge/cal_day.nc"))
RUNS = {"round 4 best objective": (read(BEST/"best_obj/cal_day.nc", "averageRoutedRunoff_mean"), "#D55E00"),
        SECOND[0]: (read(SECOND[1], "averageRoutedRunoff_mean"), "#0072B2"),
        "round 3": (read(F.OUT/"suite3_cal/wb/cal_day.nc", "averageRoutedRunoff_mean"), "#999999"),
        "emulator round 4": (emulator(), "#009E73"),
        "paper baseline": (read(BASE+"/out/baseline_timestep.nc", "averageRoutedRunoff"), "#CC79A7")}
DAILY = ["round 4 best objective", "round 3", "emulator round 4"] if not PATCH else ["round 4 best objective", SECOND[0], "round 3"]
if R5:                                                                            # round 5 (snow liquid drainage) vs round 4
    RUNS = {"round 5 best objective": (read(F.OUT/"suite5_cal/best/best_obj/cal_day.nc", "averageRoutedRunoff_mean"), "#D55E00"),
            "round 4 best objective": (RUNS["round 4 best objective"][0], "#0072B2"), "paper baseline": RUNS["paper baseline"]}
    DAILY = ["round 5 best objective", "round 4 best objective"]
if R5P:
    RUNS = {"round 5 best objective": RUNS["round 5 best objective"],
            "round 5 + snowpatch": (read(F.OUT/"suite5_cal/best/best_obj_patch/cal_day_28.nc", "averageRoutedRunoff_mean"), "#0072B2"), "paper baseline": RUNS["paper baseline"]}
    DAILY = ["round 5 best objective", "round 5 + snowpatch"]


def low7(q): return pd.Series({y: q[f"{y}-08-01":f"{y}-11-30"].rolling(7, center=True, min_periods=7).mean().min() for y in range(2013, 2020)})


def kge(s, o):
    a = s.corr(o); return 1-np.sqrt((a-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)


def metrics(s):
    j = pd.concat([o.rename("o"), s.rename("s")], axis=1).dropna(); e = 0.01*j.o.mean(); mo = j.index.month; aj = j[np.isin(mo, [4, 5, 6, 7])]
    pct = lambda m: 100*(j.s[m].sum()/j.o[m].sum()-1)
    l, lo = low7(s), low7(o)
    return dict(KGE=kge(j.s, j.o), KGE_AprJul=kge(aj.s, aj.o), NSE=1-((j.s-j.o)**2).sum()/((j.o-j.o.mean())**2).sum(),
                logNSE=1-((np.log(j.s+e)-np.log(j.o+e))**2).sum()/((np.log(j.o+e)-np.log(j.o+e).mean())**2).sum(),
                vol_bias=pct(mo > 0), AprMay_bias=pct(np.isin(mo, [4, 5])), JunJul_bias=pct(np.isin(mo, [6, 7])), AugSep_bias=pct(np.isin(mo, [8, 9])),
                OctNov_bias=pct(np.isin(mo, [10, 11])), min7_median_ratio=(l/lo).median(), min7_rank_r=l.corr(lo, method="spearman"))


T = pd.DataFrame({k: metrics(v[0]) for k, v in RUNS.items()}).T; T.to_csv(FIG+NAME+".csv"); pd.set_option("display.width", 220); print(T.round(2).to_string())
fig = plt.figure(figsize=(13, 12)); g = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1]); ax0 = fig.add_subplot(g[0, :]); axl = fig.add_subplot(g[1, :], sharex=ax0)
ax1, ax2, ax3 = (fig.add_subplot(g[2, i]) for i in range(3))
for a, title in ((ax0, "a  daily flow, log"), (axl, "b  daily flow, linear (Apr–Jul shaded: the round-4 objective)")):
    a.plot(o.index, o, color="black", lw=1.6, zorder=5)
    for k in DAILY: a.plot(RUNS[k][0].index, RUNS[k][0], color=RUNS[k][1], lw=1.0)
    a.set_ylabel("flow (m$^3$/s)"); a.set_title(title, loc="left")
ax0.set_yscale("log"); ax0.set_ylim(1e-2, 400)                                                  # round 4 best and the emulator reach ~0 flow; clip
for y in range(2013, 2021): axl.axvspan(pd.Timestamp(f"{y}-04-01"), pd.Timestamp(f"{y}-07-31"), color="#f2f2f2", zorder=0)
order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]; cl = lambda s: s.groupby(s.index.month).mean().reindex(order).values
ax1.plot(range(12), cl(o), "-o", color="black", lw=2, ms=4, zorder=5)
for k, (s, c) in RUNS.items(): ax1.plot(range(12), cl(s), color=c, lw=1.6)
ax1.set_xticks(range(12)); ax1.set_xticklabels(list("ONDJFMAMJJAS")); ax1.set_ylabel("mean flow (m$^3$/s)"); ax1.set_title("c  mean monthly flow", loc="left")
doy = lambda s: s[s.index.month.isin([8, 9, 10, 11])].groupby(lambda d: (d.month, d.day)).mean().values
ax2.plot(doy(o), color="black", lw=2, zorder=5)
for k, (s, c) in RUNS.items(): ax2.plot(doy(s), color=c, lw=1.4)
ax2.set_yscale("log"); ax2.set_xticks([0, 31, 61, 92]); ax2.set_xticklabels(["Aug", "Sep", "Oct", "Nov"]); ax2.set_ylabel("mean flow (m$^3$/s)"); ax2.set_title("d  low-flow season, mean by day", loc="left")
yy = low7(o).index; ax3.plot(yy, low7(o), "-o", color="black", lw=2, zorder=5)
for k, (s, c) in RUNS.items(): ax3.plot(yy, low7(s), "-o", ms=3.5, color=c, lw=1.4)
ax3.set_yscale("log"); ax3.set_ylim(1e-3, 10); ax3.set_xlabel("year"); ax3.set_ylabel("7-day minimum (m$^3$/s)"); ax3.set_title("e  Aug–Nov 7-day minimum", loc="left")
for a in (ax0, axl, ax1, ax2, ax3):
    a.grid(alpha=.2, lw=.6); a.set_axisbelow(True); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
h = [plt.Line2D([], [], color="black", lw=2, label="observed")]+[plt.Line2D([], [], color=c, lw=2, label=k) for k, (_, c) in RUNS.items()]
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.legend(handles=h, frameon=False, ncol=6, loc="lower center", fontsize=10); fig.savefig(FIG+NAME+".png", dpi=150, bbox_inches="tight", facecolor="white")
