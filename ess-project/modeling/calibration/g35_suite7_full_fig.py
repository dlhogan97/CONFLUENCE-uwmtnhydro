"""
Streamflow comparison for the full-record runs (g35_suite7_full.py), 2026-11-09.
Row 1: mean daily hydrograph by day of water year (7-day centred mean) for WY1983-2012, WY2013-2020 (calibration), WY2021-2025: observed, round-7 control,
       round-7 + snowpatch.  Row 2: same on a log axis (low flows).  Row 3: Aug-Sep runoff per water year, WY1983-2025.
Observed = the calibration's merged record (splices before 2012: low flows there need caution).
-> figures/draft/tuo_suite7_full_streamflow.png
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import g35_suite7_full_table as T
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/tuo_suite7_full_streamflow.png"
S = {"observed": (T.obs_mm_full(), "#1a1a1a", "-", 1.6), "round 7": (T.sim("control"), "#0072B2", "-", 1.4), "round 7 + snowpatch": (T.sim("patch"), "#E69F00", "--", 1.4)}
ERAS = {"WY1983-2012 (out of sample)": ("1982-10-01", "2012-09-30"), "WY2013-2020 (calibration)": ("2012-10-01", "2020-09-30"),
        "WY2021-2025 (out of sample)": ("2020-10-01", "2025-09-30")}
obs = S["observed"][0]
def clim(q, a0, a1):
    q = q[a0:a1].reindex(obs[a0:a1].dropna().index); dowy = (q.index - pd.to_datetime([f"{y - (m < 10)}-10-01" for y, m in zip(q.index.year, q.index.month)])).days
    c = q.groupby(np.minimum(dowy, 364)).mean(); return c.rolling(7, center=True, min_periods=1).mean()
fig = plt.figure(figsize=(13, 9.5)); gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.38, wspace=0.18)
ticks = [0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335]; labs = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]
for k, (era, (a0, a1)) in enumerate(ERAS.items()):
    for row, log in ((0, False), (1, True)):
        ax = fig.add_subplot(gs[row, k])
        for name, (q, col, ls, lw) in S.items(): ax.plot(clim(q, a0, a1), color=col, ls=ls, lw=lw, label=name)
        if log: ax.set_yscale("log"); ax.set_ylim(0.05, 20)
        ax.set_xticks(ticks); ax.set_xticklabels(labs); ax.set_xlim(0, 364); ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if row == 0:
            ax.set_title(era, fontsize=10.5)
            j = pd.concat([S["round 7"][0].rename("s"), obs.rename("o")], axis=1)[a0:a1].dropna(); r = np.corrcoef(j.s, j.o)[0, 1]
            kge = 1 - np.sqrt((r - 1)**2 + (j.s.std()/j.o.std() - 1)**2 + (j.s.mean()/j.o.mean() - 1)**2)
            ax.text(0.02, 0.95, f"KGE {kge:.2f}   volume {100*(j.s.sum()/j.o.sum() - 1):+.0f}%", transform=ax.transAxes, fontsize=9, va="top", color="#333333")
        if k == 0: ax.set_ylabel("runoff (mm/d)" + (", log" if log else ""))
        if row == 0 and k == 0: ax.legend(frameon=False, fontsize=9, loc="center left")
ax = fig.add_subplot(gs[2, :]); wy = lambda q: q[q.index.month.isin([8, 9])].groupby(q.index.year[q.index.month.isin([8, 9])]).sum()
for name, (q, col, ls, lw) in S.items():
    v = wy(q["1982-10-01":"2025-09-30"]); ax.plot(v.index, v.values, color=col, ls=ls, lw=lw, marker="o", ms=3.5, label=name)
ax.axvspan(2012.5, 2020.5, color="#999999", alpha=0.12, lw=0); ax.text(2016.5, ax.get_ylim()[1]*0.93, "calibration", ha="center", fontsize=9, color="#555555")
ax.set_ylabel("Aug-Sep runoff (mm)"); ax.set_xlabel("water year"); ax.grid(alpha=0.25, lw=0.6); ax.set_xlim(1982.5, 2025.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Late-season volume: the snowpatch adds 0.07 mm/yr on average (it overlaps the round-7 line)", fontsize=10.5)
fig.suptitle("Tuolumne, round-7 best over the full record: observed vs model (7-day smoothed mean annual hydrograph)", fontsize=12, y=0.995)
fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)
