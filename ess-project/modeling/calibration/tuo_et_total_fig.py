"""
Tuolumne basin evaporative loss, round-3 feasible trial (g35_suite3_wb.py output) vs OpenET (2026-11-03).
User: the true total (ET + sublimation) should be ~400 mm/yr; OpenET is a pattern guide only. SUMMA's scalarTotalET excludes sublimation, so the two
are stacked: ET (transpiration + canopy/ground evaporation) and sublimation (ground + canopy).
(a) monthly basin means WY2013-2020   (b) mean annual cycle, OpenET overlap WY2017-2020   (c) water-year totals
-> figures/draft/tuo_et_sublimation_timeseries.png (+ .csv)
"""
import glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tuo_snow_emulator as SE

FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
RUN = "/scratch/dlhogan/ess-project-data/calibration/diagnostics/full_basin_albedo/suite3_cal/wb/cal_day.nc"
OET = glob.glob("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/observations/et/openet_et_gridded_*_monthly.csv")[0]
C = dict(et="#2a78d6", subl="#eb6834", openet="#222222", grid="#e6e6e6", ink="#555555")
P0, P1, O0 = "2012-10-01", "2020-09-30", "2016-10-01"

H = SE.hrus(); w = H.w; X = SE.load(RUN)
B = lambda df: (df[P0:P1]*w.values).sum(1)
d = pd.DataFrame({"et": B(X["et"]), "subl": B(-X["subl"] - X["csubl"])})
m = d.resample("MS").sum()                                                     # mm/month
o = pd.read_csv(OET, index_col=0, parse_dates=True); o.columns = [int(c.split("_")[1]) for c in o.columns]
m["openet"] = ((o*w).sum(1)/o.notna().mul(w, axis=1).sum(1)).reindex(m.index)
m.to_csv(FIG + "tuo_et_sublimation_timeseries.csv")
wy = lambda i: i.year + (i.month >= 10)
A = m.groupby(wy(m.index)).sum(min_count=12); A.loc[A.index < 2017, "openet"] = np.nan
cl = m[O0:].groupby(m[O0:].index.month).mean().loc[[10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

fig = plt.figure(figsize=(13, 9.4)); g = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.18)
ax = [fig.add_subplot(g[0, :]), fig.add_subplot(g[1, 0]), fig.add_subplot(g[1, 1])]
x = m.index
ax[0].fill_between(x, 0, m.et, step="mid", color=C["et"], lw=0, label="SUMMA ET (transpiration + evaporation)")
ax[0].fill_between(x, m.et, m.et + m.subl, step="mid", color=C["subl"], lw=0, label="SUMMA sublimation (ground + canopy)")
ax[0].plot(x, m.openet, color=C["openet"], lw=2, drawstyle="steps-mid", label="OpenET ensemble")
ax[0].set_ylabel("mm / month"); ax[0].set_xlim(x[0], x[-1]); ax[0].set_ylim(0)
ax[0].set_title("a  monthly basin mean, WY2013–2020", loc="left", fontsize=12, pad=6)
ax[0].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.06), fontsize=10.5, borderaxespad=0)

k = np.arange(12)
ax[1].bar(k, cl.et, 0.72, color=C["et"]); ax[1].bar(k, cl.subl, 0.72, bottom=cl.et, color=C["subl"], edgecolor="white", lw=1.5)
ax[1].plot(k, cl.openet, "-o", color=C["openet"], lw=2, ms=6, mfc="white", mew=1.8)
ax[1].set_xticks(k); ax[1].set_xticklabels(list("ONDJFMAMJJAS")); ax[1].set_ylabel("mm / month")
ax[1].set_title("b  mean annual cycle, WY2017–2020 (OpenET overlap)", loc="left", fontsize=12)

j = np.arange(len(A))
ax[2].bar(j, A.et, 0.68, color=C["et"]); ax[2].bar(j, A.subl, 0.68, bottom=A.et, color=C["subl"], edgecolor="white", lw=1.5)
ax[2].plot(j, A.openet, "o", color=C["openet"], ms=8, mfc="white", mew=2)
ax[2].axhline(400, color=C["ink"], lw=1.2, ls="--"); ax[2].text(-0.5, 392, "expected ~400", ha="left", va="top", fontsize=10, color=C["ink"], bbox=dict(fc="white", ec="none", pad=1))
for jj, (e, s) in enumerate(zip(A.et, A.subl)): ax[2].text(jj, e + s/2, f"{e + s:.0f}", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
ax[2].set_xticks(j); ax[2].set_xticklabels([f"{y % 100:02d}" for y in A.index]); ax[2].set_xlabel("water year"); ax[2].set_ylabel("mm / yr")
ax[2].set_title(f"c  water-year totals  (mean {A.et.mean() + A.subl.mean():.0f} = ET {A.et.mean():.0f} + subl {A.subl.mean():.0f})", loc="left", fontsize=12)
ax[2].set_ylim(0, max(A[["openet"]].max().max(), (A.et + A.subl).max())*1.12)
for a in ax:
    a.grid(axis="y", color=C["grid"], lw=0.8); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.savefig(FIG + "tuo_et_sublimation_timeseries.png", dpi=150, bbox_inches="tight", facecolor="white"); print("wrote", FIG + "tuo_et_sublimation_timeseries.png")
print(A.round(0).to_string())
