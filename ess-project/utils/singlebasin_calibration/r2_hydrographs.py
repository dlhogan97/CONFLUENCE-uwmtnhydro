"""Dry / average / wet water-year hydrographs, round-2 best trial vs observed, both basins; linear and log discharge. -> ess-project/figures/draft/r2_vs_previous_dry_avg_wet_hydrographs.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
L = "/scratch/dlhogan/ess-project-data/calibration/runs/"
DOM = {"East River": dict(old="/scratch/dlhogan/ess-project-data/calibration/bestruns/East_River_baseline_1981_2025/run", run="East_River_r2_20261015_110603", obs="/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/observations/streamflow/preprocessed/East_River_USGS09112500_dailyQ_WY1970_2025.csv"),
       "Tuolumne River": dict(old="/scratch/dlhogan/ess-project-data/calibration/bestruns/Tuolumne_River_baseline_1981_2025/run", run="Tuolumne_River_r2_20261015_110603", obs="/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/preprocessed/Tuolumne_River_merged_dailyQ_WY1971_2025.csv")}
INK, GRID, MODEL, OLD = "#0b0b0b", "#d9d8d4", "#2a78d6", "#eb6834"
def wy(i): return i.year + (i.month >= 10)
data = {}
for nm, c in DOM.items():
    d = xr.open_dataset(L+c["run"]+"/best_full/rank0/out/best_timestep.nc"); t = pd.DatetimeIndex(d["time"].values)
    A = xr.open_dataset(L+c["run"]+"/best_full/rank0/attributes.nc")["HRUarea"].values.sum()
    sim = (pd.Series(d["averageRoutedRunoff"].isel(gru=0).values, index=t)*A).resample("D").mean()
    dd = xr.open_dataset(c["old"]+"/out/baseline_timestep.nc"); to = pd.DatetimeIndex(dd["time"].values)
    old = (pd.Series(dd["averageRoutedRunoff"].isel(gru=0).values, index=to)*xr.open_dataset(c["old"]+"/attributes.nc")["HRUarea"].values.sum()).resample("D").mean()
    o = pd.read_csv(c["obs"], parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
    j = pd.concat([o.rename("obs"), sim.rename("sim"), old.rename("old")], axis=1).dropna(); j = j[(j.index >= "2008-10-01") & (j.index <= "2020-09-30")]
    ann = j.groupby(wy(j.index)).obs.mean(); ann = ann[j.groupby(wy(j.index)).size() >= 360]
    yrs = {"Dry": ann.idxmin(), "Average": (ann-ann.median()).abs().idxmin(), "Wet": ann.idxmax()}
    data[nm] = (j, yrs, ann)
    print(nm, {k: (int(v), round(float(ann[v]), 1)) for k, v in yrs.items()}, "| sim/obs annual mean over all 12 yr:", round(j.sim.mean()/j.obs.mean(), 3))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(4, 3, figsize=(13.5, 11.5), sharex=True)
for bi, (nm, (j, yrs, ann)) in enumerate(data.items()):
    for ci, (cls, y) in enumerate(yrs.items()):
        g = j[wy(j.index) == y]; x = (g.index - pd.Timestamp(int(y)-1, 10, 1)).days
        for ri, log in enumerate((False, True)):
            a = ax[2*bi+ri, ci]
            a.plot(x, g.obs, color=INK, lw=1.9, label="Observed", zorder=3); a.plot(x, g.old, color=OLD, lw=1.3, ls=(0, (4, 2)), label="Previous calibration (WY1989/93-2002 fit)", zorder=1); a.plot(x, g.sim, color=MODEL, lw=1.7, label="Round-2 best (WY2009-2020 fit)", zorder=2)
            if log: a.set_yscale("log"); a.set_ylim(max(0.05, min(g.obs.min(), g.sim.min(), g.old.min())*0.7), max(g.obs.max(), g.sim.max(), g.old.max())*1.4)
            else: a.set_ylim(0, max(g.obs.max(), g.sim.max(), g.old.max())*1.08)
            a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
            a.set_xlim(0, 365); a.set_xticks([0, 92, 182, 274]); a.set_xticklabels(["Oct 1", "Jan 1", "Apr 1", "Jul 1"])
            if ci == 0: a.set_ylabel(f"{nm}\n" + ("discharge (m³/s)" if not log else "discharge (m³/s), log"), color="#52514e")
            if ri == 0:
                bias = 100*(g.sim.mean()/g.obs.mean()-1); bo = 100*(g.old.mean()/g.obs.mean()-1)
                a.set_title(f"{cls}: WY{int(y)}  (obs mean {g.obs.mean():.1f} m³/s)\nannual bias: round-2 {bias:+.0f}%, previous {bo:+.0f}%", fontsize=9, color=INK, loc="left")
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.suptitle("Round-2 recalibration vs observed: dry, average and wet water years (calibration window WY2009–2020)", y=1.025, fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.975)); out = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/r2_vs_previous_dry_avg_wet_hydrographs.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
