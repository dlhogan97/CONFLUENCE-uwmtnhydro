"""East stage-A best trial: dry / average / wet water-year hydrographs vs observed (linear + log), calibration window WY2009-2020 and held-out WY2021-2025; round-2 best shown for reference. -> ess-project/figures/draft/"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
L = "/scratch/dlhogan/ess-project-data/calibration/runs/"
SA = L+"East_River_stageA_20261019_142114/best_full_work/keep"; R2 = L+"East_River_r2_20261015_110603/best_full/rank0"
OBS = "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/observations/streamflow/preprocessed/East_River_USGS09112500_dailyQ_WY1970_2025.csv"
INK, GRID, MODEL, OLD = "#0b0b0b", "#d9d8d4", "#2a78d6", "#eb6834"
def wy(i): return i.year + (i.month >= 10)
def load(d, f):
    ds = xr.open_dataset(f"{d}/out/{f}"); t = pd.DatetimeIndex(ds["time"].values); A = xr.open_dataset(f"{d}/attributes.nc")["HRUarea"].values.sum()
    return (pd.Series(ds["averageRoutedRunoff"].isel(gru=0).values, index=t)*A).resample("D").mean()
sa = load(SA, "best_timestep.nc"); r2 = load(R2, "best_timestep.nc")
o = pd.read_csv(OBS, parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
J = pd.concat([o.rename("obs"), sa.rename("sa"), r2.rename("r2")], axis=1)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
def fig_for(t0, t1, title, out, r2_ok):
    j = J[(J.index >= t0) & (J.index <= t1)].dropna(subset=["obs", "sa"]); g_ = j.groupby(wy(j.index)); ann = j.groupby(wy(j.index)).obs.mean(); ann = ann[g_.size() >= 360]
    yrs = {"Dry": ann.idxmin(), "Average": (ann-ann.median()).abs().idxmin(), "Wet": ann.idxmax()}
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.4), sharex=True)
    for ci, (cls, y) in enumerate(yrs.items()):
        g = j[wy(j.index) == y]; x = (g.index - pd.Timestamp(int(y)-1, 10, 1)).days
        for ri, log in enumerate((False, True)):
            a = ax[ri, ci]; a.plot(x, g.obs, color=INK, lw=1.9, label="Observed", zorder=3)
            if r2_ok: a.plot(x, g.r2, color=OLD, lw=1.2, ls=(0, (4, 2)), label="Round-2 best (not equilibrium-consistent)", zorder=1)
            a.plot(x, g.sa, color=MODEL, lw=1.7, label="Stage-A best", zorder=2)
            if log: a.set_yscale("log"); a.set_ylim(max(0.05, g[["obs", "sa"]].min().min()*0.7), g[["obs", "sa"]].max().max()*1.4)
            else: a.set_ylim(0, g[["obs", "sa"]].max().max()*1.08)
            a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True); a.set_xlim(0, 365); a.set_xticks([0, 92, 182, 274]); a.set_xticklabels(["Oct 1", "Jan 1", "Apr 1", "Jul 1"])
            if ci == 0: a.set_ylabel("discharge (m³/s)" + (", log" if log else ""))
            if ri == 0: a.set_title(f"{cls}: WY{int(y)}  (obs mean {g.obs.mean():.1f} m³/s)\nannual bias: stage-A {100*(g.sa.mean()/g.obs.mean()-1):+.0f}%", fontsize=9, color=INK, loc="left")
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.04, fontsize=12, color=INK); fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(out, {k: int(v) for k, v in yrs.items()}, "annual bias over window %+.1f%%" % (100*(j.sa.mean()/j.obs.mean()-1)))
D = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
fig_for("2008-10-01", "2020-09-30", "East River stage-A best vs observed: dry, average, wet water years (calibration WY2009-2020)", D+"east_stageA_dry_avg_wet_calibration.png", True)
fig_for("2020-10-01", "2025-09-30", "East River stage-A best vs observed: dry, average, wet water years (held out WY2021-2025)", D+"east_stageA_dry_avg_wet_heldout.png", False)
