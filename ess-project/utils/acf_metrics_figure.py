"""Figure: lag-1..5 ACF on two low-flow metrics x two basins, obs vs baseline vs adjusted."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

SC = Path("/scratch/dlhogan/ess-project-data")
EAST_SW = SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/sweep_sf_20260812"
TUO_SW = SC/"domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_sweep_20260812"
FIG = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft")

OBS_C, DESTEP_C, BASE_C = "#1b1b1b", "#B0177A", "#8a8a8a"
ADJ = ["#1F5FA9", "#C2571A"]
LAGS = np.arange(1, 6)

mpl.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "figure.dpi": 150})

ea = pd.read_csv(EAST_SW/"acf_two_metrics.csv")
ta = pd.read_csv(TUO_SW/"acf_two_metrics.csv")

COLS = [("East River", ea, [("sf=7", "aquifer sf = 7", ADJ[0]), ("sf=24", "aquifer sf = 24", ADJ[1])]),
        ("Tuolumne River", ta, [("m090_q089", "snowpatch m090_q089", ADJ[0]),
                                ("m100_q320", "snowpatch m100_q320", ADJ[1])])]
ROWS = ["7-day min", "September mean"]

fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.9), sharex=True)
fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.075, hspace=0.30, wspace=0.19)

for i, metric in enumerate(ROWS):
    for j, (basin, tab, adj) in enumerate(COLS):
        ax = axes[i, j]
        t = tab[tab.metric == metric].set_index("model")
        band = float(t.loc["OBSERVED", "band"])
        ax.axhspan(-band, band, color="#000000", alpha=0.055, lw=0, zorder=0)
        ax.axhline(0, color="#bdbdbd", lw=0.8, zorder=1)

        wanted = [("OBSERVED", "observed", OBS_C, "-", 2.2, 6.5)]
        if "OBS, era-normalised" in t.index:
            wanted.append(("OBS, era-normalised", "observed, era-normalised",
                           DESTEP_C, "-", 1.8, 5.5))
        wanted.append(("baseline", "baseline", BASE_C, "--", 2.0, 5.5))
        wanted += [(k, l, c, "-", 2.0, 5.5) for k, l, c in adj]

        for key, lab, c, ls, lw, ms in wanted:
            if key not in t.index:
                continue
            r = np.array([t.loc[key, f"rho{k}"] for k in LAGS], float)
            p = np.array([t.loc[key, f"p{k}"] for k in LAGS], float)
            ax.plot(LAGS, r, ls, color=c, lw=lw, marker="o", ms=ms, mfc=c,
                    mec="white", mew=1.1, zorder=3, label=lab)
            sig = p < 0.05
            if sig.any():
                ax.plot(LAGS[sig], r[sig], "o", ms=ms+4.5, mfc="none", mec=c,
                        mew=1.3, zorder=2)

        ax.set_xticks(LAGS)
        ax.set_ylim(-0.55, 0.80)
        if i == 1:
            ax.set_xlabel("lag (years)")
        if j == 0:
            ax.set_ylabel(f"Spearman ACF\n{metric}")
        if i == 0:
            ax.set_title(basin, loc="left", fontweight="bold")
        ax.legend(loc="upper right", frameon=False, handlelength=1.5,
                  borderpad=0.15, labelspacing=0.22)

fig.text(0.085, 0.975, "Observed low-flow memory is metric- and record-dependent",
         fontsize=11.5, fontweight="bold", va="top")
fig.text(0.085, 0.943,
         "East is a single USGS gauge (09112500) and its lag structure is stable across both metrics.  "
         "The Tuolumne record splices a Hetch Hetchy\nreconstruction (WY≤2006, 2.6× wetter at low flow) "
         "onto the real gauge (WY≥2007); that step alone manufactures significant ACF at every lag.",
         fontsize=8, color="#555555", va="top", linespacing=1.5)
fig.text(0.985, 0.975, "shaded = ±1.96/√n\nring = permutation p < 0.05",
         fontsize=7.6, color="#666666", va="top", ha="right", linespacing=1.5)

FIG.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(FIG/f"memory_acf_two_metrics.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", FIG/"memory_acf_two_metrics.png")
