"""Figure: lag-1..5 low-flow ACF (obs vs baseline vs adjusted) + skill cost of the adjustment."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

SC = Path("/scratch/dlhogan/ess-project-data")
EAST_SW = SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/sweep_sf_20260812"
TUO_SW = SC/"domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_sweep_20260812"
FIG = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft")

OBS_C, BASE_C = "#1b1b1b", "#8a8a8a"
ADJ = ["#1F5FA9", "#C2571A"]          # blue = minimal adjustment, orange = strong
LAGS = np.arange(1, 6)

mpl.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "figure.dpi": 150})

ea = pd.read_csv(EAST_SW/"acf_lag1to5.csv").set_index("model")
es = pd.read_csv(EAST_SW/"skill_scores.csv").set_index("model")
ta = pd.read_csv(TUO_SW/"acf_lag1to5.csv").set_index("model")
ts = pd.read_csv(TUO_SW/"skill_scores.csv").set_index("model")

PANELS = [
    ("East River", ea, [("sf=7", "aquifer sf = 7"), ("sf=9", "aquifer sf = 9")]),
    ("Tuolumne River", ta, [("m090_q089", "snowpatch m090_q089"), ("m100_q320", "snowpatch m100_q320")]),
]

fig = plt.figure(figsize=(9.6, 6.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.24,
                      left=0.075, right=0.985, top=0.885, bottom=0.08)

# ---------------------------------------------------------------- row 1: ACF
for j, (basin, tab, adj) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, j])
    band = float(tab.loc["OBSERVED", "band"])
    ax.axhspan(-band, band, color="#000000", alpha=0.055, lw=0, zorder=0)
    ax.axhline(0, color="#bdbdbd", lw=0.8, zorder=1)

    def series(key):
        return np.array([tab.loc[key, f"rho{k}"] for k in LAGS]), \
               np.array([tab.loc[key, f"p{k}"] for k in LAGS])

    for key, lab, c, lw, ls, ms in [
            ("OBSERVED", "observed", OBS_C, 2.0, "-", 6.5),
            ("baseline", "baseline", BASE_C, 2.0, "--", 6.0)] + [
            (k, l, ADJ[i], 2.0, "-", 6.0) for i, (k, l) in enumerate(adj)]:
        r, p = series(key)
        ax.plot(LAGS, r, ls, color=c, lw=lw, zorder=3,
                marker="o", ms=ms, mfc=c, mec="white", mew=1.2, label=lab)
        sig = p < 0.05
        if sig.any():
            ax.plot(LAGS[sig], r[sig], "o", ms=ms + 4.5, mfc="none", mec=c,
                    mew=1.3, zorder=2)

    ax.set_xticks(LAGS)
    ax.set_xlabel("lag (years)")
    if j == 0:
        ax.set_ylabel("Spearman ACF of 7-day min low flow")
    ax.set_ylim(-0.55, 0.78)
    ax.set_title(basin, loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderpad=0.2,
              labelspacing=0.25)

fig.text(0.075, 0.965, "Low-flow memory: does the added store reproduce the observed lag structure?",
         fontsize=11, fontweight="bold", va="top")
fig.text(0.075, 0.933, "shaded = ±1.96/√n noise band   ·   ringed marker = permutation p < 0.05"
                       "   ·   skill from WY1984 (meta.json analysis_start)",
         fontsize=8, color="#666666", va="top")

# ---------------------------------------------------------------- row 2: skill cost
SPEC = [("East River", es, "baseline", ["sf=7", "sf=8", "sf=9", "sf=10", "sf=11", "sf=12",
                                        "sf=14", "sf=16", "sf=18", "sf=21", "sf=24"],
         ["sf=7", "sf=9"], "Sep–Nov"),
        ("Tuolumne River", ts, "baseline",
         [i for i in ts.index if i != "baseline"], ["m090_q089", "m100_q320"], "Aug–Oct")]

for j, (basin, tab, base, adjs, hi, win) in enumerate(SPEC):
    ax = fig.add_subplot(gs[1, j])
    mets = ["NSE", "KGE", "logNSE", "KGE_low"]
    x = np.arange(len(mets))
    bv = tab.loc[base, mets].values.astype(float)

    ax.axhline(0, color="#bdbdbd", lw=0.8, zorder=1)
    # full range spanned by every adjusted member
    lo = tab.loc[adjs, mets].values.astype(float).min(0)
    hiv = tab.loc[adjs, mets].values.astype(float).max(0)
    ax.bar(x, hiv - lo, bottom=lo, width=0.46, color="#d8d8d8", zorder=2,
           label="range across all members")
    for i, k in enumerate(hi):
        v = tab.loc[k, mets].values.astype(float)
        ax.plot(x + (i - 0.5) * 0.16, v, "o", ms=7, color=ADJ[i], mec="white",
                mew=1.2, zorder=4, label=k)
    ax.plot(x, bv, "_", ms=17, mew=2.4, color=BASE_C, zorder=3, label="baseline")
    for xi, v in zip(x, bv):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color="#555555")

    ax.set_xticks(x); ax.set_xticklabels(["NSE", "KGE", "log NSE", f"KGE\n({win})"])
    ax.set_xlim(-0.55, len(mets) - 0.45)
    if j == 0:
        ax.set_ylabel("daily streamflow skill")
    ax.set_title(f"{basin} — skill cost", loc="left", fontweight="bold")
    ax.legend(loc="lower center", frameon=False, ncol=2, handlelength=1.1,
              columnspacing=1.4, borderpad=0.2, labelspacing=0.3,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_ylim(min(-0.95, lo.min() - 0.35), 1.05)

FIG.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(FIG/f"memory_acf_lag5_and_skill.{ext}", bbox_inches="tight",
                facecolor="white")
print("wrote", FIG/"memory_acf_lag5_and_skill.png")
