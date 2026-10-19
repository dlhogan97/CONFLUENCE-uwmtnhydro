"""Figure: 7-day min low-flow, WY2007-2021, every source -- series, lag structure, agreement."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

SW = Path("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect"
          "/simulations/snowpatch_sweep_20260812")
FIG = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft")

M = pd.read_csv(SW/"min7_allsources_2007_2021.csv", index_col=0)
A = pd.read_csv(SW/"acf_7daymin_allsources_2007_2021.csv").set_index("series")

# class -> (colour, members[(col, label, linestyle)])
GRP = [
    ("unregulated headwater gauges", "#127A63", [
        ("nested LBM", "LBM", "-"), ("nested LABvTB", "LABvTB", "--"),
        ("nested DFkBC", "DFkBC", ":")]),
    ("T120 — 99× step at WY2017", "#C2271A", [("nested T120", "T120", "-")]),
    ("HH unimpaired (raw ≡ scaled)", "#7A44B5", [("HH raw (unimpaired)", "HH", "-")]),
    ("regulated downstream", "#9A9A9A", [("regulated downstream", "regulated", "-")]),
    ("SUMMA", "#1F5FA9", [("SUMMA baseline", "baseline", "-"),
                          ("SUMMA snowpatch m090", "snowpatch", "--")]),
]
LAGS = np.arange(1, 6)

mpl.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 7.4,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "figure.dpi": 150})

fig = plt.figure(figsize=(11.4, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.35, 1],
                      left=0.065, right=0.985, top=0.855, bottom=0.075,
                      hspace=0.38, wspace=0.22)

# ---------------------------------------------------------------- A: the series
ax = fig.add_subplot(gs[0, :])
for _, c, members in GRP:
    for col, lab, ls in members:
        ax.plot(M.index, M[col], ls, color=c, lw=1.9, marker="o", ms=4.5, mfc=c,
                mec="white", mew=0.8, label=lab)
ax.set_yscale("log")
ax.axvspan(2016.5, 2021.5, color="#C2271A", alpha=0.055, lw=0, zorder=0)
ax.annotate("T120 jumps ~99×\n(unapplied cfs→cms)", xy=(2018, 1.6), fontsize=7.6,
            color="#C2271A", ha="center")
ax.set_ylabel("7-day min, Aug–Oct  (m³/s, log)")
ax.set_xticks(M.index); ax.set_xlim(2006.5, 2021.5)
ax.set_title("A · The 15 years, every source", loc="left", fontweight="bold")
ax.legend(loc="lower left", frameon=False, ncol=5, handlelength=1.6,
          columnspacing=1.1, borderpad=0.2)

# ---------------------------------------------------------------- B: lag structure
ax = fig.add_subplot(gs[1, 0])
band = float(A.loc["SUMMA baseline", "band"])
ax.axhspan(-band, band, color="#000000", alpha=0.055, lw=0, zorder=0)
ax.axhline(0, color="#bdbdbd", lw=0.8, zorder=1)
for _, c, members in GRP:
    for col, lab, ls in members:
        r = np.array([A.loc[col, f"rho{k}"] for k in LAGS], float)
        p = np.array([A.loc[col, f"p{k}"] for k in LAGS], float)
        ax.plot(LAGS, r, ls, color=c, lw=1.9, marker="o", ms=5, mfc=c,
                mec="white", mew=0.9, zorder=3, label=lab)
        sig = p < 0.05
        if sig.any():
            ax.plot(LAGS[sig], r[sig], "o", ms=9.5, mfc="none", mec=c, mew=1.3, zorder=2)
ax.set_xticks(LAGS); ax.set_xlabel("lag (years)")
ax.set_ylabel("Spearman ACF of the 7-day min")
ax.set_ylim(-0.95, 0.95)
ax.set_title("B · Lag structure  (n = 15; ring = p < 0.05)", loc="left", fontweight="bold")

# ---------------------------------------------------------------- C: agreement
ax = fig.add_subplot(gs[1, 1])
order = ["nested LBM", "nested LABvTB", "nested DFkBC", "nested T120",
         "HH raw (unimpaired)", "regulated downstream",
         "SUMMA baseline", "SUMMA snowpatch m090"]
short = ["LBM", "LABvTB", "DFkBC", "T120", "HH", "regul.", "SUMMA", "snowp."]
C = M[order].corr(method="spearman")
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=45, ha="right")
ax.set_yticks(range(len(short))); ax.set_yticklabels(short)
for i in range(len(short)):
    for j in range(len(short)):
        v = C.values[i, j]
        ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                fontsize=6.6, color="white" if abs(v) > 0.55 else "#333333")
ax.set_title("C · Do they agree which years were low?", loc="left", fontweight="bold")
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("Spearman", fontsize=8)

fig.text(0.065, 0.975, "Tuolumne 7-day minimum low flow, WY2007–2021: nine sources, one window",
         fontsize=12, fontweight="bold", va="top")
fig.text(0.065, 0.938,
         "Only filter is Q > 0 — no low-flow cap.  HH raw and HH scaled differ by a constant "
         "K = 0.5051, so they are rank-identical and give the same ACF to 2 decimals.\n"
         "Real unregulated gauges vary 96–1086× across these 15 years; HH unimpaired varies 2.8× "
         "and the regulated gauge 2.5× — neither resolves low flow.",
         fontsize=8, color="#555555", va="top", linespacing=1.5)

FIG.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(FIG/f"tuolumne_7daymin_allsources_2007_2021.{ext}",
                bbox_inches="tight", facecolor="white")
print("wrote", FIG/"tuolumne_7daymin_allsources_2007_2021.png")
