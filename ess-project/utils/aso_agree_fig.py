"""ASO vs Landsat band-8 fSCA agreement, all 38 matched flights.

Replaces the 3-point panel: aso_landsat_pairs_all.csv holds only the 3 late-season
matches, while aso_landsat_pairs_spring.csv holds 38 (Mar-Aug, |lag| <= 6 d).

Landsat NDSI detects snow PRESENCE, so the matching ASO definition is "any snow"
(SWE > 0.01 m) -- not the 0.25 m threshold the 3-flight set appeared to endorse.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

D = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
FIG = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft")
pairs = pd.read_csv(D / "aso_landsat_pairs_spring.csv", parse_dates=["aso_date", "ls_date"])

# sequential single hue, light (Mar) -> dark (Aug): season is a magnitude, not an identity
CMAP = LinearSegmentedColormap.from_list(
    "season", ["#d6e6f2", "#94c1de", "#4a8fc0", "#1f6094", "#0b3d62"])
NORM = BoundaryNorm(np.arange(2.5, 9.5, 1.0), CMAP.N)

x, y = pairs.aso_b8_t001.to_numpy(float), pairs.ls_b8.to_numpy(float)
k0 = (x * y).sum() / (x * x).sum()
r = np.corrcoef(x, y)[0, 1]
bias = (y - x).mean()
rmse = float(np.sqrt(((y - x) ** 2).mean()))

with plt.style.context("seaborn-v0_8-poster"):
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    lim = (0.0, 1.06)

    ax.plot(lim, lim, ls="--", lw=1.8, color="#8a8a8a", zorder=2)
    sc = ax.scatter(x, y, c=pairs.month, cmap=CMAP, norm=NORM,
                    s=150, edgecolor="white", lw=1.4, zorder=4)

    # marginal rugs: where the flights actually concentrate -- inside the axes,
    # rising off the x-axis and running right off the y-axis
    rug = 0.026 * (lim[1] - lim[0])
    ax.vlines(x, lim[0], lim[0] + rug, color="#5c5c5c", lw=1.3, alpha=0.6, zorder=3)
    ax.hlines(y, lim[0], lim[0] + rug, color="#5c5c5c", lw=1.3, alpha=0.6, zorder=3)

    # upper-left is the empty half (Landsat never far exceeds ASO); borderpad insets the
    # box clear of the y-rug running up the left edge
    at = AnchoredText(f"slope  {k0:.2f}\nr        {r:+.2f}\nbias   {bias:+.3f}\nRMSE  {rmse:.3f}",
                      loc="upper left", frameon=True, borderpad=1.6,
                      prop=dict(fontsize=15, family="DejaVu Sans Mono", color="#3a3a3a"))
    at.patch.set(boxstyle="round,pad=0.4", fc="white", ec="#d8d8d8", lw=1.0, alpha=0.92)
    at.set_zorder(6)
    ax.add_artist(at)

    # label the reference line in the empty lower-left, along its own slope
    ax.annotate("1:1", (0.145, 0.145), fontsize=15, color="#8a8a8a",
                rotation=45, rotation_mode="anchor", va="bottom", ha="center")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("ASO fSCA  (SWE > 0.01 m)")
    ax.set_ylabel("Landsat fSCA")
    ax.grid(color="#e2e2e2", lw=0.9); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cb = fig.colorbar(sc, ax=ax, ticks=range(3, 9), pad=0.03, fraction=0.045)
    cb.set_label("flight month", fontsize=16)
    cb.ax.set_yticklabels(["Mar", "Apr", "May", "Jun", "Jul", "Aug"], fontsize=14)
    cb.outline.set_visible(False)

    ax.set_title(f"ASO vs Landsat band-8 fSCA\n{len(pairs)} matched flights, "
                 f"{pairs.aso_date.min():%Y}–{pairs.aso_date.max():%Y}, |lag| ≤ 6 d",
                 loc="left", fontsize=16, fontweight="bold")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "aso_landsat_fsca_agreement_38flights.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")

print("wrote", out)
print(f"  n={len(pairs)}  slope0 {k0:.3f}  r {r:+.3f}  bias {bias:+.3f}  RMSE {rmse:.3f}")
