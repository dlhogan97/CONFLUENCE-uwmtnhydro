"""Pillow-vs-model SWE skill: the HRU containing the pillow vs the basin mean.

One value per basin-water-year, restricted to the year set every (basin, source) cell
shares, so all four rows are built from identical years. Line spans the interquartile
range over years, dot marks the median, gap either side.

Peak SWE error is a percentage of the record's own mean observed peak; the IQR enters
only as the span of the line, not as the normaliser. Timing metrics stay in days --
a day-of-water-year has an arbitrary 1 Oct origin, so normalising it is misleading.
Duration (peak -> melt-out) separates a melt-RATE error from a pure shift.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

D   = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
FIG = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft")
df  = pd.read_csv(D/"snotel_hru_swe_annual.csv")

BASINS = ["East", "Tuolumne"]
COLOR  = {"East": "#0072B2", "Tuolumne": "#D55E00"}     # matches the MRC facet figure
LO, HI = 25, 75                                          # interquartile range


def shade(hex_color, amount):
    """Blend toward white (amount>0) or black (amount<0). |amount| in [0, 1]."""
    r, g, b = mcolors.to_rgb(hex_color)
    if amount >= 0:
        return tuple(c + (1.0 - c)*amount for c in (r, g, b))
    return tuple(c*(1.0 + amount) for c in (r, g, b))


# hue = basin identity, lightness + marker = which model series -- either channel alone
# identifies the row, so the figure survives greyscale and CVD
SRC = [("hru", "HRU", -0.10, "o"), ("basin", "basin mean", 0.42, "s")]
ROWS = [(b, s, lab, sh, mk) for b in BASINS for s, lab, sh, mk in SRC]

PCT = lambda x, _: f"{100*x:g}"
PANELS = [
    ("nd_peak", "Peak SWE error", "% of mean observed peak",
     lambda v: f"nRMSE {100*np.sqrt((v**2).mean()):.1f}%", PCT),
    ("d_dur",   "Melt duration error  (peak → melt-out)", "days",
     lambda v: f"ME {v.mean():+.1f} d", None),
    ("d_tmelt", "Melt-out date error", "days",
     lambda v: f"ME {v.mean():+.1f} d", None),
]

with plt.style.context("seaborn-v0_8-poster"):
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.4))

    for ax, (col, name, unit, agg, fmt) in zip(axes, PANELS):
        vals = {(b, s): df.loc[(df.basin == b) & (df.source == s), col].dropna()
                for b, s, *_ in ROWS}
        allv = pd.concat(vals.values())
        pad  = 0.10*(allv.quantile(HI/100) - allv.quantile(LO/100))
        xlo  = min(v.quantile(LO/100) for v in vals.values()) - pad
        xhi  = max(v.quantile(HI/100) for v in vals.values()) + pad
        xlo, xhi = min(xlo, -pad), max(xhi, pad)      # keep the zero line on screen
        gap  = 0.014*(xhi - xlo)

        ax.axvline(0, ls="--", lw=1.4, color="#9a9a9a", zorder=1)
        for i, (b, s, lab, sh, mk) in enumerate(ROWS):
            v = vals[(b, s)]
            c = shade(COLOR[b], sh)
            lo, med, hi = v.quantile(LO/100), v.median(), v.quantile(HI/100)
            y = len(ROWS) - 1 - i                      # East on top
            ax.plot([lo, med - gap], [y, y], lw=3.0, color=c, solid_capstyle="butt", zorder=3)
            ax.plot([med + gap, hi], [y, y], lw=3.0, color=c, solid_capstyle="butt", zorder=3)
            ax.plot([med], [y], mk, ms=12, color=c, markeredgecolor="white",
                    markeredgewidth=1.6, zorder=4)
            ax.annotate(f"{agg(v)}   n={len(v)}", (1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(8, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=12, color=c)

        ax.set_yticks(range(len(ROWS)-1, -1, -1))
        ax.set_yticklabels([f"{b} · {lab}" for b, s, lab, sh, mk in ROWS], fontsize=13)
        ax.set_ylim(-0.7, len(ROWS)-0.3)
        ax.set_xlim(xlo, xhi)
        ax.set_title(f"{name}  ({unit})", loc="left", fontsize=14.5, fontweight="bold")
        if fmt is not None:
            ax.xaxis.set_major_formatter(fmt)
        ax.grid(axis="x", color="#e4e4e4", lw=0.9); ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=12.5)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("SUMMA vs snow pillow — the containing HRU vs the basin mean",
                 x=0.008, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.008, 0.004,
             f"line = interquartile range over {int(df.wy.nunique())} shared water years · "
             f"dot = median · model − observed · "
             + " · ".join(f"{b} = {df.loc[df.basin == b, 'site'].iloc[0]}" for b in BASINS),
             ha="left", fontsize=11.5, color="#5c5c5c")
    fig.tight_layout(rect=(0, 0.03, 0.86, 0.965))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG/"snotel_hru_vs_basin_dotline.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")

print("wrote", out)
print(f"  {df.wy.nunique()} shared water years, {int(df.wy.min())}-{int(df.wy.max())}")
for b, s, lab, *_ in ROWS:
    g = df[(df.basin == b) & (df.source == s)]
    print(f"  {b:9s} {lab:11s} n={len(g)}")
