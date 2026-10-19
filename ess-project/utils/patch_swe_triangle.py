"""The three pairwise relations among patch area, peak SWE and late-summer low flow.

(a) low flow vs end-of-summer patch area
(b) low flow vs peak SWE
(c) patch area vs peak SWE

All three are OBSERVED only -- this is a statement about the real basin, not about a model.
Spearman rho is the headline in every panel because it is rank-based and therefore
comparable across the three despite the mixed axis transforms (area and flow are strictly
positive and drawn on log axes; the peak-SWE index is a z-score and cannot be logged).

Below the panels, partial rank correlations answer the question the pairwise numbers cannot:
does patch area still track low flow once the snow year is held fixed, and vice versa? That
is the test of "stronger relationship" -- a raw rho difference could just be the two
predictors sharing the same wet-year/dry-year signal.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr, rankdata

REPO = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro")
sys.path.insert(0, str(REPO/"ess-project/utils"))
import trend_acf_summary as T
from patch_swe import peak_swe_index

FIG = REPO/"ess-project/figures/draft"
DROUGHT = (1987, 1992)
HUE = "#D55E00"


def shade(hexc, amt):
    r, g, b = mcolors.to_rgb(hexc)
    return tuple(c + (1.0-c)*amt for c in (r, g, b)) if amt >= 0 else tuple(c*(1+amt) for c in (r, g, b))


C = shade(HUE, -0.45)


def partial_spearman(x, y, z):
    """Rank correlation of x and y with the rank of z projected out of both."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    Z = sm.add_constant(rz)
    ex = rx - sm.OLS(rx, Z).fit().fittedvalues
    ey = ry - sm.OLS(ry, Z).fit().fittedvalues
    r, p = spearmanr(ex, ey)
    return r, p, len(rx)


def mark(p):
    return "**" if p < 0.05 else "+" if p < 0.10 else ""


def panel(ax, x, y, xlab, ylab, tag, note, logx):
    """One scatter with its fit; returns the Spearman stats for the caption line."""
    r, p = spearmanr(x, y)
    ax.scatter(x, y, s=110, color=C, edgecolor="white", lw=1.4, zorder=5, alpha=0.95)

    # fit in whatever space the axes are drawn in, so the line is straight on screen
    xx = np.log(x) if logx else np.asarray(x, float)
    m = sm.OLS(np.log(y), sm.add_constant(xx)).fit()
    g = np.linspace(xx.min(), xx.max(), 50)
    ax.plot(np.exp(g) if logx else g, np.exp(m.params.iloc[0] + m.params.iloc[1]*g),
            color=C, lw=2.6, zorder=4)

    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(f"{tag}  {note}\nρ = {r:+.2f}{mark(p)}   p = {p:.3f}   n = {len(x)}",
                 loc="left", fontsize=12)
    ax.grid(alpha=0.25, which="both")
    return r, p, len(x)


def main(drop_drought=False):
    ns = T.notebook_ns(fast=True); yr_low = ns["yr_low"]
    eos = pd.read_csv(T.SCRATCH/"geospatial-data/snowpatch_fsca/eos_final.csv").set_index("year")

    D = pd.concat({"area": eos["A_shadow"],
                   "flow": yr_low(ns["tuolumne_obs_augoct_low_flow"]),
                   "swe":  peak_swe_index()}, axis=1).dropna()
    D = D[(D.area > 0) & (D.flow > 0)]
    if drop_drought:
        D = D[~D.index.to_series().between(*DROUGHT)]

    plt.style.use("seaborn-v0_8-poster")
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.0))

    A = "end-of-summer patch area (km²)"
    Q = "Aug–Oct 7-day min flow (m³/s)"
    S = "peak-SWE anomaly (z)"

    s_aq = panel(axes[0], D.area, D.flow, A, Q, "a", "low flow vs patch area", logx=True)
    s_sq = panel(axes[1], D.swe,  D.flow, S, Q, "b", "low flow vs peak SWE",   logx=False)
    s_sa = panel(axes[2], D.swe,  D.area, S, A, "c", "patch area vs peak SWE", logx=False)

    p_aq = partial_spearman(D.area, D.flow, D.swe)     # area -> flow, snow year held fixed
    p_sq = partial_spearman(D.swe,  D.flow, D.area)    # snow  -> flow, patch area held fixed

    fig.suptitle(
        f"Tuolumne, WY{D.index.min()}–{D.index.max()}"
        + (f" (WY{DROUGHT[0]}–{DROUGHT[1]} excluded)" if drop_drought else "")
        + f"   |   partial ρ, controlling for the third variable:"
          f"   area→flow {p_aq[0]:+.2f}{mark(p_aq[1])} (p={p_aq[1]:.3f})"
          f"    SWE→flow {p_sq[0]:+.2f}{mark(p_sq[1])} (p={p_sq[1]:.3f})"
          f"      ** p<0.05, + p<0.10",
        fontsize=13, y=1.02)

    fig.tight_layout()
    tag = "_nodrought" if drop_drought else ""
    out = FIG/f"patch_swe_lowflow_triangle{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)
    for nm, s in (("a  flow ~ area", s_aq), ("b  flow ~ SWE ", s_sq), ("c  area ~ SWE ", s_sa)):
        print(f"  {nm}   rho={s[0]:+.3f}  p={s[1]:.4f}  n={s[2]}")
    print(f"  partial area->flow | SWE   rho={p_aq[0]:+.3f}  p={p_aq[1]:.4f}  n={p_aq[2]}")
    print(f"  partial SWE ->flow | area  rho={p_sq[0]:+.3f}  p={p_sq[1]:.4f}  n={p_sq[2]}")
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drop-drought", action="store_true")
    a = ap.parse_args()
    main(a.drop_drought)
