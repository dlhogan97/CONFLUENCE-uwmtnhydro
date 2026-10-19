"""Patch area vs late-summer low flow, observed and modelled, with the elasticity as a slope.

Both axes are logged, so the slope of a fitted line IS the elasticity: the percent change in
Aug-Oct 7-day minimum flow per percent change in end-of-summer patch area.

TWO SLOPES ARE DRAWN, because they are different quantities and only one of them is a line you
can honestly fit to this scatter:

  solid   RAW elasticity -- the OLS slope of these points. What the eye sees.
  dashed  SWE-CONTROLLED elasticity -- the coefficient on log(area) in
          log Q = a + e log A + b log SWE. It is a PARTIAL slope, so it does not belong to any
          line through this cloud; it is drawn through the centroid only so the two can be
          compared. It is shallower because peak SWE is an upstream cause of patch area
          (rho ~ +0.75), so part of the raw slope is the snow year, not the patch.

The controlled number is the one to report. The raw one is shown so a reader who measures the
slope off the page is not surprised.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

REPO = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro")
sys.path.insert(0, str(REPO/"ess-project/utils"))
import trend_acf_summary as T

FIG = REPO/"ess-project/figures/draft"
DATA = T.SCRATCH/"geospatial-data/snowpatch_fsca"
SNOW = (T.SCRATCH/"domain_Tuolumne_River_distributed_elevAspect/simulations"
        "/snowpatch_anchor_20260812/anchor_m080")
HUE = "#D55E00"                                   # Tuolumne, matching every other figure
Y0, Y1 = 1989, 2023


def shade(hexc, amt):
    r, g, b = mcolors.to_rgb(hexc)
    return tuple(c + (1.0-c)*amt for c in (r, g, b)) if amt >= 0 else tuple(c*(1+amt) for c in (r, g, b))


def panel(ax, A, Q, S, col, tag, title):
    d = pd.concat([np.log(A).rename("la"), np.log(Q).rename("lq"),
                   np.log(S).rename("ls")], axis=1).dropna().loc[Y0:Y1]
    raw = sm.OLS(d.lq, sm.add_constant(d.la)).fit()
    ctl = sm.OLS(d.lq, sm.add_constant(d[["la", "ls"]])).fit(cov_type="HC3")
    e_raw, e_ctl, p_ctl = raw.params.iloc[1], ctl.params["la"], ctl.pvalues["la"]
    rho, p_rho = spearmanr(d.la, d.lq)

    ax.scatter(np.exp(d.la), np.exp(d.lq), s=95, color=col, edgecolor="white",
               lw=1.3, zorder=5, alpha=0.95)
    g = np.linspace(d.la.min(), d.la.max(), 50)
    ax.plot(np.exp(g), np.exp(raw.params.iloc[0] + e_raw*g), color=col, lw=2.4, zorder=4,
            label=f"raw  ε = {e_raw:+.2f}")
    # partial slope, anchored at the centroid so only the SLOPE is being compared
    ax.plot(np.exp(g), np.exp(d.lq.mean() + e_ctl*(g - d.la.mean())), color=col, lw=2.2,
            ls=(0, (5, 3)), zorder=4, label=f"SWE-controlled  ε = {e_ctl:+.2f}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax._lims = (d.la.min(), d.la.max(), d.lq.min(), d.lq.max())
    ax.set_xlabel("end-of-summer patch area (km²)")
    ax.set_ylabel("Aug–Oct 7-day minimum flow (m³/s)")
    ax.set_title(f"{tag}  {title}\nρ = {rho:+.2f}   ε = {e_ctl:+.2f} (p = {p_ctl:.3f})   n = {len(d)}",
                 loc="left", fontsize=12)
    ax.grid(alpha=0.22, which="both", lw=0.6)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    return e_raw, e_ctl, p_ctl, rho, len(d)


def main():
    eos = pd.read_csv(DATA/"eos_final.csv").set_index("year")
    swe = pd.read_csv(DATA/"dan_peak_swe.csv", comment="#", index_col="water_year")["peak_swe_mm"]
    ns = T.notebook_ns(fast=True); yr_low = ns["yr_low"]

    plt.style.use("seaborn-v0_8-poster")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True, sharey=True)
    o = panel(axes[0], eos["A_shadow"], yr_low(ns["tuolumne_obs_augoct_low_flow"]), swe,
              shade(HUE, -0.45), "a", "observed (Landsat)")
    m = panel(axes[1], T.modelled_patch_area(SNOW, SNOW/"bin_spec.csv"),
              yr_low(ns["tuolumne_adj_augoct_low_flow"]), swe,
              shade(HUE, 0.15), "b", "modelled (snowpatch run)")
    # SHARED, EQUAL-DECADE AXES. Non-negotiable for this figure: the reader is being asked to
    # compare two slopes by eye, and a slope only looks like its value if both panels span the
    # same number of log decades per unit of plot. Different auto-ranges would make eps=+0.11 and
    # eps=+0.21 look far more alike than they are.
    xs = [a._lims[0] for a in axes] + [a._lims[1] for a in axes]
    ys = [a._lims[2] for a in axes] + [a._lims[3] for a in axes]
    px, py = 0.10*(max(xs)-min(xs)), 0.10*(max(ys)-min(ys))
    axes[0].set_xlim(np.exp(min(xs)-px), np.exp(max(xs)+px))
    axes[0].set_ylim(np.exp(min(ys)-py), np.exp(max(ys)+py))
    for a in axes:
        a.legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle(f"Tuolumne, WY{Y0}–{Y1}   |   log–log slope = elasticity of low flow to patch area",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out = FIG/"patch_elasticity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("wrote", out)
    for nm, s in (("observed", o), ("model   ", m)):
        print(f"  {nm}  raw ε={s[0]:+.3f}  controlled ε={s[1]:+.3f} (p={s[2]:.3f})  "
              f"ρ={s[3]:+.3f}  n={s[4]}")
    return fig


if __name__ == "__main__":
    main()
