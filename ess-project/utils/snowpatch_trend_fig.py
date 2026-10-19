"""Observed Tuolumne trends, one series per panel, normalised to the series median.

Four stacked panels -- patch area, late-season low flow, peak SWE, water-year precipitation.
One series per panel, so no dual axes and no shared scale.

NORMALISED BY THE MEDIAN, so every panel is centred on 1.0 and the four are directly
comparable in relative terms despite carrying four different units. The median in native
units is printed in each legend so nothing is lost.

THE CI WEDGE AND THE REFERENCE LINE SHARE AN ANCHOR, AND THAT MAKES THE FIGURE A TEST.
The wedge traces  anchor + b*(x - x_mid)  for b between the slope bounds, so the horizontal
line at the anchor lies INSIDE the wedge exactly when the slope interval contains zero. Read
it directly: wedge swallows the flat line -> no significant trend; wedge clears it -> trend
significant at this confidence level. The anchor is the MEDIAN, which is also the right
partner for Theil-Sen (both are median-based), and normalisation puts it at 1.0 everywhere.

Caveat: that is the Theil-Sen SLOPE interval, a different test from the Spearman and
Mann-Kendall p-values printed by main(). They usually agree; near the boundary they need not.

PATCH AREA AND LOW FLOW ARE ON LOG AXES -- both are strongly right-skewed (patch area spans
0.02x to 16x its own median), and on a linear axis the spike years own the panel and the
trend through the low years is unreadable. SWE and precipitation are near-symmetric and stay
linear from 0.

BLACK ONLY, separated by linestyle: solid = observations, dashed = Theil-Sen trend,
dotted = median, grey wedge = CI on the slope.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro")
sys.path.insert(0, str(REPO/"ess-project/utils"))
import trend_acf_summary as T
import lowflow_acf as L

FIG = REPO/"ess-project/figures/draft"
DATA = T.SCRATCH/"geospatial-data/snowpatch_fsca"
Y0, Y1 = 1993, 2025
CI_LEVEL = 0.95        # confidence level for the Theil-Sen slope interval
C = dict(patch="#D55E00", flow="#332288", swe="#56B4E9", ppt="#009E73")   # CVD-validated pairs


def draw(ax, s, label, unit, logy):
    """One normalised series: observations, Theil-Sen trend, median, slope-CI wedge."""
    s = s.loc[Y0:Y1].dropna()
    t = L.trend_stats(s, alpha=CI_LEVEL)
    med = float(np.median(s.values))
    z = s/med                                     # median -> 1.0; wedge anchor is log(1) = 0
    x = z.index.values.astype(float)
    xm = np.median(x)
    g = np.linspace(x.min(), x.max(), 200)
    pct = int(round(CI_LEVEL*100))
    ax.fill_between(g, np.exp(t["lo_s"]*(g-xm)), np.exp(t["hi_s"]*(g-xm)),
                    color="0.55", alpha=0.22, lw=0, zorder=1,
                    label=f"{pct}% CI  [{t['lo']:+.2f}, {t['hi']:+.2f}] %/yr")
    ax.plot(z.index, z.values, color="black", lw=1.6, ls="-", marker="o", ms=5.5,
            mfc="white", mew=1.4, mec="black", zorder=4, label=label)
    ax.plot(g, np.exp(t["slope"]*(g-xm)), color="black", lw=2.6, ls="--", zorder=5,
            label=f"trend  {t['pct']:+.2f} %/yr")
    mfmt = f"{med:,.0f}" if med >= 100 else f"{med:.1f}" if med >= 10 else f"{med:.2f}"
    ax.axhline(1.0, color="black", lw=2.0, ls=":", zorder=3,
               label=f"median  {mfmt} {unit}")
    if logy:
        ax.set_yscale("log")
        # subs from the span: over 2 decades every minor tick labelled swamps the axis; under
        # one decade the lone decade line can leave the panel with a single label
        dec = np.log10(z.max()/z.min())
        subs = ((1.,) if dec >= 2.5 else (1., 2., 5.) if dec >= 1.2
                else (1., 1.5, 2., 3., 5., 7.))
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs, numticks=18))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.set_ylim(0, z.max()*1.12)
    ax.set_ylabel(f"{label}\n(\u00f7 median)", fontsize=13)
    ax.grid(alpha=0.18, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # handles come out in draw order (CI, obs, trend, median); reorder to reading order
    h, lb = ax.get_legend_handles_labels()
    o = [1, 2, 0, 3]
    ax.legend([h[i] for i in o], [lb[i] for i in o], frameon=False, fontsize=11, ncol=2,
              loc="upper left", handlelength=2.6, columnspacing=1.8, labelspacing=0.35)
    return t


def main(y0=None, y1=None, ci=None):
    global Y0, Y1, CI_LEVEL
    if y0: Y0 = y0
    if y1: Y1 = y1
    if ci: CI_LEVEL = ci
    eos = pd.read_csv(DATA/"eos_final.csv").set_index("year")
    swe = pd.read_csv(DATA/"dan_peak_swe.csv", comment="#", index_col="water_year")["peak_swe_mm"]
    ppt = pd.read_csv(DATA/"tum_wy_precip.csv", comment="#", index_col="water_year")["wyppt_mm"]
    ns = T.notebook_ns(fast=True)
    flow = ns["yr_low"](ns["tuolumne_obs_augoct_low_flow"])

    plt.style.use("seaborn-v0_8-poster")
    fig, ax = plt.subplots(4, 1, figsize=(13.5, 16.5), sharex=True)
    R = {}
    R["patch"] = draw(ax[0], eos["A_shadow"], "patch area", "km\u00b2", True)
    R["flow"] = draw(ax[1], flow, "Aug\u2013Oct 7-day min flow", "m\u00b3/s", True)
    R["swe"] = draw(ax[2], swe, "peak SWE (DAN)", "mm", False)
    R["ppt"] = draw(ax[3], ppt, "WY precipitation (TUM)", "mm", False)
    ax[3].set_xlabel("water year", fontsize=14)
    ax[3].set_xlim(Y0-0.6, Y1+0.6)
    fig.tight_layout(h_pad=1.6)
    out = FIG/"snowpatch_trends.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("wrote", out, f" WY{Y0}-{Y1}  CI={int(round(CI_LEVEL*100))}%")
    for k, v in R.items():
        excl = "CI EXCLUDES 0" if (v["lo_s"] > 0) == (v["hi_s"] > 0) else "includes 0"
        print(f"  {k:6s} {v['pct']:+7.2f} %/yr [{v['lo']:+7.2f},{v['hi']:+7.2f}]  {excl:13s}"
              f"  p={v['p']:.3f}  MK p={v['p_mk']:.3f}  n={v['n']}")
    return fig


def indexed(y0=None, y1=None):
    """Single-axis variant: every series divided by its own GEOMETRIC mean over the window.

    No second scale, so nothing about where the lines sit or cross is an artefact of axis
    choice -- the only thing being compared is the angle, which is what the trends are. The
    geometric mean is the right centre for a multiplicative trend (it is the mean of log v),
    and on a log axis it puts every series at 1.0 with equal %/yr reading the same distance.
    """
    global Y0, Y1
    if y0: Y0 = y0
    if y1: Y1 = y1
    eos = pd.read_csv(DATA/"eos_final.csv").set_index("year")
    S = {"patch area (Landsat)":
             (eos["A_shadow"], C["patch"], "o"),
         "Aug-Oct 7-day min flow":
             (T.notebook_ns(fast=True)["yr_low"](
                 T.notebook_ns(fast=True)["tuolumne_obs_augoct_low_flow"]), C["flow"], "s"),
         "peak SWE (DAN pillow)":
             (pd.read_csv(DATA/"dan_peak_swe.csv", comment="#",
                          index_col="water_year")["peak_swe_mm"], C["swe"], "^"),
         "WY precipitation (TUM gauge)":
             (pd.read_csv(DATA/"tum_wy_precip.csv", comment="#",
                          index_col="water_year")["wyppt_mm"], C["ppt"], "D")}
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    R = {}
    for lab, (ser, col, mk) in S.items():
        v = ser.loc[Y0:Y1].dropna()
        t = L.trend_stats(v)
        idx = v/np.exp(np.log(v.values).mean())            # geometric mean -> 1.0
        x = idx.index.values.astype(float)
        xm = np.median(x)
        g = np.linspace(x.min(), x.max(), 100)
        anch = np.median(np.log(idx.values))
        ax.fill_between(g, np.exp(anch + t["lo_s"]*(g-xm)), np.exp(anch + t["hi_s"]*(g-xm)),
                        color=col, alpha=0.11, lw=0, zorder=1)
        ax.plot(g, np.exp(anch + t["slope"]*(g-xm)), color=col, lw=2.4, zorder=3)
        ax.plot(idx.index, idx.values, color=col, lw=1.6, marker=mk, ms=5.5, mfc="white",
                mew=1.5, mec=col, zorder=4, alpha=0.85,
                label=f"{lab}   {t['pct']:+.2f} %/yr  [{t['lo']:+.2f}, {t['hi']:+.2f}]")
        R[lab] = t
    ax.set_yscale("log")
    ax.axhline(1.0, color="#999999", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1., 2., 3., 5.), numticks=15))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("value ÷ series geometric mean", fontsize=11)
    ax.set_xlabel("water year", fontsize=11)
    ax.set_xlim(Y0-0.6, Y1+0.6)
    ax.grid(alpha=0.20, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, 1.01), handlelength=2.2, columnspacing=2.4,
              borderaxespad=0.0)
    fig.tight_layout()
    out = FIG/"snowpatch_trends_indexed.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("wrote", out, f" WY{Y0}-{Y1}")
    return fig


if __name__ == "__main__":
    main()
