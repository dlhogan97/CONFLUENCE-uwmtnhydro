"""Snowpatch decline and its link to low flow, modelled vs observed.

(a) end-of-summer patch area against water year, observed and modelled, with Theil-Sen
    slopes fitted twice: on every year (thin dashed) and with the 1987-1992 drought
    EXCLUDED (thick solid).
(b) low flow against patch area, observed and modelled, coloured by that year's peak-SWE
    anomaly so the reader can see the relation is not just wet-year/dry-year sorting.

Why the 1987-1992 years come out. That drought is a six-year depressed block sitting at the
START of both records. Theil-Sen is robust to outliers in the response, not to where they
sit along the predictor, so an early block of low values flattens the fitted slope. In the
model it is worse than leverage: the drought drains the anchor bin to its 0.043 km2 floor
and pins it there for all six years, cancelling the later decline outright (+0.0%/yr with
them, -9%/yr without).

They are EXCLUDED from every fit and every statistic here, but still drawn as small hollow
grey markers, so the reader can see what was removed rather than take it on trust. Set
show_dropped=False to hide them.
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
import lowflow_acf as L
from patch_swe import peak_swe_index

FIG = REPO/"ess-project/figures/draft"
DROUGHT = (1987, 1992)            # excluded from all fits and statistics
HUE = "#D55E00"                   # Tuolumne, matching every other figure


def shade(hexc, amt):
    r, g, b = mcolors.to_rgb(hexc)
    return tuple(c + (1.0-c)*amt for c in (r, g, b)) if amt >= 0 else tuple(c*(1+amt) for c in (r, g, b))


C_OBS, C_MOD = shade(HUE, -0.45), shade(HUE, 0.38)
MK = {"observed": "o", "snowpatch": "*"}       # marker carries obs-vs-model on its own


def keep(s, drought):
    """Series with the drought years removed."""
    return s[~s.index.to_series().between(*drought)].dropna()


def sen_line(ax, s, color, solid):
    """Theil-Sen fit in log space over whatever years `s` contains."""
    w = s.dropna()
    tr = L.trend_stats(w)
    x = np.array([w.index.min(), w.index.max()], float)
    ax.plot(x, np.exp(tr["slope"]*x + tr["icept"]), color=color,
            lw=3.0 if solid else 1.8, ls="-" if solid else (0, (6, 3)),
            alpha=0.95 if solid else 0.55, zorder=7 if solid else 6)
    return tr


def fmt(tr):
    return (f"{tr['pct']:+.1f}%/yr (p={tr['p']:.3f}"
            + ("**)" if tr["p"] < 0.05 else "+)" if tr["p"] < 0.10 else ")"))


def short(tr):
    """Compact slope + significance mark, for a three-column row in the panel title."""
    return (f"{tr['pct']:+5.1f}"
            + ("**" if tr["p"] < 0.05 else "+ " if tr["p"] < 0.10 else "  "))


def main(drought=DROUGHT, show_dropped=True):
    ns = T.notebook_ns(fast=True); yr_low = ns["yr_low"]
    eos = pd.read_csv(T.SCRATCH/"geospatial-data/snowpatch_fsca/eos_final.csv").set_index("year")
    SNOW = (T.SCRATCH/"domain_Tuolumne_River_distributed_elevAspect/simulations"
                      "/snowpatch_anchor_20260812/anchor_m080")

    area = {"observed": eos["A_shadow"].dropna(),
            "snowpatch": T.modelled_patch_area(SNOW, SNOW/"bin_spec.csv").dropna()}
    swe_idx = peak_swe_index().dropna()
    flow = {"observed": yr_low(ns["tuolumne_obs_augoct_low_flow"]).dropna(),
            "snowpatch": yr_low(ns["tuolumne_adj_augoct_low_flow"]).dropna()}
    swe = peak_swe_index().dropna()

    plt.style.use("seaborn-v0_8-poster")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.2, 6.4))

    # ---------------- (a) area against time ----------------
    ax_a.axvspan(drought[0]-0.5, drought[1]+0.5, color="#9a9a9a", alpha=0.16, zorder=0)
    ax_a.annotate(f"{drought[0]}–{drought[1]}\nexcluded", xy=(np.mean(drought), 0.965),
                  xycoords=("data", "axes fraction"), ha="center", va="top",
                  fontsize=11, color="#5f5f5f")
    lines = []
    for lab, c in (("observed", C_OBS), ("snowpatch", C_MOD)):
        s, k = area[lab], keep(area[lab], drought)
        d = s[s.index.to_series().between(*drought)]
        # the kept record is drawn broken across the gap, so no line implies data there
        for seg in (k[k.index < drought[0]], k[k.index > drought[1]]):
            if len(seg):
                ax_a.plot(seg.index, seg.values, color=c, lw=1.3, marker=MK[lab], ms=5.5,
                          mec="white", mew=1.0, alpha=0.9, zorder=4)
        ax_a.plot([], [], color=c, lw=1.3, marker=MK[lab], ms=5.5, mec="white", mew=1.0,
                  label=lab)
        if show_dropped and len(d):
            ax_a.plot(d.index, d.values, ls="none", marker=MK[lab], ms=5.0, mfc="none",
                      mec="#9a9a9a", mew=1.2, zorder=3)
        t_all = sen_line(ax_a, s, c, solid=False)
        t_cut = sen_line(ax_a, k, c, solid=True)

        # SWE-adjusted slope: Theil-Sen on the part of log(area) that peak SWE does NOT
        # explain. Peak SWE has no trend of its own, so this cannot manufacture one -- it
        # only strips the year-to-year snowpack variance. Reported as a number, not a line:
        # the residual series lives on a different scale and drawing it here would mislead.
        D = pd.concat({"y": k, "z": swe_idx}, axis=1).dropna()
        D = D[D.y > 0]
        beta = sm.OLS(np.log(D.y), sm.add_constant(D[["z"]])).fit().params["z"]
        resid = np.log(D.y) - beta*D.z
        t_swe = L.trend_stats(pd.Series(np.exp(resid - resid.mean() + np.log(D.y).mean()),
                                        index=D.index))
        lines.append(f"{lab:10s} {short(t_all)}  {short(t_cut)}  {short(t_swe)}"
                     f"   β={beta:+.2f}")
    if show_dropped:
        ax_a.plot([], [], ls="none", marker="o", ms=5.0, mfc="none", mec="#9a9a9a",
                  mew=1.2, label=f"{drought[0]}–{drought[1]} (excluded)")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("water year"); ax_a.set_ylabel("end-of-summer patch area (km²)")
    ax_a.set_title("a  Patch area — dashed = all years, solid = drought removed\n"
                   "Theil–Sen %/yr:   all yrs   no drought   SWE-adj\n"
                   + "\n".join(lines) + "\n** p<0.05,  + p<0.10",
                   loc="left", fontsize=11, fontfamily="monospace")
    ax_a.grid(alpha=0.25); ax_a.legend(loc="lower left", fontsize=12, framealpha=0.9, ncols=2)

    # ---------------- (b) low flow against area ----------------
    stats = []
    for lab, c in (("observed", C_OBS), ("snowpatch", C_MOD)):
        D = pd.concat({"a": area[lab], "q": flow[lab], "z": swe}, axis=1).dropna()
        D = D[(D.a > 0) & (D.q > 0)]
        if show_dropped:
            Dd = D[D.index.to_series().between(*drought)]
            ax_b.scatter(Dd.a, Dd.q, facecolor="none", edgecolor="#9a9a9a", lw=1.2,
                         s=95, marker=MK[lab], zorder=3)
        D = D[~D.index.to_series().between(*drought)]        # excluded from the fit
        m = sm.OLS(np.log(D.q), sm.add_constant(np.log(D.a))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3})
        el, ci = m.params["a"], m.conf_int().loc["a"]
        xx = np.linspace(np.log(D.a).min(), np.log(D.a).max(), 50)
        ax_b.plot(np.exp(xx), np.exp(m.params["const"] + el*xx), color=c, lw=2.8, zorder=4)
        sc = ax_b.scatter(D.a, D.q, c=D.z, cmap="RdBu", vmin=-2, vmax=2, s=105,
                          marker=MK[lab], edgecolor=c, lw=1.6, zorder=5, label=lab)
        stats.append(f"{lab:10s} {el:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  "
                     f"ρ={spearmanr(D.a, D.q)[0]:+.2f}  n={len(D)}")
    cb = fig.colorbar(sc, ax=ax_b, pad=0.02, shrink=0.85)
    cb.set_label("peak-SWE anomaly (z)", fontsize=12); cb.ax.tick_params(labelsize=11)
    ax_b.set_xscale("log"); 
    ax_b.set_yscale("log")
    
    ax_b.set_xlabel("end-of-summer patch area (km²)")
    ax_b.set_ylabel("Aug–Oct 7-day min flow (m³/s)")
    ax_b.set_title("b  Low flow vs patch area — marker = source, colour = snow year\n"
                   "elasticity [95% CI]:\n" + "\n".join(stats),
                   loc="left", fontsize=11, fontfamily="monospace")
    ax_b.grid(alpha=0.25, which="both")
    ax_b.legend(loc="upper left", fontsize=12, framealpha=0.9)

    fig.tight_layout()
    out = FIG/f"snowpatch_decline_lowflow_no{drought[0]}_{drought[1]}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)
    for l in lines + stats:
        print("  " + l)
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drought", type=int, nargs=2, default=list(DROUGHT))
    ap.add_argument("--hide-dropped", action="store_true")
    a = ap.parse_args()
    main(tuple(a.drought), not a.hide_dropped)
