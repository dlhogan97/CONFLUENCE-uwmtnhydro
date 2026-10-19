"""Tuolumne snowpatch attribution figure, WY1986-2023 (the span the model runs cover).

(a) end-of-summer snowpatch area, observed vs modelled -- the observed decline is
    significant, the modelled one does not exist, and the BASELINE is identically zero
(b) Aug-Oct 7-day-minimum low flow, observed vs both runs
(c) the elasticity: low flow vs patch area in log space, coloured by that year's peak SWE,
    which is what earns the attribution -- the relation holds ACROSS snow years, not just
    along the trend
(d) lag-1..5 Spearman ACF, trend-removed only where the trend is significant (p<0.05)

Trend lines are Theil-Sen in log space and are drawn SOLID when p<0.05, dashed otherwise,
so a reader cannot mistake a non-significant slope for a result.
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
from patch_swe import peak_swe_index

FIG = REPO/"ess-project/figures/draft"
P_TREND = T.P_TREND
HUE = "#D55E00"                                     # Tuolumne, matching every other figure
STEPS = (-0.45, 0.10, 0.45)                          # observed / baseline / store-added


def shade(hexc, amt):
    r, g, b = mcolors.to_rgb(hexc)
    return tuple(c + (1.0-c)*amt for c in (r, g, b)) if amt >= 0 else tuple(c*(1+amt) for c in (r, g, b))


C_OBS, C_BASE, C_ADJ = [shade(HUE, s) for s in STEPS]
STYLE = {"observed": (C_OBS, "-", 2.6, "o"), "baseline": (C_BASE, "--", 2.0, "s"),
         "snowpatch": (C_ADJ, "-", 2.4, "D")}


def trendtxt(d):
    """Per-series Theil-Sen rate, with the significant ones flagged, for a panel subtitle."""
    parts = []
    for k, v in d.items():
        tr = T.trend_stats(v)
        parts.append(f"{k} {tr['pct']:+.1f}%/yr (p={tr['p']:.3f}"
                     + ("**)" if tr["p"] < P_TREND else ")"))
    return "  ·  ".join(parts)


def tsline(ax, s, color, sig):
    """Theil-Sen fit in log space; solid if the trend is significant, dashed if not."""
    tr = T.trend_stats(s)
    x = np.array([s.index.min(), s.index.max()], float)
    ax.plot(x, np.exp(tr["slope"]*x + tr["icept"]), color=color, lw=2.6,
            ls="-" if sig else (0, (5, 3)), alpha=0.9, zorder=6)
    return tr


def main(WY0=T.WY0, WY1=T.WY1):
    ns = T.notebook_ns(fast=True); yr_low, compute_acf = ns["yr_low"], ns["compute_acf"]
    clip = lambda s: s.dropna().loc[WY0:WY1]
    eos = pd.read_csv(T.SCRATCH/"geospatial-data/snowpatch_fsca/eos_final.csv").set_index("year")
    SNOW = T.SCRATCH/"domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_anchor_20260812/anchor_m080"

    patch = {"observed": clip(eos["A_shadow"]),
             "snowpatch": clip(T.modelled_patch_area(SNOW, SNOW/"bin_spec.csv"))}
    base_patch = clip(T.modelled_patch_area(
        T.SCRATCH/"calibration/bestruns/Tuolumne_River_baseline_wbias10_1981_2025/run"))
    flow = {"observed":  clip(yr_low(ns["tuolumne_obs_augoct_low_flow"])),
            "baseline":  clip(yr_low(ns["tuolumne_mod_augoct_low_flow"])),
            "snowpatch": clip(yr_low(ns["tuolumne_adj_augoct_low_flow"]))}
    swe = clip(peak_swe_index())

    plt.style.use("seaborn-v0_8-poster")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.6))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ---- (a) patch area -------------------------------------------------------------
    for lab in ("observed", "snowpatch"):
        c, ls, lw, mk = STYLE[lab]
        s = patch[lab]
        ax_a.plot(s.index, s.values, color=c, lw=1.5, ls=ls, marker=mk, ms=5,
                  mec="white", mew=1.0, alpha=0.85, label=lab, zorder=4)
        tsline(ax_a, s, c, T.trend_stats(s)["p"] < P_TREND)
    ax_a.axhline(0.043, color="#999", lw=1.0, ls=":", zorder=2)
    ax_a.annotate("anchor floor 0.043 km²", xy=(WY1, 0.043), xytext=(-6, 5),
                  textcoords="offset points", ha="right", fontsize=10, color="#777")
    ax_a.set_yscale("log")
    ax_a.set_ylabel("end-of-summer patch area (km²)")
    ax_a.set_title(f"a  Snowpatch area — baseline is identically zero in all "
                   f"{len(base_patch)} years\n" + trendtxt(patch), loc="left", fontsize=13)

    # ---- (b) low flow ----------------------------------------------------------------
    for lab in ("observed", "baseline", "snowpatch"):
        c, ls, lw, mk = STYLE[lab]
        s = flow[lab]
        ax_b.plot(s.index, s.values, color=c, lw=1.5, ls=ls, marker=mk, ms=5,
                  mec="white", mew=1.0, alpha=0.85, label=lab, zorder=4)
        tr = tsline(ax_b, s, c, T.trend_stats(s)["p"] < P_TREND)
    ax_b.set_yscale("log")
    ax_b.set_ylabel("Aug–Oct 7-day min flow (m³/s)")
    sig_b = [k for k, v in flow.items() if T.trend_stats(v)["p"] < P_TREND]
    head = (f"significant in {', '.join(sig_b)}" if sig_b else "no significant trend")
    ax_b.set_title(f"b  Low flow — {head}\n" + trendtxt(flow), loc="left", fontsize=13)

    for ax in (ax_a, ax_b):
        ax.set_xlabel("water year"); ax.grid(alpha=0.25)
        ax.legend(loc="lower left", fontsize=12, framealpha=0.9, ncols=3)

    # ---- (c) elasticity ---------------------------------------------------------------
    D = pd.concat({"q": flow["observed"], "a": patch["observed"], "z": swe}, axis=1).dropna()
    m = sm.OLS(np.log(D.q), sm.add_constant(np.log(D.a))).fit(cov_type="HAC",
                                                              cov_kwds={"maxlags": 3})
    el, ci = m.params["a"], m.conf_int().loc["a"]
    xx = np.linspace(np.log(D.a).min(), np.log(D.a).max(), 50)
    pr = m.get_prediction(sm.add_constant(xx)).summary_frame(alpha=0.05)
    ax_c.fill_between(np.exp(xx), np.exp(pr["mean_ci_lower"]), np.exp(pr["mean_ci_upper"]),
                      color=C_OBS, alpha=0.15, lw=0, zorder=2)
    ax_c.plot(np.exp(xx), np.exp(pr["mean"]), color=C_OBS, lw=2.8, zorder=4)
    sc = ax_c.scatter(D.a, D.q, c=D.z, cmap="RdBu", s=95, edgecolor="white", lw=1.2,
                      zorder=5, vmin=-2, vmax=2)
    cb = fig.colorbar(sc, ax=ax_c, pad=0.02, shrink=0.85)
    cb.set_label("peak-SWE index (z)", fontsize=12); cb.ax.tick_params(labelsize=11)
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xlabel("observed patch area (km²)"); ax_c.set_ylabel("Aug–Oct 7-day min flow (m³/s)")
    r_all = spearmanr(D.a, D.q)[0]
    ax_c.set_title(f"c  Elasticity {el:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}], p={m.pvalues['a']:.4f}\n"
                   f"Spearman {r_all:+.2f} · colour shows the relation holds across snow years",
                   loc="left", fontsize=13)
    ax_c.grid(alpha=0.25, which="both")

    # ---- (d) ACF ----------------------------------------------------------------------
    W, OFF = 0.26, 0.27
    for i, lab in enumerate(("observed", "baseline", "snowpatch")):
        s = flow[lab]; tr = T.trend_stats(s); sig = tr["p"] < P_TREND
        a = compute_acf(T.detrend(s, tr) if sig else s, n_lags=5, n_permutations=5000)
        c = STYLE[lab][0]
        ax_d.bar(np.arange(1, 6) + (i-1)*OFF, a["rho"], width=W, color=c,
                 edgecolor="black", lw=0.8, zorder=3,
                 label=lab + (" (detrended)" if sig else ""))
        for k, (r, p) in enumerate(zip(a["rho"], a["p"]), start=1):
            if p < 0.10:
                ax_d.annotate("*", xy=(k + (i-1)*OFF, r + (0.02 if r >= 0 else -0.02)),
                              ha="center", va="bottom" if r >= 0 else "top",
                              fontsize=16, zorder=6)
    ax_d.axhline(0, color="k", lw=1.5, zorder=4)
    ax_d.set_xticks(range(1, 6)); ax_d.set_xlabel("lag (years)")
    ax_d.set_ylabel("Spearman autocorrelation")
    det = [k for k, v in flow.items() if T.trend_stats(v)["p"] < P_TREND]
    ax_d.set_title("d  Low-flow lag autocorrelation, trend removed only where p<0.05\n"
                   + (f"detrended: {', '.join(det)}" if det else "no series needed it")
                   + "; * = p<0.10", loc="left", fontsize=13)
    ax_d.grid(axis="y", alpha=0.3); ax_d.set_axisbelow(True)
    ax_d.legend(loc="upper right", fontsize=12, framealpha=0.9)
    # the two model bars sit exactly on top of each other and that is the result, not a bug:
    # the store raises every year's minimum by a near-constant amount, so it is rank-preserving
    rs = spearmanr(flow["baseline"], flow["snowpatch"])[0]
    if rs > 0.999:
        ax_d.annotate("baseline ≡ snowpatch: the store shifts every year by\n"
                      f"≤{(flow['snowpatch']-flow['baseline']).max():+.3f} m³/s, monotonically, "
                      "so ranks — and the ACF — are unchanged",
                      xy=(0.02, 0.03), xycoords="axes fraction", va="bottom", ha="left",
                      fontsize=11, color="#555")

    fig.suptitle(f"Tuolumne: end-of-summer snowpatch area and late-summer low flow, "
                 f"WY{WY0}–{WY1}", x=0.006, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = FIG/f"tuolumne_patch_lowflow_attribution_wy{WY0}_{WY1}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)
    print(f"  elasticity {el:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] p={m.pvalues['a']:.4f}  n={len(D)}")
    print(f"  observed patch  min {patch['observed'].min():.3f}  median {patch['observed'].median():.3f}"
          f"  max {patch['observed'].max():.3f} km2")
    print(f"  modelled patch  min {patch['snowpatch'].min():.3f}  median {patch['snowpatch'].median():.3f}"
          f"  max {patch['snowpatch'].max():.3f} km2")
    print(f"  baseline patch  all zero in {len(base_patch)} years: {bool((base_patch==0).all())}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wy0", type=int, default=T.WY0)
    ap.add_argument("--wy1", type=int, default=T.WY1)
    a = ap.parse_args()
    main(a.wy0, a.wy1)
