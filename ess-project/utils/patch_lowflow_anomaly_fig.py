"""Observed Tuolumne patch-area and low-flow anomalies through time, on one axis.

(a) annual anomalies, (b) the same anomalies as a centred 5-year running mean, which is the
panel that speaks to a shared DECLINE rather than to shared wet and dry years.

ANOMALY = z-score of log10(value) over the plotted window. Both series are right-skewed, so a
linear anomaly would be set by the spike years; the log z-score puts two different units on
one axis without a second scale.

The statistics separate the two claims. Raw and first-difference rho measure year-to-year
co-variation; rho after removing each series' linear trend asks whether it survives without
the trends; the trend pair itself is the decline. Running means are autocorrelated, so no
p-value is attached to panel (b).
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro")
sys.path.insert(0, str(REPO/"ess-project/utils"))
import trend_acf_summary as T
import lowflow_acf as L

FIG = REPO/"ess-project/figures/draft"
DATA = T.SCRATCH/"geospatial-data/snowpatch_fsca"
C = dict(patch="#D55E00", flow="#332288")
WIN = 5


def zlog(s):
    v = np.log10(s)
    return (v-v.mean())/v.std(ddof=1)


def detrend(s):
    return s-np.polyval(np.polyfit(s.index.values.astype(float), s.values, 1), s.index.values.astype(float))


def stats(d):
    """Spearman raw, detrended and first-differenced, and each series' Theil-Sen trend."""
    a, f = zlog(d.patch), zlog(d.flow)
    da, df = a.diff().dropna(), f.diff().dropna()
    tp, tf = L.trend_stats(d.patch), L.trend_stats(d.flow)
    return dict(n=len(d), rho=spearmanr(a, f)[0], p=spearmanr(a, f)[1],
                rho_detr=spearmanr(detrend(a), detrend(f))[0], p_detr=spearmanr(detrend(a), detrend(f))[1],
                rho_diff=spearmanr(da, df)[0], p_diff=spearmanr(da, df)[1],
                patch_pct=tp["pct"], patch_lo=tp["lo"], patch_hi=tp["hi"],
                flow_pct=tf["pct"], flow_lo=tf["lo"], flow_hi=tf["hi"])


def main():
    eos = pd.read_csv(DATA/"eos_final.csv").set_index("year")["A_shadow"]
    ns = T.notebook_ns(fast=True)
    flow = ns["yr_low"](ns["tuolumne_obs_augoct_low_flow"])
    d = pd.concat([eos.rename("patch"), flow.rename("flow")], axis=1).dropna()
    d.index = d.index.astype(int)

    W = {"full": (d.index.min(), d.index.max()), "WY1993+": (1993, d.index.max()),
         "WY<=2006": (d.index.min(), 2006), "WY>=2007": (2007, d.index.max())}
    S = pd.DataFrame({k: stats(d.loc[a:b]) for k, (a, b) in W.items()}).T
    for yrs, lab in (([2011, 2017, 2023], "no 2011/2017/2023"),):
        S.loc[lab] = stats(d.drop(index=yrs))
    pd.set_option("display.width", 220); print(S.round(2).to_string())

    z = pd.DataFrame({"patch": zlog(d.patch), "flow": zlog(d.flow)})
    r = z.rolling(WIN, center=True).mean().dropna()
    z.join(r, rsuffix=f"_{WIN}yr").join(d, rsuffix="_raw").to_csv(FIG/"tuo_patch_lowflow_anomaly.csv")
    S.to_csv(FIG/"tuo_patch_lowflow_anomaly_stats.csv")
    s = S.loc["full"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 2])
    lab = dict(patch="end-of-summer patch area (Landsat)", flow="Aug–Oct 7-day min flow (HH unimpaired)")
    for k, mk in (("patch", "o"), ("flow", "s")):
        ax[0].plot(z.index, z[k], color=C[k], lw=2, marker=mk, ms=7, mfc="white", mew=1.6, label=lab[k])
        ax[1].plot(r.index, r[k], color=C[k], lw=2, marker=mk, ms=6, mfc="white", mew=1.4)
    ax[0].set_title(f"a  annual anomalies   ρ = {s.rho:+.2f} (p {'< 0.001' if s.p < 0.001 else f'= {s.p:.3f}'}), "
                    f"detrended {s.rho_detr:+.2f} (p = {s.p_detr:.3f}), "
                    f"year-to-year change {s.rho_diff:+.2f} (p = {s.p_diff:.3f}), n = {int(s.n)}",
                    loc="left", fontsize=10.5)
    ax[1].set_title(f"b  centred {WIN}-year running mean   trends: patch {s.patch_pct:+.1f} %/yr "
                    f"[{s.patch_lo:+.1f}, {s.patch_hi:+.1f}], flow {s.flow_pct:+.1f} %/yr "
                    f"[{s.flow_lo:+.1f}, {s.flow_hi:+.1f}]", loc="left", fontsize=10.5)
    for a in ax:
        a.axhline(0, color="0.45", lw=0.9, zorder=0)
        a.set_ylabel("anomaly (z of log₁₀)", fontsize=11)
        a.grid(alpha=0.2, lw=0.6); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(frameon=False, fontsize=10, ncol=2, loc="lower left")
    ax[1].set_xlabel("water year", fontsize=11)
    fig.tight_layout()
    out = FIG/"tuo_patch_lowflow_anomaly.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)

    # independent check: the real gauge and the unregulated headwater gauges
    src = open(REPO/"ess-project/utils/tuo_augnov_lag1_acf.py").read(); g = {}
    exec(compile(src[:src.index("L = pd.DataFrame({k: low7")], "base", "exec"), g)
    for name in g["names"]:
        q = g["low7"](g["G"][name]); x = pd.concat([eos.rename("patch"), q.rename("flow")], axis=1).dropna()
        x = x[x.flow > 0]
        if len(x) < 8: continue
        a, f = zlog(x.patch), zlog(x.flow); t = L.trend_stats(x.flow)
        print(f"{name:15s} WY{x.index.min()}-{x.index.max()} n={len(x):2d} rho={spearmanr(a,f)[0]:+.2f} "
              f"detr={spearmanr(detrend(a),detrend(f))[0]:+.2f} diff={spearmanr(a.diff().dropna(),f.diff().dropna())[0]:+.2f} "
              f"flow trend {t['pct']:+.1f} [{t['lo']:+.1f},{t['hi']:+.1f}] %/yr")


if __name__ == "__main__":
    main()
