"""Elasticity of late-season low flow to store size, drawn as a slope in log-log space.

Both axes are logged, so a fitted slope IS the elasticity: percent change in low flow per
percent change in the store. Three panels ask three questions with the same instrument:

  a  Tuolumne, SAME year    does a bigger patch mean more flow that summer?      yes
  b  Tuolumne, NEXT year    does it mean more flow the FOLLOWING summer?          no
  c  East, LAGGED           does storage carried into the year raise the next
                            autumn's minimum, and does adding the slow store
                            create that where the baseline had none?             yes

THE LINE DRAWN IS THE SWE-CONTROLLED (PARTIAL) SLOPE, not the OLS fit to the visible points.
It is the coefficient on log(store) in  log Q = a + e log S + b log SWE, anchored at each
series' centroid so only the ANGLE is being read. It is shallower than the naked point cloud
because peak SWE is an upstream cause of patch area (rho ~ +0.75), and part of the apparent
slope is the snow year rather than the store. Drawing the raw fit instead would show a steeper
line than the number reported in the text.

Both basins condition on an OBSERVED point pillow -- DAN for the Tuolumne, Butte SNOTEL for
the East -- so the control is the same kind of measurement in both, not modelled snow in one
and observed in the other.

Panels a and b share axes deliberately: the reader is comparing two angles, and a slope only
looks like its value if both panels span the same log decades per unit of plot.
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
HUE = {"Tuolumne": "#D55E00", "East": "#0072B2"}      # matching every other figure
TUO0, TUO1 = 1989, 2023
EAST0, EAST1 = 1982, 2024


def shade(hexc, amt):
    r, g, b = mcolors.to_rgb(hexc)
    return tuple(c + (1.0-c)*amt for c in (r, g, b)) if amt >= 0 else tuple(c*(1+amt) for c in (r, g, b))


def series(ax, x, y, s, col, mk, label):
    """Scatter one series and draw its SWE-controlled slope through the centroid."""
    d = pd.concat([np.log(x).rename("lx"), np.log(y).rename("ly"),
                   np.log(s).rename("ls")], axis=1).dropna()
    m = sm.OLS(d.ly, sm.add_constant(d[["lx", "ls"]])).fit(cov_type="HC3")
    e, p = m.params["lx"], m.pvalues["lx"]
    rho = spearmanr(d.lx, d.ly)[0]
    ax.scatter(np.exp(d.lx), np.exp(d.ly), s=88, color=col, marker=mk, edgecolor="white",
               lw=1.2, zorder=5, alpha=0.92)
    g = np.linspace(d.lx.min(), d.lx.max(), 50)
    star = "**" if p < 0.05 else "*" if p < 0.10 else ""
    ax.plot(np.exp(g), np.exp(d.ly.mean() + e*(g - d.lx.mean())), color=col, lw=2.6, zorder=4,
            label=f"{label}   ε = {e:+.2f}{star}")
    return dict(e=e, p=p, rho=rho, n=len(d), lims=(d.lx.min(), d.lx.max(), d.ly.min(), d.ly.max()))


def main():
    eos = pd.read_csv(DATA/"eos_final.csv").set_index("year")
    swe = pd.read_csv(DATA/"dan_peak_swe.csv", comment="#", index_col="water_year")["peak_swe_mm"]
    # East snow control: Butte SNOTEL (NRCS 380, 3106 m) Feb-May peak -- an OBSERVED point
    # pillow, matching the Tuolumne's use of DAN. Substituting the modelled basin mean moves
    # the slow-store elasticity +0.40 -> +0.43 (p 0.110 -> 0.068); the point estimate is
    # insensitive, the pillow is just a noisier proxy for basin-mean snow so the CI widens.
    pk = pd.read_csv(DATA/"butte_peak_swe.csv", comment="#",
                     index_col="water_year")["peak_swe_mm"]
    aq = pd.read_csv(DATA/"east_aquifer_oct1.csv", index_col="oct1_year")
    ns = T.notebook_ns(fast=True); yr_low = ns["yr_low"]

    Ao, Am = eos["A_shadow"], T.modelled_patch_area(SNOW, SNOW/"bin_spec.csv")
    Qo = yr_low(ns["tuolumne_obs_augoct_low_flow"])
    Qm = yr_low(ns["tuolumne_adj_augoct_low_flow"])
    slowQ = pd.read_csv(T.SCRATCH/"streamflow-data/EastRiver"
                        "/EastRiver_SUMMA_slowstore_sf10b2p5_dailyDischarge_1981_2025.csv",
                        parse_dates=["time"]).set_index("time")["discharge"]
    east = pd.DataFrame({
        "Qb": yr_low(ns["east_mod_augoct_low_flow"]),
        "Qs": yr_low(slowQ.where(slowQ.index.month.isin([9, 10, 11]))
                     .rolling(7, center=True, min_periods=7).mean().resample("YE-DEC").min())})
    east["Sb"] = aq["baseline_basin"].reindex(east.index-1).values      # 1 Oct of the PRIOR yr
    east["Ss"] = aq["slowstore_basin"].reindex(east.index-1).values
    east = east.loc[EAST0:EAST1]

    plt.style.use("seaborn-v0_8-poster")
    fig, ax = plt.subplots(1, 3, figsize=(20.5, 6.4))
    cT = {"obs": shade(HUE["Tuolumne"], -0.45), "mod": shade(HUE["Tuolumne"], 0.30)}
    cE = {"base": shade(HUE["East"], 0.35), "slow": shade(HUE["East"], -0.40)}
    R = {}
    sl = slice(TUO0, TUO1)
    R["a_obs"] = series(ax[0], Ao.loc[sl], Qo.loc[sl], swe, cT["obs"], "o", "observed")
    R["a_mod"] = series(ax[0], Am.loc[sl], Qm.loc[sl], swe, cT["mod"], "^", "model")
    sl2 = slice(TUO0+1, TUO1)
    R["b_obs"] = series(ax[1], Ao.loc[sl2], Qo.loc[sl2].shift(1), swe, cT["obs"], "o", "observed")
    R["b_mod"] = series(ax[1], Am.loc[sl2], Qm.loc[sl2].shift(1), swe, cT["mod"], "^", "model")
    R["c_base"] = series(ax[2], east.Sb, east.Qb, pk, cE["base"], "o", "baseline")
    R["c_slow"] = series(ax[2], east.Ss, east.Qs, pk, cE["slow"], "^", "slow store")

    for a in ax:
        a.set_xscale("log"); a.set_yscale("log")
        
        a.grid(alpha=0.22, which="both", lw=0.6)
        a.legend(frameon=False, fontsize=11, loc="upper left")
    # panels a and b on identical axes -- the whole point is comparing two angles
    L = [R[k]["lims"] for k in ("a_obs", "a_mod", "b_obs", "b_mod")]
    xs = [v for l in L for v in l[:2]]; ys = [v for l in L for v in l[2:]]
    px, py = 0.10*(max(xs)-min(xs)), 0.10*(max(ys)-min(ys))
    for a in ax[:2]:
        a.set_xlim(np.exp(min(xs)-px), np.exp(max(xs)+px))
        a.set_ylim(np.exp(min(ys)-py), np.exp(max(ys)+py))
        a.set_xlabel("end-of-summer patch area (km²)")
    ax[0].set_ylabel("Aug–Oct 7-day minimum flow (m³/s)")
    ax[1].set_ylabel("")
    ax[2].set_xlabel("aquifer storage on 1 Oct, prior year (m)")
    ax[2].set_ylabel("Sep–Nov 7-day minimum flow (m³/s)")
    ax[0].set_title(f"a  Tuolumne — same year\nWY{TUO0}–{TUO1}, n = {R['a_obs']['n']}",
                    loc="left", fontsize=12)
    ax[1].set_title(f"b  Tuolumne — following year\nWY{TUO0+1}–{TUO1}, n = {R['b_obs']['n']}",
                    loc="left", fontsize=12)
    ax[2].set_title(f"c  East — storage carried into the year\nWY{EAST0}–{EAST1}, n = {R['c_slow']['n']}",
                    loc="left", fontsize=12)
    fig.suptitle("Elasticity of late-season low flow to store size  —  log–log slope = ε, "
                 "controlled for observed peak SWE    (** p<0.05, * p<0.10)", fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIG/"store_elasticity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("wrote", out)
    for k, v in R.items():
        print(f"  {k:8s} ε={v['e']:+.3f} (p={v['p']:.3f})  ρ={v['rho']:+.3f}  n={v['n']}")
    return fig


if __name__ == "__main__":
    main()
