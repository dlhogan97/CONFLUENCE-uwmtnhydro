"""Trends before and after removing the effect of peak SWE.

The claim to support is a three-part one:
  * late-season low flow and end-of-summer patch area both DECLINE
  * peak SWE has NO trend
  * yet big-SWE years still have bigger patches and higher low flows

Those are only in tension if you think the decline must come from the snowpack. The test
that separates them is a trend fitted to the part of each series that peak SWE does NOT
explain:

    log Y_t = a + b*SWE_t + c*t + e_t

`b` is the sensitivity to that year's snowpack, `c` the trend once that sensitivity is
accounted for. Because SWE itself has no trend, conditioning on it cannot CREATE a trend in
Y -- it can only remove variance, which tightens the confidence interval on `c`. So a
SWE-adjusted decline that survives (and sharpens) is direct evidence the loss is not a
snowfall-supply signal.

Two estimators, reported side by side because they fail differently: OLS with HAC errors
(efficient, assumes a linear-in-logs form) and Theil-Sen on the residuals of log Y ~ SWE
(robust to outliers, no distributional assumption).
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

REPO = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro")
sys.path.insert(0, str(REPO/"ess-project/utils"))
import trend_acf_summary as T
import lowflow_acf as L
from patch_swe import peak_swe_index

EOS_CSV = T.SCRATCH/"geospatial-data/snowpatch_fsca/eos_final.csv"
DROUGHT = (1987, 1992)

DROUGHT_YEARS = set(range(DROUGHT[0], DROUGHT[1] + 1))


def thin_years(max_scenes):
    """Water years with <= max_scenes usable Landsat scenes in the 1 Aug - 10 Oct window.

    Read from eos_final.csv rather than hardcoded, so the set cannot drift from the data --
    a hand-written list of the <=3 years is easy to get wrong (1996 also has 3, and is the
    one most often missed).

    An annual MINIMUM taken over very few scenes is biased HIGH: the true late-summer
    minimum is unlikely to have been sampled. Because the thin years cluster early in the
    record, that bias flattens a declining trend rather than creating one."""
    e = pd.read_csv(EOS_CSV).set_index("year")
    return set(e.index[e["n_scenes"] <= max_scenes].astype(int))

# Windows to report. Each is (tag, first WY, last WY, years to exclude). The window IS the
# treatment -- nothing is excised unless this table says so.
def windows():
    thin2, thin3 = thin_years(2), thin_years(3)
    return [
        ("1986-2024",                  1986, 2024, set()),
        ("1986-2024 less n_scenes<=2", 1986, 2024, thin2),
        ("1986-2024 less n_scenes<=3", 1986, 2024, thin3),
        ("1993-2024",                  1993, 2024, set()),
        ("1993-2024 less n_scenes<=3", 1993, 2024, thin3),
        ("1985-2025 (full)",           1985, 2025, set()),
        ("1985-2025 less n_scenes<=2", 1985, 2025, thin2),
        ("1985-2025 less n_scenes<=3", 1985, 2025, thin3),
        ("1985-2025 less 87-92",       1985, 2025, DROUGHT_YEARS),
    ]


def build():
    ns = T.notebook_ns(fast=True); yr_low = ns["yr_low"]
    eos = pd.read_csv(EOS_CSV).set_index("year")
    SNOW = (T.SCRATCH/"domain_Tuolumne_River_distributed_elevAspect/simulations"
                      "/snowpatch_anchor_20260812/anchor_m080")
    return {
        ("observed", "patch area"):  eos["A_shadow"].dropna(),
        ("observed", "low flow"):    yr_low(ns["tuolumne_obs_augoct_low_flow"]).dropna(),
        ("modelled", "patch area"):  T.modelled_patch_area(SNOW, SNOW/"bin_spec.csv").dropna(),
        ("modelled", "low flow"):    yr_low(ns["tuolumne_adj_augoct_low_flow"]).dropna(),
    }, peak_swe_index().dropna()


def adjusted_trend(y, swe, years=None):
    """Raw and SWE-adjusted trend in %/yr, plus how much variance SWE explains.

    `years` is the ONLY year filter -- any drought excision must already be baked into it,
    so a window that says it keeps 1987-1992 really does."""
    D = pd.concat({"y": y, "s": swe}, axis=1).dropna()
    D = D[D.y > 0]
    if years is not None:
        D = D.loc[D.index.intersection(years)]
    D["t"] = D.index.values.astype(float) - D.index.values.mean()
    ly = np.log(D.y)

    raw = L.trend_stats(D.y)                                   # Theil-Sen, no covariate
    m_t = sm.OLS(ly, sm.add_constant(D[["t"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    m_st = sm.OLS(ly, sm.add_constant(D[["s", "t"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    resid = ly - (m_st.params["const"] + m_st.params["s"]*D.s)  # SWE removed, trend retained
    ts_adj = L.trend_stats(np.exp(resid))

    pc = lambda b: 100*(np.exp(b)-1)
    ci_t, ci_st = m_t.conf_int().loc["t"], m_st.conf_int().loc["t"]
    return dict(
        n=len(D),
        ols_raw=pc(m_t.params["t"]), ols_raw_p=m_t.pvalues["t"],
        ols_raw_lo=pc(ci_t[0]), ols_raw_hi=pc(ci_t[1]),
        ols_adj=pc(m_st.params["t"]), ols_adj_p=m_st.pvalues["t"],
        ols_adj_lo=pc(ci_st[0]), ols_adj_hi=pc(ci_st[1]),
        swe_beta=m_st.params["s"], swe_p=m_st.pvalues["s"],
        r2_swe=sm.OLS(ly, sm.add_constant(D[["s"]])).fit().rsquared,
        sen_raw=raw["pct"], sen_raw_p=raw["p"],
        sen_adj=ts_adj["pct"], sen_adj_p=ts_adj["p"],
        rho_swe=spearmanr(D.s, D.y)[0])


def common_years(series, swe, lo, hi, drop):
    """Years where every series AND the SWE index exist, inside [lo, hi].

    Forced to a single set per window so the four rows are comparable to each other: the
    low-flow record reaches back to 1970 and patch area only to 1985, so a per-series
    window would silently compare different periods."""
    c = set(swe.index)
    for v in series.values():
        c &= set(v.index)
    c = {y for y in c if lo <= y <= hi} - set(drop)
    return sorted(c)


def main():
    series, swe = build()
    rows = []
    for tag, lo, hi, drop in windows():
        yrs = common_years(series, swe, lo, hi, drop)
        sw = swe.loc[yrs]
        ts = L.trend_stats(pd.Series(np.exp(sw), index=sw.index))
        gone = sorted(set(drop) & set(range(lo, hi+1)))
        print(f"=== {tag}   realized WY{min(yrs)}-{max(yrs)}, n={len(yrs)}"
              + (f"   [excluded: {', '.join(str(y) for y in gone)}]" if gone else "") + " ===")
        print(f"    peak-SWE index vs year: Spearman {ts['rho']:+.2f} (p={ts['p']:.3f})"
              + ("  <- SIGNIFICANT, the adjustment can move the trend here"
                 if ts["p"] < 0.05 else "  -- no trend to transmit"))
        print(f"    {'series':22s} {'n':>3s} {'SWE beta':>9s} {'r2':>5s} {'raw OLS':>21s} "
              f"{'SWE-adjusted OLS':>21s} {'adj 95% CI':>18s} {'Sen adj':>21s}")
        for (src, var), y in series.items():
            r = adjusted_trend(y, swe, years=yrs)
            rows.append(dict(window=tag, wy0=min(yrs), wy1=max(yrs),
                             source=src, variable=var, **r))
            f = lambda a, b: f"{a:+6.2f}%/yr (p={b:.3f})"
            print(f"    {src + ' ' + var:22s} {r['n']:3d} {r['swe_beta']:+9.3f} {r['r2_swe']:5.2f} "
                  f"{f(r['ols_raw'], r['ols_raw_p']):>21s} "
                  f"{f(r['ols_adj'], r['ols_adj_p']):>21s} "
                  f"{'[%+.1f,%+.1f]' % (r['ols_adj_lo'], r['ols_adj_hi']):>18s} "
                  f"{f(r['sen_adj'], r['sen_adj_p']):>21s}")
        print()

    out = pd.DataFrame(rows)
    dst = REPO/"ess-project/figures/draft/table_swe_adjusted_trends.csv"
    out.to_csv(dst, index=False)
    print(f"wrote {dst}")
    return out


if __name__ == "__main__":
    main()
