"""Trend-aware low-flow autocorrelation, with a null that knows about the detrending.

Why this exists. A permutation p-value shuffles the VALUES of a series, which destroys any
trend along with everything else. That is the wrong null for a series we have just
detrended: removing a trend forces the residuals to be uncorrelated with time, which pushes
autocorrelation negative at longer lags all by itself. Testing the detrended series against
a shuffled-values null therefore reports significance that the procedure manufactured.

So the null here is generated through the SAME pipeline as the data:

    surrogate  ->  test its trend  ->  remove it if p < P_TREND  ->  Spearman ACF

Two surrogates, because two different questions get asked at different lags:

    N0  white noise + the observed trend      -- "is there ANY year-to-year structure?"
                                                 the right null for lag 1
    N1  AR(1) at the observed rho_1 + trend   -- "is there structure BEYOND one-year
                                                 persistence?" the right null for lags >= 2

Reporting the N1 p-value at lag 5 matters: on the Tuolumne the permutation p is 0.051 and
the N1 p is 0.14, i.e. the apparent long-lag signal is what a short, mildly persistent,
detrended record produces on its own. On the East the same test gives p = 0.005, so that
one is real.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, theilslopes, kendalltau

P_TREND = 0.05          # remove a trend only when it is significant at this level
N_SIM = 2000            # surrogates per null
N_LAGS = 5


# ---------------- trend ----------------
def trend_stats(s, alpha=0.95):
    """Monotonic-trend summary. Theil-Sen slope is fitted in LOG space when the series is
    strictly positive, so the trend is multiplicative and cannot drive a flow negative.

    alpha is the CONFIDENCE LEVEL of the returned slope interval (scipy convention), so 0.95
    gives a 95% CI and 0.90 a 90% one. Default unchanged at 0.95 -- existing callers keep
    exactly the numbers they had."""
    s = s.dropna()
    x = s.index.values.astype(float); v = s.values.astype(float)
    rho, p = spearmanr(x, v)
    tau, p_mk = kendalltau(x, v)
    pos = v.min() > 0
    sl, ic, lo, hi = theilslopes(np.log(v) if pos else v, x, alpha=alpha)
    return dict(n=len(s), rho=rho, p=p, tau=tau, p_mk=p_mk, slope=sl, icept=ic, pos=pos,
                lo_s=lo, hi_s=hi, alpha=alpha,   # raw Theil-Sen slope CI (log space when pos)
                pct=100*(np.exp(sl)-1) if pos else np.nan,
                lo=100*(np.exp(lo)-1) if pos else np.nan,
                hi=100*(np.exp(hi)-1) if pos else np.nan,
                significant=bool(p < P_TREND and pos))


def detrend(s, tr=None):
    """Divide out the fitted multiplicative trend, rescaled to preserve the series mean."""
    tr = tr or trend_stats(s)
    x = s.index.values.astype(float)
    out = s / np.exp(tr["slope"]*x + tr["icept"])
    return out * (s.mean()/out.mean())


# ---------------- autocorrelation ----------------
def _rho_by_year(s, n_lags=N_LAGS):
    """Spearman rho_k pairing on the TRUE year lag, so a missing year drops the pair
    rather than shifting every subsequent one."""
    s = s.dropna(); pos = {int(y): i for i, y in enumerate(s.index)}
    v = s.values.astype(float)
    out = np.full(n_lags, np.nan)
    for k in range(1, n_lags+1):
        ip = [pos[int(y)-k] for y in s.index if int(y)-k in pos]
        ic = [pos[int(y)] for y in s.index if int(y)-k in pos]
        if len(ip) > 2:
            out[k-1] = spearmanr(v[ip], v[ic])[0]
    return out


def _pipeline(v, years, n_lags=N_LAGS):
    """The exact rule applied to the data, so surrogates and data are treated identically."""
    s = pd.Series(v, index=years)
    tr = trend_stats(s)
    return _rho_by_year(detrend(s, tr) if tr["significant"] else s, n_lags)


def _ar1(n, rho, slope_log, rng):
    e = rng.standard_normal(n)
    x = np.empty(n); x[0] = e[0]
    for i in range(1, n):
        x[i] = rho*x[i-1] + np.sqrt(max(1-rho**2, 0))*e[i]
    return np.exp(0.5*x + slope_log*np.arange(n))


def acf_trend_aware(s, n_lags=N_LAGS, n_sim=N_SIM, seed=0):
    """Conditionally-detrended ACF with p-values from surrogates run through the same rule.

    Returns lags, rho, p (N0 at lag 1, N1 at lags >= 2), both raw p-value sets, the trend
    summary, and whether the detrending actually fired.
    """
    s = s.dropna()
    tr = trend_stats(s)
    adj = detrend(s, tr) if tr["significant"] else s
    rho = _rho_by_year(adj, n_lags)

    rng = np.random.default_rng(seed)
    years = s.index.values
    n = len(s)
    slope = tr["slope"] if tr["pos"] else 0.0
    r1 = float(np.clip(rho[0], -0.95, 0.95))
    null = {}
    for tag, r in (("N0", 0.0), ("N1", r1)):
        sims = np.array([_pipeline(_ar1(n, r, slope, rng), years, n_lags)
                         for _ in range(n_sim)])
        mu = np.nanmean(sims, axis=0)
        null[tag] = np.array([np.nanmean(np.abs(sims[:, k]-mu[k]) >= abs(rho[k]-mu[k]))
                              for k in range(n_lags)])
    p = np.where(np.arange(n_lags) == 0, null["N0"], null["N1"])
    return dict(lags=np.arange(1, n_lags+1), rho=rho, p=p, n=n,
                p_N0=null["N0"], p_N1=null["N1"], trend=tr, detrended=tr["significant"])


def summarize(series, n_sim=N_SIM, n_lags=N_LAGS, seed=0):
    """series: {(basin, label): pd.Series}. Returns (trend_table, acf_table, results)."""
    res, trows, arows = {}, [], []
    for key, s in series.items():
        basin, lab = key
        a = acf_trend_aware(s, n_lags=n_lags, n_sim=n_sim, seed=seed)
        res[key] = a
        t = a["trend"]
        trows.append(dict(basin=basin, series=lab, n=t["n"], rho=t["rho"], p=t["p"],
                          tau=t["tau"], pct_per_yr=t["pct"], ci_lo=t["lo"], ci_hi=t["hi"],
                          detrended=a["detrended"]))
        arows.append(dict(basin=basin, series=lab, n=a["n"], detrended=a["detrended"],
                          **{f"rho{k}": a["rho"][k-1] for k in range(1, n_lags+1)},
                          **{f"p{k}": a["p"][k-1] for k in range(1, n_lags+1)}))
    return pd.DataFrame(trows), pd.DataFrame(arows), res
