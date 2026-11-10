"""
Uncertainty of the OBSERVED East River low-flow memory statistic (2026-10-27):
  rho1 = Spearman lag-1 of the residual of ln(Aug1-Nov30 7-d min) regressed on ln(Apr-Jul 7-d peak)   (= emu_core.memory res1)
Three resampling views, 10 000 replicates each:
  pair bootstrap      resample the (r_t, r_t+1) pairs with replacement (residuals fixed); ignores dependence beyond adjacent years
  block bootstrap     circular moving blocks of whole years (low flow + peak together), lengths 2-4; the peak regression is REFIT in every
                      replicate, so its uncertainty propagates. Block joins add artificial adjacent pairs, which biases rho1 toward 0.
  permutation (null)  shuffle the residual series (regression residuals do not depend on year order) -> distribution of rho1 with no memory
usage: python obs_lag1_uncertainty.py
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import emu_core as E

NB = 10000; rng = np.random.default_rng(42)
WINDOWS = {"WY1971-2024": (1971, 2024), "WY1987-2024": (1987, 2024), "WY2009-2024": (2009, 2024)}


def data(y0, y1):
    lf, pk = E.series(E.obs, y0, y1); d = pd.concat([lf.rename("lf"), pk.rename("pk")], axis=1).dropna()
    return d.loc[[y for y in d.index if y0 <= y <= y1]]


def resid(lf, pk): b = np.polyfit(pk, lf, 1); return lf - np.polyval(b, pk)
def rho1(r): return spearmanr(r[:-1], r[1:])[0]


def consecutive(d):
    yrs = d.index.values; return np.all(np.diff(yrs) == 1)


rows = []
for w, (y0, y1) in WINDOWS.items():
    d = data(y0, y1); n = len(d); assert consecutive(d), f"{w}: gap in years"
    r = resid(d.lf.values, d.pk.values); est = rho1(r)
    # pair bootstrap
    x, y = r[:-1], r[1:]; m = len(x); pb = np.array([spearmanr(x[i], y[i])[0] for i in rng.integers(0, m, (NB, m))])
    out = dict(window=w, n_years=n, rho1=est, pair_lo=np.nanpercentile(pb, 2.5), pair_hi=np.nanpercentile(pb, 97.5), pair_P_gt0=np.mean(pb > 0))
    # circular block bootstrap of years, regression refit per replicate
    for L in (2, 3, 4):
        bs = []
        for _ in range(NB):
            idx = np.concatenate([(s + np.arange(L)) % n for s in rng.integers(0, n, int(np.ceil(n/L)))])[:n]
            bs.append(rho1(resid(d.lf.values[idx], d.pk.values[idx])))
        bs = np.array(bs); out[f"block{L}_lo"], out[f"block{L}_hi"], out[f"block{L}_median"] = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), np.nanmedian(bs)
    # permutation null
    pn = np.array([rho1(rng.permutation(r)) for _ in range(NB)])
    out.update(perm_null_95=np.percentile(pn, 95), p_one_sided=(np.sum(pn >= est) + 1)/(NB + 1), p_two_sided=(np.sum(np.abs(pn) >= abs(est)) + 1)/(NB + 1))
    rows.append(out)
T = pd.DataFrame(rows); T.to_csv("/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/stageB/obs_lag1_uncertainty.csv", index=False)
pd.set_option("display.width", 250); print(T.round(3).T.to_string())
