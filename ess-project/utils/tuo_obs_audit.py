"""Is the Tuolumne observed low-flow memory real, or an artifact of the reconstruction?

The merged record is TWO records spliced: WY1970-2006 = 'HH_scaled' (Hetch Hetchy
reservoir inflow, area-scaled -- a mass-balance reconstruction), WY2007-2025 = 'gauge'.
Test the ACF on each era separately, and on September mean volume vs the 7-day min.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

SC = Path("/scratch/dlhogan/ess-project-data")
TUO = SC/"domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/preprocessed"
NLAG = 5


def compute_acf(series, n_lags=NLAG, n_permutations=5000, seed=0):
    def _sp(a, b):
        ra = rankdata(a)-(len(a)+1)/2.; rb = rankdata(b)-(len(b)+1)/2.
        den = np.sqrt((ra*ra).sum()*(rb*rb).sum())
        return (ra*rb).sum()/den if den > 0 else 0.0
    rng = np.random.default_rng(seed)
    s = series.dropna().astype(float)
    yrs = s.index.values.astype(int); v = s.values; n = v.size
    pos = {int(y): i for i, y in enumerate(yrs)}
    rho = np.full(n_lags, np.nan); p = np.ones(n_lags)
    for k in range(1, n_lags+1):
        ip = np.array([pos[int(y)-k] for y in yrs if int(y)-k in pos], int)
        ic = np.array([i for i, y in enumerate(yrs) if int(y)-k in pos], int)
        if ip.size <= 2: continue
        rho[k-1] = _sp(v[ip], v[ic])
        pm = np.array([abs(_sp(r[ip], r[ic])) for r in
                       (rng.permutation(v) for _ in range(n_permutations))])
        p[k-1] = np.mean(pm >= abs(rho[k-1]))
    return dict(rho=rho, p=p, band=1.96/np.sqrt(n), n=n)


d = pd.read_csv(TUO/"Tuolumne_River_merged_dailyQ_WY1971_2025.csv",
                parse_dates=["datetime"]).set_index("datetime")
q = d["discharge_cms"]
src = d["source"]
lump = pd.read_csv(TUO/"Tuolumne_River_lumped_streamflow_processed.csv",
                   parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]


def min7(s):
    x = (s.where(s.index.month.isin([8, 9, 10]))
          .rolling(7, center=True, min_periods=7).mean().resample("YS-OCT").min())
    x[x < 0.01] = np.nan; x.index = x.index.year; return x


def sepmean(s):
    x = s.where(s.index.month == 9).resample("YS-OCT").mean()
    x.index = x.index.year; return x


def augoctmean(s):
    x = s.where(s.index.month.isin([8, 9, 10])).resample("YS-OCT").mean()
    x.index = x.index.year; return x


def show(tag, s):
    a = compute_acf(s.dropna())
    stars = "".join(f"{r:+.2f}{'*' if pp < 0.05 else ' '}" for r, pp in zip(a["rho"], a["p"]))
    print(f"{tag:<40s} n={a['n']:3d}  band±{a['band']:.2f}   {stars}")


print("="*104)
print("A. METRIC CHOICE  (full merged record, WY1970-2024)")
print("="*104)
print(f"{'series':<40s} {'':>4s}  {'':>8s}   " + "".join(f"  lag{k}  " for k in range(1, 6)))
for tag, f in [("7-day min (Aug-Oct)", min7), ("September MEAN", sepmean),
               ("Aug-Oct MEAN", augoctmean)]:
    show(tag, f(q))

print("\n" + "="*104)
print("B. SPLICE  -- reconstruction era vs real gauge era")
print("="*104)
rec = q[src == "HH_scaled"]
gau = q[src == "gauge"]
for tag, s in [("HH_scaled only (WY1970-2006)", rec), ("gauge only (WY2007-2025)", gau)]:
    for mtag, f in [("7-day min", min7), ("Sept mean", sepmean)]:
        show(f"{tag} | {mtag}", f(s))

print("\n" + "="*104)
print("C. THE OTHER FILE  (lumped_processed = downstream regulated gauge)")
print("="*104)
for mtag, f in [("7-day min", min7), ("Sept mean", sepmean)]:
    show(f"lumped_processed | {mtag}", f(lump))

print("\n" + "="*104)
print("D. IS THE RECONSTRUCTION SMOOTH AT LOW FLOW?")
print("="*104)
for tag, s in [("HH_scaled (WY1984-2006)", rec[rec.index.year >= 1984]),
               ("gauge     (WY2007-2025)", gau)]:
    lo = s[s.index.month.isin([8, 9, 10])]
    dd = lo.diff().abs()
    print(f"{tag}  n={len(lo):5d}  mean={lo.mean():6.2f}  "
          f"|dQ/dt|/Q = {(dd/lo).mean():.4f}   "
          f"frac exactly-repeated day-to-day = {(lo.diff() == 0).mean():.3f}   "
          f"n unique values = {lo.round(3).nunique():5d} ({lo.round(3).nunique()/len(lo):.2f}/day)")

print("\n" + "="*104)
print("E. YEAR-TO-YEAR SEPTEMBER MEANS -- does the reconstruction drift in level?")
print("="*104)
sm = sepmean(q)
for era, lab in [(range(1970, 2007), "HH_scaled"), (range(2007, 2025), "gauge")]:
    v = sm[sm.index.isin(era)].dropna()
    print(f"  {lab:<10s} n={len(v):2d}  mean={v.mean():6.2f}  cv={v.std()/v.mean():.3f}  "
          f"min={v.min():.2f}  max={v.max():.2f}")
print("\n  September mean by WY (merged):")
print("  " + "  ".join(f"{y}:{v:.1f}" for y, v in sm.dropna().items()))
