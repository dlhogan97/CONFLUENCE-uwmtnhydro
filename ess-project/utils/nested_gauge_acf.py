"""Independent check: do the unregulated nested headwater gauges show low-flow memory?

These four gauges (Tuolumne Meadows area) are independent of both the Hetch Hetchy
reconstruction and the regulated downstream gauge, so they are the only clean test of
whether the Tuolumne actually carries interannual low-flow memory.

Short records (n ~ 18-21), so the noise band is wide -- treat as corroboration, not proof.
"""
from pathlib import Path
import numpy as np, pandas as pd, netCDF4 as nc4, glob
from scipy.stats import rankdata

SC = Path("/scratch/dlhogan/ess-project-data")
NEST = SC/"streamflow-data/TuolumneRiver/TuolumneStreamflow_Hydroclimate_2002_2021.csv"
TUO = SC/"domain_Tuolumne_River_distributed_elevAspect"
TUO_SW = TUO/"simulations/snowpatch_sweep_20260812"
NLAG = 4
MIN_SEP_DAYS = 25          # a WY counts only if September is essentially complete


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


def sep_mean(s, min_days=MIN_SEP_DAYS):
    x = s[s.index.month == 9]
    g = x.groupby(x.index.year)
    m, c = g.mean(), g.count()
    return m[c >= min_days]


def show(tag, a):
    print(f"{tag:<26s} n={a['n']:3d} ±{a['band']:.2f}   "
          + " ".join(f"{r:+.2f}{'*' if p < 0.05 else ' '}" for r, p in zip(a["rho"], a["p"])))


d = pd.read_csv(NEST, parse_dates=["Date"]).set_index("Date")
d.columns = [c.replace("Estimated_Discharge_", "") for c in d.columns]
GAUGES = list(d.columns)

print("="*84)
print("A. DATA QUALITY -- is any of this spurious?")
print("="*84)
for g in GAUGES:
    s = d[g].dropna()
    lo = s[s.index.month.isin([8, 9, 10])]
    print(f"  {g:8s} n={len(s):5d}  zeros={int((s == 0).sum()):4d}  "
          f"flat day-to-day={float((lo.diff() == 0).mean()):.3f}  "
          f"Sep median={lo[lo.index.month == 9].median():.3f}  max={s.max():.1f}")

print("\n" + "="*84)
print("B. DO THE NESTED GAUGES AGREE WITH EACH OTHER ON SEPTEMBER RANKING?")
print("   (they are nested, so disagreement = data problem, not hydrology)")
print("="*84)
sm = pd.DataFrame({g: sep_mean(d[g]) for g in GAUGES})
print("\n  September mean by WY (m3/s):")
print(sm.round(2).to_string())
print("\n  Spearman between gauges (September mean):")
print(sm.corr(method="spearman").round(2).to_string())

print("\n" + "="*84)
print("C. LAG ACF OF SEPTEMBER MEAN -- each independent gauge")
print("="*84)
print(f"{'':<26s} {'':>9s}   " + "  ".join(f"lag{k} " for k in range(1, NLAG+1)))
for g in GAUGES:
    show(g, compute_acf(sm[g].dropna()))

# composite: median rank across the gauges available each year
rk = sm.rank(axis=0)
comp = rk.mean(axis=1).dropna()
show("COMPOSITE (mean rank)", compute_acf(comp))

print("\n" + "="*84)
print("D. SAME YEARS, THE OTHER RECORDS  (WY2005-2023, gauge era only)")
print("="*84)
yrs = comp.index
t = pd.read_csv(TUO/"observations/streamflow/preprocessed/"
                "Tuolumne_River_merged_dailyQ_WY1971_2025.csv",
                parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
lump = pd.read_csv(TUO/"observations/streamflow/preprocessed/"
                   "Tuolumne_River_lumped_streamflow_processed.csv",
                   parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]


def qdaily(run):
    run = Path(run)
    at = nc4.Dataset(run/"attributes.nc"); a = float(np.array(at["HRUarea"][:]).sum()); at.close()
    f = sorted(glob.glob(str(run/"output"/"*.nc")) + glob.glob(str(run/"out"/"*.nc")))[0]
    ds = nc4.Dataset(f); tt = ds["time"]
    tv = nc4.num2date(tt[:], tt.units, only_use_cftime_datetimes=False)
    q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a; ds.close()
    return pd.Series(q, index=pd.DatetimeIndex(tv)).resample("1D").mean()


others = {"merged obs (HH+gauge)": sep_mean(t), "lumped obs (regulated)": sep_mean(lump),
          "SUMMA baseline": sep_mean(qdaily(SC/"calibration/bestruns/"
                                            "Tuolumne_River_baseline_wbias10_1981_2025/run")),
          "SUMMA m090_q089": sep_mean(qdaily(TUO_SW/"m090_q089"))}
print("\n  Spearman vs the nested composite, same years:")
for k, v in others.items():
    j = pd.DataFrame({"a": comp, "b": v.reindex(comp.index)}).dropna()
    r = compute_acf(j.a)  # placeholder to reuse spearman
    ra = rankdata(j.a); rb = rankdata(j.b)
    rr = np.corrcoef(ra, rb)[0, 1]
    print(f"    {k:<24s} n={len(j):2d}  rho={rr:+.2f}")
print("\n  Their own ACF restricted to these years:")
for k, v in others.items():
    show(k, compute_acf(v.reindex(comp.index).dropna()))
