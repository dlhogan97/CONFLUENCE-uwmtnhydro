"""Score the East aquiferBaseflowExp sweep: does the exponent break the memory-vs-bias trade?

The uniform-sf sweep slid along one line -- every gain in rho_1 cost autumn volume,
monotonically. The exponent makes the bucket's timescale flow-dependent, so the question
here is whether any member lands ABOVE that line: significant rho_1 at a PBIAS_low near
the baseline's +29%.
"""
from pathlib import Path
import glob
import numpy as np, pandas as pd, netCDF4 as nc4
from scipy.stats import rankdata

SC = Path("/scratch/dlhogan/ess-project-data")
EAST = SC/"domain_East_River_distributed_elevAspect"
SB = EAST/"simulations/store_memory_sandbox/East"
OUT = SB/"sweep_exp_20260812"
SPINUP, LOW = "1983-09-30", [9, 10, 11]
YR0, YR1 = 1970, 2024


def compute_acf(s, n_lags=5, nperm=5000, seed=0):
    def _sp(a, b):
        ra = rankdata(a)-(len(a)+1)/2.; rb = rankdata(b)-(len(b)+1)/2.
        d = np.sqrt((ra*ra).sum()*(rb*rb).sum())
        return (ra*rb).sum()/d if d > 0 else 0.0
    rng = np.random.default_rng(seed)
    s = s.dropna().astype(float); v = s.values; n = v.size
    pos = {int(y): i for i, y in enumerate(s.index)}
    rho = np.full(n_lags, np.nan); p = np.ones(n_lags)
    for k in range(1, n_lags+1):
        ip = np.array([pos[int(y)-k] for y in s.index if int(y)-k in pos], int)
        ic = np.array([i for i, y in enumerate(s.index) if int(y)-k in pos], int)
        if ip.size <= 2: continue
        rho[k-1] = _sp(v[ip], v[ic])
        pm = np.array([abs(_sp(r[ip], r[ic])) for r in
                       (rng.permutation(v) for _ in range(nperm))])
        p[k-1] = np.mean(pm >= abs(rho[k-1]))
    return dict(rho=rho, p=p, n=n, band=1.96/np.sqrt(n))


def low7(d):
    s = (d.where(d.index.month.isin(LOW))
          .rolling(7, center=True, min_periods=7).mean().resample("YE-NOV").min())
    s[s < 0.01] = np.nan; s.index = s.index.year
    return s[(s.index >= YR0) & (s.index <= YR1)]


def kge(o, m):
    r = np.corrcoef(o, m)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (m.std(ddof=0)/o.std(ddof=0)-1)**2 + (m.mean()/o.mean()-1)**2)


def qdaily(run):
    run = Path(run)
    with nc4.Dataset(run/"attributes.nc") as d:
        a = float(np.array(d["HRUarea"][:]).sum())
    f = sorted(glob.glob(str(run/"output"/"*.nc")) + glob.glob(str(run/"out"/"*.nc")))
    if not f: return None
    with nc4.Dataset(f[0]) as ds:
        t = ds["time"]
        tt = nc4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
        q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a
    return pd.Series(q, index=pd.DatetimeIndex(tt)).resample("1D").mean()


obs = pd.read_csv(EAST/"observations/streamflow/preprocessed/"
                  "East_River_USGS09112500_dailyQ_WY1970_2025.csv",
                  parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
obs = obs.groupby(obs.index.normalize()).mean()
a_ob = compute_acf(low7(obs))
owy = obs.resample("YS-OCT").sum(); owy.index = owy.index.year + 1
owy = owy[(owy.index >= 1984) & (owy.index <= 2025)]
dry = set(owy[owy <= owy.quantile(0.25)].index)


def score(q):
    j = pd.DataFrame({"o": obs, "m": q}).dropna()
    j = j[(j.index > SPINUP) & (j.o > 0)]
    if len(j) < 3000: return None
    lo = j[j.index.month.isin(LOW)]
    a = compute_acf(low7(q))
    mwy = q.resample("YS-OCT").sum(); mwy.index = mwy.index.year + 1
    dj = pd.DataFrame({"o": owy, "m": mwy}).dropna(); dj = dj[dj.index.isin(dry)]
    return dict(rho1=a["rho"][0], p1=a["p"][0], rho2=a["rho"][1], rho3=a["rho"][2],
                mae=float(np.nanmean(np.abs(a["rho"]-a_ob["rho"]))),
                PBIAS_low=100*(lo.m.sum()-lo.o.sum())/lo.o.sum(),
                KGE_low=kge(lo.o.values, lo.m.values),
                dryPB=100*(dj.m.sum()-dj.o.sum())/dj.o.sum() if len(dj) >= 3 else np.nan,
                KGE=kge(j.o.values, j.m.values),
                NSE=1-np.sum((j.m-j.o)**2)/np.sum((j.o-j.o.mean())**2))


rows = []
for tag, path in ([("baseline", SC/"calibration/bestruns/East_River_baseline_1981_2025/run"),
                   ("uniform sf=7", SB/"sweep_sf_20260812/sf07p0"),
                   ("uniform sf=10", SB/"sweep_sf_20260812/sf10p0")]
                  + [(Path(d).name, d) for d in sorted(glob.glob(str(OUT/"exp*")))]):
    q = qdaily(path)
    if q is None:
        print(f"  (no output yet: {tag})"); continue
    s = score(q)
    if s: rows.append(dict(run=tag, **s))

t = pd.DataFrame(rows).set_index("run")
print("observed  " + " ".join(f"{r:+.2f}{'*' if p < 0.05 else ' '}"
                              for r, p in zip(a_ob["rho"], a_ob["p"])) + f"   n={a_ob['n']}\n")
print(t.round(3).to_string())
b = t.loc["baseline"]
print(f"\nbaseline: rho1={b.rho1:+.2f} (p={b.p1:.3f})  PBIAS_low={b.PBIAS_low:+.1f}%  KGE={b.KGE:.3f}")
win = t[(t.p1 < 0.05) & (t.PBIAS_low < 45)]
print("\n=== members with SIGNIFICANT rho1 and PBIAS_low < 45% ===")
print(win.round(3).to_string() if len(win) else "  none -- the trade is not broken by the exponent")
t.to_csv(OUT/"scores.csv")
print(f"\n-> {OUT/'scores.csv'}")
