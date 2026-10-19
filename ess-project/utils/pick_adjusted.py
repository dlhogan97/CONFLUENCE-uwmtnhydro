"""Re-score every sweep member against the CORRECTED observations and pick one per basin.

Criterion (the user's): most memory improvement for the least performance degradation.
  memory   = MAE of rho_1..5 vs observed  (whole lag SHAPE, not just lag-1)
  skill    = daily KGE over the obs overlap, WY1984+ (meta.json analysis_start)
Reported as deltas from that basin's baseline so the trade-off is explicit.
"""
from pathlib import Path
import glob, re, json
import numpy as np, pandas as pd, netCDF4 as nc4
from scipy.stats import rankdata

SC = Path("/scratch/dlhogan/ess-project-data")
EAST = SC/"domain_East_River_distributed_elevAspect"
TUO = SC/"domain_Tuolumne_River_distributed_elevAspect"
EAST_SW = EAST/"simulations/store_memory_sandbox/East/sweep_sf_20260812"
TUO_SW = TUO/"simulations/snowpatch_sweep_20260812"
SD = SC/"streamflow-data/TuolumneRiver"
CFS, K = 35.3146667, 0.505111
YR0, YR1, SPINUP = 1970, 2024, "1983-09-30"


def compute_acf(series, n_lags=5, n_permutations=5000, seed=0):
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


def low(d, months, rs):
    s = (d.where(d.index.month.isin(months))
          .rolling(7, center=True, min_periods=7).mean().resample(rs).min())
    s[s < 0.01] = np.nan; s.index = s.index.year
    return s[(s.index >= YR0) & (s.index <= YR1)]


def kge(o, m):
    r = np.corrcoef(o, m)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (m.std(ddof=0)/o.std(ddof=0)-1)**2 + (m.mean()/o.mean()-1)**2)


def skill(obs, mod, months):
    j = pd.DataFrame({"o": obs, "m": mod}).dropna()
    j = j[(j.index > SPINUP) & (j.o > 0)]
    lo = j[j.index.month.isin(months)]
    nse = 1 - np.sum((j.m-j.o)**2)/np.sum((j.o-j.o.mean())**2)
    return dict(KGE=kge(j.o.values, j.m.values), NSE=nse,
                KGE_low=kge(lo.o.values, lo.m.values),
                PBIAS_low=100*(lo.m.sum()-lo.o.sum())/lo.o.sum())


def qdaily(run):
    run = Path(run)
    at = nc4.Dataset(run/"attributes.nc"); a = float(np.array(at["HRUarea"][:]).sum()); at.close()
    f = sorted(glob.glob(str(run/"output"/"*.nc")) + glob.glob(str(run/"out"/"*.nc")))[0]
    ds = nc4.Dataset(f); t = ds["time"]
    tt = nc4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
    q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a; ds.close()
    return pd.Series(q, index=pd.DatetimeIndex(tt)).resample("1D").mean()


def load_hh():
    d = pd.read_csv(SD/"TuolumneRiver_HH_unimpaired_19702024.csv")
    d = d[d.DATE.astype(str).str.match(r"^\d{1,2}/\d{1,2}$", na=False)]
    out = {}
    for c in [c for c in d.columns if c.startswith("WY ")]:
        wy = int(c.split()[1])
        v = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
        for ds_, val in zip(d.DATE, v):
            m, day = (int(x) for x in ds_.split("/"))
            try: out[pd.Timestamp(wy-1 if m >= 10 else wy, m, day)] = val
            except ValueError: pass
    s = (pd.Series(out).sort_index().dropna()/CFS)*K
    end = pd.Timestamp(s.index[-1].year + (1 if s.index[-1].month >= 10 else 0), 9, 30)
    if end > s.index[-1]: end = pd.Timestamp(end.year-1, 9, 30)
    return s.loc[:end]


def run(basin, obs, baseline, members, months, rs):
    print("\n" + "="*112)
    print(f"{basin}   (obs re-scored: corrected record)")
    print("="*112)
    o = low(obs, months, rs)
    a_ob = compute_acf(o)
    print("observed  " + " ".join(f"{r:+.2f}{'*' if p < 0.05 else ' '}"
                                  for r, p in zip(a_ob["rho"], a_ob["p"])) + f"   n={a_ob['n']}")
    rows = []
    for tag, path in [("baseline", baseline)] + members:
        q = qdaily(path)
        a = compute_acf(low(q, months, rs))
        s = skill(obs, q, months)
        rows.append(dict(model=tag, **{f"rho{k+1}": a["rho"][k] for k in range(5)},
                         p1=a["p"][0], mae=float(np.nanmean(np.abs(a["rho"]-a_ob["rho"]))), **s))
    t = pd.DataFrame(rows).set_index("model")
    b = t.loc["baseline"]
    t["d_mae"] = t.mae - b.mae            # negative = memory shape improved
    t["d_KGE"] = t.KGE - b.KGE            # negative = skill lost
    t["ratio"] = (-t.d_mae)/(-t.d_KGE).clip(lower=1e-6)   # memory gained per unit KGE lost
    cols = ["rho1", "p1", "rho2", "rho3", "mae", "d_mae", "KGE", "d_KGE", "ratio",
            "NSE", "KGE_low", "PBIAS_low"]
    print(t[cols].round(3).to_string())
    adj = t.drop(index="baseline")
    imp = adj[adj.d_mae < 0]
    if len(imp):
        print(f"\n  best memory shape      : {adj.mae.idxmin()}")
        print(f"  least skill lost       : {adj.d_KGE.idxmax()}")
        print(f"  best trade-off (ratio) : {imp.ratio.idxmax()}")
    return t


east_obs = pd.read_csv(EAST/"observations/streamflow/preprocessed/"
                       "East_River_USGS09112500_dailyQ_WY1970_2025.csv",
                       parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
east_mem = []
for d in sorted(glob.glob(str(EAST_SW/"sf*"))):
    if glob.glob(d+"/output/*.nc"):
        sf = float(re.search(r"sf(\d+)p(\d+)", Path(d).name).expand(r"\1.\2"))
        east_mem.append((f"sf={sf:g}", d))
te = run("EAST RIVER", east_obs.groupby(east_obs.index.normalize()).mean(),
         SC/"calibration/bestruns/East_River_baseline_1981_2025/run",
         east_mem, [9, 10, 11], "YE-NOV")

tuo_mem = [(Path(d).name, d) for d in sorted(glob.glob(str(TUO_SW/"m*")))
           if glob.glob(d+"/output/*.nc")]
tt = run("TUOLUMNE RIVER", load_hh(),
         SC/"calibration/bestruns/Tuolumne_River_baseline_wbias10_1981_2025/run",
         tuo_mem, [8, 9, 10], "YE-OCT")

te.to_csv(EAST_SW/"pick_adjusted.csv"); tt.to_csv(TUO_SW/"pick_adjusted.csv")
print("\nwrote pick_adjusted.csv in both sweep dirs")
