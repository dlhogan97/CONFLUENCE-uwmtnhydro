"""Lag-1..5 ACF on TWO low-flow metrics, obs vs baseline vs adjusted.

Motivated by the Tuolumne obs audit: the merged record splices a Hetch-Hetchy
reconstruction (WY1970-2006, 2.6x wetter at low flow) onto the real gauge
(WY2007-2025). That step manufactures significant ACF at every lag out to 5 in the
7-day min. September mean volume is far less sensitive to it, so it is reported
alongside, plus an era-normalised version of the 7-day min for Tuolumne obs.
"""
from pathlib import Path
import glob, re
import numpy as np, pandas as pd, netCDF4 as nc4
from scipy.stats import rankdata

SC = Path("/scratch/dlhogan/ess-project-data")
EAST = SC/"domain_East_River_distributed_elevAspect"
TUO = SC/"domain_Tuolumne_River_distributed_elevAspect"
EAST_SW = EAST/"simulations/store_memory_sandbox/East/sweep_sf_20260812"
TUO_SW = TUO/"simulations/snowpatch_sweep_20260812"
YR0, YR1, NLAG = 1970, 2024, 5
SPLICE_WY = 2006          # last water year of the Tuolumne HH_scaled reconstruction


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


def clip(x):
    x.index = x.index.year
    return x[(x.index >= YR0) & (x.index <= YR1)]


def min7(s, months, rs):
    x = (s.where(s.index.month.isin(months))
          .rolling(7, center=True, min_periods=7).mean().resample(rs).min())
    x[x < 0.01] = np.nan
    return clip(x)


def sepvol(s, months, rs):
    """Mean flow over the single core low-flow month (September, both basins)."""
    x = s.where(s.index.month == 9).resample(rs).mean()
    return clip(x)


def destep(x, split=SPLICE_WY):
    """Divide each splice era by its own median so a level shift can't fake persistence."""
    x = x.dropna().astype(float).copy()
    a, b = x.index <= split, x.index > split
    if a.sum() > 2 and b.sum() > 2:
        x[a] = x[a]/x[a].median(); x[b] = x[b]/x[b].median()
    return x


def qdaily(run):
    run = Path(run)
    at = nc4.Dataset(run/"attributes.nc"); a = float(np.array(at["HRUarea"][:]).sum()); at.close()
    f = sorted(glob.glob(str(run/"output"/"*.nc")) + glob.glob(str(run/"out"/"*.nc")))[0]
    ds = nc4.Dataset(f); t = ds["time"]
    tt = nc4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
    q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a; ds.close()
    return pd.Series(q, index=pd.DatetimeIndex(tt)).resample("1D").mean()


def line(tag, a):
    return (f"{tag:<34s} n={a['n']:3d} ±{a['band']:.2f}  "
            + " ".join(f"{r:+.2f}{'*' if p < 0.05 else ' '}" for r, p in zip(a["rho"], a["p"])))


def block(basin, obs, runs, months, rs, extra_obs=(), out=None):
    print("\n" + "="*88)
    print(f"{basin}")
    print("="*88)
    rows = []

    def rec(metric, tag, a):
        rows.append(dict(metric=metric, model=tag, n=a["n"], band=a["band"],
                         **{f"rho{k+1}": a["rho"][k] for k in range(NLAG)},
                         **{f"p{k+1}": a["p"][k] for k in range(NLAG)}))
        return a

    for mtag, f in [("7-day min", min7), ("September mean", sepvol)]:
        print(f"\n--- {mtag.upper()} ---")
        print(f"{'':<34s} {'':>7s}   " + "  ".join(f"lag{k} " for k in range(1, 6)))
        print(line("OBSERVED", rec(mtag, "OBSERVED", compute_acf(f(obs, months, rs)))))
        for tag, s in extra_obs:
            print(line(f"  {tag}", rec(mtag, tag, compute_acf(destep(f(s, months, rs))))))
        for label, path in runs:
            print(line(label, rec(mtag, label, compute_acf(f(qdaily(path), months, rs)))))

    if out:
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  -> {out}")


def main():
    o = pd.read_csv(EAST/"observations/streamflow/preprocessed/"
                    "East_River_USGS09112500_dailyQ_WY1970_2025.csv",
                    parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
    east_runs = [("baseline", SC/"calibration/bestruns/East_River_baseline_1981_2025/run")]
    for d in sorted(glob.glob(str(EAST_SW/"sf*"))):
        if glob.glob(d+"/output/*.nc"):
            sf = float(re.search(r"sf(\d+)p(\d+)", Path(d).name).expand(r"\1.\2"))
            if sf in (7, 9, 12, 24):
                east_runs.append((f"sf={sf:g}", d))
    block("EAST RIVER  (single USGS gauge 09112500, no splice)",
          o.groupby(o.index.normalize()).mean(), east_runs, [9, 10, 11], "YE-NOV",
          out=EAST_SW/"acf_two_metrics.csv")

    t = pd.read_csv(TUO/"observations/streamflow/preprocessed/"
                    "Tuolumne_River_merged_dailyQ_WY1971_2025.csv",
                    parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
    t = t.groupby(t.index).mean()
    tuo_runs = [("baseline", SC/"calibration/bestruns/Tuolumne_River_baseline_wbias10_1981_2025/run")]
    for nm in ("m090_q089", "m095_q160", "m100_q320"):
        d = TUO_SW/nm
        if glob.glob(str(d/"output"/"*.nc")):
            tuo_runs.append((nm, d))
    block("TUOLUMNE RIVER  (HH_scaled WY<=2006 spliced to gauge WY>=2007)",
          t, tuo_runs, [8, 9, 10], "YS-OCT",
          extra_obs=[("OBS, era-normalised", t)], out=TUO_SW/"acf_two_metrics.csv")
    print("\n  * permutation p < 0.05    'era-normalised' divides each splice era by its own median")


if __name__ == "__main__":
    main()
