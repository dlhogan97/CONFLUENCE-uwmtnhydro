"""7-day minimum low-flow lag structure, WY2007-2021, EVERY available source.

Nine series on one footing: four unregulated nested headwater gauges, Hetch Hetchy
unimpaired inflow (raw + the area-scaled version the merged obs record uses), the
downstream regulated gauge, the SUMMA baseline, and a snowpatch member.

Per instruction: NO low-flow cap. The only filter is Q > 0.
"""
from pathlib import Path
import glob
import numpy as np, pandas as pd, netCDF4 as nc4
from scipy.stats import rankdata

SC = Path("/scratch/dlhogan/ess-project-data")
TUO = SC/"domain_Tuolumne_River_distributed_elevAspect"
TUO_SW = TUO/"simulations/snowpatch_sweep_20260812"
SD = SC/"streamflow-data/TuolumneRiver"
WY0, WY1 = 2007, 2021
NLAG = 5
CFS = 35.3146667


def compute_acf(series, n_lags=NLAG, n_permutations=20000, seed=0):
    def _sp(a, b):
        ra = rankdata(a)-(len(a)+1)/2.; rb = rankdata(b)-(len(b)+1)/2.
        den = np.sqrt((ra*ra).sum()*(rb*rb).sum())
        return (ra*rb).sum()/den if den > 0 else 0.0
    rng = np.random.default_rng(seed)
    s = series.dropna().astype(float)
    yrs = s.index.values.astype(int); v = s.values; n = v.size
    pos = {int(y): i for i, y in enumerate(yrs)}
    rho = np.full(n_lags, np.nan); p = np.full(n_lags, np.nan); npair = np.zeros(n_lags, int)
    for k in range(1, n_lags+1):
        ip = np.array([pos[int(y)-k] for y in yrs if int(y)-k in pos], int)
        ic = np.array([i for i, y in enumerate(yrs) if int(y)-k in pos], int)
        npair[k-1] = ip.size
        if ip.size <= 3: continue
        rho[k-1] = _sp(v[ip], v[ic])
        pm = np.array([abs(_sp(r[ip], r[ic])) for r in
                       (rng.permutation(v) for _ in range(n_permutations))])
        p[k-1] = np.mean(pm >= abs(rho[k-1]))
    return dict(rho=rho, p=p, band=1.96/np.sqrt(n), n=n, npair=npair)


def min7(s):
    """7-day centred rolling mean, minimum over Aug-Oct, by water year. Only filter: Q>0."""
    s = s[s > 0]
    x = (s.where(s.index.month.isin([8, 9, 10]))
          .rolling(7, center=True, min_periods=7).mean().resample("YS-OCT").min())
    x.index = x.index.year
    return x[(x.index >= WY0) & (x.index <= WY1)]


# ------------------------------------------------------------------ sources
def load_nested():
    d = pd.read_csv(SD/"TuolumneStreamflow_Hydroclimate_2002_2021.csv",
                    parse_dates=["Date"]).set_index("Date")
    d.columns = [c.replace("Estimated_Discharge_", "") for c in d.columns]
    return {g: d[g].dropna() for g in d.columns}


def load_hh_raw():
    """Wide WY-matrix, rows = day of water year (10/1 .. 9/30), values in cfs."""
    d = pd.read_csv(SD/"TuolumneRiver_HH_unimpaired_19702024.csv")
    d = d[d.DATE.astype(str).str.match(r"^\d{1,2}/\d{1,2}$", na=False)]
    out = {}
    for c in [c for c in d.columns if c.startswith("WY ")]:
        wy = int(c.split()[1])
        v = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
        for ds, val in zip(d.DATE, v):
            m, day = (int(x) for x in ds.split("/"))
            yr = wy-1 if m >= 10 else wy
            try:
                out[pd.Timestamp(yr, m, day)] = val
            except ValueError:
                pass                                   # 2/29 in non-leap WY columns
    s = pd.Series(out).sort_index().dropna()
    return s/CFS                                       # cfs -> m3/s


def qdaily(run):
    run = Path(run)
    at = nc4.Dataset(run/"attributes.nc"); a = float(np.array(at["HRUarea"][:]).sum()); at.close()
    f = sorted(glob.glob(str(run/"output"/"*.nc")) + glob.glob(str(run/"out"/"*.nc")))[0]
    ds = nc4.Dataset(f); t = ds["time"]
    tt = nc4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
    q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a; ds.close()
    return pd.Series(q, index=pd.DatetimeIndex(tt)).resample("1D").mean()


hh_raw = load_hh_raw()
merged = pd.read_csv(TUO/"observations/streamflow/preprocessed/"
                     "Tuolumne_River_merged_dailyQ_WY1971_2025.csv",
                     parse_dates=["datetime"]).set_index("datetime")
hh_scaled_hist = merged.loc[merged.source == "HH_scaled", "discharge_cms"]

# recover the scale factor the merged record applied, then extend it to WY2007-2021
j = pd.DataFrame({"raw": hh_raw, "scl": hh_scaled_hist}).dropna()
K = float(np.median(j.scl/j.raw))
print(f"HH area-scale factor recovered from the merged record: K = {K:.4f}  "
      f"(n={len(j)} overlapping days, IQR {np.percentile(j.scl/j.raw,25):.4f}-"
      f"{np.percentile(j.scl/j.raw,75):.4f})")

lump = pd.read_csv(TUO/"observations/streamflow/preprocessed/"
                   "Tuolumne_River_lumped_streamflow_processed.csv",
                   parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]

nested = load_nested()
SERIES = {}
for g in ("T120", "LBM", "LABvTB", "DFkBC"):
    SERIES[f"nested {g}"] = nested[g]
SERIES["HH raw (unimpaired)"] = hh_raw
SERIES["HH corrected (scaled)"] = hh_raw*K
SERIES["gauge (merged obs)"] = merged["discharge_cms"]
SERIES["regulated downstream"] = lump
SERIES["SUMMA baseline"] = qdaily(SC/"calibration/bestruns/"
                                  "Tuolumne_River_baseline_wbias10_1981_2025/run")
SERIES["SUMMA snowpatch m090"] = qdaily(TUO_SW/"m090_q089")

M = pd.DataFrame({k: min7(v) for k, v in SERIES.items()})

print("\n" + "="*118)
print(f"7-DAY MINIMUM (Aug-Oct), WY{WY0}-{WY1}   [m3/s]   -- only filter is Q > 0")
print("="*118)
print(M.round(3).to_string())
print("\nyears present per series:")
print("  " + "   ".join(f"{c}:{int(M[c].notna().sum())}" for c in M.columns))

print("\n" + "="*118)
print("LAG STRUCTURE OF THE 7-DAY MIN")
print("="*118)
print(f"{'series':<24s}{'n':>3s} {'band':>6s}   " + "".join(f"{'lag'+str(k):>9s}" for k in range(1, 6)))
print("-"*118)
rows = []
for c in M.columns:
    a = compute_acf(M[c])
    cells = "".join(("     n/a " if np.isnan(r) else f"{r:>+8.2f}{'*' if (pp == pp and pp < 0.05) else ' '}")
                    for r, pp in zip(a["rho"], a["p"]))
    print(f"{c:<24s}{a['n']:>3d} ±{a['band']:.2f}   {cells}")
    rows.append(dict(series=c, n=a["n"], band=a["band"],
                     **{f"rho{k+1}": a["rho"][k] for k in range(NLAG)},
                     **{f"p{k+1}": a["p"][k] for k in range(NLAG)},
                     **{f"npair{k+1}": a["npair"][k] for k in range(NLAG)}))
print(f"\n  * permutation p < 0.05.  n={M.notna().sum().max()} years -> lag5 rests on "
      f"{compute_acf(M['HH raw (unimpaired)'])['npair'][4]} pairs; treat lags 4-5 as indicative only.")

print("\n" + "="*118)
print("DO THE SERIES AGREE ON WHICH YEARS WERE LOW?  (Spearman of the 7-day min itself)")
print("="*118)
print(M.corr(method="spearman").round(2).to_string())

out = TUO_SW/"acf_7daymin_allsources_2007_2021.csv"
pd.DataFrame(rows).to_csv(out, index=False)
M.to_csv(TUO_SW/"min7_allsources_2007_2021.csv")
print(f"\n-> {out}")
