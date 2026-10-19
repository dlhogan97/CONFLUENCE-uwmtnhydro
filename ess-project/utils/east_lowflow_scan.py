"""Scan every full-period East run for one question: does anything beat the baseline
on LOW FLOWS?

Reported per run:
  PBIAS_low  Sep-Nov volume bias (baseline is already +29%, so nearer 0 is better)
  KGE_low    KGE on Sep-Nov daily flow
  dryPB      bias on the lowest observed quartile of WY totals (the zoom-panel metric)
  KGE / NSE  whole-record daily, so we can see what a low-flow gain costs
"""
from pathlib import Path
import glob, re
import numpy as np, pandas as pd, netCDF4 as nc4

SC = Path("/scratch/dlhogan/ess-project-data")
EAST = SC/"domain_East_River_distributed_elevAspect"
SB = EAST/"simulations/store_memory_sandbox/East"
SPINUP = "1983-09-30"
LOW_MONTHS = [9, 10, 11]


def kge(o, m):
    r = np.corrcoef(o, m)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (m.std(ddof=0)/o.std(ddof=0)-1)**2 + (m.mean()/o.mean()-1)**2)


def qdaily(run):
    run = Path(run)
    at = run/"attributes.nc"
    if not at.exists():
        return None
    d = nc4.Dataset(at); a = float(np.array(d["HRUarea"][:]).sum()); d.close()
    f = sorted(glob.glob(str(run/"output"/"*.nc")) + glob.glob(str(run/"out"/"*.nc")))
    if not f:
        return None
    ds = nc4.Dataset(f[0])
    if "averageRoutedRunoff" not in ds.variables:
        ds.close(); return None
    t = ds["time"]
    tt = nc4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
    q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a; ds.close()
    return pd.Series(q, index=pd.DatetimeIndex(tt)).resample("1D").mean(), a


def params(run):
    p = Path(run)/"trialParams.nc"
    if not p.exists():
        return {}
    d = nc4.Dataset(p)
    out = {}
    # These families modify a SUBSET of HRUs, so record the range and how many HRUs
    # were changed -- reading element 0 alone reports the baseline value and hides it.
    for v in ("aquiferScaleFactor", "aquiferBaseflowRate", "aquiferBaseflowExp"):
        if v in d.variables:
            a = np.array(d[v][:]).ravel()
            out[v + "_min"] = float(a.min())
            out[v + "_max"] = float(a.max())
            out[v + "_nmod"] = int((a != np.median(a)).sum())
    d.close()
    return out


obs = pd.read_csv(EAST/"observations/streamflow/preprocessed/"
                  "East_River_USGS09112500_dailyQ_WY1970_2025.csv",
                  parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
obs = obs.groupby(obs.index.normalize()).mean()

# observed WY totals + the dry-quartile year set
owy = obs.resample("YS-OCT").sum(); owy.index = owy.index.year + 1
owy = owy[(owy.index >= 1984) & (owy.index <= 2025)]
dry_thresh = owy.quantile(0.25)
dry_years = set(owy[owy <= dry_thresh].index)


def score(q, area):
    j = pd.DataFrame({"o": obs, "m": q}).dropna()
    j = j[(j.index > SPINUP) & (j.o > 0)]
    if len(j) < 3000:
        return None
    lo = j[j.index.month.isin(LOW_MONTHS)]
    mwy = q.resample("YS-OCT").sum(); mwy.index = mwy.index.year + 1
    dj = pd.DataFrame({"o": owy, "m": mwy}).dropna()
    dj = dj[dj.index.isin(dry_years)]
    dpb = 100*(dj.m.sum()-dj.o.sum())/dj.o.sum() if len(dj) >= 3 else np.nan
    return dict(n=len(j),
                PBIAS_low=100*(lo.m.sum()-lo.o.sum())/lo.o.sum(),
                KGE_low=kge(lo.o.values, lo.m.values),
                dryPB=dpb, n_dry=len(dj),
                KGE=kge(j.o.values, j.m.values),
                NSE=1-np.sum((j.m-j.o)**2)/np.sum((j.o-j.o.mean())**2))


CANDS = [("baseline", SC/"calibration/bestruns/East_River_baseline_1981_2025/run")]
CANDS += [("current_best", SC/"calibration/bestruns/East_River_current_best/run")]
CANDS += [("bigaq10m", SC/"calibration/bestruns/East_River_baseline_bigaq10m_1981_2025/run")]
for d in sorted(glob.glob(str(SC/"calibration/bestruns/experiments/East_River_*"))):
    CANDS.append((Path(d).name.replace("East_River_", ""), Path(d)/"run"))
for pat in ("east_slow_*_fp", "ens_bw_*_fp", "ens_scale_*_fp", "sweep_sf_20260812/sf*"):
    for d in sorted(glob.glob(str(SB/pat))):
        CANDS.append((Path(d).name, d))
for d in sorted(glob.glob(str(SB/"ens_tau_*"))):
    if not Path(d).name.endswith("_11yr"):
        CANDS.append((Path(d).name, d))

rows = []
for name, path in CANDS:
    r = qdaily(path)
    if r is None:
        continue
    q, a = r
    s = score(q, a)
    if s is None:
        continue
    rows.append(dict(run=name, **s, **params(path)))

t = pd.DataFrame(rows).set_index("run")
b = t.loc["baseline"]
print(f"observed dry-quartile threshold: {dry_thresh:.0f} m3/s-days  "
      f"({len(dry_years)} WYs: {sorted(dry_years)})\n")
print(f"BASELINE   PBIAS_low={b.PBIAS_low:+.1f}%  KGE_low={b.KGE_low:.3f}  "
      f"dryPB={b.dryPB:+.1f}%  KGE={b.KGE:.3f}  NSE={b.NSE:.3f}\n")

t["absPB"] = t.PBIAS_low.abs()
better = t[(t.absPB < abs(b.PBIAS_low)) | (t.KGE_low > b.KGE_low)]
cols = ["PBIAS_low", "KGE_low", "dryPB", "KGE", "NSE",
        "aquiferScaleFactor_min", "aquiferScaleFactor_max", "aquiferScaleFactor_nmod",
        "aquiferBaseflowRate_min", "aquiferBaseflowRate_max"]
print("=== runs that beat baseline on |PBIAS_low| OR KGE_low ===")
if len(better):
    print(better.sort_values("absPB")[cols].round(4).to_string())
else:
    print("  none")
print(f"\n=== all {len(t)} scored runs, sorted by |PBIAS_low| ===")
print(t.sort_values("absPB")[cols].round(4).to_string())
print()
print("=== sweep_sf_20260812 members (uniform sf) ===")
print(t[t.index.str.startswith("sf")][cols].round(4).to_string())
t.to_csv(SB/"east_lowflow_scan.csv")
print(f"\n-> {SB/'east_lowflow_scan.csv'}")
