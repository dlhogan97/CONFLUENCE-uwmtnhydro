"""Score the anchor snowpatch design against the two things it was built to fix:
(1) the store must never fully melt out, (2) the perennial bins must stay distinct."""
import warnings; warnings.filterwarnings("ignore")
import glob
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

RUNS = Path("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect"
            "/simulations/snowpatch_anchor_20260812")
DES  = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/designs_anchor_20260812")
DATA = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
SWE_TH, A_HRU2 = 0.05, 8.611236
K, S0 = 4.043, 1.331

eos = pd.read_csv(DATA/"eos_final.csv", index_col="year")
obs = eos.f_shadow
swe_design = pd.read_csv(DATA/"model_hru2_peak_swe.csv", index_col="year").peak_swe_mm/1000

rows = []
for d in sorted(RUNS.glob("anchor_m*")):
    if not (d/"output").is_dir():
        continue
    nc = sorted(glob.glob(str(d/"output/*.nc")))
    if not nc:
        continue
    b = pd.read_csv(DES/f"{d.name}.csv")
    ds = xr.open_dataset(nc[0])
    hid = np.asarray(ds.hruId.values).ravel()
    t = pd.DatetimeIndex(ds.time.values)
    swe = pd.DataFrame({int(r.bin): np.asarray(
        ds["scalarSWE"].isel(hru=int(np.where(hid == r.hruId)[0][0])).values).ravel()/1000.0
        for r in b.itertuples()}, index=t)
    w = b.set_index("bin").area_frac

    sep = swe[(t.month == 9) & (t.day == 30)]
    sep = sep.groupby(sep.index.year).last()      # hourly output -> 24 rows per 30 Sep
    f = pd.Series(((sep > SWE_TH).values * w.values).sum(axis=1), index=sep.index)

    j = pd.DataFrame({"mod": f, "obs": obs}).dropna()
    a = sep[1]                                              # the anchor bin
    per = sep[[2, 3, 4, 5]]
    rows.append(dict(
        member=d.name, n=len(j),
        fsca_ratio=j["mod"].mean()/j["obs"].mean(),
        r=j["mod"].corr(j["obs"]),
        zero_years=int((f == 0).sum()),
        fsca_min=f.min(),
        anchor_end=a.iloc[-1], anchor_min=a.min(),
        anchor_drift=(a.iloc[-1]-20.0)/len(a),
        per_spread_end=per.iloc[-1].max()-per.iloc[-1].min(),
        per_identical_yrs=int((per.nunique(axis=1) == 1).sum()),
        store_growth=float(sep.iloc[-1].sum()/max(sep.iloc[0].sum(), 1e-9))))

t = pd.DataFrame(rows).set_index("member")
print("=== anchor members ===")
print(t.round(3).to_string())
print("\nobserved: fSCA mean %.3f  min %.4f  zero-years 0 of %d"
      % (obs.mean(), obs.min(), len(obs)))
print("sweep best (m090_q089) for reference: fSCA ratio 1.52, r 0.825, 7 zero-years,"
      " perennial bins identical in ALL years, store_growth 0.002")
