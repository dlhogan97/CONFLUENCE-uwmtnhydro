"""Export the two selected adjusted runs to the CSV layout the mrc_analysis configs expect."""
from pathlib import Path
import glob
import numpy as np, pandas as pd, netCDF4 as nc4

SC = Path("/scratch/dlhogan/ess-project-data")
OUT = SC/"streamflow-data"

RUNS = {
    "EastRiver/EastRiver_SUMMA_aquifer_sf7_dailyDischarge_1981_2025.csv": (
        SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/"
           "East/sweep_sf_20260812/sf07p0", "SUMMA-adjusted_East_aquifer_sf7_1981_2025"),
    "TuolumneRiver/TuolumneRiver_SUMMA_snowpatch_m090q089_dailyDischarge_1981_2025.csv": (
        SC/"domain_Tuolumne_River_distributed_elevAspect/simulations/"
           "snowpatch_sweep_20260812/m090_q089", "SUMMA-adjusted_Tuolumne_snowpatch_m090q089_1981_2025"),
}


def qdaily(run):
    run = Path(run)
    at = nc4.Dataset(run/"attributes.nc"); a = float(np.array(at["HRUarea"][:]).sum()); at.close()
    f = sorted(glob.glob(str(run/"output"/"*.nc")) + glob.glob(str(run/"out"/"*.nc")))[0]
    ds = nc4.Dataset(f); t = ds["time"]
    tt = nc4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
    q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a; ds.close()
    return pd.Series(q, index=pd.DatetimeIndex(tt)).resample("1D").mean()


for rel, (run, sid) in RUNS.items():
    s = qdaily(run)
    df = pd.DataFrame({"time": s.index.strftime("%Y-%m-%d"), "station_id": sid,
                       "discharge": s.values})
    p = OUT/rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"{p.name:62s} n={len(df):5d}  {df.time.iloc[0]}..{df.time.iloc[-1]}  "
          f"mean={s.mean():.2f} m3/s  min={s.min():.3f}")
