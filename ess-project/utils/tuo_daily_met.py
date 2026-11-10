"""Daily basin-mean and high-elevation air temperature / shortwave from the Tuolumne METSIM forcing (basin_averaged_data, 28 HRUs, 1981-2025; local-clock hourly).
'basin' = HRU-area-weighted mean over 28 HRUs; 'high' = HRU-area-weighted mean over HRUs above 3600 m (hru 1-2, the perennial-snow headwall zone).
Note: forcing SW is biased -11% Mar-Jun vs the Vogelsang (CDEC FLV) radiometer (see project notes) -> treat SW as an index. -> modeling/calibration/observations/tuolumne_daily_met.csv"""
import numpy as np, pandas as pd, xarray as xr, glob
SC = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/"
A = xr.open_dataset(SC+"settings/SUMMA/attributes.nc"); area = np.array(A["HRUarea"].values, float); elev = np.array(A["elevation"].values, float)
wb = area/area.sum(); hi = elev > 3600; wh = np.where(hi, area, 0.0); wh = wh/wh.sum(); print("high HRUs:", np.where(hi)[0]+1, "mean elev %.0f" % (elev[hi].mean()))
rows = []
for f in sorted(glob.glob(SC+"forcing/basin_averaged_data/*_metsim_*.nc")):
    d = xr.open_dataset(f); t = pd.DatetimeIndex(d["time"].values); T = d["airtemp"].values.astype(float)-273.15; S = d["SWRadAtm"].values.astype(float); P = d["pptrate"].values.astype(float)*3600
    df = pd.DataFrame({"T_basin": T@wb, "T_high": T@wh, "SW_basin": S@wb, "SW_high": S@wh, "P_basin": P@wb}, index=t); rows.append(df.resample("D").agg({"T_basin": "mean", "T_high": "mean", "SW_basin": "mean", "SW_high": "mean", "P_basin": "sum"})); d.close()
out = pd.concat(rows); out.index.name = "date"; out.to_csv("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/tuolumne_daily_met.csv"); print(out.shape, out.index.min(), out.index.max()); print(out.describe().round(1))
