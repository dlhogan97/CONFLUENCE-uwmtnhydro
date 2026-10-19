"""Water supply vs snowpack by elevation band: accumulated precipitation (Oct 1 -> flight date) and modelled SWE vs ASO SWE, best-so-far v2 trial.
usage: band_swe_precip.py <East_River|Tuolumne_River> <best_full/rankN dir>"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, netCDF4 as nc, xarray as xr
from pathlib import Path
dom, R = sys.argv[1], Path(sys.argv[2]); OBS = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations")
d = xr.open_dataset(R/"out"/"best_timestep.nc"); t = pd.DatetimeIndex(d["time"].values); hid = d["hruId"].values.ravel().astype(int)
swe = pd.DataFrame(d["scalarSWE"].values, index=t, columns=hid).resample("D").mean()
pr = pd.DataFrame(d["pptrate"].values, index=t, columns=hid)                     # mm/s (sum over the hourly step)
a = xr.open_dataset(R/"attributes.nc"); A = pd.Series(a["HRUarea"].values.ravel(), index=a["hruId"].values.ravel().astype(int)); E = pd.Series(a["elevation"].values.ravel(), index=A.index)
ann = (pr.resample("D").sum()*3600*0+pr.resample("D").mean()*86400).groupby(lambda x: x.year+(x.month>=10)).sum()      # mm / water year (daily mean rate x 86400)
print(f"basin mean precipitation (mm/yr, area-weighted, WY complete years): {(ann.loc[ann.index[1:-1]]*(A/A.sum())).sum(axis=1).mean():.0f}")
pday = pr.resample("D").mean()*86400
if dom == "East_River":
    aso = pd.read_csv(OBS/"east_aso_hru_swe.csv", parse_dates=["date"]); cuts = [0, 2800, 3200, 3500, 4000]; labs = ["~2700", "~3000", "~3350", "~3700"]
    band = pd.cut(E, cuts, labels=labs)
else:
    import geopandas as gpd
    g = gpd.read_file("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp")
    cls = dict(zip(g.HRU_ID, g.elevClass)); band = pd.Series({h: f"b{cls[h]}" for h in A.index})
    aso = pd.read_csv(OBS/"tuolumne_aso_hru_swe.csv", parse_dates=["date"]).drop_duplicates(["date", "hru"])
aso = aso[(aso.covf >= 0.7) & aso.aso_mm.notna() & aso.date.isin(swe.index)].copy()
print("flights inside the model run & >=70% HRU coverage:", sorted(set(aso.date.dt.strftime('%Y-%m-%d')))[:60])
rows = []
for dt, gp in aso.groupby("date"):
    wy0 = pd.Timestamp(dt.year - (dt.month < 10), 10, 1); psum = pday.loc[wy0:dt].sum()
    for h, r in gp.set_index("hru").iterrows():
        if h in A.index: rows.append(dict(date=dt, hru=h, band=band[h], area=A[h], aso=r.aso_mm, mdl=swe.loc[dt, h], precip=psum[h]))
df = pd.DataFrame(rows); df["spring"] = df.date.dt.month <= 4
def agg(x):
    w = x.area/x.area.sum(); return pd.Series(dict(nHRUflt=len(x), precip_mm=(x.precip*w).sum(), ASO_mm=(x.aso*w).sum(), model_mm=(x.mdl*w).sum(), model_over_ASO=(x["mdl"]*w).sum()/(x.aso*w).sum(),
                                                  ASO_frac_of_precip=(x.aso*w).sum()/(x.precip*w).sum(), model_frac_of_precip=(x["mdl"]*w).sum()/(x.precip*w).sum()))
pd.set_option("display.width", 220)
for nm, sub in (("SPRING flights (<=Apr, near peak)", df[df.spring]), ("MELT-SEASON flights (>=May)", df[~df.spring])):
    print("\n", nm, "-- by elevation band (area-weighted over HRU-flights)"); print(sub.groupby("band").apply(agg).round(2).to_string())
    print(" all bands:", agg(sub).round(2).to_dict())
    print(" by flight date:"); print(sub.groupby("date").apply(agg).round(2).to_string())
df.to_csv(R/"band_swe_precip.csv", index=False)
