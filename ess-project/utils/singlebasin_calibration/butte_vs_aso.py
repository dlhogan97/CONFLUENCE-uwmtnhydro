"""How well does the Butte SNOTEL pillow represent basin SWE on the ASO flight dates (East River)?"""
import warnings; warnings.filterwarnings("ignore")
import re, numpy as np, pandas as pd, geopandas as gpd, rasterio, netCDF4 as nc, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import Point
ASO = Path("/scratch/dlhogan/ess-project-data/geospatial-data/ASO/East")
SHP = "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/shapefiles/catchment/East_River_distributed_elevAspect_HRUs_elevation_aspect.shp"
R = "/scratch/dlhogan/ess-project-data/calibration/bestruns/East_River_baseline_1981_2025/run"
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/butte_vs_aso_basin_swe.png"
# --- Butte pillow (mm) ---
d = pd.read_csv("/scratch/dlhogan/ess-project-data/meteorological-data/SNOTEL/380_Butte_sntl_obs.csv", usecols=["datetime", "geometry", "SWE", "SWE_units"])
lon, lat, _ = map(float, re.findall(r"[-\d.]+", d.geometry.iloc[0]))
t = pd.to_datetime(d.datetime, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
butte = pd.Series(pd.to_numeric(d.SWE, errors="coerce").values*(25.4 if str(d.SWE_units.dropna().iloc[0]).lower().startswith("in") else 1.0), index=t).dropna(); butte = butte[~butte.index.duplicated()]
# --- model ---
ds = nc.Dataset(R+"/out/baseline_timestep.nc"); tm = ds["time"]; tt = pd.DatetimeIndex(nc.num2date(tm[:], tm.units, only_use_cftime_datetimes=False)); hid = np.array(ds["hruId"][:]).ravel().astype(int)
msw = pd.DataFrame(np.array(ds["scalarSWE"][:]), index=tt, columns=hid).resample("1D").mean(); A = pd.Series(np.array(nc.Dataset(R+"/attributes.nc")["HRUarea"][:]), index=hid)
def wmedian(v, w):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]; c = np.cumsum(w)/w.sum(); return float(v[np.searchsorted(c, 0.5)])
g = gpd.read_file(SHP); basin = g.dissolve().geometry.iloc[0]; hru13 = g[g.HRU_ID == 13].geometry.iloc[0]
basin_km2 = gpd.GeoSeries([basin], crs=g.crs).to_crs("EPSG:32613").area.iloc[0]/1e6
print("East River catchment (union of the 18 HRU polygons): %.1f km2 ; HRU 13: %.1f km2" % (basin_km2, gpd.GeoSeries([hru13], crs=g.crs).to_crs("EPSG:32613").area.iloc[0]/1e6))
rows = []
for f in sorted(ASO.glob("*.tif")):
    date = pd.Timestamp(re.search(r"(\d{8})", f.name).group(1))
    with rasterio.open(f) as src:
        gb = gpd.GeoSeries([basin], crs=g.crs).to_crs(src.crs).iloc[0]; g13 = gpd.GeoSeries([hru13], crs=g.crs).to_crs(src.crs).iloc[0]
        pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        def stats(geom):
            try: a, tr = mask(src, [geom], crop=True, filled=True, nodata=-9999.0)
            except Exception: return np.array([]), 0.0
            a = a[0].astype(float)
            ins = geometry_mask([geom], out_shape=a.shape, transform=tr, invert=True); ok = ins & np.isfinite(a) & (a >= 0) & (a < 50) & (a != -9999.0)
            return (1000*a[ok]) if ok.any() else np.array([]), ok.sum()/max(ins.sum(), 1)
        vb, cov = stats(basin if False else gb); v13, cov13 = stats(g13); vp, covp = stats(pt.buffer(150)); vp5, _ = stats(pt.buffer(500))
    rows.append(dict(date=date, basin_cov=cov, valid_km2=cov*basin_km2, raster_km2=(src.width*src.height*abs(src.res[0]*src.res[1]))/1e6, aso_basin_median=np.median(vb) if len(vb) else np.nan, aso_basin_mean=vb.mean() if len(vb) else np.nan,
                     aso_hru13_mean=v13.mean() if len(v13) else np.nan, aso_pillow_150m=vp.mean() if len(vp) else np.nan, aso_pillow_500m=vp5.mean() if len(vp5) else np.nan,
                     butte_obs=float(butte.get(date, np.nan))))
df = pd.DataFrame(rows)
df["mod_hru13"] = [msw.loc[d_, 13] if d_ in msw.index else np.nan for d_ in df.date]
df["mod_basin_mean"] = [float((msw.loc[d_].values*A.values).sum()/A.sum()) if d_ in msw.index else np.nan for d_ in df.date]
df["mod_basin_median"] = [wmedian(msw.loc[d_].values, A.values) if d_ in msw.index else np.nan for d_ in df.date]
df = df[df.basin_cov >= 0.7].reset_index(drop=True); df.to_csv("butte_vs_aso_flight_dates.csv", index=False)
df["ratio_butte_to_basin_median"] = df.butte_obs/df.aso_basin_median; df["ratio_model_hru13_to_basin_median"] = df.mod_hru13/df.mod_basin_median
pd.set_option("display.width", 250); print(df.drop(columns=["basin_cov"]).round(1).to_string(index=False))
v = df[(df.date.dt.year <= 2025)].dropna(subset=["butte_obs", "aso_basin_median"])
print("\nCLIPPING CHECK -- valid ASO pixels inside the East River polygon, by flight (all stats below use only these pixels):")
print(df[["date","basin_cov","valid_km2","raster_km2"]].assign(basin_cov=lambda x: (100*x.basin_cov).round(0)).rename(columns={"basin_cov":"% of catchment covered","valid_km2":"valid km2 in catchment","raster_km2":"whole raster km2"}).round(0).to_string(index=False))
v = df[(df.date.dt.year <= 2025)].dropna(subset=["butte_obs", "aso_basin_median"]).copy()
sp = v[v.date.dt.month <= 4]; ml = v[v.date.dt.month >= 5]
print("\nSPRING flights (Mar-Apr, n=%d): Butte / ASO basin median = %s ; mean %.2f" % (len(sp), np.round(sp.ratio_butte_to_basin_median.values, 2), sp.ratio_butte_to_basin_median.mean()))
print("  corr(Butte, ASO basin median) = %.2f ; slope through origin = %.2f ; Butte / ASO basin MEAN = %.2f (mean) ; Butte / ASO HRU13 mean = %.2f" % (np.corrcoef(sp.butte_obs, sp.aso_basin_median)[0, 1], (sp.butte_obs*sp.aso_basin_median).sum()/(sp.butte_obs**2).sum(), (sp.butte_obs/sp.aso_basin_mean).mean(), (sp.butte_obs/sp.aso_hru13_mean).mean()))
print("  ASO at the pillow (150 m) / Butte obs = %.2f ; ASO HRU13 mean / ASO pillow = %.2f" % ((sp.aso_pillow_150m/sp.butte_obs).mean(), (sp.aso_hru13_mean/sp.aso_pillow_150m).mean()))
print("  model HRU13 / model basin median = %.2f (sd %.2f) vs observed Butte / ASO basin median %.2f" % (sp.ratio_model_hru13_to_basin_median.mean(), sp.ratio_model_hru13_to_basin_median.std(), sp.ratio_butte_to_basin_median.mean()))
print("\nMELT-SEASON flights (May-Jun): what each says at / around the Butte pillow (mm)")
print(ml[["date","butte_obs","aso_pillow_150m","aso_hru13_mean","mod_hru13","aso_basin_median","aso_basin_mean","mod_basin_mean"]].round(0).to_string(index=False))
# --- figure ---
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
v = v.sort_values("date").reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.9), constrained_layout=True)
a = ax[0]; lim = 1.05*max(v.butte_obs.max(), v.aso_basin_median.max())
a.plot([0, lim], [0, lim], color="#7a7a76", lw=1, ls="--"); sl = (sp.butte_obs*sp.aso_basin_median).sum()/(sp.butte_obs**2).sum(); a.plot([0, lim], [0, sl*lim], color="#2a78d6", lw=1.2, alpha=0.6)
for _, r in v.iterrows():
    c = "#2a78d6" if r.date.month <= 4 else "#eb6834"; a.plot(r.butte_obs, r.aso_basin_median, "o", ms=8, color=c, mec="#fcfcfb", mew=1.2)
    if r.aso_basin_median > 50: a.annotate(r.date.strftime("%Y-%m-%d"), (r.butte_obs, r.aso_basin_median), xytext=(6, -11), textcoords="offset points", fontsize=7.5)
a.annotate("4 late-melt flights with\nmedian < 10 mm and Butte at 0-3 mm", (3, 8), xytext=(190, 25), textcoords="data", fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#7a7a76"))
a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_xlabel("Butte SNOTEL SWE on the flight date (mm)"); a.set_ylabel("ASO median SWE inside the East River catchment (mm)")
a.set_title("Butte vs the catchment median, on ASO flight dates\nblue: Mar-Apr (spring) flights, orange: May-Jun; dashed 1:1; blue line: spring slope %.2f" % sl, fontsize=10, loc="left")
b = ax[1]; w_ = v[v.aso_basin_median >= 50].reset_index(drop=True); x = np.arange(len(w_))
b.bar(x-0.19, w_.ratio_butte_to_basin_median, 0.36, color="#0b0b0b", label="observed: Butte / ASO basin median"); b.bar(x+0.19, w_.ratio_model_hru13_to_basin_median, 0.36, color="#2a78d6", label="model: HRU 13 / model basin median")
b.axhline(1, color="#7a7a76", lw=0.8, ls="--"); b.set_ylim(0, 1.75); b.set_xticks(x); b.set_xticklabels([d_.strftime("%Y-%m-%d") for d_ in w_.date], rotation=45, ha="right", fontsize=8); b.set_ylabel("pillow (or its HRU) / catchment median")
b.set_title("How representative is the pillow? (flights with catchment median >= 50 mm)", fontsize=10, loc="left"); b.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
fig.savefig(FIG, dpi=140); print("saved", FIG)
