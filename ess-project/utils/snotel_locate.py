"""Which HRU contains each SNOTEL / CDEC pillow, and does the elevation match?"""
import warnings; warnings.filterwarnings("ignore")
import re, glob
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point

SN = Path("/scratch/dlhogan/ess-project-data/meteorological-data/SNOTEL")
SHP = {
 "East":     "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/shapefiles/catchment/East_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
 "Tuolumne": "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
}
STATIONS = {
 "East":     [("380_Butte_sntl_obs.csv", "Butte")],
 "Tuolumne": [("DAN_DANA_MEADOWS_cdec_obs.csv", "Dana Meadows"),
              ("GYP_GAYLOR_PIT_cdec_obs.csv", "Gaylor Pit"),
              ("SLI_SLIDE_CANYON_cdec_obs.csv", "Slide Canyon"),
              ("TUM_TUOLUMNE_MEADOWS_cdec_obs.csv", "Tuolumne Meadows")],
}
FT = 0.3048

for basin, shp in SHP.items():
    g = gpd.read_file(shp)
    print(f"\n===== {basin} =====")
    print("  cols:", [c for c in g.columns if c != "geometry"])
    print(f"  n HRU {len(g)}   crs {g.crs}")
    for fn, name in STATIONS[basin]:
        df = pd.read_csv(SN/fn, nrows=1)
        lon, lat, elev_ft = map(float, re.findall(r"[-\d.]+", df.geometry.iloc[0]))
        pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(g.crs)
        hit = gpd.sjoin(pt, g, predicate="within")
        if len(hit) == 0:
            print(f"  {name:18s} ({lat:.4f},{lon:.4f}) {elev_ft*FT:6.0f} m  -> OUTSIDE domain")
            continue
        r = hit.iloc[0]
        cols = {c: r[c] for c in g.columns if c not in ("geometry",)}
        print(f"  {name:18s} ({lat:.4f},{lon:.4f}) {elev_ft*FT:6.0f} m  -> {cols}")
