"""Per-HRU mean ASO SWE (mm) for every Tuolumne-area flight, clipped to the Tuolumne HRU polygons only. -> modeling/calibration/observations/tuolumne_aso_hru_swe.csv"""
import warnings; warnings.filterwarnings("ignore")
import re, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from pathlib import Path
D = Path("/scratch/dlhogan/ess-project-data/geospatial-data/ASO/Tuolumne")
SHP = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp"
OUT = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/tuolumne_aso_hru_swe.csv")
g = gpd.read_file(SHP).sort_values("HRU_ID").reset_index(drop=True); rows = []
for f in sorted(D.glob("*.tif")):
    date = pd.Timestamp(re.search(r"(\d{8})\.tif$", f.name).group(1))
    with rasterio.open(f) as src:
        gg = g.to_crs(src.crs); nod = src.nodata
        for _, r in gg.iterrows():
            try: arr, tr = mask(src, [r.geometry], crop=True, filled=True, nodata=-9999.0)
            except Exception: continue
            a = arr[0].astype(float); inside = geometry_mask([r.geometry], out_shape=a.shape, transform=tr, invert=True)
            ok = inside & np.isfinite(a) & (a >= 0) & (a < 50) & (a != -9999.0)
            if nod is not None: ok &= (a != nod)
            rows.append(dict(file=f.name, date=date, hru=int(r.HRU_ID), aso_mm=1000*a[ok].mean() if ok.any() else np.nan, covf=ok.sum()/max(inside.sum(), 1)))
    print(f.name, flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False); print("wrote", OUT, len(rows))
