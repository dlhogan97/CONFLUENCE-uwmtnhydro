"""Save cropped per-scene snow class rasters over the Lyell catchments (LABvTB bbox, contains LBM) for GIFs / within-season analysis.
Classes (uint8): 0 outside LABvTB catchment, 1 snow-free valid, 2 snow (sensor rule NDSI 0.08 OLI | 0.11 TM/ETM+, NIR>0.11), 3 invalid (cloud/shadow/snow-QA bits 1-3, or green<=0.05). Also DEM + catchment masks. Scenes Jul 1 - Oct 10, 1985-2025, WRS 042/034.
-> snowpatch_fsca/catchment_masks/<yyyymmdd>_<platform>.npz, catchment_masks/static.npz"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
import landsat_fsca as L, landsat_grid as G
from fsca_by_band import basin_grid, PX_KM2
from fsca_thresh import DATA, SENSOR_THRESH
from fsca_catchments import catch_masks
OUT = DATA/"catchment_masks"; OUT.mkdir(exist_ok=True)
DEM = "/scratch/dlhogan/ess-project-data/geospatial-data/MERIT-Hydro/elv/elv_n30w120/n35w120_elv.tif"

def main():
    t0 = time.time(); grid, _ = basin_grid(); CM = catch_masks(grid); m = CM["LABvTB"]; rr, cc = np.where(m); pad = 10
    r0, r1, c0, c1 = max(rr.min()-pad, 0), rr.max()+pad, max(cc.min()-pad, 0), cc.max()+pad; sl = (slice(r0, r1), slice(c0, c1))
    tr = grid["transform"]*rasterio.Affine.translation(c0, r0); H, W = r1-r0, c1-c0; dem = np.zeros((H, W), "float32")
    with rasterio.open(DEM) as s: reproject(rasterio.band(s, 1), dem, dst_transform=tr, dst_crs=grid["crs"], resampling=Resampling.bilinear)
    np.savez_compressed(OUT/"static.npz", dem=dem, affine=np.array(tr)[:6], **{f"mask_{k}": v[sl] for k, v in CM.items()}); print("crop", H, W, flush=True)
    items = [i for i in L.search(1985, 2025, mon_lo=(7, 1), mon_hi=(10, 10), cloud_lt=35) if i.properties.get("landsat:wrs_path") == "042" and i.properties.get("landsat:wrs_row") == "034"]
    def job(it):
        d = pd.Timestamp(it.datetime).tz_localize(None).normalize(); plat = it.properties.get("platform")
        try:
            qa = G.read_on_grid(it.assets["qa_pixel"].href, grid).astype("uint32")[sl]; g = (G.read_on_grid(it.assets["green"].href, grid)*L.SR_SCALE+L.SR_OFF)[sl]
            sw = (G.read_on_grid(it.assets["swir16"].href, grid)*L.SR_SCALE+L.SR_OFF)[sl]; n = (G.read_on_grid(it.assets["nir08"].href, grid)*L.SR_SCALE+L.SR_OFF)[sl]
        except Exception: return
        fill = (qa != 0) & ((qa & 1) == 0); bad = np.zeros(qa.shape, bool)
        for b in (1, 2, 3): bad |= ((qa >> b) & 1).astype(bool)
        valid = fill & ~bad & (g > 0) & (sw > 0) & (g > 0.05)
        with np.errstate(invalid="ignore", divide="ignore"): ndsi = (g-sw)/(g+sw)
        snow = (ndsi > SENSOR_THRESH[plat]) & (n > L.NIR_THRESH); c = np.zeros(qa.shape, "uint8"); inm = m[sl]
        c[inm & valid & ~snow] = 1; c[inm & valid & snow] = 2; c[inm & ~valid] = 3
        np.savez_compressed(OUT/f"{d:%Y%m%d}_{plat}.npz", cls=c)
    with ThreadPoolExecutor(8) as ex: list(ex.map(job, items))
    print(f"done {len(items)} scenes ({time.time()-t0:.0f}s)")
if __name__ == "__main__": main()
