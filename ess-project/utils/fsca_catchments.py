"""Landsat end-of-season snow area inside the nested Tuolumne catchments (LBM, DFkBC, LABvTB, T120), same scenes / QA / MIN_COVER / MIN_VALID / window (Aug 1 - Oct 10) as fsca_thresh.py.
Classes: t40 (NDSI>0.40 & NIR>0.11) and sensor (NDSI>0.08 OLI | >0.11 TM/ETM+, & NIR>0.11); strict (QA bits 1-4) and shadow-inclusive (bits 1-3, green>0.05).
Snow area = snow px / valid px * catchment px area. -> snowpatch_fsca/scenes_by_catchment_thresh.csv, eos_by_catchment.csv"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
import landsat_fsca as L, landsat_grid as G
from fsca_by_band import basin_grid, MIN_COVER, MIN_VALID, PX_KM2
from fsca_thresh import DATA, SENSOR_THRESH
R = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"

def catch_masks(grid):
    gdf = gpd.read_file(R+"modeling/calibration/observations/tuolumne_nested_catchments.geojson").to_crs(grid["crs"])
    return {r.id: rasterize([(r.geometry, 1)], out_shape=(grid["height"], grid["width"]), transform=grid["transform"], fill=0, dtype="uint8").astype(bool) for r in gdf.itertuples()}

def stats(it, grid, CM):
    try:
        qa = G.read_on_grid(it.assets["qa_pixel"].href, grid).astype("uint32"); g = G.read_on_grid(it.assets["green"].href, grid)*L.SR_SCALE+L.SR_OFF
        sw = G.read_on_grid(it.assets["swir16"].href, grid)*L.SR_SCALE+L.SR_OFF; n = G.read_on_grid(it.assets["nir08"].href, grid)*L.SR_SCALE+L.SR_OFF
    except Exception as e: return dict(status=f"read fail {type(e).__name__}")
    with np.errstate(invalid="ignore", divide="ignore"): ndsi = (g-sw)/(g+sw)
    th = SENSOR_THRESH[it.properties.get("platform")]; cls = {"t40": (ndsi > L.NDSI_THRESH) & (n > L.NIR_THRESH), "sensor": (ndsi > th) & (n > L.NIR_THRESH)}
    out = dict(status="ok"); fill = (qa != 0) & ((qa & 1) == 0)
    for k, m in CM.items():
        cover = (m & fill).sum()/m.sum(); out[f"cover_{k}"] = float(cover)
        if cover < MIN_COVER: continue
        for tag, bits in (("strict", (1, 2, 3, 4)), ("shadow", (1, 2, 3))):
            bad = np.zeros(qa.shape, bool)
            for b in bits: bad |= ((qa >> b) & 1).astype(bool)
            valid = m & fill & ~bad & (g > 0) & (sw > 0)
            if tag == "shadow": valid &= g > 0.05
            vf = valid.sum()/m.sum(); out[f"valid_{k}_{tag}"] = float(vf)
            if vf < MIN_VALID: continue
            for c, s in cls.items(): out[f"area_{k}_{c}_{tag}"] = (valid & s).sum()/valid.sum()*m.sum()*PX_KM2
    return out

def main():
    t0 = time.time(); grid, _ = basin_grid(); CM = catch_masks(grid); print({k: round(v.sum()*PX_KM2, 1) for k, v in CM.items()}, "km2 on Landsat grid", flush=True)
    items = [i for i in L.search(1985, 2025, mon_lo=(7, 1), mon_hi=(10, 10), cloud_lt=35) if i.properties.get("landsat:wrs_path") == "042" and i.properties.get("landsat:wrs_row") == "034"]
    def job(it):
        r = stats(it, grid, CM); r.update(date=pd.Timestamp(it.datetime).tz_localize(None).normalize(), platform=it.properties.get("platform")); return r
    with ThreadPoolExecutor(8) as ex: df = pd.DataFrame(list(ex.map(job, items))).sort_values("date")
    df["year"] = df.date.dt.year; df.to_csv(DATA/"scenes_by_catchment_thresh.csv", index=False)
    d = df[(df.status == "ok") & df.apply(lambda r: (r.date.month, r.date.day) >= (8, 1), axis=1)]; rows = []
    for k in CM:
        for c in ("t40", "sensor"):
            for tag in ("strict", "shadow"):
                col = f"area_{k}_{c}_{tag}"
                if col not in d: continue
                s = d[col].dropna(); pick = s.groupby(d.loc[s.index, "year"]).idxmin()
                for y, i in pick.items(): rows.append(dict(catch=k, cls=c, tier=tag, year=y, area_km2=d.loc[i, col], date=d.loc[i, "date"], platform=d.loc[i, "platform"], n_scenes=int(s[d.loc[s.index, "year"] == y].size)))
    pd.DataFrame(rows).to_csv(DATA/"eos_by_catchment.csv", index=False); print(f"done {len(df)} scenes, ok {int((df.status=='ok').sum())} ({time.time()-t0:.0f}s)")
if __name__ == "__main__": main()
