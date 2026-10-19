"""
End-of-season fractional snow-covered area over Tuolumne HRU 2 from
Landsat Collection 2 Level-2 surface reflectance (Planetary Computer STAC).

Binary NDSI snow classification per 30 m pixel, aggregated to an area
fraction over the HRU -- at ~9,570 pixels the aggregation is what gives the
fraction, so no sub-pixel unmixing is needed (and its ~10% error would be
comparable to a late-September signal).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from pystac_client import Client
import planetary_computer as pc

SHP = Path("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_"
           "elevAspect/shapefiles/catchment/"
           "Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [-119.3888, 37.7396, -119.1996, 38.0938]

# Landsat C2 L2 surface-reflectance scaling
SR_SCALE, SR_OFF = 2.75e-5, -0.2
NDSI_THRESH = 0.40        # standard snow test
NIR_THRESH = 0.11         # rejects water, which also has high NDSI
MIN_VALID = 0.70          # need this fraction of the HRU cloud-free
MIN_COVER = 0.95          # ...and this fraction actually inside the scene
N_HRU = 8.611236e6 / 900  # ~9568 Landsat 30 m pixels in HRU 2
# The WRS-2 boundary cuts through HRU 2, so many single scenes cover only part
# of it and the remainder comes back as QA fill (qa == 1), NOT as cloud. Both
# fractions are therefore measured against N_HRU, never against scene pixels.


def hru_geom_wgs():
    g = gpd.read_file(SHP)
    return g[g.HRU_ID == 2]


def search(y0=1984, y1=2025, mon_lo=(8, 15), mon_hi=(10, 10), cloud_lt=35):
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    items = list(cat.search(
        collections=["landsat-c2-l2"], bbox=BBOX,
        datetime=f"{y0}-01-01/{y1}-12-31",
        query={"eo:cloud_cover": {"lt": cloud_lt}}).items())
    keep = []
    for it in items:
        d = it.datetime
        if (d.month, d.day) < mon_lo or (d.month, d.day) > mon_hi:
            continue
        keep.append(it)
    return keep


def _read(item, band, geom_scene):
    href = item.assets[band].href
    with rasterio.open(href) as src:
        arr, _ = rio_mask(src, geom_scene, crop=True, filled=True,
                          nodata=0, all_touched=False)
    return arr[0]


def scene_fsca(item, gdf):
    """Binary NDSI fSCA for one Landsat item over the HRU. None if unusable."""
    try:
        crs = item.properties.get("proj:code") or item.properties.get("proj:epsg")
        with rasterio.open(item.assets["green"].href) as src:
            scene_crs = src.crs
        geom = list(gdf.to_crs(scene_crs).geometry)

        green = _read(item, "green", geom).astype("float32")
        swir = _read(item, "swir16", geom).astype("float32")
        nir = _read(item, "nir08", geom).astype("float32")
        qa = _read(item, "qa_pixel", geom).astype("uint16")
    except Exception as e:                       # noqa: BLE001
        return dict(status=f"read fail: {type(e).__name__}")

    clipped = qa != 0                            # 0 = outside the clipped polygon
    fill = (qa & 1) > 0                          # bit0 fill = outside the SCENE
    in_scene = clipped & ~fill
    cover = in_scene.sum() / N_HRU
    if cover < MIN_COVER:
        return dict(status="partial scene", cover=round(float(cover), 3))

    dil_cloud = (qa & (1 << 1)) > 0
    cirrus = (qa & (1 << 2)) > 0
    cloud = (qa & (1 << 3)) > 0
    shadow = (qa & (1 << 4)) > 0
    bad = dil_cloud | cirrus | cloud | shadow

    valid = in_scene & ~bad & (green > 0) & (swir > 0)
    vfrac = valid.sum() / N_HRU
    if vfrac < MIN_VALID:
        return dict(status="too cloudy", valid_frac=round(float(vfrac), 3),
                    cover=round(float(cover), 3))

    g = green * SR_SCALE + SR_OFF
    s = swir * SR_SCALE + SR_OFF
    n = nir * SR_SCALE + SR_OFF
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi = (g - s) / (g + s)
    snow = valid & (ndsi > NDSI_THRESH) & (n > NIR_THRESH)

    return dict(status="ok",
                fsca=round(float(snow.sum() / valid.sum()), 4),
                valid_frac=round(float(vfrac), 3),
                cover=round(float(cover), 3),
                n_valid=int(valid.sum()),
                cloud_scene=item.properties.get("eo:cloud_cover"))


def run(items, gdf, workers=8):
    def job(it):
        r = scene_fsca(it, gdf)
        r.update(date=pd.Timestamp(it.datetime).tz_localize(None),
                 platform=it.properties.get("platform"),
                 wrs=f"{it.properties.get('landsat:wrs_path')}"
                     f"{it.properties.get('landsat:wrs_row')}",
                 id=it.id)
        return r
    with ThreadPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(job, items))).sort_values("date")


if __name__ == "__main__":
    import sys, time
    gdf = hru_geom_wgs()
    y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2015, 2017)
    t = time.time()
    items = search(y0, y1)
    print(f"{len(items)} scenes {y0}-{y1} in window")
    df = run(items, gdf)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    print(f"\nelapsed {time.time()-t:.0f}s  "
          f"({(time.time()-t)/max(len(items),1):.1f}s/scene)")
    ok = df[df.status == "ok"]
    if len(ok):
        print("\nper-year minimum fSCA:")
        ok = ok.copy(); ok["year"] = ok.date.dt.year
        print(ok.loc[ok.groupby("year").fsca.idxmin()]
              [["year", "date", "fsca", "valid_frac", "platform"]].to_string(index=False))
