"""
Same-date mosaicking of Landsat scenes onto a fixed 30 m HRU-2 grid.

The WRS-2 boundary splits HRU 2, so a single scene rarely covers it. Rows
043033 + 043034 are acquired on the same overpass and tile the HRU N-S (with
overlap), so they must be combined on a COMMON grid -- summing per-scene
counts would double-count the overlap band.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from landsat_fsca import (hru_geom_wgs, search, SR_SCALE, SR_OFF,
                          NDSI_THRESH, NIR_THRESH, MIN_VALID)

REF_CRS = "EPSG:32611"
RES = 30.0
MIN_COVER = 0.95
BANDS = ("green", "swir16", "nir08", "qa_pixel")


def reference_grid():
    """Fixed 30 m UTM-11N grid + rasterised HRU-2 mask."""
    gdf = hru_geom_wgs().to_crs(REF_CRS)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin, ymin = np.floor(xmin / RES) * RES - RES, np.floor(ymin / RES) * RES - RES
    xmax, ymax = np.ceil(xmax / RES) * RES + RES, np.ceil(ymax / RES) * RES + RES
    w, h = int((xmax - xmin) / RES), int((ymax - ymin) / RES)
    transform = from_origin(xmin, ymax, RES, RES)
    mask = rasterize([(g, 1) for g in gdf.geometry], out_shape=(h, w),
                     transform=transform, fill=0, dtype="uint8",
                     all_touched=False).astype(bool)
    return dict(crs=REF_CRS, transform=transform, width=w, height=h,
                mask=mask, n_hru=int(mask.sum()))


def read_on_grid(href, grid, dtype="float32"):
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=grid["crs"], transform=grid["transform"],
                       width=grid["width"], height=grid["height"],
                       resampling=Resampling.nearest, src_nodata=None,
                       nodata=0) as vrt:
            return vrt.read(1).astype(dtype)


def date_fsca(items, grid):
    """Mosaic all items from one date, then classify. items share a date."""
    m = grid["mask"]
    n_hru = grid["n_hru"]
    acc = {b: np.zeros(m.shape, dtype="float32") for b in BANDS}
    have = np.zeros(m.shape, dtype=bool)

    for it in items:
        try:
            layers = {b: read_on_grid(it.assets[b].href, grid) for b in BANDS}
        except Exception as e:                    # noqa: BLE001
            return dict(status=f"read fail: {type(e).__name__}")
        qa = layers["qa_pixel"].astype("uint32")
        good = (qa != 0) & ((qa & 1) == 0)        # in-scene, not fill
        take = good & ~have                       # first valid scene wins
        for b in BANDS:
            acc[b][take] = layers[b][take]
        have |= good

    in_hru = m & have
    cover = in_hru.sum() / n_hru
    if cover < MIN_COVER:
        return dict(status="partial", cover=round(float(cover), 3))

    qa = acc["qa_pixel"].astype("uint32")
    bad = ((qa >> 1) & 1) | ((qa >> 2) & 1) | ((qa >> 3) & 1) | ((qa >> 4) & 1)
    valid = in_hru & (bad == 0) & (acc["green"] > 0) & (acc["swir16"] > 0)
    vfrac = valid.sum() / n_hru
    if vfrac < MIN_VALID:
        return dict(status="too cloudy", valid_frac=round(float(vfrac), 3),
                    cover=round(float(cover), 3))

    g = acc["green"] * SR_SCALE + SR_OFF
    s = acc["swir16"] * SR_SCALE + SR_OFF
    n = acc["nir08"] * SR_SCALE + SR_OFF
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi = (g - s) / (g + s)
    snow = valid & (ndsi > NDSI_THRESH) & (n > NIR_THRESH)
    return dict(status="ok", fsca=round(float(snow.sum() / valid.sum()), 4),
                valid_frac=round(float(vfrac), 3), cover=round(float(cover), 3),
                n_valid=int(valid.sum()))


def run(y0, y1, workers=6, **kw):
    grid = reference_grid()
    items = search(y0, y1, **kw)
    by_date = {}
    for it in items:
        by_date.setdefault(pd.Timestamp(it.datetime).tz_localize(None).date(),
                           []).append(it)

    def job(kv):
        d, its = kv
        r = date_fsca(its, grid)
        r.update(date=pd.Timestamp(d), n_scenes=len(its),
                 wrs=",".join(sorted({f"{i.properties['landsat:wrs_path']}"
                                      f"{i.properties['landsat:wrs_row']}"
                                      for i in its})),
                 platform=",".join(sorted({i.properties["platform"] for i in its})))
        return r

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(job, sorted(by_date.items())))
    return pd.DataFrame(rows).sort_values("date"), grid


if __name__ == "__main__":
    import sys, time
    y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2015, 2017)
    t = time.time()
    df, grid = run(y0, y1)
    print(f"reference grid {grid['width']}x{grid['height']}  "
          f"N_HRU = {grid['n_hru']} px "
          f"({grid['n_hru']*900/1e6:.2f} km2 vs 8.61 actual)")
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    ok = df[df.status == "ok"]
    print(f"\n{len(ok)}/{len(df)} dates usable   elapsed {time.time()-t:.0f}s")
