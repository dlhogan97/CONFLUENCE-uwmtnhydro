"""
Where does late-season snow ACTUALLY persist?

One pass over a grid covering the whole basin; per-elevation-band and per-HRU
fSCA computed from the same rasters, so the vertical profile of persistent
snow comes out of a single read rather than one retrieval per region.

Reports BOTH fSCA (fraction, depends on how the region is drawn) and
snow-covered AREA in km2 (invariant to the region definition) -- the latter
is the honest metric when comparing regions of different size.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin

import landsat_fsca as L
import landsat_grid as G

OUT = Path(__file__).parent
RES = 30.0
MIN_COVER = 0.75
MIN_VALID = 0.35
PX_KM2 = RES * RES / 1e6


def basin_grid():
    gdf = gpd.read_file(L.SHP).to_crs(G.REF_CRS)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin, ymin = np.floor(xmin / RES) * RES - RES, np.floor(ymin / RES) * RES - RES
    xmax, ymax = np.ceil(xmax / RES) * RES + RES, np.ceil(ymax / RES) * RES + RES
    w, h = int((xmax - xmin) / RES), int((ymax - ymin) / RES)
    transform = from_origin(xmin, ymax, RES, RES)
    grid = dict(crs=G.REF_CRS, transform=transform, width=w, height=h)
    # per-HRU label raster (0 = outside basin)
    lab = rasterize([(g, int(i)) for g, i in zip(gdf.geometry, gdf.HRU_ID)],
                    out_shape=(h, w), transform=transform, fill=0,
                    dtype="int32", all_touched=False)
    grid["labels"] = lab
    grid["mask"] = lab > 0
    grid["n_hru"] = int(grid["mask"].sum())
    grid["band_of_hru"] = dict(zip(gdf.HRU_ID, gdf.elevClass))
    grid["elev_of_band"] = gdf.groupby("elevClass").elev_mean.mean().to_dict()
    return grid, gdf


def scene_stats(it, grid, hru_ids, band_ids):
    """fSCA + snow area for every HRU and every elevation band, one read."""
    m, lab = grid["mask"], grid["labels"]
    try:
        qa = G.read_on_grid(it.assets["qa_pixel"].href, grid).astype("uint32")
        g = G.read_on_grid(it.assets["green"].href, grid) * L.SR_SCALE + L.SR_OFF
        sw = G.read_on_grid(it.assets["swir16"].href, grid) * L.SR_SCALE + L.SR_OFF
        n = G.read_on_grid(it.assets["nir08"].href, grid) * L.SR_SCALE + L.SR_OFF
    except Exception as e:                         # noqa: BLE001
        return dict(status=f"read fail {type(e).__name__}")

    ins = m & (qa != 0) & ((qa & 1) == 0)
    cover = ins.sum() / m.sum()
    if cover < MIN_COVER:
        return dict(status="partial", cover=round(float(cover), 3))
    bad = (((qa >> 1) & 1) | ((qa >> 2) & 1) | ((qa >> 3) & 1) |
           ((qa >> 4) & 1)).astype(bool)
    valid = ins & ~bad & (g > 0) & (sw > 0)
    vfrac = valid.sum() / m.sum()
    if vfrac < MIN_VALID:
        return dict(status="too cloudy", valid_frac=round(float(vfrac), 3),
                    cover=round(float(cover), 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi = (g - sw) / (g + sw)
    snow = valid & (ndsi > L.NDSI_THRESH) & (n > L.NIR_THRESH)

    r = dict(status="ok", cover=round(float(cover), 3),
             valid_frac=round(float(vfrac), 3))
    lab_v, lab_s = lab[valid], lab[snow]
    nv = np.bincount(lab_v, minlength=30).astype(float)
    ns = np.bincount(lab_s, minlength=30).astype(float)
    for h in hru_ids:
        r[f"v_hru{h}"], r[f"s_hru{h}"] = nv[h], ns[h]
    bands = {}
    for h in hru_ids:
        b = grid["band_of_hru"][h]
        bands.setdefault(b, [0.0, 0.0])
        bands[b][0] += nv[h]; bands[b][1] += ns[h]
    for b, (v_, s_) in bands.items():
        r[f"fsca_b{b}"] = s_ / v_ if v_ else np.nan
        r[f"area_b{b}"] = s_ * PX_KM2
    r["fsca_basin"] = ns[1:].sum() / nv[1:].sum()
    r["area_basin"] = ns[1:].sum() * PX_KM2
    return r


def main():
    t0 = time.time()
    grid, gdf = basin_grid()
    hru_ids = sorted(gdf.HRU_ID)
    band_ids = sorted(gdf.elevClass.unique())
    print(f"basin grid {grid['width']}x{grid['height']}  "
          f"N={grid['n_hru']} px ({grid['n_hru']*PX_KM2:.1f} km2 vs 774.5)")

    items = [i for i in L.search(1985, 2025, mon_lo=(7, 1), mon_hi=(10, 10),
                                 cloud_lt=35)
             if i.properties.get("landsat:wrs_path") == "042"
             and i.properties.get("landsat:wrs_row") == "034"]
    print(f"{len(items)} scenes Jul 1 - Oct 10")

    def job(it):
        r = scene_stats(it, grid, hru_ids, band_ids)
        r.update(date=pd.Timestamp(it.datetime).tz_localize(None).normalize(),
                 platform=it.properties.get("platform"),
                 scene_cloud=it.properties.get("eo:cloud_cover"))
        return r

    with ThreadPoolExecutor(8) as ex:
        df = pd.DataFrame(list(ex.map(job, items))).sort_values("date")
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df.to_csv(OUT / "scenes_by_band.csv", index=False)
    ok = df[df.status == "ok"].copy()
    print(f"usable {len(ok)}/{len(df)}   ({time.time()-t0:.0f}s)")

    # annual EOS = scene minimising BASIN snow area, Aug 15 onward
    win = ok[ok.apply(lambda r: (r.date.month, r.date.day) >= (8, 15), axis=1)]
    eos = win.loc[win.groupby("year").area_basin.idxmin()].set_index("year")
    eos.to_csv(OUT / "eos_by_band.csv")

    bcols = [c for c in eos.columns if c.startswith("fsca_b")]
    acols = [c for c in eos.columns if c.startswith("area_b")]
    prof = pd.DataFrame({
        "band": [int(c[-1]) for c in bcols],
        "elev_m": [round(grid["elev_of_band"][int(c[-1])]) for c in bcols],
        "area_km2": [round(gdf[gdf.elevClass == int(c[-1])].HRU_area.sum()/1e6, 1)
                     for c in bcols],
        "fSCA_mean": [round(eos[c].mean(), 4) for c in bcols],
        "fSCA_max": [round(eos[c].max(), 4) for c in bcols],
        "snowkm2_mean": [round(eos[c.replace("fsca", "area")].mean(), 3)
                         for c in bcols],
        "snowkm2_max": [round(eos[c.replace("fsca", "area")].max(), 3)
                        for c in bcols],
    }).sort_values("band", ascending=False)
    prof["pct_of_basin_snow"] = (100 * prof.snowkm2_mean /
                                 prof.snowkm2_mean.sum()).round(1)
    pd.set_option("display.width", 200)
    print(f"\n=== end-of-season snow by elevation band ({len(eos)} years) ===")
    print(prof.to_string(index=False))

    print(f"\nbasin EOS fSCA mean {eos.fsca_basin.mean():.4f}   "
          f"snow area mean {eos.area_basin.mean():.2f} km2  "
          f"max {eos.area_basin.max():.2f} km2")

    # per-HRU, top two bands
    top = [h for h in hru_ids if grid["band_of_hru"][h] >= 7]
    rows = []
    for h in top:
        v = eos[f"v_hru{h}"]; s = eos[f"s_hru{h}"]
        rows.append(dict(hru=h, band=grid["band_of_hru"][h],
                         elev=round(float(gdf.loc[gdf.HRU_ID == h, "elev_mean"].iloc[0])),
                         aspect=round(float(gdf.loc[gdf.HRU_ID == h, "aspectClas"].iloc[0])),
                         area_km2=round(float(gdf.loc[gdf.HRU_ID == h, "HRU_area"].iloc[0])/1e6, 1),
                         fSCA_mean=round(float((s / v).mean()), 4),
                         snowkm2_mean=round(float((s * PX_KM2).mean()), 3)))
    ph = pd.DataFrame(rows).sort_values("snowkm2_mean", ascending=False)
    print("\n=== per-HRU, top two bands (ranked by persistent snow area) ===")
    print(ph.to_string(index=False))
    prof.to_csv(OUT / "band_profile.csv", index=False)
    ph.to_csv(OUT / "hru_profile.csv", index=False)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
