#!/usr/bin/env python3
"""
regen_intersections.py — Rebuild the catchment_intersection shapefiles
(with_soilgrids, with_landclass) against the current HRU catchment shapefile.

The forcing pipeline was re-run against a new HRU discretization, but the
intersection shapefiles (and hence attributes.nc) were left on the old one.
This recomputes categorical zonal statistics of the soil and land-cover
rasters over the *current* catchment polygons, reproducing the schema that
update_settings_for_hrus.py expects:

    with_soilgrids : USGS_<c>, USGS_P<c>, count, soilClass, soilPct
    with_landclass : IGBP_<c>, IGBP_P<c>, count, landClass, landPct

`soilClass` / `landClass` are the majority class; `*Pct` its areal fraction.

Run this BEFORE update_settings_for_hrus.py.

Usage
-----
    python regen_intersections.py --domain-dir /scratch/.../domain_X [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats


def _find_catchment(domain_dir: Path) -> Path:
    d = domain_dir / "shapefiles" / "catchment"
    for pat in ("*_HRUs_elevation_aspect.shp", "*_HRUs_elevation*.shp", "*.shp"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No catchment shapefile under {d}")


# Fill / nodata sentinels that appear in these products and must never be
# treated as a class.  The Tuolumne MODIS land-cover download is entirely 255.
FILL_VALUES = {255, 250, -9999, -999, -32768}

# Plausible class ranges, used to score candidate rasters.
VALID_RANGE = {"landclass": (1, 17), "soilclass": (1, 16)}


def _to_raster_crs(catchment: gpd.GeoDataFrame, raster: Path) -> gpd.GeoDataFrame:
    """Reproject the catchment into the raster's CRS.

    rasterstats does NOT reproject — it assumes vectors and raster share a CRS.
    The Tuolumne NLCD land raster is Albers (metres) while the catchments are
    EPSG:4326, which silently yields zero overlap and an all-fill result.
    """
    import rasterio
    with rasterio.open(raster) as r:
        rcrs = r.crs
    if rcrs is None or catchment.crs is None or catchment.crs == rcrs:
        return catchment
    return catchment.to_crs(rcrs)


def _pixel_size_m(path: Path) -> float:
    """Approximate pixel size in metres (degrees are converted)."""
    import rasterio
    with rasterio.open(path) as r:
        res = r.res[0]
        if r.crs and r.crs.is_geographic:
            return res * 111_320.0
        return res


def _valid_frac_over_catchment(catchment: gpd.GeoDataFrame, path: Path,
                               kind: str) -> float:
    """Fraction of in-catchment pixels that are a plausible class."""
    lo, hi = VALID_RANGE.get(kind, (1, 255))
    try:
        cat = _to_raster_crs(catchment, path)
        stats = zonal_stats(cat, str(path), categorical=True)
    except Exception:
        return -1.0
    good = tot = 0
    for s in stats:
        for k, n in s.items():
            if k is None:
                continue
            k = int(k)
            tot += n
            if k not in FILL_VALUES and lo <= k <= hi:
                good += n
    return good / tot if tot else 0.0


def _pick_raster(domain_dir: Path, kind: str, catchment: gpd.GeoDataFrame,
                 override: Path | None = None) -> Path:
    """Choose a raster, scoring on valid classes *over the catchment*.

    Candidates are restricted to the canonical products; the annual MODIS
    MCD12Q1 tiles are excluded (there are 100+ and they all tie).
    """
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(override)
        print(f"    using override: {override.name}")
        return override

    d = domain_dir / "attributes" / kind
    if kind == "elevation":
        tifs = [t for t in sorted(d.rglob("*.tif"))
                if "aspect" not in t.name.lower()]
        if not tifs:
            raise FileNotFoundError(f"No DEM under {d}")
        stem = domain_dir.name.replace("domain_", "")
        exact = [t for t in tifs if stem in t.name]
        return (exact or tifs)[0]

    # Match on a substring, not a suffix: the only Tuolumne raster with IGBP
    # codes is "..._NLCD_land_classes_modis.tif" (NLCD reclassified to MODIS).
    # The plain "..._NLCD_land_classes.tif" carries raw NLCD codes (31/42/52),
    # which SUMMA's MODIFIED_IGBP_MODIS_NOAH table cannot interpret — the
    # in-catchment validity score below rejects it.
    key = "land_classes" if kind == "landclass" else "soil_classes"
    tifs = [t for t in sorted(d.rglob("*.tif")) if key in t.name]
    if not tifs:
        raise FileNotFoundError(f"No *{key}* raster under {d}")

    # Rank by validity first, then by resolution: a coarse (~900 m) MODIS
    # product cannot resolve alpine barren and smears it into grassland, while
    # the 30 m NLCD->IGBP product reproduces the barren/evergreen/shrub gradient.
    # Both score ~1.0 on validity, so resolution is the discriminator.
    cand = [(_valid_frac_over_catchment(catchment, t, kind), _pixel_size_m(t), t)
            for t in tifs]
    scored = sorted(cand, key=lambda x: (-round(x[0], 2), x[1]))
    for s, px, t in scored:
        print(f"    candidate {t.name:55s} valid={s:.3f} px={px:7.1f}m"
              + ("  <-- chosen" if t is scored[0][2] else ""))
    best_score, _, best = scored[0]
    if best_score <= 0:
        raise ValueError(
            f"No usable {kind} raster for {domain_dir.name}: every candidate is "
            f"fill/nodata over the catchment.\n  "
            + "\n  ".join(t.name for t in tifs)
        )
    return best


def _backup(path: Path) -> None:
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        src = path.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, dst.with_suffix(ext))
    print(f"    backed up -> {dst.name}")


def intersect(catchment: gpd.GeoDataFrame, raster: Path, prefix: str,
              class_col: str, pct_col: str) -> gpd.GeoDataFrame:
    cat_r = _to_raster_crs(catchment, raster)
    stats = zonal_stats(cat_r, str(raster), categorical=True, all_touched=False)

    classes = sorted({
        int(k) for s in stats for k in s.keys()
        if k is not None and int(k) not in FILL_VALUES
    })
    if not classes:
        raise ValueError(f"{raster.name}: no valid classes over the catchment "
                         f"(all pixels are fill)")
    out = catchment.copy()
    counts = np.zeros((len(catchment), len(classes)), dtype=float)
    for i, s in enumerate(stats):
        for j, c in enumerate(classes):
            counts[i, j] = float(s.get(c, 0) or 0)

    total = counts.sum(axis=1)
    total_safe = np.where(total > 0, total, 1.0)

    for j, c in enumerate(classes):
        out[f"{prefix}_{c}"] = counts[:, j]
        out[f"{prefix}_P{c}"] = 100.0 * counts[:, j] / total_safe
    out["count"] = total

    maj = counts.argmax(axis=1)
    out[class_col] = [classes[m] for m in maj]
    out[pct_col] = 100.0 * counts[np.arange(len(counts)), maj] / total_safe
    # HRUs with no raster coverage keep a majority of 0 counts — flag them
    empty = np.where(total == 0)[0]
    if len(empty):
        print(f"    WARNING: {len(empty)} HRU(s) had no raster coverage: "
              f"HRU_ID={[int(catchment.iloc[i]['HRU_ID']) for i in empty]}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain-dir", required=True, type=Path)
    p.add_argument("--land-raster", type=Path, default=None,
                   help="explicit land-cover raster (overrides auto-selection)")
    p.add_argument("--soil-raster", type=Path, default=None,
                   help="explicit soil-class raster (overrides auto-selection)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    dom = args.domain_dir
    cat_shp = _find_catchment(dom)
    cat = gpd.read_file(cat_shp).sort_values("HRU_ID").reset_index(drop=True)
    print(f"\nDomain     : {dom.name}")
    print(f"Catchment  : {cat_shp.name}  ({len(cat)} HRUs)")

    jobs = [
        ("soilclass", "with_soilgrids", "catchment_with_soilclass.shp",
         "USGS", "soilClass", "soilPct"),
        ("landclass", "with_landclass", "catchment_with_landclass.shp",
         "IGBP", "landClass", "landPct"),
    ]

    for kind, subdir, fname, prefix, class_col, pct_col in jobs:
        raster = _pick_raster(dom, kind, cat, getattr(args, f"{kind[:4]}_raster", None))
        print(f"\n  {kind}: {raster.name}")
        gdf = intersect(cat, raster, prefix, class_col, pct_col)
        summary = ", ".join(
            f"HRU{int(r['HRU_ID'])}:{int(r[class_col])}({r[pct_col]:.0f}%)"
            for _, r in gdf.iterrows()
        )
        print(f"    majority -> {summary}")

        outdir = dom / "shapefiles" / "catchment_intersection" / subdir
        outpath = outdir / fname
        if args.dry_run:
            print(f"    [dry-run] would write {outpath}")
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        _backup(outpath)
        gdf.to_file(outpath)
        print(f"    wrote {outpath}  ({len(gdf)} HRUs)")

    # Also refresh with_dem, which update_settings_for_hrus does not read but
    # other tooling may; it is just the catchment with elevation stats.
    dem = _pick_raster(dom, "elevation", cat)
    print(f"\n  elevation (for with_dem): {dem.name}")
    zs = zonal_stats(_to_raster_crs(cat, dem), str(dem), stats=["mean", "min", "max"])
    gdf = cat.copy()
    gdf["elev_mean"] = [z["mean"] for z in zs]
    gdf["elev_min"] = [z["min"] for z in zs]
    gdf["elev_max"] = [z["max"] for z in zs]
    outpath = dom / "shapefiles" / "catchment_intersection" / "with_dem" / "catchment_with_dem.shp"
    if args.dry_run:
        print(f"    [dry-run] would write {outpath}")
    else:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _backup(outpath)
        gdf.to_file(outpath)
        print(f"    wrote {outpath}  ({len(gdf)} HRUs)")

    print("\nDone. Next: python update_settings_for_hrus.py --domain-dir "
          f"{dom} --dry-run")


if __name__ == "__main__":
    main()
