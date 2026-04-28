#!/usr/bin/env python3
"""
update_settings_for_hrus.py — Rebuild attributes.nc, coldState.nc, and
trialParams.nc to match a discretized HRU catchment shapefile.

Spatial attribute values are computed using the same methods as
SummaPreProcessor in summa_utils.py:
  - soilTypeIndex / vegTypeIndex  → catchment_intersection shapefiles
  - elevation / area / lon / lat  → catchment shapefile columns
  - aspect                        → np.gradient on the raw DEM (same formula
                                    as _calculate_aspect_from_dem)
  - tan_slope                     → Horn (1981) kernel on the raw DEM (same
                                    formula as _calculate_tan_slope_from_dem)
  - contourLength                 → sqrt(HRU_area) from shapefile attribute
                                    (summa_utils currently defaults to 100 m;
                                    this script uses the actual area instead)

coldState and trialParams are expanded from 1 HRU to N HRUs by broadcasting
the single existing value uniformly to all HRUs.

Usage
-----
    python update_settings_for_hrus.py \\
        --domain-dir /scratch/.../domain_Tuolumne_River_distributed_elev \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import netCDF4 as nc4
import numpy as np
import rasterio
import rasterstats
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backup(path: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
    shutil.copy2(path, dst)
    print(f"  backed up → {dst.name}")


def _find_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Shapefile loaders
# ---------------------------------------------------------------------------

def _load_catchment(domain_dir: Path) -> gpd.GeoDataFrame:
    shp_dir = domain_dir / "shapefiles" / "catchment"
    shp = _find_file(shp_dir, "*_HRUs_elevation*.shp") or _find_file(shp_dir, "*.shp")
    if shp is None:
        raise FileNotFoundError(f"No catchment shapefile in {shp_dir}")
    gdf = gpd.read_file(shp).sort_values("HRU_ID").reset_index(drop=True)
    print(f"Catchment shapefile : {shp.name}  ({len(gdf)} HRUs)")
    return gdf


def _load_soil_intersection(domain_dir: Path) -> Optional[gpd.GeoDataFrame]:
    p = domain_dir / "shapefiles" / "catchment_intersection" / "with_soilgrids"
    shp = _find_file(p, "*.shp")
    if shp is None:
        print("  WARNING: no soil intersection shapefile — soilTypeIndex will use default (6=loam)")
        return None
    gdf = gpd.read_file(shp)
    print(f"Soil shapefile      : {shp.name}")
    return gdf


def _load_land_intersection(domain_dir: Path) -> Optional[gpd.GeoDataFrame]:
    p = domain_dir / "shapefiles" / "catchment_intersection" / "with_landclass"
    shp = _find_file(p, "*.shp")
    if shp is None:
        print("  WARNING: no land intersection shapefile — vegTypeIndex will use default (7=shrublands)")
        return None
    gdf = gpd.read_file(shp)
    print(f"Land shapefile      : {shp.name}")
    return gdf


# ---------------------------------------------------------------------------
# Aspect — matches SummaPreProcessor._calculate_aspect_from_dem exactly
#   np.gradient on raw DEM → arctan2(-dy, dx) → compass bearing → +180° offset
#   → sin/cos circular averaging via rasterstats
# ---------------------------------------------------------------------------

def _compute_aspect(catchment_gdf: gpd.GeoDataFrame,
                    dem_path: Path,
                    azimuth_offset_deg: float = 180.0) -> Dict[int, float]:
    print(f"  Aspect from DEM : {dem_path.name}  (offset={azimuth_offset_deg}°)")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        transform = src.transform
        nodata = src.nodata

    cell_x = abs(transform[0])
    cell_y = abs(transform[4])

    # np.gradient matches summa_utils._calculate_aspect_from_dem
    dy, dx = np.gradient(dem, cell_y, cell_x)
    aspect_rad = np.arctan2(-dy, dx)
    upslope_deg = (90.0 - np.degrees(aspect_rad)) % 360.0
    aspect_deg = (upslope_deg + azimuth_offset_deg) % 360.0

    # Flat-area mask (same condition as summa_utils)
    flat_mask = (np.abs(dx) < 1e-8) & (np.abs(dy) < 1e-8)
    valid_mask = np.isfinite(dem)
    if nodata is not None:
        valid_mask &= (dem != nodata)
    valid_mask &= ~flat_mask

    FILL = -9999.0
    sin_a = np.full(dem.shape, FILL, dtype=np.float64)
    cos_a = np.full(dem.shape, FILL, dtype=np.float64)
    sin_a[valid_mask] = np.sin(np.deg2rad(aspect_deg[valid_mask]))
    cos_a[valid_mask] = np.cos(np.deg2rad(aspect_deg[valid_mask]))

    sin_stats = rasterstats.zonal_stats(catchment_gdf.geometry, sin_a,
                                        affine=transform, stats=["mean"], nodata=FILL)
    cos_stats = rasterstats.zonal_stats(catchment_gdf.geometry, cos_a,
                                        affine=transform, stats=["mean"], nodata=FILL)

    results: Dict[int, float] = {}
    for row, ss, cs in zip(catchment_gdf.itertuples(), sin_stats, cos_stats):
        sv, cv = ss["mean"], cs["mean"]
        if sv is None or cv is None or np.isnan(sv) or np.isnan(cv) or (abs(sv) < 1e-12 and abs(cv) < 1e-12):
            results[int(row.HRU_ID)] = 180.0
        else:
            results[int(row.HRU_ID)] = float((np.degrees(np.arctan2(sv, cv)) + 360.0) % 360.0)
    return results


# ---------------------------------------------------------------------------
# Tan slope — matches SummaPreProcessor._calculate_tan_slope_from_dem exactly
#   Horn (1981) 3×3 kernel, geographic cell-size correction, zonal mean
# ---------------------------------------------------------------------------

def _compute_tan_slope(catchment_gdf: gpd.GeoDataFrame, dem_path: Path) -> Dict[int, float]:
    print(f"  Tan-slope from DEM: {dem_path.name}")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    nodata_mask = ~np.isfinite(dem)
    if nodata is not None:
        nodata_mask |= (dem == nodata)
    dem[nodata_mask] = np.nan

    cell_x = abs(transform[0])
    cell_y = abs(transform[4])
    if crs is not None and crs.is_geographic:
        nrows = dem.shape[0]
        lat_top = transform.f
        lats = lat_top - (np.arange(nrows) + 0.5) * cell_y
        dy_m = cell_y * 111320.0
        dx_m = cell_x * 111320.0 * np.cos(np.deg2rad(lats))[:, np.newaxis]
    else:
        dy_m, dx_m = cell_y, cell_x

    # Horn (1981) 3×3 weighted gradient kernel
    p = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    dz_dx = ((p[0:-2, 2:] + 2*p[1:-1, 2:] + p[2:, 2:]) -
             (p[0:-2, 0:-2] + 2*p[1:-1, 0:-2] + p[2:, 0:-2])) / (8.0 * dx_m)
    dz_dy = ((p[2:, 0:-2] + 2*p[2:, 1:-1] + p[2:, 2:]) -
             (p[0:-2, 0:-2] + 2*p[0:-2, 1:-1] + p[0:-2, 2:])) / (8.0 * dy_m)

    slope = np.maximum(np.sqrt(dz_dx**2 + dz_dy**2), 1e-6)
    slope[nodata_mask] = np.nan

    FILL = -9999.0
    slope_fill = np.where(np.isfinite(slope), slope, FILL)
    stats = rasterstats.zonal_stats(catchment_gdf.geometry, slope_fill,
                                    affine=transform, stats=["mean"], nodata=FILL)

    return {int(row.HRU_ID): float(s["mean"] if s["mean"] is not None else 0.1)
            for row, s in zip(catchment_gdf.itertuples(), stats)}


# ---------------------------------------------------------------------------
# attributes.nc
# ---------------------------------------------------------------------------

def update_attributes(
    catchment_gdf: gpd.GeoDataFrame,
    soil_gdf: Optional[gpd.GeoDataFrame],
    land_gdf: Optional[gpd.GeoDataFrame],
    dem_path: Optional[Path],
    settings_dir: Path,
    dry_run: bool,
) -> None:
    attr_path = settings_dir / "attributes.nc"
    print(f"\n--- attributes.nc ---")

    n_hru = len(catchment_gdf)
    n_gru = int(catchment_gdf["GRU_ID"].nunique())
    hru_ids = catchment_gdf["HRU_ID"].values.astype(np.int32)
    gru_ids = catchment_gdf["GRU_ID"].values.astype(np.int32)

    # soilTypeIndex from intersection shapefile
    soil_idx = np.full(n_hru, 6, dtype=np.int32)
    if soil_gdf is not None and "soilClass" in soil_gdf.columns:
        for i, hid in enumerate(hru_ids):
            row = soil_gdf[soil_gdf["HRU_ID"].astype(int) == int(hid)]
            if not row.empty:
                soil_idx[i] = int(row["soilClass"].iloc[0])

    # vegTypeIndex from intersection shapefile
    veg_idx = np.full(n_hru, 7, dtype=np.int32)
    if land_gdf is not None and "landClass" in land_gdf.columns:
        for i, hid in enumerate(hru_ids):
            row = land_gdf[land_gdf["HRU_ID"].astype(int) == int(hid)]
            if not row.empty:
                veg_idx[i] = int(row["landClass"].iloc[0])

    # contourLength: sqrt(HRU_area) — HRU_area is already in m²
    contour = np.sqrt(catchment_gdf["HRU_area"].values.astype(np.float64))

    # tan_slope and aspect from raw DEM
    if dem_path is not None:
        tan_slope_map = _compute_tan_slope(catchment_gdf, dem_path)
        aspect_map = _compute_aspect(catchment_gdf, dem_path)
    else:
        print("  WARNING: no DEM — using defaults: tan_slope=0.1, aspect=180.0")
        tan_slope_map = {int(h): 0.1 for h in hru_ids}
        aspect_map = {int(h): 180.0 for h in hru_ids}

    tan_slope = np.array([tan_slope_map.get(int(h), 0.1) for h in hru_ids])
    aspect = np.array([aspect_map.get(int(h), 180.0) for h in hru_ids])

    print(f"  HRUs={n_hru}  GRUs={n_gru}")
    for i, hid in enumerate(hru_ids):
        print(f"  HRU {int(hid):2d}: elev={catchment_gdf.iloc[i]['elev_mean']:.0f}m  "
              f"area={catchment_gdf.iloc[i]['HRU_area']:.2e}m²  "
              f"soil={soil_idx[i]}  veg={veg_idx[i]}  "
              f"aspect={aspect[i]:.1f}°  tan_slope={tan_slope[i]:.4f}  "
              f"contour={contour[i]:.0f}m")

    if dry_run:
        print("  [dry-run] no files written"); return

    _backup(attr_path)

    mheight = 2.0
    if attr_path.exists():
        try:
            with xr.open_dataset(attr_path) as s:
                if "mHeight" in s:
                    mheight = float(s["mHeight"].values.flat[0])
        except Exception:
            pass

    with nc4.Dataset(attr_path, "w", format="NETCDF4") as ds:
        ds.setncattr("Author", "update_settings_for_hrus.py")
        ds.setncattr("History", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        ds.createDimension("hru", n_hru)
        ds.createDimension("gru", n_gru)

        def _v(name, dtype, dims, units, long_name, data):
            v = ds.createVariable(name, dtype, dims, fill_value=False)
            v.setncattr("units", units)
            v.setncattr("long_name", long_name)
            v[:] = data

        _v("hruId",         "i4", "hru", "-",     "HRU ID",                        hru_ids)
        _v("gruId",         "i4", "gru", "-",     "GRU ID",                        np.unique(gru_ids))
        _v("hru2gruId",     "i4", "hru", "-",     "GRU ID each HRU belongs to",    gru_ids)
        _v("downHRUindex",  "i4", "hru", "-",     "index of downstream HRU (0=outlet)", np.zeros(n_hru, np.int32))
        _v("longitude",     "f8", "hru", "dd",    "longitude of HRU centroid",     catchment_gdf["center_lon"].values)
        _v("latitude",      "f8", "hru", "dd",    "latitude of HRU centroid",      catchment_gdf["center_lat"].values)
        _v("elevation",     "f8", "hru", "m",     "mean HRU elevation",            catchment_gdf["elev_mean"].values)
        _v("HRUarea",       "f8", "hru", "m^2",   "HRU area",                      catchment_gdf["HRU_area"].values)
        _v("tan_slope",     "f8", "hru", "m m-1", "average tangent slope of HRU",  tan_slope)
        _v("contourLength", "f8", "hru", "m",     "contour length of HRU",         contour)
        _v("slopeTypeIndex","i4", "hru", "-",     "slope type index",              np.ones(n_hru, np.int32))
        _v("soilTypeIndex", "i4", "hru", "-",     "soil type index",               soil_idx)
        _v("vegTypeIndex",  "i4", "hru", "-",     "vegetation type index",         veg_idx)
        _v("mHeight",       "f8", "hru", "m",     "measurement height above ground", np.full(n_hru, mheight))
        _v("aspect",        "f8", "hru", "dd",    "mean aspect (degrees from N)",  aspect)

    print(f"  wrote → {attr_path}")


# ---------------------------------------------------------------------------
# coldState.nc — broadcast 1-HRU values to N HRUs
# ---------------------------------------------------------------------------

def update_cold_state(catchment_gdf: gpd.GeoDataFrame, settings_dir: Path, dry_run: bool) -> None:
    cs_path = settings_dir / "coldState.nc"
    print(f"\n--- coldState.nc ---")
    hru_ids = catchment_gdf["HRU_ID"].values.astype(np.int32)
    n_hru = len(hru_ids)

    # Use nc4 directly so orphan dimensions (e.g. midSoil, which has no variables
    # pointing to it but is queried by name in SUMMA's read_icond) are preserved.
    with nc4.Dataset(cs_path) as _nc:
        dims_src = {k: len(v) for k, v in _nc.dimensions.items()}
    print(f"  hru: 1 → {n_hru}   other dims: { {k: v for k, v in dims_src.items() if k != 'hru'} }")

    if dry_run:
        print("  [dry-run] no files written"); return

    _backup(cs_path)
    with xr.open_dataset(cs_path) as src:
        src_loaded = {name: (da.dims, da.values.copy(), dict(da.attrs))
                      for name, da in src.data_vars.items()}

    with nc4.Dataset(cs_path, "w", format="NETCDF4") as dst:
        dst.setncattr("Author", "update_settings_for_hrus.py")
        dst.setncattr("History", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        for dim, size in dims_src.items():
            dst.createDimension(dim, n_hru if dim == "hru" else size)
        for name, (dims, arr, attrs) in src_loaded.items():
            new_arr = hru_ids if name == "hruId" else arr
            if name != "hruId":
                for axis, dim in enumerate(dims):
                    if dim == "hru" and arr.shape[axis] == 1:
                        new_arr = np.repeat(new_arr, n_hru, axis=axis)
            dtype = "i4" if np.issubdtype(arr.dtype, np.integer) else "f8"
            v = dst.createVariable(name, dtype, dims, fill_value=False)
            for attr, val in attrs.items():
                v.setncattr(attr, val)
            v[:] = new_arr
    print(f"  wrote → {cs_path}")


# ---------------------------------------------------------------------------
# trialParams.nc — broadcast 1-HRU values to N HRUs
# ---------------------------------------------------------------------------

def update_trial_params(catchment_gdf: gpd.GeoDataFrame, settings_dir: Path, dry_run: bool) -> None:
    tp_path = settings_dir / "trialParams.nc"
    print(f"\n--- trialParams.nc ---")
    hru_ids = catchment_gdf["HRU_ID"].values.astype(np.int32)
    gru_unique = np.unique(catchment_gdf["GRU_ID"].values.astype(np.int32))
    n_hru, n_gru = len(hru_ids), len(gru_unique)
    print(f"  hru: 1 → {n_hru}   gru: 1 → {n_gru}")

    if dry_run:
        print("  [dry-run] no files written"); return

    _backup(tp_path)
    with xr.open_dataset(tp_path) as src:
        dims_src = dict(src.sizes)
        src_loaded = {name: (da.dims, da.values.copy(), dict(da.attrs))
                      for name, da in src.data_vars.items()}

    with nc4.Dataset(tp_path, "w", format="NETCDF4") as dst:
        dst.setncattr("Author", "update_settings_for_hrus.py")
        dst.setncattr("History", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        for dim in dims_src:
            dst.createDimension(dim, n_hru if dim == "hru" else n_gru if dim == "gru" else dims_src[dim])
        for name, (dims, arr, attrs) in src_loaded.items():
            if name == "hruId":
                new_arr = hru_ids
            elif name == "gruId":
                new_arr = gru_unique
            else:
                new_arr = arr
                for axis, dim in enumerate(dims):
                    if dim == "hru" and arr.shape[axis] == 1:
                        new_arr = np.repeat(new_arr, n_hru, axis=axis)
                    elif dim == "gru" and arr.shape[axis] == 1:
                        new_arr = np.repeat(new_arr, n_gru, axis=axis)
            dtype = "i4" if np.issubdtype(arr.dtype, np.integer) else "f8"
            v = dst.createVariable(name, dtype, dims, fill_value=False)
            for attr, val in attrs.items():
                v.setncattr(attr, val)
            v[:] = new_arr
    print(f"  wrote → {tp_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Rebuild SUMMA settings files to match HRU shapefile discretization")
    p.add_argument("--domain-dir", required=True,
                   help="Root domain directory")
    p.add_argument("--dem-name", default="",
                   help="Elevation DEM filename in attributes/elevation/dem/ "
                        "(auto-detected as domain_<name>_elv.tif if omitted)")
    p.add_argument("--aspect-offset", type=float, default=180.0,
                   help="Azimuth offset in degrees added to upslope aspect "
                        "(default 180 = downslope-facing, matches ASPECT_AZIMUTH_OFFSET_DEG)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without writing any files")
    args = p.parse_args()

    domain_dir = Path(args.domain_dir).expanduser().resolve()
    settings_dir = domain_dir / "settings" / "SUMMA"
    dem_dir = domain_dir / "attributes" / "elevation" / "dem"

    # Auto-detect raw elevation DEM (exclude pre-computed aspect rasters)
    if args.dem_name:
        dem_path = dem_dir / args.dem_name
    else:
        candidates = sorted(dem_dir.glob("*_elv.tif")) if dem_dir.exists() else []
        dem_path = candidates[0] if candidates else None
        if dem_path:
            print(f"Auto-detected DEM : {dem_path.name}")
        else:
            print("WARNING: no *_elv.tif found — slope/aspect will use defaults")

    catchment_gdf = _load_catchment(domain_dir)
    soil_gdf = _load_soil_intersection(domain_dir)
    land_gdf = _load_land_intersection(domain_dir)

    update_attributes(catchment_gdf, soil_gdf, land_gdf, dem_path, settings_dir, args.dry_run)
    update_cold_state(catchment_gdf, settings_dir, args.dry_run)
    update_trial_params(catchment_gdf, settings_dir, args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
