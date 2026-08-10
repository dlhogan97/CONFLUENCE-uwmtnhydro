#!/usr/bin/env python3
"""
download_gridded_et.py — Download OpenET as a lat/lon GRID over a basin.

Unlike ``download_distributed_et.py`` (which asks OpenET to spatially reduce ET
per HRU via one polygon API call each), this script pulls the native gridded ET
raster over the whole basin *once*. The grid can then be area-weight-remapped
onto any HRU discretization (elevAspect, elevTPI, elevation) with easymore --
see ``remap_et_to_hrus.py``.

Endpoint: OpenET ``/raster/geotiff/stack`` (monthly, ensemble ET).
Limits handled automatically:
  * 31 timesteps / request  -> the period is chunked into <=31-month blocks.
  * per-request area cap (50k acres free / 200k acres tier-2) -> the whole-basin
    polygon is tried first; on an area-limit error the bounding box is tiled
    into rectangles under the safe cap and mosaicked back together.

Output (per basin):
  <domain_*_elevAspect>/observations/et/
      openet_et_ensemble_<basin>_gridded_monthly.nc   (time, lat, lon; mm/month)

Usage
-----
    # See the plan (tiles, requests, quota) WITHOUT calling the API:
    python download_gridded_et.py --domain East --dry-run

    # Download the full available monthly record:
    python download_gridded_et.py --domain East
    python download_gridded_et.py --domain Tuolumne

    # Custom window / coarsening:
    python download_gridded_et.py --domain East --start-date 2016-01-01 \
        --end-date 2020-12-31 --coarsen 8
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root so utils.data.openet_utils is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from utils.data.openet_utils import OpenETClient, AreaLimitError  # noqa: E402

# ---------------------------------------------------------------------------
# Basin configurations. The gridded download depends only on the basin
# boundary, so one download per basin serves every HRU discretization.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/scratch/dlhogan/ess-project-data")

BASIN_CONFIGS = {
    "East": {
        "boundary_shp": DATA_ROOT / "domain_East_River_distributed_elevAspect/shapefiles/catchment/East_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
        "output_dir": DATA_ROOT / "domain_East_River_distributed_elevAspect/observations/et",
        "grid_filename": "openet_et_ensemble_East_gridded_monthly.nc",
        "label": "East River",
    },
    "Tuolumne": {
        "boundary_shp": DATA_ROOT / "domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
        "output_dir": DATA_ROOT / "domain_Tuolumne_River_distributed_elevAspect/observations/et",
        "grid_filename": "openet_et_ensemble_Tuolumne_gridded_monthly.nc",
        "label": "Tuolumne River",
    },
}

ENV_FILE = REPO_ROOT / ".env"

# OpenET Landsat-era ET is reliable from ~2016; data lags real time ~1-2 months.
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_COARSEN = 8  # 8 * ~30 m ~= 240 m; keeps the easymore source grid light


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _month_chunks(start: str, end: str, max_steps: int = OpenETClient.MAX_STEPS_PER_STACK):
    """Yield (chunk_start, chunk_end) ISO strings each spanning <= max_steps months."""
    from dateutil.relativedelta import relativedelta

    s = date.fromisoformat(start).replace(day=1)
    e = date.fromisoformat(end).replace(day=1)
    cur = s
    while cur <= e:
        chunk_end = min(cur + relativedelta(months=max_steps - 1), e)
        # end date = last day of chunk_end's month
        last = chunk_end + relativedelta(months=1) - relativedelta(days=1)
        yield cur.isoformat(), last.isoformat()
        cur = chunk_end + relativedelta(months=1)


def _default_end_date() -> str:
    """Latest complete month, minus a 2-month processing lag."""
    from dateutil.relativedelta import relativedelta

    lagged = (date.today().replace(day=1) - relativedelta(months=2))
    last = lagged + relativedelta(months=1) - relativedelta(days=1)
    return last.isoformat()


def _load_boundary(shp_path: Path):
    """Return (dissolved_polygon_geojson, bounds, area_acres, shapely_polygon) in EPSG:4326."""
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError(f"{shp_path} has no CRS")
    gdf_ea = gdf.to_crs(5070)
    area_acres = gdf_ea.geometry.area.sum() / 4046.8564224
    gdf_wgs = gdf.to_crs(4326)
    geom = gdf_wgs.dissolve().geometry.iloc[0]
    import json
    geojson = json.loads(gpd.GeoSeries([geom], crs=4326).to_json())["features"][0]["geometry"]
    return geojson, tuple(gdf_wgs.total_bounds), area_acres, geom


def _tiles_intersecting_basin(tiles, basin_poly):
    """Drop bbox tiles that don't touch the basin; submit the rest as rectangles.

    Returns a list of (geometry_coords, tag). We submit the *rectangle* (5
    vertices), not the basin-clipped polygon: the clipped boundary has hundreds
    of vertices and the OpenET API intermittently rejects it with
    ``400 Invalid geometry``. Rectangles are already sized under the area cap by
    ``tile_bounds`` and parse reliably; grid cells outside the basin come back as
    nodata and are ignored by easymore.
    """
    from shapely.geometry import box

    kept = []
    for i, (minx, miny, maxx, maxy) in enumerate(tiles):
        if box(minx, miny, maxx, maxy).intersection(basin_poly).is_empty:
            continue
        coords = OpenETClient.bbox_from_bounds(minx, miny, maxx, maxy)
        kept.append((coords, f"t{i}"))
    return kept


def _chunk_month_starts(cs: str, ce: str):
    """List of month-start date strings (YYYY-MM-01) spanned by a chunk."""
    from dateutil.relativedelta import relativedelta

    m = date.fromisoformat(cs).replace(day=1)
    end = date.fromisoformat(ce).replace(day=1)
    out = []
    while m <= end:
        out.append(m.isoformat())
        m += relativedelta(months=1)
    return out


def _plan_tiles(bounds, area_acres, max_acres):
    """Decide the request geometry strategy.

    Returns (mode, geometries) where mode is 'whole' (single basin polygon) or
    'tiled' (list of bbox rectangles). We always *plan* the tiled fallback so
    --dry-run can report the worst case; the actual download tries 'whole' first.
    """
    minx, miny, maxx, maxy = bounds
    tiles = OpenETClient.tile_bounds(minx, miny, maxx, maxy, max_acres=max_acres)
    return tiles


# ---------------------------------------------------------------------------
# GeoTIFF stack -> NetCDF grid assembly
# ---------------------------------------------------------------------------
def _tif_to_dataarray(tif_paths):
    """Open one or more single-band GeoTIFFs (same date), mosaic, return 2-D DataArray.

    nodata (-inf) is converted to NaN.
    """
    import rioxarray
    from rioxarray.merge import merge_arrays

    arrs = []
    for p in tif_paths:
        da = rioxarray.open_rasterio(p, masked=True).squeeze("band", drop=True)
        arrs.append(da)
    da = arrs[0] if len(arrs) == 1 else merge_arrays(arrs)
    da = da.where(np.isfinite(da))
    return da


def _build_grid_netcdf(date_to_tifs: dict, out_path: Path, coarsen: int, label: str):
    """Assemble {date: [tif,...]} into a (time, lat, lon) NetCDF of ET (mm/month)."""
    import xarray as xr

    times = sorted(date_to_tifs)
    layers = []
    ref = None
    for d in times:
        da = _tif_to_dataarray(date_to_tifs[d])
        if coarsen and coarsen > 1:
            da = da.coarsen(x=coarsen, y=coarsen, boundary="trim").mean(skipna=True)
        # align all layers to the first grid to guard against off-by-one merges
        if ref is None:
            ref = da
        else:
            da = da.reindex_like(ref, method="nearest", tolerance=abs(float(ref.x[1] - ref.x[0])) / 2)
        layers.append(da)

    cube = xr.concat(layers, dim=pd.DatetimeIndex(times, name="time"))
    cube = cube.rename("et").rename({"x": "lon", "y": "lat"})
    # strip rioxarray metadata that isn't NetCDF-serializable
    for k in ("grid_mapping", "_FillValue", "AREA_OR_POINT"):
        cube.attrs.pop(k, None)
    cube.attrs.update({
        "long_name": "OpenET ensemble evapotranspiration",
        "units": "mm/month",
        "source": "OpenET /raster/geotiff/stack, model=ensemble, reference_et=gridmet",
        "basin": label,
    })
    cube["lon"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})
    cube["lat"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    ds = cube.to_dataset()
    # drop the rio CRS coord (spatial_ref) and any other stray coords
    for v in list(ds.coords):
        if v not in ("time", "lat", "lon"):
            ds = ds.drop_vars(v)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"et": {"zlib": True, "complevel": 4, "_FillValue": -9999.0}}
    ds.to_netcdf(out_path, encoding=encoding)
    return ds


# ---------------------------------------------------------------------------
# Main download flow
# ---------------------------------------------------------------------------
def run(basin: str, start_date: str, end_date: str, coarsen: int,
        max_acres: float, keep_tifs: bool, dry_run: bool) -> None:
    if basin not in BASIN_CONFIGS:
        raise ValueError(f"Unknown basin '{basin}'. Choose from {list(BASIN_CONFIGS)}")
    cfg = BASIN_CONFIGS[basin]
    shp = Path(cfg["boundary_shp"])
    if not shp.exists():
        print(f"Boundary shapefile not found: {shp}")
        sys.exit(1)

    geojson, bounds, area_acres, basin_poly = _load_boundary(shp)
    chunks = list(_month_chunks(start_date, end_date))
    tiles = _plan_tiles(bounds, area_acres, max_acres)
    kept_tiles = _tiles_intersecting_basin(tiles, basin_poly)  # clipped, empties dropped

    n_months = sum(
        (date.fromisoformat(ce).year - date.fromisoformat(cs).year) * 12
        + (date.fromisoformat(ce).month - date.fromisoformat(cs).month) + 1
        for cs, ce in chunks
    )
    print(f"=== Gridded OpenET plan: {cfg['label']} ===")
    print(f"  Period:        {start_date} -> {end_date}  ({n_months} months)")
    print(f"  Basin area:    {area_acres:,.0f} acres  (bbox {bounds})")
    print(f"  Time chunks:   {len(chunks)}  (<= {OpenETClient.MAX_STEPS_PER_STACK} months each)")
    print(f"  Whole-basin fits free-tier 50k cap? {'yes' if area_acres <= 50000 else 'no'}")
    print(f"  Whole-basin fits tier-2 200k cap?   {'yes' if area_acres <= 200000 else 'no'}")
    print(f"  Fallback tiling (bbox @ <= {max_acres:,.0f} ac): {len(tiles)} tiles, "
          f"{len(kept_tiles)} intersect the basin")
    best = len(chunks)                          # tier-2 / whole-basin: 1 request/chunk
    worst = len(chunks) * len(kept_tiles)       # free-tier fallback: kept tiles/chunk
    print(f"  API requests:  best {best} (whole-basin, tier-2)  /  worst {worst} (tiled, free tier)")
    print(f"  Monthly quota: 100 (free) / 400 (tier-2)")

    if dry_run:
        print("\n[dry-run] no API calls made.")
        return

    if not ENV_FILE.exists():
        print(f"OpenET API key not found. Copy .env.template -> .env and add your key.")
        sys.exit(1)

    client = OpenETClient(env_file=str(ENV_FILE))
    out_dir = Path(cfg["output_dir"])
    tif_dir = out_dir / "_gridded_tifs"
    tif_dir.mkdir(parents=True, exist_ok=True)

    whole_coords = OpenETClient._flatten_polygon(geojson)
    # Auto-detect tier: always try the whole-basin polygon first. On tier-2 this
    # succeeds (area cap 200k). On the free tier the first chunk raises
    # AreaLimitError, after which we tile for this and all remaining chunks.
    use_tiled = False
    date_to_tifs: dict = {}

    for ci, (cs, ce) in enumerate(chunks, 1):
        print(f"\n--- chunk {ci}/{len(chunks)}: {cs} -> {ce} ---")
        if not use_tiled:
            try:
                got = client.download_geotiff_stack(
                    whole_coords, cs, ce, output_dir=tif_dir, prefix="whole_",
                )
                for d, p in got.items():
                    date_to_tifs.setdefault(d, []).append(p)
                continue
            except AreaLimitError as e:
                print(f"  whole-basin exceeded area cap ({e}); switching to tiled mode")
                use_tiled = True
        # tiled path (empties already dropped; rectangles submitted)
        months = _chunk_month_starts(cs, ce)
        for ti, (coords, tag) in enumerate(kept_tiles):
            # Resume: skip the API call if every month's tif already exists.
            expected = {m: tif_dir / f"{tag}_et_ensemble_{m}.tif" for m in months}
            if all(p.exists() for p in expected.values()):
                print(f"    tile {ti + 1}/{len(kept_tiles)} ({tag}) — cached, skipping")
                for m, p in expected.items():
                    date_to_tifs.setdefault(m, []).append(p)
                continue
            print(f"    tile {ti + 1}/{len(kept_tiles)} ({tag})")
            got = client.download_geotiff_stack(
                coords, cs, ce, output_dir=tif_dir, prefix=f"{tag}_",
            )
            for d, p in got.items():
                date_to_tifs.setdefault(d, []).append(p)
            time.sleep(3)  # stay well under 20 req/min

    if not date_to_tifs:
        print("No data downloaded.")
        return

    out_path = out_dir / cfg["grid_filename"]
    print(f"\nAssembling {len(date_to_tifs)} monthly layers -> {out_path} (coarsen={coarsen})")
    ds = _build_grid_netcdf(date_to_tifs, out_path, coarsen, cfg["label"])
    print(f"  grid shape: {dict(ds.sizes)}")
    et = ds["et"]
    print(f"  ET mm/month  min={float(et.min()):.1f} mean={float(et.mean()):.1f} max={float(et.max()):.1f}")

    if not keep_tifs:
        import shutil
        shutil.rmtree(tif_dir, ignore_errors=True)

    print(f"\nWrote {out_path}")
    print(f"Next: python remap_et_to_hrus.py --domain {basin}_elevAspect")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", "--basin", dest="basin", required=True,
                    choices=list(BASIN_CONFIGS), help="Basin to download")
    ap.add_argument("--start-date", default=DEFAULT_START_DATE)
    ap.add_argument("--end-date", default=None, help="default: latest complete month (~2 mo lag)")
    ap.add_argument("--coarsen", type=int, default=DEFAULT_COARSEN,
                    help="spatial coarsening factor for the easymore source grid (1 = native 30 m)")
    ap.add_argument("--max-acres", type=float, default=OpenETClient.SAFE_MAX_ACRES,
                    help="per-tile area cap for the free-tier fallback")
    ap.add_argument("--keep-tifs", action="store_true", help="keep the raw per-month GeoTIFFs")
    ap.add_argument("--dry-run", action="store_true", help="print the request plan and exit")
    args = ap.parse_args()

    end_date = args.end_date or _default_end_date()
    run(args.basin, args.start_date, end_date, args.coarsen,
        args.max_acres, args.keep_tifs, args.dry_run)


if __name__ == "__main__":
    main()
