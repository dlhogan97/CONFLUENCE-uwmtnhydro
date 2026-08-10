#!/usr/bin/env python3
"""
remap_et_to_hrus.py — Area-weight the gridded OpenET onto HRU polygons via easymore.

Takes the basin-scale gridded ET NetCDF produced by ``download_gridded_et.py``
and remaps it onto a specific HRU discretization (elevAspect, elevTPI,
elevation) using EASYMORE's area-weighted intersection. Because the grid is
basin-scale, the same download serves every discretization -- only the target
shapefile changes.

Outputs (per domain), into <domain>/observations/et/:
  openet_et_gridded_remapped_<domain>_monthly.nc   (time, hru; mm/month)
  openet_et_gridded_remapped_<domain>_monthly.csv  (date, hru_<HRU_ID>, ...)
The EASYMORE remap-weights file is cached and reused across runs.

Usage
-----
    python remap_et_to_hrus.py --domain East_elevAspect
    python remap_et_to_hrus.py --domain Tuolumne_elevAspect
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_gridded_et import BASIN_CONFIGS, DATA_ROOT  # noqa: E402

# ---------------------------------------------------------------------------
# Discretization targets. `basin` points at the gridded download; `target_shp`
# is the HRU shapefile to remap onto.
# ---------------------------------------------------------------------------
DOMAIN_CONFIGS = {
    "East_elevAspect": {
        "basin": "East",
        "target_shp": DATA_ROOT / "domain_East_River_distributed_elevAspect/shapefiles/catchment/East_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
        "output_dir": DATA_ROOT / "domain_East_River_distributed_elevAspect/observations/et",
        "hru_id": "HRU_ID",
    },
    "Tuolumne_elevAspect": {
        "basin": "Tuolumne",
        "target_shp": DATA_ROOT / "domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
        "output_dir": DATA_ROOT / "domain_Tuolumne_River_distributed_elevAspect/observations/et",
        "hru_id": "HRU_ID",
    },
}


def _ensure_wgs84(shp_path: Path, tmp_dir: Path) -> Path:
    """Return a path to the shapefile in EPSG:4326 (reprojecting into tmp if needed)."""
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() == 4326:
        return shp_path
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{shp_path.stem}_wgs84.shp"
    gdf.to_crs(4326).to_file(out)
    return out


def remap(domain: str, coarsen_note: str = "") -> None:
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from {list(DOMAIN_CONFIGS)}")
    cfg = DOMAIN_CONFIGS[domain]
    basin_cfg = BASIN_CONFIGS[cfg["basin"]]

    source_nc = Path(basin_cfg["output_dir"]) / basin_cfg["grid_filename"]
    if not source_nc.exists():
        print(f"Gridded ET NetCDF not found: {source_nc}")
        print(f"Run first: python download_gridded_et.py --domain {cfg['basin']}")
        sys.exit(1)

    target_shp = Path(cfg["target_shp"])
    if not target_shp.exists():
        print(f"HRU shapefile not found: {target_shp}")
        sys.exit(1)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / f"_easymore_tmp_{domain}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    target_wgs84 = _ensure_wgs84(target_shp, temp_dir)

    from easymore import Easymore

    case = f"openet_et_gridded_{domain}"
    esmr = Easymore()
    esmr.author_name = "CONFLUENCE ess-project"
    esmr.license = "OpenET data (https://openetdata.org)"
    esmr.case_name = case

    esmr.source_nc = str(source_nc)
    esmr.var_names = ["et"]
    esmr.var_lon = "lon"
    esmr.var_lat = "lat"
    esmr.var_time = "time"

    esmr.target_shp = str(target_wgs84)
    esmr.target_shp_ID = cfg["hru_id"]

    esmr.temp_dir = str(temp_dir) + "/"
    esmr.output_dir = str(out_dir) + "/"
    esmr.remapped_dim_id = "hru"
    esmr.remapped_var_id = "hruId"
    esmr.format_list = ["f4"]
    esmr.fill_value_list = ["-9999"]
    esmr.save_csv = False
    esmr.sort_ID = False
    esmr.clip_source_shp = True          # trim source grid to the basin -> faster
    esmr.skip_outside_shape = True

    # Reuse cached remap weights if present.
    remap_file = out_dir / f"{case}_remapping.nc"
    if remap_file.exists():
        print(f"Reusing cached remap weights: {remap_file.name}")
        esmr.remap_nc = str(remap_file)

    print(f"Remapping {source_nc.name} -> {target_shp.name} ({cfg['hru_id']}) via easymore {coarsen_note}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        esmr.nc_remapper()

    # easymore writes <case>_remapped_<source_stem>.nc into output_dir
    remapped = sorted(out_dir.glob(f"{case}_remapped_*.nc"))
    if not remapped:
        print("EASYMORE produced no remapped file; check logs.")
        sys.exit(1)
    remapped_nc = remapped[-1]

    # Move/keep the remap weights for reuse
    tmp_remap = temp_dir / f"{case}_remapping.nc"
    if tmp_remap.exists() and not remap_file.exists():
        import shutil
        shutil.copy(str(tmp_remap), str(remap_file))

    _write_wide_csv(remapped_nc, out_dir / f"{case}_monthly.csv", cfg["hru_id"])

    # tidy temp
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\nWrote:\n  {remapped_nc}\n  {out_dir / (case + '_monthly.csv')}\n  {remap_file} (weights)")


def _write_wide_csv(remapped_nc: Path, csv_path: Path, hru_id_field: str) -> None:
    """Convert the easymore (time, hru) NetCDF to a wide CSV: date, hru_<id>, ..."""
    import xarray as xr

    ds = xr.open_dataset(remapped_nc)
    # HRU identifiers live in 'hruId' (remapped_var_id)
    if "hruId" in ds:
        hru_ids = np.asarray(ds["hruId"].values).ravel()
    else:
        hru_ids = np.arange(ds.sizes["hru"])
    et = ds["et"].values  # (time, hru)
    times = pd.to_datetime(ds["time"].values)
    df = pd.DataFrame(et, index=times, columns=[f"hru_{int(h)}" for h in hru_ids])
    df.index.name = "date"
    df.reset_index().to_csv(csv_path, index=False)
    ds.close()
    print(f"  wide CSV: {df.shape[0]} months x {df.shape[1]} HRUs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=list(DOMAIN_CONFIGS))
    args = ap.parse_args()
    remap(args.domain)


if __name__ == "__main__":
    main()
