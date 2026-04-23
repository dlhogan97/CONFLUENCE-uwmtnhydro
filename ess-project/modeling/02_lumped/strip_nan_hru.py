#!/usr/bin/env python3
"""
strip_nan_hru.py — Remove the all-NaN HRU from Tuolumne basin_averaged_data files.

Each file has 2 HRUs: index 0 is an artifact polygon (all NaN), index 1 is the
real basin-averaged data. This script drops index 0 and reassigns hruId=1 so
downstream tools see a clean single-HRU file.

Modifies files in-place (writes to a temp file then replaces the original).

Usage:
    python strip_nan_hru.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

BASIN_DIR = Path(
    "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_lumped"
    "/forcing/basin_averaged_data"
)

GOOD_HRU_IDX = 1   # index 1 = real data
TARGET_HRU_ID = 1  # reassign so hruId matches attributes.nc


def strip_file(src: Path, dry_run: bool = False) -> str:
    """Strip NaN HRU from a single file. Returns 'already_fixed' or 'fixed'."""
    ds = xr.open_dataset(src, decode_times=False)

    n_hru = ds.sizes.get("hru", 1)
    if n_hru == 1:
        ds.close()
        return "already_fixed"

    # isel with a slice keeps the hru dimension (size 1) instead of collapsing it
    out = ds.isel(hru=slice(GOOD_HRU_IDX, GOOD_HRU_IDX + 1))

    # Reassign hruId so it matches attributes.nc
    if "hruId" in out:
        out["hruId"] = xr.Variable(
            out["hruId"].dims,
            np.array([TARGET_HRU_ID], dtype=ds["hruId"].dtype),
            ds["hruId"].attrs,
        )

    enc = {"time": {k: v for k, v in ds["time"].encoding.items()
                    if k in ("units", "calendar", "dtype")}}
    ds.close()

    if dry_run:
        return "fixed"

    # Write to temp file beside the original, then atomically replace
    fd, tmp = tempfile.mkstemp(dir=src.parent, suffix=".tmp.nc")
    os.close(fd)
    try:
        out.to_netcdf(tmp, mode="w", encoding=enc)
        os.replace(tmp, src)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

    return "fixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip NaN HRU from basin_averaged_data files")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()

    files = sorted(BASIN_DIR.glob("*_METSIM_*.nc"))
    print(f"Directory : {BASIN_DIR}")
    print(f"Files     : {len(files)}")
    if args.dry_run:
        print("[dry-run] no files will be written\n")

    counts = {"fixed": 0, "already_fixed": 0, "error": 0}
    for f in tqdm(files, unit="file"):
        try:
            result = strip_file(f, dry_run=args.dry_run)
            counts[result] = counts.get(result, 0) + 1
        except Exception as e:
            tqdm.write(f"  ERROR {f.name}: {e}")
            counts["error"] += 1

    verb = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{verb}         : {counts.get('fixed', 0)}")
    print(f"Already 1-HRU : {counts.get('already_fixed', 0)}")
    if counts["error"]:
        print(f"Errors        : {counts['error']}")


if __name__ == "__main__":
    main()
