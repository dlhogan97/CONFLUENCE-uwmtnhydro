#!/usr/bin/env python3
"""
prep_distributed_lw_correction.py — Replace METSIM LWRadAtm with Dilley & O'Brien (1998)
empirical estimate for a multi-HRU distributed SUMMA forcing directory.

METSIM longwave output is unreliable; this replaces it using air temperature,
pressure, and specific humidity following the same approach as prep_metsim_forcing.py
for the lumped domain.

Input:  any SUMMA-ready forcing directory with (time, hru) netCDF files
Output: same files with LWRadAtm replaced, written to OUT_DIR

Usage:
    python prep_distributed_lw_correction.py [--force]

    --force   Overwrite existing output files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "utils" / "custom"))
from calc import empirical_lw_dilley_obrien

IN_DIR = Path(
    "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elev"
    "/forcing/SUMMA_input"
)
OUT_DIR = Path(
    "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elev"
    "/forcing/SUMMA_input_lwcorr"
)

SUMMA_TIME_UNITS = "seconds since 1990-01-01 00:00:00"
SUMMA_CALENDAR = "standard"


def process_file(src: Path, dst: Path) -> None:
    # decode_times=False keeps raw float64 time values so encoding round-trips cleanly
    ds = xr.open_dataset(src, decode_times=False)

    lw_new = empirical_lw_dilley_obrien(
        ds["airtemp"].values,   # (time, hru) float32 — vectorized
        ds["airpres"].values,
        ds["spechum"].values,
    )

    ds_out = ds.copy()
    ds_out["LWRadAtm"] = xr.Variable(
        ds["LWRadAtm"].dims,
        lw_new.astype(np.float32),
        {"long_name": "incoming longwave radiation", "units": "W m-2",
         "source": "Dilley & O'Brien (1998) empirical from T, p, q"},
    )

    encoding = {"time": {"dtype": "float64"}}
    ds_out.to_netcdf(dst, mode="w", encoding=encoding)
    ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace METSIM LWRadAtm with Dilley & O'Brien for distributed SUMMA forcing"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.nc"))
    print(f"Source : {IN_DIR}")
    print(f"Output : {OUT_DIR}")
    print(f"Files  : {len(files)}")
    print(f"LW     : Dilley & O'Brien (1998) empirical from T, p, q")

    n_ok = n_skip = 0
    for src in tqdm(files, unit="file"):
        dst = OUT_DIR / src.name
        if dst.exists() and not args.force:
            n_skip += 1
            continue
        try:
            process_file(src, dst)
            n_ok += 1
        except Exception as e:
            tqdm.write(f"  SKIP {src.name}: {e}")

    print(f"\nDone. Written={n_ok}  Skipped(exist)={n_skip}")


if __name__ == "__main__":
    main()
