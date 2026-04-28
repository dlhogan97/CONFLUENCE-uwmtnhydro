#!/usr/bin/env python3
"""
prep_metsim_forcing.py — Extract single-HRU METSIM forcing for SUMMA.

The basin_averaged_data METSIM files have 2 HRUs: index 0 is all NaN
(artifact polygon), index 1 has the real basin-averaged data. This script:
  1. Extracts HRU index 1 from each monthly METSIM file.
  2. Reassigns hruId=1 to match the Tuolumne lumped attributes.nc.
  3. Re-encodes time as "seconds since 1990-01-01 00:00:00" (SUMMA standard).
  4. Adds the scalar data_step=3600 variable.
  5. Replaces LWRadAtm with the Dilley & O'Brien (1998) empirical estimate
     computed from airtemp, airpres, and spechum (METSIM LW is unreliable).
  6. Writes SUMMA-ready single-HRU files to metsim_SUMMA_input/.

Run once before the parameter sweep. Only processes the files needed for
the simulation period to keep the output directory small.

Usage:
    python prep_metsim_forcing.py [--all] [--force]

    --all     Process all METSIM files in basin_averaged_data/ instead of
              only those needed for the sweep period (201210–201709).
    --force   Overwrite existing output files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "utils" / "custom"))
from calc import empirical_lw_dilley_obrien

BASIN_DIR = Path(
    "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_lumped"
    "/forcing/basin_averaged_data"
)
OUT_DIR = Path(
    "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_lumped"
    "/forcing/metsim_SUMMA_input"
)

FORCING_VARS = ["airpres", "LWRadAtm", "SWRadAtm", "pptrate", "airtemp", "spechum", "windspd"]
GOOD_HRU_IDX = 1          # HRU index 1 = real basin-averaged data
TARGET_HRU_ID = 1         # hruId to assign (matches attributes.nc)
SUMMA_TIME_UNITS = "seconds since 1990-01-01 00:00:00"
SUMMA_CALENDAR = "standard"

# Months needed for sweep: SIM_START=2012-10-01, SIM_END=2017-09-30
SWEEP_MONTHS = {
    f"{y}{m:02d}"
    for y in range(2012, 2018)
    for m in range(1, 13)
    if not (y == 2012 and m < 10) and not (y == 2017 and m > 9)
}


def process_file(src: Path, dst: Path) -> None:
    ds = xr.open_dataset(src, decode_times=True)

    # Extract the good HRU slice → shape (time,) then expand to (time, hru=1)
    single = ds.isel(hru=GOOD_HRU_IDX)

    # Build output dataset with (hru=1, time) dimension ordering SUMMA expects
    time_coord = single["time"]

    data_vars = {}
    for v in FORCING_VARS:
        if v not in single:
            continue
        arr = single[v].values  # shape (time,)
        data_vars[v] = xr.Variable(
            ["time", "hru"],
            arr[:, np.newaxis],
            ds[v].attrs,
        )

    # Replace LWRadAtm with Dilley & O'Brien (1998) empirical estimate.
    # METSIM LW output is not reliable; this formula uses T, p, q instead.
    lw_new = empirical_lw_dilley_obrien(
        single["airtemp"].values,
        single["airpres"].values,
        single["spechum"].values,
    )
    data_vars["LWRadAtm"] = xr.Variable(
        ["time", "hru"],
        lw_new[:, np.newaxis].astype(np.float64),
        {"long_name": "incoming longwave radiation", "units": "W m-2"},
    )

    # lat/lon — take scalar from the good HRU
    for aux in ("latitude", "longitude"):
        if aux in single:
            data_vars[aux] = xr.Variable(
                ["hru"],
                np.atleast_1d(float(single[aux].values)),
                ds[aux].attrs if aux in ds else {},
            )

    data_vars["hruId"] = xr.Variable(["hru"], np.array([TARGET_HRU_ID], dtype=np.int32), {})
    data_vars["data_step"] = xr.Variable([], np.int32(3600), {"units": "seconds", "long_name": "data step"})

    out = xr.Dataset(data_vars, coords={"time": time_coord})
    ds.close()

    # Write with SUMMA-standard time encoding
    encoding = {
        "time": {
            "dtype": "float64",
            "units": SUMMA_TIME_UNITS,
            "calendar": SUMMA_CALENDAR,
        }
    }
    out.to_netcdf(dst, mode="w", encoding=encoding)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep single-HRU METSIM forcing for SUMMA")
    parser.add_argument("--all",   action="store_true", help="Process all METSIM files (not just sweep period)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(BASIN_DIR.glob("*_METSIM_*.nc"))
    if not args.all:
        files = [
            f for f in files
            if any(month in f.name for month in SWEEP_MONTHS)
        ]

    print(f"Source : {BASIN_DIR}")
    print(f"Output : {OUT_DIR}")
    print(f"Files  : {len(files)} to process")
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
    print(f"\nAdd to forcingFileList.txt:")
    for f in sorted(OUT_DIR.glob("*.nc")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
