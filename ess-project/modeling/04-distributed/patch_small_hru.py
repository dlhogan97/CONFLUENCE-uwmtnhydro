#!/usr/bin/env python3
"""
patch_small_hru.py — Fill a tiny/bad HRU's forcing values with those from
an adjacent donor HRU, in all basin_averaged_data netCDF files.

Run this BEFORE preprocess_models so the fix is applied before NaN-filling
and data_step assignment.

Usage
-----
    python patch_small_hru.py \
        --basin-dir /scratch/.../forcing/basin_averaged_data \
        --donor-hru-id 3 \
        --target-hru-id 4 \
        [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm


FORCING_VARS = ["airpres", "LWRadAtm", "SWRadAtm", "pptrate", "airtemp", "spechum", "windspd"]


def patch_file(path: Path, out_path: Path, donor_idx: int, target_idx: int) -> bool:
    """Copy donor HRU slice → target HRU slice for all forcing vars. Returns True if changed."""
    ds = xr.load_dataset(path)

    changed = False
    for var in FORCING_VARS:
        if var not in ds:
            continue
        arr = ds[var].values.copy()        # shape (time, hru)
        donor_vals = arr[:, donor_idx]
        if not np.allclose(arr[:, target_idx], donor_vals, equal_nan=True):
            arr[:, target_idx] = donor_vals
            ds[var] = xr.Variable(ds[var].dims, arr, ds[var].attrs)
            changed = True

    if not changed:
        ds.close()
        return False

    ds.to_netcdf(out_path, mode="w")
    ds.close()
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Fill small/bad HRU forcing with donor HRU values")
    p.add_argument("--basin-dir",      required=True,
                   help="Path to forcing/basin_averaged_data/")
    p.add_argument("--donor-hru-id",   type=int, required=True,
                   help="HRU ID to copy FROM (1-indexed, as in hruId variable)")
    p.add_argument("--target-hru-id",  type=int, required=True,
                   help="HRU ID to overwrite (1-indexed, as in hruId variable)")
    p.add_argument("--output-dir",     default="",
                   help="Write patched files here instead of modifying originals in place. "
                        "Directory is created if it does not exist. Unpatched files are "
                        "symlinked (or copied if --copy-unchanged) so the output dir is "
                        "a complete drop-in replacement for basin_averaged_data.")
    p.add_argument("--copy-unchanged", action="store_true",
                   help="Copy unchanged files to --output-dir instead of symlinking them")
    p.add_argument("--dry-run",        action="store_true",
                   help="Report what would change without writing")
    args = p.parse_args()

    basin_dir = Path(args.basin_dir).expanduser().resolve()
    files = sorted(basin_dir.glob("*.nc"))
    if not files:
        print(f"No .nc files found in {basin_dir}")
        return

    # Determine 0-based array indices from hruId — scan files until we find one intact
    hru_ids = None
    for f in files:
        try:
            ds = xr.load_dataset(f)
            key = "hruId" if "hruId" in ds else None
            if key is None and "hruId" in ds.coords:
                key = "hruId"
            if key and len(ds.data_vars) > 0:
                hru_ids = ds[key].values.astype(int)
                ds.close()
                break
            ds.close()
        except Exception:
            pass

    if hru_ids is None:
        print("ERROR: could not read hruId from any file — all may be corrupted.")
        return

    id_to_idx = {int(h): i for i, h in enumerate(hru_ids)}
    donor_idx  = id_to_idx[args.donor_hru_id]
    target_idx = id_to_idx[args.target_hru_id]

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir : {out_dir}")
    else:
        print("Output dir : in-place (originals overwritten)")

    print(f"Donor  HRU ID={args.donor_hru_id}  → array index {donor_idx}")
    print(f"Target HRU ID={args.target_hru_id} → array index {target_idx}")
    print(f"Files to process: {len(files)}")
    if args.dry_run:
        print("[dry-run] no files will be written\n")

    n_changed = 0
    n_skipped = 0
    for f in tqdm(files, unit="file"):
        out_path = (out_dir / f.name) if out_dir else f
        try:
            if args.dry_run:
                ds = xr.load_dataset(f)
                has_data = any(v in ds for v in FORCING_VARS)
                ds.close()
                if has_data:
                    n_changed += 1
            else:
                patched = patch_file(f, out_path, donor_idx, target_idx)
                if patched:
                    n_changed += 1
                elif out_dir and not patched:
                    # File unchanged — symlink or copy into output dir
                    if not out_path.exists():
                        if args.copy_unchanged:
                            import shutil
                            shutil.copy2(f, out_path)
                        else:
                            out_path.symlink_to(f)
        except Exception as e:
            tqdm.write(f"  SKIP {f.name}: {e}")
            n_skipped += 1

    print(f"\n{'Would patch' if args.dry_run else 'Patched'} {n_changed}/{len(files)} files.")
    if n_skipped:
        print(f"Skipped (corrupted): {n_skipped} — listed above.")
    if out_dir:
        print(f"\nTo use patched forcing, point FORCING_SUMMA_PATH (or basin_averaged_data) "
              f"at:\n  {out_dir}")


if __name__ == "__main__":
    main()
