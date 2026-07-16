#!/usr/bin/env python3
"""
add_data_step.py — Add the scalar `data_step` variable to SUMMA forcing files.

The July METSIM regeneration dropped `data_step` from the forcing.  Without it
SUMMA reports "number of time steps = 1", the time-delay routing histogram
collapses to a zero fraction, and the solver fails to converge on the first
step.  The variable is a scalar (no dimensions) giving the forcing timestep in
seconds.

Usage
-----
    python add_data_step.py --forcing-dir /scratch/.../forcing/SUMMA_input [--step 3600]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np


def add_data_step(path: Path, step: int) -> str:
    with nc.Dataset(path, "a") as ds:
        if "data_step" in ds.variables:
            existing = int(ds.variables["data_step"][...])
            if existing == step:
                return "ok"
            ds.variables["data_step"][...] = step
            return "updated"
        v = ds.createVariable("data_step", "i8", ())
        v.long_name = "data step length in seconds"
        v.units = "s"
        v[...] = step
    return "added"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forcing-dir", required=True, type=Path)
    p.add_argument("--step", type=int, default=3600)
    args = p.parse_args()

    files = sorted(args.forcing_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc under {args.forcing_dir}")

    counts: dict[str, int] = {}
    for f in files:
        r = add_data_step(f, args.step)
        counts[r] = counts.get(r, 0) + 1

    print(f"{args.forcing_dir}:  {len(files)} files  ->  {counts}")

    # verify
    with nc.Dataset(files[0]) as ds:
        assert "data_step" in ds.variables, "data_step still missing!"
        print(f"  verified: data_step = {int(ds.variables['data_step'][...])} s")


if __name__ == "__main__":
    main()
