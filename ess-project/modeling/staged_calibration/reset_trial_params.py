#!/usr/bin/env python3
"""Reset trialParams.nc to clean physical starting values for staged calibration.

The optimizer searches  param_hru_i = base_i × spatial_weight_i × M.
Base values here are domain-mean references — spatial weights in the stage
YAML files carry the per-HRU differentiation.  Setting sensible bases ensures
the multiplier search bounds map to physically meaningful absolute ranges.

Usage
-----
    python reset_trial_params.py                        # East River distributed (bigBuckt)
    python reset_trial_params.py --mode qtopmodel       # East River distributed (qTopmodl)
    python reset_trial_params.py --dry-run              # print values, don't write
    python reset_trial_params.py --output /path/to/trialParams.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Target file — override with --output
# ---------------------------------------------------------------------------
DEFAULT_PATH = Path(
    "/scratch/dlhogan/ess-project-data"
    "/domain_East_River_distributed/settings/SUMMA/trialParams.nc"
)

# ---------------------------------------------------------------------------
# HRU ordering (elevation descending, 5 HRUs)
#   idx 0 → ~3994 m  barren/alpine (MODIS 16)
#   idx 1 → ~3696 m  open shrublands (MODIS 7)
#   idx 2 → ~3364 m  evergreen forest (MODIS 1)
#   idx 3 → ~3010 m  evergreen forest (MODIS 1)
#   idx 4 → ~2705 m  open shrublands (MODIS 7)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 1 — Snow
# Uniform across HRUs: spatial variation handled by SUMMA slope/aspect geometry.
# ---------------------------------------------------------------------------
SNOW_PARAMS = {
    "albedoDecayRate":    1.19e5,   # s  (localParamInfo default; Colorado dust → optimizer pulls toward ~3e5)
    # "Frad_direct":        0.70,    # fraction  (localParamInfo default)
    # "Frad_vis":           0.50,    # fraction  (localParamInfo default)
    "frozenPrecipMultip": 1.00,    # dimensionless  (no bias correction assumed at start)
}

# ---------------------------------------------------------------------------
# Stage 2 — Soil hydraulics and ET
# Domain-mean reference values; stage2 spatial weights differentiate by veg class.
# ---------------------------------------------------------------------------
SOIL_ET_PARAMS = {
    # Saturated hydraulic conductivity — ROSETTA sandy loam ~4e-5 m/s as domain mean.
    # Spatial weights: barren×0.55 → 2.2e-5, forest×1.5 → 6.0e-5, shrub×1.0 → 4.0e-5
    "k_soil":       4.0e-4,   # m/s

    # Van Genuchten — ROSETTA sandy loam defaults
    "vGn_alpha":   -2.17,     # m⁻¹  (stored negative in SUMMA)
    "vGn_n":        1.30,     # dimensionless

    # Surface saturation-excess scale.  Domain mean ~1.0; spatial weights push
    # barren up (×1.6) and forest down (×0.9).
    "qSurfScale":   1.00,     # dimensionless

    # Rooting depth domain mean.  Spatial weights: barren×0.4 → 0.4m, forest×1.5 → 1.5m
    "rootingDepth": 1.00,     # m

    # Porosity — ROSETTA sandy loam ~0.43 as domain mean
    "theta_sat":    0.45,     # m³/m³
}

# ---------------------------------------------------------------------------
# Stage 3 — Groundwater: bigBuckt HRU-level aquifer
# These are HRU-level (localColumn scheme).  No spatial weights in stage3_groundwater.yaml.
# ---------------------------------------------------------------------------
AQUIFER_BIGBUCKT_PARAMS = {
    # Storage scale — 5 m gives a physically plausible aquifer for fractured East River rock
    "aquiferScaleFactor":  5.00,    # m  (localParamInfo default 0.35 is too small)

    # Recession nonlinearity: 2.0 = convex recession, typical mountain catchment
    "aquiferBaseflowExp":  3.00,    # dimensionless  (localParamInfo default)

    # Baseflow rate at S = scaleFactor: ~1e-3 m/s gives reasonable low-flow magnitudes
    "aquiferBaseflowRate": 1.0e-6,  # m/s
}

# ---------------------------------------------------------------------------
# Stage 3 — Groundwater: qTopmodl HRU-level params
# k_soil is shared with stage 2 but calibrated here under qTopmodl.
# ---------------------------------------------------------------------------
AQUIFER_QTOPMODEL_PARAMS = {
    "k_soil":          4.0e-4,   # m/s  (same domain mean as soil stage)
    "kAnisotropic":    1.00,     # dimensionless  (localParamInfo default; forest→higher)
    "fieldCapacity":   0.20,     # m³/m³  (localParamInfo default for loam)
    "zScale_TOPMODEL": 10.00,    # m  (East River fractured rock; localParamInfo default ~15 m)
}

# ---------------------------------------------------------------------------
# Stage 4 — Routing (GRU-level)
# basinParamInfo defaults are good physical starting points.
# ---------------------------------------------------------------------------
ROUTING_PARAMS = {
    "routingGammaShape": 2.50,      # dimensionless  (basinParamInfo default)
    "routingGammaScale": 46000.0,   # s  (~12.8 hours mean travel time; basinParamInfo default)
}

# ---------------------------------------------------------------------------
BASIN_PARAMS = {}


# ---------------------------------------------------------------------------

def _set_hru(ds: xr.Dataset, name: str, value: float, n_hru: int) -> None:
    """Set a uniform value for an HRU-dimension variable."""
    if name not in ds:
        print(f"  [skip] {name} — not in file")
        return
    arr = np.full(n_hru, value, dtype=np.float64)
    ds[name].values[:] = arr
    print(f"  {name:<25} = {value}")


def _set_gru(ds: xr.Dataset, name: str, value: float) -> None:
    """Set a value for a GRU-dimension variable."""
    if name not in ds:
        print(f"  [skip] {name} — not in file")
        return
    ds[name].values[:] = value
    print(f"  {name:<25} = {value}  (GRU)")


def main() -> None:
    p = argparse.ArgumentParser(description="Reset trialParams.nc to clean calibration starting values")
    p.add_argument("--output", type=Path, default=DEFAULT_PATH,
                   help="Path to trialParams.nc to write")
    p.add_argument("--mode", choices=["bigbuckt", "qtopmodel"], default="bigbuckt",
                   help="Groundwater scheme: bigbuckt (default) or qtopmodel")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned values without writing the file")
    args = p.parse_args()

    if not args.output.exists():
        raise FileNotFoundError(f"trialParams.nc not found: {args.output}")

    with xr.open_dataset(args.output) as _ds:
        ds = _ds.load()

    # Detect HRU and GRU counts
    n_hru = int(ds.sizes.get("hru", ds.sizes.get("hru", 1)))
    n_gru = int(ds.sizes.get("gru", 1))
    print(f"File   : {args.output}")
    print(f"Mode   : {args.mode}")
    print(f"HRUs   : {n_hru}   GRUs: {n_gru}")
    print()

    if args.dry_run:
        print("[dry-run] Values that WOULD be written:\n")

    all_hru_params = {**SNOW_PARAMS, **SOIL_ET_PARAMS}
    if args.mode == "bigbuckt":
        all_hru_params.update(AQUIFER_BIGBUCKT_PARAMS)
    else:
        all_hru_params.update(AQUIFER_QTOPMODEL_PARAMS)

    print("── HRU parameters ──────────────────────────────────────────")
    for name, val in all_hru_params.items():
        if not args.dry_run:
            _set_hru(ds, name, val, n_hru)
        else:
            print(f"  {name:<25} = {val}")

    print("\n── GRU parameters ──────────────────────────────────────────")
    for name, val in ROUTING_PARAMS.items():
        if not args.dry_run:
            _set_gru(ds, name, val)
        else:
            print(f"  {name:<25} = {val}  (GRU)")

    print("\n── Basin params (unchanged from basinParamInfo defaults) ───")
    for name, val in BASIN_PARAMS.items():
        if not args.dry_run:
            _set_gru(ds, name, val)
        else:
            print(f"  {name:<25} = {val}  (GRU)")

    if args.dry_run:
        print("\n[dry-run] File NOT written.")
        return

    # Back up original before overwriting
    backup = args.output.with_suffix(".nc.bak")
    import shutil
    shutil.copy2(args.output, backup)
    print(f"\nBackup : {backup}")

    ds.to_netcdf(args.output)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
