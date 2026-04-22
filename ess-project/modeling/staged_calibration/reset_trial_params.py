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

    # Single value applied to all HRUs:
    python reset_trial_params.py --override k_soil=0.0005

    # Per-HRU values (must match n_hru, ordered elevation-descending):
    python reset_trial_params.py --override rootingDepth=0.4,1.0,1.5,1.5,1.0

    # Multiple overrides:
    python reset_trial_params.py --override k_soil=0.0005 --override albedoDecayRate=1.5e5

    # Seed an elev×aspect trialParams from an elevation-only run (repeats each
    # elevation band's params for every aspect class within that band):
    python reset_trial_params.py \\
        --output /path/to/elevAspect/trialParams.nc \\
        --seed-elev /path/to/elevation_only/trialParams.nc
    # Default 5 aspects per band (NE/SE/NW/SW/Flat); override with --aspects-per-band N
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
    "/domain_Tuolumne_River_lumped/settings/SUMMA/trialParams.nc"
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
    "albedoDecayRate":    8.19e5,   # s  (localParamInfo default; Colorado dust → optimizer pulls toward ~3e5)
    # "Frad_direct":        0.70,    # fraction  (localParamInfo default)
    # "Frad_vis":           0.50,    # fraction  (localParamInfo default)
    "frozenPrecipMultip": 1.0,    # dimensionless  (no bias correction assumed at start)
}

# ---------------------------------------------------------------------------
# Stage 2 — Soil hydraulics and ET
# Domain-mean reference values; stage2 spatial weights differentiate by veg class.
# ---------------------------------------------------------------------------
SOIL_ET_PARAMS = {
    'k_soil': 5e-4,    # m/s

    # Van Genuchten — ROSETTA sandy loam defaults
    "vGn_alpha":   -2.7,     # m⁻¹  (stored negative in SUMMA)
    "vGn_n":        1.3,     # dimensionless

    # Surface saturation-excess scale.  Domain mean ~1.0; spatial weights push
    # barren up (×1.6) and forest down (×0.9).

    # Porosity — ROSETTA sandy loam ~0.43 as domain mean
    "theta_sat":    0.45,     # m³/m³

        # Minimum stomatal resistance — domain mean ~100 s/m; spatial weights push
        # barren up (×1.5 → 150 s/m) and forest down (×0.7 → 70 s/m).
    "minStomatalResistance": 10.0,    # s/m
    'qSurfScale': 5.5,
    'zScale_TOPMODEL': 1.0,
    'kAnisotropic': 0.1,
    'rootingDepth': 0.5,
    'fieldCapacity': 0.2,
    # # 'vGn_alpha': [-3, -3, -3, -3, -3],
    # # 'theta_sat': [0.35, 0.40, 0.45, 0.45, 0.45],
}


# ---------------------------------------------------------------------------
# Stage 3 — Groundwater: bigBuckt HRU-level aquifer
# These are HRU-level (localColumn scheme).  No spatial weights in stage3_groundwater.yaml.
# ---------------------------------------------------------------------------
AQUIFER_BIGBUCKT_PARAMS = {
    # Storage scale — 5 m gives a physically plausible aquifer for fractured East River rock
    "aquiferScaleFactor":  5.00,    # m  (localParamInfo default 0.35 is too small)

    # Recession nonlinearity: 2.0 = convex recession, typical mountain catchment
    "aquiferBaseflowExp":  2.00,    # dimensionless  (localParamInfo default)

    # Baseflow rate at S = scaleFactor: ~1e-3 m/s gives reasonable low-flow magnitudes
    "aquiferBaseflowRate": 1.0e-6,  # m/s
}

# ---------------------------------------------------------------------------
# Stage 3 — Groundwater: qTopmodl HRU-level params
# k_soil is shared with stage 2 but calibrated here under qTopmodl.
# ---------------------------------------------------------------------------
AQUIFER_QTOPMODEL_PARAMS = {
    "kAnisotropic":    1.00,     # dimensionless  (localParamInfo default; forest→higher)
    "zScale_TOPMODEL": 2.50,    # m  (East River fractured rock; localParamInfo default ~15 m)
}

# ---------------------------------------------------------------------------
# Stage 4 — Routing (GRU-level)
# basinParamInfo defaults are good physical starting points.
# ---------------------------------------------------------------------------
ROUTING_PARAMS = {
    "routingGammaShape": 2.50,      # dimensionless  (basinParamInfo default)
    "routingGammaScale": 46000.0,   # s  (~12.8 hours mean travel time; basinParamInfo default)
}

# Parameters that should be removed entirely from trialParams.nc.
DROP_PARAMS = ["critSoilTranspire", "critSoilWilting"]

# ---------------------------------------------------------------------------

def _set_hru(ds: xr.Dataset, name: str, value: "float | list[float]", n_hru: int) -> None:
    """Set a value for an HRU-dimension variable, creating it if absent.

    value may be a scalar (applied to all HRUs) or a list of length n_hru.
    If n_hru=1 and a list is provided, the geometric mean is used for log-scale
    params (conductivities) and arithmetic mean for others.
    """
    if isinstance(value, (list, np.ndarray)):
        arr = np.array(value, dtype=np.float64)
        if len(arr) != n_hru:
            if n_hru == 1:
                # Collapse to a single representative value for lumped domains.
                # Use geometric mean for conductivity-like params (all positive, wide range),
                # arithmetic mean otherwise.
                if np.all(arr > 0) and (arr.max() / arr.min()) > 10:
                    scalar = float(np.exp(np.mean(np.log(arr))))
                else:
                    scalar = float(np.mean(arr))
                print(f"  {name:<25} list→scalar (lumped): {list(arr)} → {scalar:.4g}")
                arr = np.array([scalar])
            else:
                raise ValueError(
                    f"  {name}: provided {len(arr)} values but n_hru={n_hru}"
                )
        label = f"[{', '.join(f'{v:.4g}' for v in arr)}]"
    else:
        arr = np.full(n_hru, value, dtype=np.float64)
        label = str(value)

    if name not in ds:
        ds[name] = xr.DataArray(arr, dims=["hru"])
        print(f"  {name:<25} = {label}  [created]")
    else:
        ds[name].values[:] = arr
        print(f"  {name:<25} = {label}")


def _drop_params(ds: xr.Dataset, names: list[str], dry_run: bool = False) -> xr.Dataset:
    """Drop named variables from dataset when present."""
    present = [name for name in names if name in ds]
    if not present:
        return ds

    if dry_run:
        print("[dry-run] Variables that WOULD be removed from dataset:")
        for name in present:
            print(f"  {name}")
        print()
        return ds

    ds = ds.drop_vars(present)
    print("Removed variables from dataset:")
    for name in present:
        print(f"  {name}")
    print()
    return ds


def _set_gru(ds: xr.Dataset, name: str, value: float) -> None:
    """Set a value for a GRU-dimension variable, creating it if absent."""
    arr = np.array([value], dtype=np.float64)
    if name not in ds:
        ds[name] = xr.DataArray(arr, dims=["gru"])
        print(f"  {name:<25} = {value}  (GRU) [created]")
        return
    ds[name].values[:] = value
    print(f"  {name:<25} = {value}  (GRU)")


def _parse_overrides(override_args: list[str]) -> dict:
    """Parse --override name=val or name=v1,v2,...,vN entries.

    Returns a dict mapping param name → float or list[float].
    """
    result = {}
    for item in override_args:
        if "=" not in item:
            raise ValueError(f"--override must be 'name=value': {item!r}")
        name, raw = item.split("=", 1)
        parts = raw.split(",")
        if len(parts) == 1:
            result[name.strip()] = float(parts[0])
        else:
            result[name.strip()] = [float(v) for v in parts]
    return result


def _expand_from_elevation(
    src_path: Path,
    dst: xr.Dataset,
    aspects_per_band: int,
    dry_run: bool,
) -> xr.Dataset:
    """Expand elevation-only trialParams to an elev×aspect layout.

    Each elevation-band value is repeated for every aspect class within
    that band using np.repeat, preserving parameter order.
    """
    with xr.open_dataset(src_path) as _src:
        src = _src.load()

    n_src_hru = int(src.sizes.get("hru", 0))
    n_dst_hru = int(dst.sizes.get("hru", 0))
    expected_dst = n_src_hru * aspects_per_band
    if n_dst_hru != expected_dst:
        raise ValueError(
            f"Destination has {n_dst_hru} HRUs but source ({n_src_hru}) × "
            f"aspects_per_band ({aspects_per_band}) = {expected_dst}. "
            "Check --aspects-per-band or source file."
        )

    for name, da in src.data_vars.items():
        if name == "hruId":
            continue
        if "hru" in da.dims:
            expanded = np.repeat(da.values, aspects_per_band)
            if dry_run:
                print(f"  {name:<25} expand {list(da.values)} → {list(expanded)}")
            else:
                if name not in dst:
                    dst[name] = xr.DataArray(expanded.astype(da.dtype), dims=["hru"])
                    print(f"  {name:<25} [created + expanded]")
                else:
                    dst[name].values[:] = expanded.astype(dst[name].dtype)
                    print(f"  {name:<25} expanded {n_src_hru} → {n_dst_hru} HRUs")
        elif "gru" in da.dims:
            n_src_gru = int(src.sizes.get("gru", 1))
            n_dst_gru = int(dst.sizes.get("gru", 1))
            if n_src_gru != n_dst_gru:
                print(f"  [skip] {name} — GRU count mismatch ({n_src_gru} vs {n_dst_gru})")
                continue
            if dry_run:
                print(f"  {name:<25} GRU copy {list(da.values)}")
            else:
                if name not in dst:
                    dst[name] = da.copy()
                    print(f"  {name:<25} [created, GRU]")
                else:
                    dst[name].values[:] = da.values.astype(dst[name].dtype)
                    print(f"  {name:<25} copied (GRU)")

    # Always set hruId to 1-indexed sequential regardless of source
    hru_ids = np.arange(1, n_dst_hru + 1, dtype=np.int32)
    if "hruId" in dst:
        dst["hruId"].values[:] = hru_ids
    else:
        dst["hruId"] = xr.DataArray(hru_ids, dims=["hru"])
    if dry_run:
        print(f"  {'hruId':<25} = {list(hru_ids)}")
    else:
        print(f"  {'hruId':<25} = 1–{n_dst_hru}")
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description="Reset trialParams.nc to clean calibration starting values")
    p.add_argument("--output", type=Path, default=DEFAULT_PATH,
                   help="Path to trialParams.nc to write")
    p.add_argument("--mode", choices=["bigbuckt", "qtopmodel"], default="bigbuckt",
                   help="Groundwater scheme: bigbuckt (default) or qtopmodel")
    p.add_argument("--override", metavar="NAME=VALUE", action="append", default=[],
                   help="Override a parameter value. Single value → all HRUs; "
                        "comma-separated list → one value per HRU (elevation-desc order). "
                        "May be repeated: --override k_soil=5e-4 --override rootingDepth=0.4,1.0,1.5,1.5,1.0")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned values without writing the file")
    p.add_argument("--seed-elev", type=Path, default=None, metavar="SOURCE_NC",
                   help="Seed an elev×aspect trialParams from an elevation-only trialParams.nc. "
                        "Each elevation band's values are repeated for every aspect class.")
    p.add_argument("--aspects-per-band", type=int, default=5, metavar="N",
                   help="Number of aspect classes per elevation band (default: 5)")
    p.add_argument("--create", action="store_true",
                   help="Create a new trialParams.nc from scratch if the output file does not exist. "
                        "Use --n-hru and --n-gru to set dimensions.")
    p.add_argument("--n-hru", type=int, default=1, metavar="N",
                   help="Number of HRUs when creating a new file (default: 1)")
    p.add_argument("--n-gru", type=int, default=1, metavar="N",
                   help="Number of GRUs when creating a new file (default: 1)")
    args = p.parse_args()

    if args.output.is_dir():
        args.output = args.output / "trialParams.nc"

    if not args.output.exists():
        import shutil
        if args.create:
            # Build a minimal skeleton netCDF with hruId and gruId
            n_hru_new = args.n_hru
            n_gru_new = args.n_gru
            ds_new = xr.Dataset(
                {
                    "hruId": xr.DataArray(np.arange(1, n_hru_new + 1, dtype=np.int32), dims=["hru"]),
                    "gruId": xr.DataArray(np.arange(1, n_gru_new + 1, dtype=np.int32), dims=["gru"]),
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            ds_new.to_netcdf(args.output)
            print(f"Created new trialParams.nc ({n_hru_new} HRU, {n_gru_new} GRU): {args.output}")
        elif DEFAULT_PATH.exists():
            shutil.copy2(DEFAULT_PATH, args.output)
            print(f"Seeded {args.output.name} from {DEFAULT_PATH.name}")
        else:
            raise FileNotFoundError(
                f"Output file not found. Use --create to build from scratch: {args.output}"
            )

    overrides = _parse_overrides(args.override)

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

    ds = _drop_params(ds, DROP_PARAMS, dry_run=args.dry_run)

    if args.seed_elev is not None:
        # --seed-elev path: expand elevation-only params → elev×aspect layout
        if not args.seed_elev.exists():
            raise FileNotFoundError(f"--seed-elev source not found: {args.seed_elev}")
        print("── Expanding from elevation-only trialParams ───────────────")
        ds = _expand_from_elevation(args.seed_elev, ds, args.aspects_per_band, args.dry_run)
    else:
        # Default: write hard-coded parameter blocks
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

    if overrides:
        print("\n── Manual overrides (applied last) ─────────────────────────")
        for name, val in overrides.items():
            if name in DROP_PARAMS:
                print(f"  [skip] {name} is configured for removal from dataset")
                continue
            # Route to HRU or GRU setter based on the variable's dimension in the file
            if name in ds and "gru" in ds[name].dims and "hru" not in ds[name].dims:
                if isinstance(val, list):
                    raise ValueError(f"  {name} is a GRU variable — list override not supported")
                if not args.dry_run:
                    _set_gru(ds, name, val)
                else:
                    print(f"  {name:<25} = {val}  (GRU override)")
            else:
                if not args.dry_run:
                    _set_hru(ds, name, val, n_hru)
                else:
                    label = f"[{', '.join(f'{v:.4g}' for v in val)}]" if isinstance(val, list) else str(val)
                    print(f"  {name:<25} = {label}  (HRU override)")

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
