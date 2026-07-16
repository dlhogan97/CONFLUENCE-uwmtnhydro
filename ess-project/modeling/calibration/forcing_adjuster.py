#!/usr/bin/env python3
"""
forcing_adjuster.py — Apply per-trial forcing multipliers before SUMMA runs.

Used to expose two calibration knobs that SUMMA itself does not provide:

1. Per-elevation-band precipitation multiplier:
       pptrate'_h(t) = precip_mult_h × pptrate_h(t)
   Compensates for residual orographic bias after the basin-wide forcing fix.

2. Basin-wide longwave radiation multiplier:
       LWRadAtm'_h(t) = lw_mult × LWRadAtm_h(t)
   Compensates for systematic LW under/over-prediction that drives snowmelt
   timing and recession behavior.

Both are applied to the monthly forcing netCDFs before SUMMA reads them.
Files outside the simulation window are symlinked (cheap), files inside
are copied with the multipliers applied.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _empirical_lw_dilley_obrien(Tair, p, q):
    """Dilley & O'Brien (1998) empirical clear-sky downwelling LW from T, p, q.

    The vapour term is 96.96*sqrt(w/25) with precipitable water
    w [kg m-2] = 4650*e0/Tair (e0 in kPa).  Dividing by 2.5 instead of 25
    inflates it by sqrt(10) and pushes effective emissivity above 1.0.
    """
    MV_CST = 0.622
    e_0 = (q * (p / 1000)) / (MV_CST + q * (1 - MV_CST))  # vapour pressure (kPa)
    return (59.38
            + 113.7 * (Tair / 273.16) ** 6
            + 96.96 * np.sqrt((4650 * e_0) / (25.0 * Tair)))

# Match METSIM-style monthly filenames ending in _YYYYMM.nc
_MONTH_RE = re.compile(r"_(\d{4})(\d{2})\.nc$")


def _parse_month_from_filename(path: Path) -> Optional[tuple]:
    m = _MONTH_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _file_in_period(
    path: Path, sim_start_year: int, sim_start_month: int,
    sim_end_year: int, sim_end_month: int,
) -> bool:
    ym = _parse_month_from_filename(path)
    if ym is None:
        return True  # be safe: include files we can't parse
    y, m = ym
    if (y, m) < (sim_start_year, sim_start_month):
        return False
    if (y, m) > (sim_end_year, sim_end_month):
        return False
    return True


def write_adjusted_forcing(
    base_forcing_dir: Path,
    trial_forcing_dir: Path,
    sim_start: str,
    sim_end: str,
    n_hru: int,
    hru_precip_multipliers: Optional[Dict[int, float]] = None,
    basin_lw_multiplier: float = 1.0,
    hru_lw_multipliers: Optional[Dict[int, float]] = None,
    hru_temp_deltas_K: Optional[Dict[int, float]] = None,
    dilley_obrien_lw: bool = False,
    file_glob: str = "*.nc",
) -> Path:
    """Materialise an adjusted forcing directory for one trial run.

    Files outside [sim_start, sim_end] are symlinked from `base_forcing_dir`.
    Files inside are loaded, multipliers/offsets applied, and written.

    Parameters
    ----------
    base_forcing_dir, trial_forcing_dir, sim_start, sim_end, n_hru, file_glob:
        See write_adjusted_forcing prior docstring.
    hru_precip_multipliers:
        {hru_idx: multiplier} mapping for `pptrate`. Missing HRUs default to 1.0.
        If all 1.0 (or None), pptrate is left unchanged.
    basin_lw_multiplier:
        DEPRECATED in favor of `hru_lw_multipliers`, but still honored when
        `hru_lw_multipliers` is None.  Scalar multiplier applied uniformly to
        `LWRadAtm`.
    hru_lw_multipliers:
        {hru_idx: multiplier} for `LWRadAtm`. Missing HRUs default to 1.0.
        Takes precedence over `basin_lw_multiplier`.
    hru_temp_deltas_K:
        {hru_idx: delta_K} **additive** offset applied to `airtemp`. Missing
        HRUs default to 0.0.  Used for the bottom-elevations warm-bias knob
        that captures AR rain at low elevation.
    dilley_obrien_lw:
        If True, replace LWRadAtm with the Dilley & O'Brien (1998) empirical
        estimate computed from airtemp, airpres, and spechum before applying
        any hru_lw_multipliers.
    """
    base_forcing_dir = Path(base_forcing_dir)
    trial_forcing_dir = Path(trial_forcing_dir)
    trial_forcing_dir.mkdir(parents=True, exist_ok=True)

    s_y, s_m = int(sim_start[:4]), int(sim_start[5:7])
    e_y, e_m = int(sim_end[:4]),   int(sim_end[5:7])

    # Per-HRU precip multipliers
    if hru_precip_multipliers:
        precip_arr = np.array(
            [float(hru_precip_multipliers.get(i, 1.0)) for i in range(n_hru)],
            dtype=np.float32,
        )
    else:
        precip_arr = np.ones(n_hru, dtype=np.float32)

    # Per-HRU LW multipliers (override basin scalar if provided)
    if hru_lw_multipliers:
        lw_arr = np.array(
            [float(hru_lw_multipliers.get(i, 1.0)) for i in range(n_hru)],
            dtype=np.float32,
        )
    else:
        lw_arr = np.full(n_hru, float(basin_lw_multiplier), dtype=np.float32)

    # Per-HRU additive temperature offsets (Kelvin)
    if hru_temp_deltas_K:
        temp_delta_arr = np.array(
            [float(hru_temp_deltas_K.get(i, 0.0)) for i in range(n_hru)],
            dtype=np.float32,
        )
    else:
        temp_delta_arr = np.zeros(n_hru, dtype=np.float32)

    apply_precip  = not np.allclose(precip_arr, 1.0)
    apply_lw      = not np.allclose(lw_arr, 1.0)
    apply_temp    = not np.allclose(temp_delta_arr, 0.0)
    apply_any     = apply_precip or apply_lw or apply_temp or dilley_obrien_lw

    files = sorted(base_forcing_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No forcing files in {base_forcing_dir}")

    n_modified = 0
    for src in files:
        dst = trial_forcing_dir / src.name
        in_period = _file_in_period(src, s_y, s_m, e_y, e_m)

        if not in_period or not apply_any:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            continue

        with xr.open_dataset(src) as ds:
            ds = ds.load()
        if ds.sizes.get("hru", 0) != n_hru:
            raise ValueError(
                f"{src.name} has hru={ds.sizes.get('hru')} but expected {n_hru}"
            )

        if apply_precip:
            if "pptrate" not in ds:
                raise KeyError(f"pptrate missing from {src.name}")
            ds["pptrate"] = ds["pptrate"] * xr.DataArray(precip_arr, dims=["hru"])

        if apply_temp:
            if "airtemp" not in ds:
                raise KeyError(f"airtemp missing from {src.name}")
            ds["airtemp"] = ds["airtemp"] + xr.DataArray(temp_delta_arr, dims=["hru"])

        if dilley_obrien_lw:
            for v in ("airtemp", "airpres", "spechum", "LWRadAtm"):
                if v not in ds:
                    raise KeyError(f"{v} missing from {src.name} (required for Dilley-O'Brien)")
            lw_do = _empirical_lw_dilley_obrien(
                ds["airtemp"].values, ds["airpres"].values, ds["spechum"].values
            )
            ds["LWRadAtm"] = xr.DataArray(lw_do, dims=ds["LWRadAtm"].dims,
                                           attrs=ds["LWRadAtm"].attrs)

        if apply_lw:
            if "LWRadAtm" not in ds:
                raise KeyError(f"LWRadAtm missing from {src.name}")
            ds["LWRadAtm"] = ds["LWRadAtm"] * xr.DataArray(lw_arr, dims=["hru"])

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        ds.to_netcdf(dst)
        n_modified += 1

    logger.debug(
        "Adjusted forcing → %s  (%d modified, %d total)  precip=%s  dilley_obrien=%s  lw=%s  temp_dK=%s",
        trial_forcing_dir, n_modified, len(files),
        precip_arr.tolist() if apply_precip else "1.0×",
        dilley_obrien_lw,
        lw_arr.tolist() if apply_lw else "1.0×",
        temp_delta_arr.tolist() if apply_temp else "0.0",
    )
    return trial_forcing_dir


def patch_forcing_path(
    file_manager_path: Path, new_forcing_dir: Path,
) -> None:
    """Rewrite `forcingPath` in fileManager.txt to point at `new_forcing_dir`.

    The trial's settings copy already has fileManager.txt patched for output
    paths; this is an additional in-place edit.
    """
    file_manager_path = Path(file_manager_path)
    new_forcing_dir = Path(new_forcing_dir)

    lines = file_manager_path.read_text().splitlines()
    new_lines = []
    found = False
    for line in lines:
        if line.lstrip().startswith("forcingPath"):
            key_part = line[: line.index("forcingPath") + len("forcingPath")]
            new_lines.append(f"{key_part}    '{new_forcing_dir}/'")
            found = True
        else:
            new_lines.append(line)
    if not found:
        raise ValueError(f"forcingPath line not found in {file_manager_path}")
    file_manager_path.write_text("\n".join(new_lines) + "\n")
