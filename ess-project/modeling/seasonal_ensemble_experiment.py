#!/usr/bin/env python3
"""
Seasonal Ensemble Experiment
============================
Constructs and runs a seasonal forcing ensemble for 22 domain+model combinations.
The target water year is WY2021 (Sep 1 2020 – Sep 30 2021).  For each of four
seasons, the target year's forcing is replaced month-by-month with forcing from
each of 20 donor water years (2001–2020), producing 20 ensemble members per
season (80 per model).

Water-year season definitions
------------------------------
  Fall   : Sep–Nov 2020  →  donor provides calendar months (WY-1) Sep/Oct/Nov
  Winter : Dec 2020–Mar 2021  →  donor provides (WY-1) Dec and WY Jan/Feb/Mar
  Spring : Apr–Jun 2021  →  donor provides WY Apr/May/Jun
  Summer : Jul–Aug 2021  →  donor provides WY Jul/Aug

Usage
-----
  # Step 1 – Build warm states and ensemble forcing (run once per domain):
  python seasonal_ensemble_experiment.py setup --domain East_River_lumped

  # Step 2 – Run SUMMA for a specific combination:
  python seasonal_ensemble_experiment.py run \\
      --domain East_River_lumped --model bigBuckt --season fall

  # Step 3 – Post-process (linear reservoir routing + summary CSV):
  python seasonal_ensemble_experiment.py postprocess \\
      --domain East_River_lumped --model bigBuckt --season fall

  # Run all steps for a domain (all models, all seasons):
  python seasonal_ensemble_experiment.py all --domain East_River_lumped

  # List all known domain+model combinations:
  python seasonal_ensemble_experiment.py list

Progress is logged to <domain_data_dir>/simulations/seasonal_ensemble_simulations/progress.log
so that interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from utils.custom.linear_reservoir import two_reservoir_daily  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("seasonal_ensemble")

# ── Global constants ──────────────────────────────────────────────────────────
SCRATCH_BASE     = Path("/scratch/dlhogan/ess-project-data")
ESS_MODELING_DIR = Path(__file__).parent
ENSEMBLE_SUBDIR  = "seasonal_ensemble_simulations"

TARGET_WY       = 2021
WARM_STATE_DATE = "2020-08-31 23:00:00"
SIM_START       = "2020-09-01 01:00"
SIM_END         = "2021-09-30 23:00"
DONOR_YEARS     = list(range(2001, 2021))   # 20 donors: WY2001–WY2020

# All 13 target months for the simulation (year, month)
TARGET_MONTHS: List[Tuple[int, int]] = [
    (2020, 9), (2020, 10), (2020, 11), (2020, 12),
    (2021, 1), (2021, 2),  (2021, 3),  (2021, 4),
    (2021, 5), (2021, 6),  (2021, 7),  (2021, 8),  (2021, 9),
]

# Seasons: maps name to the (target_year, target_month) list to replace
SEASONS: Dict[str, List[Tuple[int, int]]] = {
    "fall":   [(2020, 9), (2020, 10), (2020, 11)],
    "winter": [(2020, 12), (2021, 1), (2021, 2), (2021, 3)],
    "spring": [(2021, 4), (2021, 5), (2021, 6)],
    "summer": [(2021, 7), (2021, 8)],
}


def donor_calendar_month(season: str, target_year: int, target_month: int,
                         donor_wy: int) -> Tuple[int, int]:
    """Return (donor_cal_year, donor_cal_month) for a given target month.

    Water-year convention:
      Fall   (Sep–Nov 2020) : donor calendar year = donor_wy - 1
      Winter Dec 2020       : donor calendar year = donor_wy - 1
      Winter Jan–Mar 2021   : donor calendar year = donor_wy
      Spring / Summer       : donor calendar year = donor_wy
    """
    if season == "fall":
        return (donor_wy - 1, target_month)
    elif season == "winter":
        if target_month == 12:
            return (donor_wy - 1, 12)
        else:
            return (donor_wy, target_month)
    else:   # spring, summer
        return (donor_wy, target_month)


# ── Domain / model registry ───────────────────────────────────────────────────
# models: dict[canonical_key → path relative to best_simulations/]
#   Nested paths (e.g. "bigBuckt_best/qTopmodl_best") are supported.
# longterm_nc: dict[canonical_key → glob pattern relative to best_simulations/]

DOMAIN_REGISTRY: Dict[str, dict] = {
    "East_River_lumped": {
        "data_dir"   : "domain_East_River_lumped",
        "linres_json": "east_linRes_params.json",
        "linres_key" : "lumped",
        "models": {
            "bigBuckt": "bigBuckt_20260425_best",
            "noXplict": "noXplicit_drainage_best",
        },
        "longterm_nc": {
            "bigBuckt": "lumped_bigBuckt_best_lon*term.nc",
            "noXplict": "lumped_noXplict_drainage_best_longterm.nc",
        },
    },
    "East_River_distributed": {
        "data_dir"   : "domain_East_River_distributed",
        "linres_json": "east_linRes_params.json",
        "linres_key" : "distributed_elev",
        "models": {
            "bigBuckt": "bigBuckt_elevation_best_Jarvis_20260414_141650",
            "noXplict": "noXplict_elevation_best_20260414_144029",
            "qTopmodl": "qTopmodl_elevation_best_20260421_223508",
        },
        "longterm_nc": {
            "bigBuckt": "distributed_elevation_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevation_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elevation_qTopmodl_best_longterm.nc",
        },
    },
    "East_River_distributed_elevAspect": {
        "data_dir"   : "domain_East_River_distributed_elevAspect",
        "linres_json": "east_linRes_params.json",
        "linres_key" : "distributed_elevAspect",
        "models": {
            "bigBuckt": "bigBuckt_elevationAspect_best_20260422_144322",
            "noXplict": "noXplict_elevAspect_best_20260421_154832",
            "qTopmodl": "qTopmodl_elevationAspect_best_20260422_204739",
        },
        "longterm_nc": {
            "bigBuckt": "distributed_elevAspect_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevAspect_best_longterm.nc",
            "qTopmodl": "distributed_elevAspect_qTopmodl_best_longterm.nc",
        },
    },
    "East_River_distributed_elevTPI": {
        "data_dir"   : "domain_East_River_distributed_elevTPI",
        "linres_json": "east_linRes_params.json",
        "linres_key" : "distributed_elevTPI",
        "models": {
            "bigBuckt": "bigBuckt_best",
            "noXplict": "noXplict_best",
            "qTopmodl": "qTopmodl_best",
        },
        "longterm_nc": {
            "bigBuckt": "distributed_elevTPI_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevTPI_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elevTPI_qTopmodl_best_longterm.nc",
        },
    },
    "Tuolumne_River_lumped": {
        "data_dir"   : "domain_Tuolumne_River_lumped",
        "linres_json": "tuolumne_linRes_params.json",
        "linres_key" : "lumped",
        "models": {
            "bigBuckt": "bigBuckt_best",
            "noXplict": "noXplict_drainage_best",
        },
        "longterm_nc": {
            "bigBuckt": "lumped_bigBuckt_best_longterm.nc",
            "noXplict": "lumped_noXplict_drainage_best_longterm.nc",
        },
    },
    "Tuolumne_River_distributed_elev": {
        "data_dir"   : "domain_Tuolumne_River_distributed_elev",
        "linres_json": "tuolumne_linRes_params.json",
        "linres_key" : "distributed_elev",
        "models": {
            "bigBuckt": "bigBuckt_best",
            "noXplict": "noXplict_best",
            "qTopmodl": "qTopmodl_best",
        },
        "longterm_nc": {
            "bigBuckt": "distributed_elev_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elev_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elev_qTopmodl_best_longterm.nc",
        },
    },
    "Tuolumne_River_distributed_elevAspect": {
        "data_dir"   : "domain_Tuolumne_River_distributed_elevAspect",
        "linres_json": "tuolumne_linRes_params.json",
        "linres_key" : "distributed_elevAspect",
        "models": {
            "bigBuckt": "bigBuckt_best",
            "noXplict": "noXplict_best",
            # qTopmodl is nested inside bigBuckt_best/
            "qTopmodl": "bigBuckt_best/qTopmodl_best",
        },
        "longterm_nc": {
            "bigBuckt": "distributed_elevAspect_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevAspect_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elevAspect_qTopmodl_best_longterm.nc",
        },
    },
    "Tuolumne_River_distributed_elevTPI": {
        "data_dir"   : "domain_Tuolumne_River_distributed_elevTPI",
        "linres_json": "tuolumne_linRes_params.json",
        "linres_key" : "distributed_elevTPI",
        "models": {
            "bigBuckt": "bigBuckt_best",
            "noXplict": "noXplict_best",
            "qTopmodl": "qTopmodl_best",
        },
        "longterm_nc": {
            "bigBuckt": "distributed_elevTPI_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevTPI_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elevTPI_qTopmodl_best_longterm.nc",
        },
    },
}

# Maps canonical model keys → key used inside the linRes JSON files
_LINRES_MODEL_KEY = {
    "bigBuckt": "bigBuckt",
    "noXplict": "noXplict_drainage",
    "qTopmodl": "qTopmodl",
}


# =============================================================================
# ProgressLogger
# =============================================================================

class ProgressLogger:
    """File-backed step tracker for restartable runs.

    Lines are written as:  STEP_KEY=DONE  or  STEP_KEY=FAILED
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._done: set = set()
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                if "=DONE" in line:
                    self._done.add(line.split("=")[0].strip())

    def is_done(self, key: str) -> bool:
        with self._lock:
            return key in self._done

    def mark_done(self, key: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._done.add(key)
            with open(self.log_path, "a") as fh:
                fh.write(f"{key}=DONE  # {ts}\n")
        logger.info("[progress] %s → DONE", key)

    def mark_failed(self, key: str, reason: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with open(self.log_path, "a") as fh:
                fh.write(f"{key}=FAILED  # {ts}  {reason}\n")
        logger.warning("[progress] %s → FAILED  %s", key, reason)

    @staticmethod
    def step_key(*parts) -> str:
        return ":".join(str(p) for p in parts)


# =============================================================================
# Path helpers
# =============================================================================

def find_file_manager(model_dir: Path) -> Optional[Path]:
    """Search for fileManager.txt up to 3 levels inside model_dir."""
    for subpath in ("fileManager.txt",
                    "settings/fileManager.txt",
                    "settings/SUMMA/fileManager.txt"):
        p = model_dir / subpath
        if p.exists():
            return p
    return None


def parse_file_manager(fm_path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in fm_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("!", "#")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            val = " ".join(parts[1:]).strip("'\"")
            result[parts[0]] = val
    return result


def find_forcing_dir(model_dir: Path, fm_values: Dict[str, str]) -> Optional[Path]:
    """Locate the monthly forcing directory.

    Priority:
      1. forcingPath from fileManager.txt (if it exists on disk)
      2. forcing_adj/ inside model_dir
      3. forcing/SUMMA_input_filtered/ inside model_dir
      4. forcing/SUMMA_input/ inside model_dir
    """
    if "forcingPath" in fm_values:
        p = Path(fm_values["forcingPath"])
        if p.exists() and any(p.glob("*.nc")):
            return p

    for candidate in ("forcing_adj",
                      "forcing/SUMMA_input_filtered",
                      "forcing/SUMMA_input"):
        p = model_dir / candidate
        if p.is_dir() and any(p.glob("*.nc")):
            return p

    return None


def find_settings_dir(model_dir: Path) -> Optional[Path]:
    fm = find_file_manager(model_dir)
    return fm.parent if fm else None


def get_forcing_prefix(forcing_dir: Path) -> str:
    """Infer forcing filename prefix (everything before the YYYYMM.nc suffix).

    When multiple prefixes exist (e.g. ERA5 + METSIM in the same dir), prefer
    the one whose files span both the earliest donor year and the target year.
    """
    nc_files = sorted(forcing_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files in {forcing_dir}")

    # Collect all prefixes and the YYYYMM dates they cover
    prefix_dates: dict = {}
    for f in nc_files:
        m = re.match(r"^(.+_)(\d{6})$", f.stem)
        if m:
            prefix_dates.setdefault(m.group(1), []).append(int(m.group(2)))

    if not prefix_dates:
        raise ValueError(f"Cannot parse forcing prefix from files in {forcing_dir}")

    if len(prefix_dates) == 1:
        return next(iter(prefix_dates))

    # Pick the prefix whose date range covers both donor start and target end
    target_start = int(SIM_START[:4]) * 100 + int(SIM_START[5:7])   # e.g. 202009
    donor_start  = min(DONOR_YEARS) * 100 + 9                        # e.g. 200109

    for prefix, dates in sorted(prefix_dates.items(),
                                 key=lambda kv: len(kv[1]), reverse=True):
        if min(dates) <= donor_start and max(dates) >= target_start:
            return prefix

    # Fallback: prefix with most files
    return max(prefix_dates, key=lambda p: len(prefix_dates[p]))


def validate_model(domain: str, model: str
                   ) -> Tuple[Path, Path, Path, Dict[str, str]]:
    """Return (model_dir, settings_dir, forcing_dir, fm_values) or raise."""
    cfg       = DOMAIN_REGISTRY[domain]
    best_sims = SCRATCH_BASE / cfg["data_dir"] / "simulations" / "best_simulations"
    model_dir = best_sims / cfg["models"][model]

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    fm_path = find_file_manager(model_dir)
    if fm_path is None:
        raise FileNotFoundError(f"fileManager.txt not found under {model_dir}")

    fm_values    = parse_file_manager(fm_path)
    settings_dir = fm_path.parent
    forcing_dir  = find_forcing_dir(model_dir, fm_values)
    if forcing_dir is None:
        raise FileNotFoundError(
            f"No forcing directory found for {domain}/{model} in {model_dir}"
        )
    return model_dir, settings_dir, forcing_dir, fm_values


# =============================================================================
# WarmStateExtractor
# =============================================================================

class WarmStateExtractor:
    """Extract Aug 31 2020 23:00 model state from a longterm output file and
    write a warmState.nc that mirrors the coldState.nc structure exactly.

    Variables NOT present in SUMMA output (handled with defaults):
      dt_init          → 3600.0 (hourly timestep)
      nSnow            → 0   (end of summer: no snow)
      nSoil            → read from coldState.nc
      scalarSnowAlbedo → 0.85  (irrelevant with nSnow=0)
      scalarCanairTemp → scalarCanopyTemp_mean from output (or 285 K)
    """

    WARM_TS = pd.Timestamp(WARM_STATE_DATE)

    # (coldState_varname, longterm_varname_or_None, scalar_default_or_None)
    _SCALAR_MAP = [
        ("scalarCanopyIce",      "scalarCanopyIce_mean",      0.0),
        ("scalarCanopyLiq",      "scalarCanopyLiq_mean",      0.0),
        ("scalarSnowDepth",      "scalarSnowDepth_mean",      0.0),
        ("scalarSWE",            "scalarSWE",                 0.0),
        ("scalarSfcMeltPond",    "scalarSfcMeltPond",         0.0),
        ("scalarAquiferStorage", "scalarAquiferStorage_mean", 0.0),
        ("scalarCanopyTemp",     "scalarCanopyTemp_mean",     285.0),
    ]
    # State variables from longterm output (all use _mean suffix from outputControl)
    _LAYER_MAP = [
        ("mLayerTemp",       "mLayerTemp_mean"),
        ("mLayerVolFracIce", "mLayerVolFracIce_mean"),
        ("mLayerVolFracLiq", "mLayerVolFracLiq_mean"),
        ("mLayerMatricHead", "mLayerMatricHead_mean"),
    ]

    def __init__(self, longterm_nc: Path, cold_state_nc: Path):
        self.longterm_nc  = longterm_nc
        self.cold_state_nc = cold_state_nc

    def extract(self, output_path: Path) -> Path:
        logger.info("Extracting warm state from %s", self.longterm_nc.name)

        lt   = xr.open_dataset(self.longterm_nc)
        cold = xr.open_dataset(self.cold_state_nc)

        # Nearest timestep to 2020-08-31 23:00
        ts_idx = pd.DatetimeIndex(lt.time.values)
        idx    = ts_idx.get_indexer([self.WARM_TS], method="nearest")[0]
        actual = pd.Timestamp(lt.time.values[idx])
        if abs((actual - self.WARM_TS).total_seconds()) > 7200:
            logger.warning(
                "Warm-state: nearest time %s is >2 h from target %s",
                actual, self.WARM_TS,
            )
        lt_t = lt.isel(time=idx)

        n_hru     = int(cold.sizes["hru"])
        n_soil    = int(cold["nSoil"].values.flat[0])
        n_snow    = 0
        n_mid     = n_snow + n_soil      # 3
        n_ifc     = n_snow + n_soil + 1  # 4

        import netCDF4 as nc4
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with nc4.Dataset(str(output_path), "w") as dst:
            dst.createDimension("hru",     n_hru)
            dst.createDimension("scalarv", 1)
            dst.createDimension("midSoil", n_soil)
            dst.createDimension("midToto", n_mid)
            dst.createDimension("ifcToto", n_ifc)

            # hruId
            v = dst.createVariable("hruId", "i4", ("hru",))
            v[:] = cold["hruId"].values

            # Integer scalars
            for name, val in [("nSoil", n_soil), ("nSnow", n_snow)]:
                v = dst.createVariable(name, "i4", ("scalarv", "hru"))
                v[:] = val

            # dt_init
            v = dst.createVariable("dt_init", "f8", ("scalarv", "hru"))
            v[:] = 3600.0

            # Scalar state variables from longterm output
            for cs_name, lt_name, default in self._SCALAR_MAP:
                v = dst.createVariable(cs_name, "f8", ("scalarv", "hru"))
                # Try requested name first, then strip _mean suffix as fallback
                actual_lt = lt_name
                if actual_lt and actual_lt not in lt_t:
                    fallback = actual_lt.removesuffix("_mean")
                    if fallback != actual_lt and fallback in lt_t:
                        actual_lt = fallback
                    else:
                        actual_lt = None
                if actual_lt:
                    arr = np.atleast_1d(lt_t[actual_lt].values).ravel()
                    if arr.shape[0] == n_hru:
                        v[0, :] = arr
                    elif arr.shape[0] == 1:
                        v[0, :] = np.full(n_hru, arr[0])
                    else:
                        v[0, :] = np.full(n_hru, float(np.nanmean(arr)))
                else:
                    v[0, :] = default
                    if lt_name:
                        logger.warning(
                            "  warmState: %s not in output → default %.3g",
                            cs_name, default,
                        )

            # scalarSnowAlbedo (not in output; irrelevant with nSnow=0)
            v = dst.createVariable("scalarSnowAlbedo", "f8", ("scalarv", "hru"))
            v[:] = 0.85

            # scalarCanairTemp (not in output → approximate from canopy temp)
            v = dst.createVariable("scalarCanairTemp", "f8", ("scalarv", "hru"))
            for candidate in ("scalarCanopyTemp_mean", "scalarSurfaceTemp_mean"):
                if candidate in lt_t:
                    arr = np.atleast_1d(lt_t[candidate].values).ravel()
                    if arr.shape[0] == n_hru:
                        v[0, :] = arr
                    else:
                        v[0, :] = np.full(n_hru, float(np.nanmean(arr)))
                    break
            else:
                v[0, :] = 285.0
                logger.warning("  warmState: scalarCanairTemp set to 285 K fallback")

            # Layer variables (midToto × hru)
            for cs_name, lt_name in self._LAYER_MAP:
                v = dst.createVariable(cs_name, "f8", ("midToto", "hru"))
                # Try requested name first, then strip _mean suffix as fallback
                actual_lt = lt_name
                if actual_lt not in lt_t:
                    fallback = actual_lt.removesuffix("_mean")
                    actual_lt = fallback if fallback != lt_name and fallback in lt_t else None
                if actual_lt:
                    arr = lt_t[actual_lt].values
                    arr = np.atleast_2d(arr)
                    # arr may be (midToto_full, hru) — take first n_mid rows
                    arr_use = arr[:n_mid, :] if arr.shape[0] >= n_mid else arr
                    if arr_use.shape[1] != n_hru:
                        arr_use = np.tile(arr_use[:, 0:1], (1, n_hru))
                    v[:arr_use.shape[0], :] = arr_use
                    if arr_use.shape[0] < n_mid:
                        v[arr_use.shape[0]:, :] = arr_use[-1, :]
                else:
                    v[:] = 0.0
                    logger.warning("  warmState: %s not found → 0", cs_name)

            # Fixed soil structure: copy mLayerDepth and iLayerHeight from original
            # coldState (soil layer thicknesses are invariant across a SUMMA run)
            for cs_name, cs_dim in [("mLayerDepth", "midToto"),
                                     ("iLayerHeight", "ifcToto")]:
                expected = n_mid if cs_dim == "midToto" else n_ifc
                v = dst.createVariable(cs_name, "f8", (cs_dim, "hru"))
                src = cold[cs_name].values
                arr = np.atleast_2d(src)
                if arr.shape[0] == 1 and arr.shape[1] >= expected:
                    arr = arr.T
                arr_use = arr[:expected, :] if arr.shape[0] >= expected else arr
                if arr_use.shape[1] != n_hru:
                    arr_use = np.tile(arr_use[:, 0:1], (1, n_hru))
                v[:] = arr_use

            dst.Author  = "seasonal_ensemble_experiment.py"
            dst.History = (
                f"Warm state from {self.longterm_nc.name} "
                f"at {actual} (created {datetime.now():%Y-%m-%d %H:%M})"
            )
            dst.Purpose = "Initial state for WY2021 seasonal ensemble"

        lt.close()
        cold.close()
        logger.info("  warmState → %s", output_path)
        return output_path


# =============================================================================
# SeasonalForcingBuilder
# =============================================================================

class SeasonalForcingBuilder:
    """Build the ensemble forcing directory for one domain.

    Output layout inside ensemble_forcing_dir/:
      target_YYYYMM.nc                   ← symlink to original
      <season>_donor<YYYY>_<YYYYMM>.nc   ← donor data with relabelled timestamps
    """

    _TIME_UNITS = "seconds since 1990-01-01"
    _CALENDAR   = "standard"

    def __init__(self, src_forcing_dir: Path, ens_forcing_dir: Path,
                 forcing_prefix: str):
        self.src_dir = src_forcing_dir
        self.dst_dir = ens_forcing_dir
        self.prefix  = forcing_prefix
        self.dst_dir.mkdir(parents=True, exist_ok=True)

    def build_target_symlinks(self):
        """Symlink the 13 target-year monthly files."""
        for yr, mo in TARGET_MONTHS:
            src = self.src_dir / f"{self.prefix}{yr:04d}{mo:02d}.nc"
            dst = self.dst_dir / f"target_{yr:04d}{mo:02d}.nc"
            if dst.exists() or dst.is_symlink():
                continue
            if not src.exists():
                raise FileNotFoundError(f"Target forcing not found: {src}")
            dst.symlink_to(src.resolve())
        logger.info("  target-year symlinks ready in %s", self.dst_dir)

    def build_donor_file(self, season: str, donor_wy: int,
                         target_year: int, target_month: int) -> Path:
        """Create one relabelled donor forcing file."""
        d_yr, d_mo = donor_calendar_month(
            season, target_year, target_month, donor_wy
        )
        src = self.src_dir / f"{self.prefix}{d_yr:04d}{d_mo:02d}.nc"
        if not src.exists():
            raise FileNotFoundError(f"Donor forcing not found: {src}")

        dst_name = (f"{season}_donor{donor_wy:04d}_"
                    f"{target_year:04d}{target_month:02d}.nc")
        dst = self.dst_dir / dst_name
        if dst.exists():
            return dst

        self._relabel_time(src, dst, target_year, target_month)
        return dst

    def build_all_donor_files(self, progress: ProgressLogger):
        """Build all donor files for every season × donor year."""
        for season, month_list in SEASONS.items():
            for donor_wy in DONOR_YEARS:
                key = progress.step_key("FORCING", season, donor_wy)
                if progress.is_done(key):
                    continue
                try:
                    for tgt_yr, tgt_mo in month_list:
                        self.build_donor_file(season, donor_wy, tgt_yr, tgt_mo)
                    progress.mark_done(key)
                except Exception as exc:
                    progress.mark_failed(key, str(exc))
                    logger.error("  Forcing %s donor %d: %s", season, donor_wy, exc)

    # ── Time relabelling ────────────────────────────────────────────────────

    def _relabel_time(self, src: Path, dst: Path,
                      target_year: int, target_month: int):
        """Copy src, replacing the time axis with target_year/month timestamps.

        Month-length mismatch handling:
          donor longer  → truncate to target length
          donor shorter → pad last timestep to fill target length
        """
        import netCDF4 as nc4
        from cftime import date2num

        with xr.open_dataset(src, mask_and_scale=False) as ds_src:
            n_hrs_donor  = ds_src.sizes["time"]
            n_days_tgt   = monthrange(target_year, target_month)[1]
            n_hrs_target = n_days_tgt * 24

            new_times = pd.date_range(
                start=f"{target_year:04d}-{target_month:02d}-01 00:00",
                periods=n_hrs_target,
                freq="h",
            )
            new_time_num = date2num(
                new_times.to_pydatetime().tolist(),
                units=self._TIME_UNITS,
                calendar=self._CALENDAR,
            )

            dst.parent.mkdir(parents=True, exist_ok=True)
            with nc4.Dataset(str(src), "r") as src_nc, \
                 nc4.Dataset(str(dst), "w") as dst_nc:

                for name, dim in src_nc.dimensions.items():
                    if name == "time":
                        dst_nc.createDimension("time", n_hrs_target)
                    else:
                        dst_nc.createDimension(
                            name, None if dim.isunlimited() else len(dim)
                        )

                src_nc.set_auto_mask(False)  # read raw bytes; avoid fill-value masking

                for name, var in src_nc.variables.items():
                    # Preserve source fill value so netCDF4 doesn't substitute its
                    # own default (9.96921e+36) when the source uses _FillValue=nan
                    fv = var._FillValue if hasattr(var, "_FillValue") else False
                    out = dst_nc.createVariable(
                        name, var.datatype, var.dimensions,
                        zlib=True, complevel=1, fill_value=fv,
                    )
                    attrs = {k: var.getncattr(k) for k in var.ncattrs()
                             if k != "_FillValue"}
                    if name == "time":
                        attrs["units"]    = self._TIME_UNITS
                        attrs["calendar"] = self._CALENDAR
                    out.setncatts(attrs)

                    if name == "time":
                        out[:] = new_time_num
                    elif "time" not in var.dimensions:
                        out[:] = var[:]
                    else:
                        t_ax  = var.dimensions.index("time")
                        data  = var[:]
                        n_use = min(n_hrs_donor, n_hrs_target)
                        sl    = [slice(None)] * data.ndim
                        sl[t_ax] = slice(0, n_use)
                        trimmed = data[tuple(sl)]

                        if n_hrs_donor < n_hrs_target:
                            # Pad by repeating last valid step
                            pad_sl       = [slice(None)] * data.ndim
                            pad_sl[t_ax] = n_hrs_donor - 1
                            last_step    = np.expand_dims(data[tuple(pad_sl)], t_ax)
                            extra = n_hrs_target - n_hrs_donor
                            tiles = [1] * data.ndim
                            tiles[t_ax] = extra
                            trimmed = np.concatenate(
                                [trimmed, np.tile(last_step, tiles)], axis=t_ax
                            )
                        out[:] = trimmed

                for k in src_nc.ncattrs():
                    dst_nc.setncattr(k, src_nc.getncattr(k))

        logger.debug("  relabelled %s → %s", src.name, dst.name)


# =============================================================================
# SeasonalSettingsManager
# =============================================================================

class SeasonalSettingsManager:
    """Set up base_settings/ and per-run directories for one domain+model.

    Layout inside <ensemble_dir>/<model>/:
      base_settings/               ← symlinks to source settings + warmState.nc
      <season>_simulations/
        donor_<YYYY>/
          fileManager.txt          ← unique per run
          forcingFileList.txt      ← unique per run
          <symlinks to base_settings files>
          output/
    """

    _SETTINGS_FILES = [
        "warmState.nc",
        "attributes.nc",
        "trialParams.nc",
        "modelDecisions.txt",
        "localParamInfo.txt",
        "basinParamInfo.txt",
        "outputControl.txt",
        "TBL_GENPARM.TBL",
        "TBL_MPTABLE.TBL",
        "TBL_SOILPARM.TBL",
        "TBL_VEGPARM.TBL",
    ]

    def __init__(self, domain: str, model: str, ensemble_dir: Path,
                 settings_src: Path, warm_state_path: Path,
                 ensemble_forcing_dir: Path):
        self.model_dir   = ensemble_dir / model
        self.base_dir    = self.model_dir / "base_settings"
        self.src_dir     = settings_src
        self.warm_state  = warm_state_path
        self.forcing_dir = ensemble_forcing_dir

    def create_base_settings(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # warmState.nc (already written by WarmStateExtractor)
        ws_link = self.base_dir / "warmState.nc"
        if not ws_link.exists() and not ws_link.is_symlink():
            ws_link.symlink_to(self.warm_state.resolve())

        for fname in self._SETTINGS_FILES:
            if fname == "warmState.nc":
                continue
            src = self.src_dir / fname
            dst = self.base_dir / fname
            if dst.exists() or dst.is_symlink():
                continue
            if src.exists():
                dst.symlink_to(src.resolve())
            else:
                logger.warning("  base_settings: %s not found in %s", fname, self.src_dir)

        # Any extra .TBL files not in the standard list
        for src in self.src_dir.glob("*.TBL"):
            dst = self.base_dir / src.name
            if not dst.exists() and not dst.is_symlink():
                dst.symlink_to(src.resolve())

        logger.info("  base_settings ready: %s", self.base_dir)

    def create_run_dir(self, season: str, donor_wy: int) -> Path:
        run_dir = self.model_dir / f"{season}_simulations" / f"donor_{donor_wy:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "output").mkdir(exist_ok=True)

        # Symlink settings from base_settings
        for fname in self._SETTINGS_FILES:
            src = self.base_dir / fname
            dst = run_dir / fname
            if not dst.exists() and not dst.is_symlink() and (src.exists() or src.is_symlink()):
                dst.symlink_to(src.resolve())

        # Unique files
        fl_path = run_dir / "forcingFileList.txt"
        if not fl_path.exists():
            fl_path.write_text(self._forcing_list(season, donor_wy) + "\n")

        fm_path = run_dir / "fileManager.txt"
        if not fm_path.exists():
            fm_path.write_text(self._file_manager(season, donor_wy, run_dir))

        return run_dir

    def _forcing_list(self, season: str, donor_wy: int) -> str:
        replaced = set(SEASONS[season])
        lines = []
        for tgt_yr, tgt_mo in TARGET_MONTHS:
            if (tgt_yr, tgt_mo) in replaced:
                fname = f"{season}_donor{donor_wy:04d}_{tgt_yr:04d}{tgt_mo:02d}.nc"
            else:
                fname = f"target_{tgt_yr:04d}{tgt_mo:02d}.nc"
            lines.append(fname)
        return "\n".join(lines)

    def _file_manager(self, season: str, donor_wy: int, run_dir: Path) -> str:
        tag = f"seasonal_{season}_d{donor_wy:04d}"
        return (
            f"controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n"
            f"simStartTime         '{SIM_START}'\n"
            f"simEndTime           '{SIM_END}'\n"
            f"tmZoneInfo           'utcTime'\n"
            f"outFilePrefix        '{tag}'\n"
            f"settingsPath         '{run_dir}/'\n"
            f"forcingPath          '{self.forcing_dir}/'\n"
            f"outputPath           '{run_dir}/output/'\n"
            f"initConditionFile    'warmState.nc'\n"
            f"attributeFile        'attributes.nc'\n"
            f"trialParamFile       'trialParams.nc'\n"
            f"forcingListFile      'forcingFileList.txt'\n"
            f"decisionsFile        'modelDecisions.txt'\n"
            f"outputControlFile    'outputControl.txt'\n"
            f"globalHruParamFile   'localParamInfo.txt'\n"
            f"globalGruParamFile   'basinParamInfo.txt'\n"
            f"vegTableFile         'TBL_VEGPARM.TBL'\n"
            f"soilTableFile        'TBL_SOILPARM.TBL'\n"
            f"generalTableFile     'TBL_GENPARM.TBL'\n"
            f"noahmpTableFile      'TBL_MPTABLE.TBL'\n"
        )


# =============================================================================
# EnsembleRunner
# =============================================================================

class EnsembleRunner:
    """Run SUMMA for each ensemble member."""

    def __init__(self, ensemble_dir: Path, domain: str, model: str,
                 progress: ProgressLogger, summa_exe: str = "summa"):
        self.ensemble_dir = ensemble_dir
        self.domain   = domain
        self.model    = model
        self.progress = progress
        self.summa    = summa_exe

    def _run_one(self, season: str, donor_wy: int, dry_run: bool) -> None:
        key = self.progress.step_key("RUN", self.domain, self.model, season, donor_wy)
        if self.progress.is_done(key):
            return

        run_dir = (self.ensemble_dir / self.model
                   / f"{season}_simulations" / f"donor_{donor_wy:04d}")
        fm_path = run_dir / "fileManager.txt"

        if not fm_path.exists():
            self.progress.mark_failed(key, "missing fileManager.txt")
            logger.error("  Missing fileManager: %s", fm_path)
            return

        if dry_run:
            logger.info("  [dry_run] %s -m %s", self.summa, fm_path)
            return

        log_file = run_dir / "summa.log"
        logger.info("  donor %d …", donor_wy)
        try:
            with open(log_file, "w") as lf:
                proc = subprocess.run(
                    [self.summa, "-m", str(fm_path)],
                    stdout=lf, stderr=subprocess.STDOUT,
                    timeout=14400,
                )
            if proc.returncode == 0:
                self.progress.mark_done(key)
            else:
                self.progress.mark_failed(key, f"rc={proc.returncode}")
                logger.error("  SUMMA rc=%d  log: %s", proc.returncode, log_file)
        except subprocess.TimeoutExpired:
            self.progress.mark_failed(key, "timeout")
        except Exception as exc:
            self.progress.mark_failed(key, str(exc))
            logger.error("  Unexpected error donor %d: %s", donor_wy, exc)

    def run_season(self, season: str, dry_run: bool = False, workers: int = 1):
        logger.info("Running %s / %s / %s  (%d members, %d workers)",
                    self.domain, self.model, season, len(DONOR_YEARS), workers)
        if workers <= 1:
            for donor_wy in DONOR_YEARS:
                self._run_one(season, donor_wy, dry_run)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._run_one, season, wy, dry_run): wy
                    for wy in DONOR_YEARS
                }
                for fut in as_completed(futures):
                    wy = futures[fut]
                    exc = fut.exception()
                    if exc:
                        logger.error("  Thread error donor %d: %s", wy, exc)


# =============================================================================
# LinearReservoirProcessor
# =============================================================================

class LinearReservoirProcessor:
    """Append basin__RoutedRunoff to SUMMA output files in-place.

    Reads averageInstantRunoff → resamples to daily → applies two-reservoir
    linear model → broadcasts back to hourly and stores as basin__RoutedRunoff.
    If JSON params are null, basin__RoutedRunoff = averageInstantRunoff (passthrough).
    """

    def __init__(self, domain: str, model: str):
        cfg       = DOMAIN_REGISTRY[domain]
        json_path = ESS_MODELING_DIR / cfg["linres_json"]
        with open(json_path) as fh:
            all_p = json.load(fh)

        params = all_p[cfg["linres_key"]].get(_LINRES_MODEL_KEY[model], {})
        self.k_fast   = params.get("k_fast")
        self.k_slow   = params.get("k_slow")
        self.f_fast   = params.get("f_fast")
        self.use_lres = (
            self.k_fast is not None
            and self.k_slow is not None
            and self.f_fast is not None
        )
        logger.info("  linRes %s/%s: k_fast=%s k_slow=%s f_fast=%s  use=%s",
                    domain, model, self.k_fast, self.k_slow,
                    self.f_fast, self.use_lres)

    def process_file(self, nc_path: Path):
        import netCDF4 as nc4

        with nc4.Dataset(str(nc_path), "a") as ds:
            if "basin__RoutedRunoff" in ds.variables:
                return

            if "averageInstantRunoff" not in ds.variables:
                logger.warning("  %s: averageInstantRunoff missing", nc_path.name)
                return

            q_raw = np.asarray(ds.variables["averageInstantRunoff"][:],
                                dtype=np.float64)
            if hasattr(q_raw, "filled"):
                q_raw = q_raw.filled(0.0)

            tv    = ds.variables["time"]
            times = xr.coding.times.decode_cf_datetime(
                tv[:], tv.units,
                calendar=getattr(tv, "calendar", "standard"),
            )
            t_idx = pd.DatetimeIndex(times)

            orig_shape = q_raw.shape          # (time,) or (time, gru)
            q2d = q_raw.reshape(len(t_idx), -1)  # → (time, n_gru)
            n_gru = q2d.shape[1]

            routed_list = []
            for g in range(n_gru):
                q_daily = (
                    pd.Series(q2d[:, g], index=t_idx)
                    .resample("1D").mean().values
                )
                if self.use_lres:
                    qr = two_reservoir_daily(
                        q_daily, self.k_fast, self.k_slow, self.f_fast
                    )
                else:
                    qr = q_daily
                # Broadcast daily → hourly (each hour gets its day's mean)
                qr_hr = np.repeat(qr, 24)[: len(t_idx)]
                routed_list.append(qr_hr)

            q_routed = np.column_stack(routed_list).reshape(orig_shape)

            src_var = ds.variables["averageInstantRunoff"]
            v = ds.createVariable(
                "basin__RoutedRunoff", "f8", src_var.dimensions,
                zlib=True, complevel=1,
            )
            v.long_name  = ("Basin-routed runoff from two-reservoir linear "
                            "model applied to averageInstantRunoff")
            v.units      = getattr(src_var, "units", "m s-1")

            v[:] = q_routed

        logger.info("  basin__RoutedRunoff added: %s", nc_path.name)

    def process_season(self, ensemble_dir: Path, model: str, season: str):
        sim_dir = ensemble_dir / model / f"{season}_simulations"
        if not sim_dir.is_dir():
            logger.warning("  No sim dir: %s", sim_dir)
            return
        for donor_dir in sorted(sim_dir.iterdir()):
            out_dir = donor_dir / "output"
            if not out_dir.is_dir():
                continue
            for nc_file in sorted(out_dir.glob("*.nc")):
                try:
                    self.process_file(nc_file)
                except Exception as exc:
                    logger.error("  linRes %s: %s", nc_file.name, exc)


# =============================================================================
# SummaryStatsComputer
# =============================================================================

class SummaryStatsComputer:
    """Compute per-run water-year metrics and write summary.csv per season."""

    def __init__(self, ensemble_dir: Path, domain: str, model: str):
        self.ensemble_dir = ensemble_dir
        self.domain = domain
        self.model  = model

    def compute_season_summary(self, season: str) -> pd.DataFrame:
        rows = []
        sim_dir = self.ensemble_dir / self.model / f"{season}_simulations"
        if not sim_dir.is_dir():
            logger.warning("  No sim dir: %s", sim_dir)
            return pd.DataFrame()

        for donor_dir in sorted(sim_dir.iterdir()):
            if not donor_dir.is_dir():
                continue
            try:
                donor_wy = int(donor_dir.name.replace("donor_", ""))
            except ValueError:
                continue
            out_files = sorted((donor_dir / "output").glob("*.nc"))
            if not out_files:
                logger.warning("  No output in %s", donor_dir / "output")
                continue
            try:
                rows.append(self._metrics(out_files[0], donor_wy, season))
            except Exception as exc:
                logger.error("  Summary %s: %s", donor_dir.name, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            csv_path = sim_dir / "summary.csv"
            df.to_csv(csv_path, index=False)
            logger.info("  summary → %s", csv_path)
        return df

    def _metrics(self, nc_path: Path, donor_wy: int, season: str) -> dict:
        ds = xr.open_dataset(nc_path)

        # Water year Oct 2020 – Sep 2021
        wy_start = pd.Timestamp("2020-10-01")
        wy_end   = pd.Timestamp("2021-09-30 23:59:59")
        ds_wy    = ds.sel(time=slice(wy_start, wy_end))

        dt_s = self._dt(ds_wy)

        # Streamflow (basin__RoutedRunoff or fallback)
        q_var = ("basin__RoutedRunoff" if "basin__RoutedRunoff" in ds_wy
                 else "averageInstantRunoff")
        q_ms  = self._collapse_spatial(ds_wy[q_var])     # m/s, (time,)

        # Precipitation from pptrate (kg/m²/s = mm/s; divide by 1000 for m/s)
        ppt_ms = self._collapse_spatial(ds_wy["pptrate"]) / 1000.0

        total_q_mm   = float(np.nansum(q_ms))   * dt_s * 1e3
        total_p_mm   = float(np.nansum(ppt_ms)) * dt_s * 1e3
        runoff_ratio = total_q_mm / total_p_mm if total_p_mm > 0 else np.nan

        # ET + sublimation: kg m-2 s-1 × dt_s = kg m-2 = mm; SUMMA sign is negative
        et = np.zeros(len(ds_wy.time))
        for vname in ("scalarTotalET",
                      "scalarCanopySublimation",
                      "scalarSnowSublimation"):
            if vname in ds_wy:
                et += self._collapse_spatial(ds_wy[vname])
        total_et_mm = -float(np.nansum(et)) * dt_s   # flip sign → positive mm lost

        # Mean aquifer storage (output variable has _mean suffix)
        mean_aq = np.nan
        for aq_name in ("scalarAquiferStorage_mean", "scalarAquiferStorage"):
            if aq_name in ds_wy:
                aq = ds_wy[aq_name]
                sp = [d for d in aq.dims if d != "time"]
                mean_aq = float(aq.mean(dim=sp).mean()) if sp else float(aq.mean())
                break

        # Mean total soil liquid (kg m-2 = mm)
        mean_soil = np.nan
        if "scalarTotalSoilLiq" in ds_wy:
            sl = ds_wy["scalarTotalSoilLiq"]
            sp = [d for d in sl.dims if d != "time"]
            mean_soil = float(sl.mean(dim=sp).mean()) if sp else float(sl.mean())

        # Mean air temperature (K → °C)
        mean_airtemp_C = np.nan
        if "airtemp" in ds_wy:
            mean_airtemp_C = float(self._collapse_spatial(ds_wy["airtemp"]).mean()) - 273.15

        # Total snowfall (kg m-2 s-1 → mm)
        total_snowfall_mm = np.nan
        if "scalarSnowfall" in ds_wy:
            sf = self._collapse_spatial(ds_wy["scalarSnowfall"])
            total_snowfall_mm = float(np.nansum(sf)) * dt_s

        # Positive SWE change: sum of hourly SWE increases (captures net snow accumulation)
        pos_swe_mm = np.nan
        swe_name = "scalarSWE" if "scalarSWE" in ds_wy else None
        if swe_name:
            swe = self._collapse_spatial(ds_wy[swe_name])
            dswe = np.diff(swe)
            pos_swe_mm = float(np.nansum(dswe[dswe > 0]))

        ds.close()
        return {
            "domain"              : self.domain,
            "model"               : self.model,
            "season"              : season,
            "donor_wy"            : donor_wy,
            "total_q_mm"          : round(total_q_mm,        2),
            "total_p_mm"          : round(total_p_mm,        2),
            "runoff_ratio"        : round(runoff_ratio,       4),
            "total_et_sublim_mm"  : round(total_et_mm,       2),
            "mean_aquifer_storage": round(mean_aq,            4),
            "mean_soil_liq_mm"    : round(mean_soil,          4),
            "mean_airtemp_C"      : round(mean_airtemp_C,     3),
            "total_snowfall_mm"   : round(total_snowfall_mm,  2) if not np.isnan(total_snowfall_mm) else np.nan,
            "pos_swe_change_mm"   : round(pos_swe_mm,         2) if not np.isnan(pos_swe_mm) else np.nan,
        }

    @staticmethod
    def _dt(ds: xr.Dataset) -> float:
        if len(ds.time) < 2:
            return 3600.0
        return (pd.Timestamp(ds.time.values[1])
                - pd.Timestamp(ds.time.values[0])).total_seconds()

    @staticmethod
    def _collapse_spatial(da: xr.DataArray) -> np.ndarray:
        sp = [d for d in da.dims if d != "time"]
        return da.mean(dim=sp).values if sp else da.values


# =============================================================================
# High-level helpers
# =============================================================================

def get_ensemble_dir(domain: str) -> Path:
    cfg = DOMAIN_REGISTRY[domain]
    return SCRATCH_BASE / cfg["data_dir"] / "simulations" / ENSEMBLE_SUBDIR


def get_progress_logger(domain: str) -> ProgressLogger:
    ens_dir = get_ensemble_dir(domain)
    ens_dir.mkdir(parents=True, exist_ok=True)
    return ProgressLogger(ens_dir / "progress.log")


def get_longterm_nc(domain: str, model: str) -> Path:
    cfg       = DOMAIN_REGISTRY[domain]
    best_sims = SCRATCH_BASE / cfg["data_dir"] / "simulations" / "best_simulations"
    pattern   = cfg["longterm_nc"][model]
    matches   = sorted(best_sims.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No longterm .nc matching '{pattern}' in {best_sims}"
        )
    return matches[0]


# =============================================================================
# Top-level step functions
# =============================================================================

def step_setup(domain: str, model: Optional[str] = None):
    """Create warm states, ensemble forcing, base_settings, and run dirs."""
    cfg      = DOMAIN_REGISTRY[domain]
    ens_dir  = get_ensemble_dir(domain)
    progress = get_progress_logger(domain)
    models   = [model] if model else list(cfg["models"].keys())

    for m in models:
        if m not in cfg["models"]:
            logger.warning("Unknown model %s for %s — skipping", m, domain)
            continue
        logger.info("=== Setup: %s / %s ===", domain, m)

        try:
            _, settings_dir, forcing_dir, _ = validate_model(domain, m)
        except FileNotFoundError as exc:
            logger.error("  validate_model failed: %s", exc)
            continue

        # ── Warm state ───────────────────────────────────────────────────
        ws_key  = progress.step_key("WARM_STATE", domain, m)
        ws_path = ens_dir / m / "base_settings" / "warmState.nc"
        if not progress.is_done(ws_key):
            try:
                longterm = get_longterm_nc(domain, m)
                cold     = settings_dir / "coldState.nc"
                if not cold.exists():
                    cands = sorted(settings_dir.glob("coldState*.nc"))
                    cold  = cands[0] if cands else cold
                WarmStateExtractor(longterm, cold).extract(ws_path)
                progress.mark_done(ws_key)
            except Exception as exc:
                progress.mark_failed(ws_key, str(exc))
                logger.error("  Warm state failed: %s", exc)
                continue

        # ── Ensemble forcing (shared across models within a domain) ──────
        try:
            prefix = get_forcing_prefix(forcing_dir)
        except Exception as exc:
            logger.error("  Cannot get forcing prefix: %s", exc)
            continue

        ens_forcing = ens_dir / "forcing_adj"
        builder     = SeasonalForcingBuilder(forcing_dir, ens_forcing, prefix)

        sym_key = progress.step_key("FORCING_SYMLINKS", domain)
        if not progress.is_done(sym_key):
            try:
                builder.build_target_symlinks()
                progress.mark_done(sym_key)
            except Exception as exc:
                progress.mark_failed(sym_key, str(exc))
                logger.error("  Target symlinks failed: %s", exc)

        builder.build_all_donor_files(progress)

        # ── base_settings ────────────────────────────────────────────────
        bs_key = progress.step_key("BASE_SETTINGS", domain, m)
        if not progress.is_done(bs_key):
            try:
                mgr = SeasonalSettingsManager(
                    domain, m, ens_dir, settings_dir, ws_path, ens_forcing
                )
                mgr.create_base_settings()
                progress.mark_done(bs_key)
            except Exception as exc:
                progress.mark_failed(bs_key, str(exc))
                logger.error("  base_settings failed: %s", exc)
                continue

        # ── Per-run directories ──────────────────────────────────────────
        mgr = SeasonalSettingsManager(
            domain, m, ens_dir, settings_dir, ws_path, ens_forcing
        )
        for season in SEASONS:
            for donor_wy in DONOR_YEARS:
                rkey = progress.step_key("RUN_DIR", domain, m, season, donor_wy)
                if progress.is_done(rkey):
                    continue
                try:
                    mgr.create_run_dir(season, donor_wy)
                    progress.mark_done(rkey)
                except Exception as exc:
                    progress.mark_failed(rkey, str(exc))
                    logger.error("  run_dir %s/%s/%d: %s", m, season, donor_wy, exc)


def step_run(domain: str, model: str, season: str,
             dry_run: bool = False, summa_exe: str = "summa", workers: int = 1):
    ens_dir  = get_ensemble_dir(domain)
    progress = get_progress_logger(domain)
    EnsembleRunner(ens_dir, domain, model, progress, summa_exe).run_season(
        season, dry_run=dry_run, workers=workers
    )


def step_postprocess(domain: str, model: str, season: str):
    ens_dir = get_ensemble_dir(domain)
    LinearReservoirProcessor(domain, model).process_season(ens_dir, model, season)
    SummaryStatsComputer(ens_dir, domain, model).compute_season_summary(season)


def step_all(domain: str, model: Optional[str] = None,
             dry_run: bool = False, summa_exe: str = "summa", workers: int = 1):
    cfg    = DOMAIN_REGISTRY[domain]
    models = [model] if model else list(cfg["models"].keys())

    step_setup(domain, model)

    for m in models:
        for season in SEASONS:
            step_run(domain, m, season, dry_run=dry_run,
                     summa_exe=summa_exe, workers=workers)
            step_postprocess(domain, m, season)


# =============================================================================
# CLI
# =============================================================================

def list_models():
    print("\nKnown domain / model combinations")
    print("=" * 60)
    total = 0
    for domain, cfg in DOMAIN_REGISTRY.items():
        for model in cfg["models"]:
            print(f"  {domain:50s}  {model}")
            total += 1
    print(f"\nTotal: {total} models  ({len(DONOR_YEARS)} donor years × 4 seasons = "
          f"{len(DONOR_YEARS) * 4} runs per model)\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Print all known domain/model combinations")

    p = sub.add_parser("setup", help="Build warm states, forcing, run dirs")
    p.add_argument("--domain", required=True, choices=list(DOMAIN_REGISTRY))
    p.add_argument("--model",  default=None, help="Specific model (default: all)")

    p = sub.add_parser("run", help="Run SUMMA ensemble members")
    p.add_argument("--domain",    required=True, choices=list(DOMAIN_REGISTRY))
    p.add_argument("--model",     required=True)
    p.add_argument("--season",    required=True, choices=list(SEASONS))
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--summa-exe", default="summa")
    p.add_argument("--workers",   type=int, default=1,
                   help="Parallel SUMMA instances (default: 1)")

    p = sub.add_parser("postprocess",
                       help="Apply linear reservoir + write summary CSV")
    p.add_argument("--domain",  required=True, choices=list(DOMAIN_REGISTRY))
    p.add_argument("--model",   required=True)
    p.add_argument("--season",  required=True, choices=list(SEASONS))

    p = sub.add_parser("all", help="Full pipeline: setup → run → postprocess")
    p.add_argument("--domain",    required=True, choices=list(DOMAIN_REGISTRY))
    p.add_argument("--model",     default=None)
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--summa-exe", default="summa")
    p.add_argument("--workers",   type=int, default=1,
                   help="Parallel SUMMA instances (default: 1)")

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "setup":
        step_setup(args.domain, args.model)
    elif args.command == "run":
        step_run(args.domain, args.model, args.season,
                 dry_run=args.dry_run, summa_exe=args.summa_exe,
                 workers=args.workers)
    elif args.command == "postprocess":
        step_postprocess(args.domain, args.model, args.season)
    elif args.command == "all":
        step_all(args.domain, args.model,
                 dry_run=args.dry_run, summa_exe=args.summa_exe,
                 workers=args.workers)


if __name__ == "__main__":
    main()
