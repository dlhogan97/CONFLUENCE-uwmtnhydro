#!/usr/bin/env python3
"""
summa_runner.py — Execute SUMMA and read its output for staged optimization.

Handles:
- Subprocess execution with timeout and failure detection
- Per-HRU and basin-level output reading
- Spin-up trimming
- Parallel run isolation (each trial gets its own output subdirectory)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# SUMMA error patterns that indicate numerical instability (not just a bad run).
_INSTABILITY_PATTERNS = [
    "dt < minstep",
    "failed to converge",
    "Jacobian is singular",
    "negative liquid water",
    "negative SWE",
]


def run_summa(
    exe_path: str,
    file_manager: str,
    run_id: Optional[str] = None,
    timeout_sec: int = 1800,
) -> bool:
    """Execute SUMMA and return True if it completed successfully.

    Parameters
    ----------
    exe_path:
        Path to the SUMMA executable (e.g. "summa" or "/path/to/summa").
    file_manager:
        Full path to the fileManager.txt for this trial run.
    run_id:
        Optional label used in log messages (e.g. "trial_042").
    timeout_sec:
        Kill the process after this many seconds (default: 30 min).

    Returns
    -------
    bool
        True if SUMMA exited with code 0 and produced output; False otherwise.
    """
    label = run_id or Path(file_manager).parent.name
    t0 = time.perf_counter()

    cmd = [exe_path, "-m", str(file_manager)]
    logger.debug("[%s] Running: %s", label, " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        logger.warning("[%s] SUMMA timed out after %ds", label, timeout_sec)
        return False
    except FileNotFoundError:
        logger.error("[%s] SUMMA executable not found: %s", label, exe_path)
        return False

    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        # Log the first 20 lines of stderr for diagnosis
        stderr_head = "\n".join(result.stderr.splitlines()[:20])
        logger.warning("[%s] SUMMA exited %d (%.1fs)\n%s", label, result.returncode, elapsed, stderr_head)
        return False

    # Check for numerical instability strings even in a zero-exit run
    combined = (result.stdout + result.stderr).lower()
    for pat in _INSTABILITY_PATTERNS:
        if pat.lower() in combined:
            logger.warning("[%s] Numerical instability detected: '%s'", label, pat)
            return False

    logger.debug("[%s] SUMMA completed in %.1fs", label, elapsed)
    return True


def read_hru_output(
    output_dir: str | Path,
    file_prefix: str,
    variable: str,
    spinup_days: int = 365,
    time_step_hours: float = 1.0,
) -> "Dict[int, pd.Series]":
    """Read a variable from SUMMA output for all HRUs.

    Returns pandas Series with DatetimeIndex so time-alignment with
    observations (which may have different frequency or start date) is trivial.

    Parameters
    ----------
    output_dir:
        Directory containing SUMMA output netCDF files.
    file_prefix:
        Output file prefix.
    variable:
        SUMMA output variable name (e.g., "scalarSWE").
    spinup_days:
        Number of leading days to trim from the start of output.
    time_step_hours:
        Output time step in hours.

    Returns
    -------
    dict
        Mapping zero-based hru_index → pd.Series(DatetimeIndex, float64).
    """
    import pandas as pd

    output_dir = Path(output_dir)
    candidates = sorted(output_dir.glob(f"{file_prefix}*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No SUMMA output files matching '{file_prefix}*.nc' in {output_dir}")

    spinup_steps = int(spinup_days * 24 / time_step_hours)

    result: Dict[int, "pd.Series"] = {}
    for nc_path in candidates:
        try:
            ds = xr.open_dataset(nc_path, decode_times=True)
        except Exception as e:
            logger.warning("Could not open %s: %s", nc_path, e)
            continue

        if variable not in ds:
            ds.close()
            continue

        # Build DatetimeIndex from SUMMA time coordinate
        times = pd.to_datetime(ds["time"].values)
        arr = ds[variable].values  # (time,) or (time, hru) or (time, gru, hru)

        ds.close()

        # Flatten GRU dimension if present (basin-level vars have shape (time, gru))
        if arr.ndim == 3:
            arr = arr[:, 0, :]  # take first (only) GRU
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]  # (time,) → (time, 1)

        n_hru = arr.shape[1]
        base_idx = max(result.keys(), default=-1) + 1
        for i in range(n_hru):
            s = pd.Series(arr[:, i], index=times)
            s = s.iloc[spinup_steps:]
            result[base_idx + i] = s

    if not result:
        raise ValueError(
            f"Variable '{variable}' not found in any SUMMA output file under {output_dir}"
        )

    logger.debug("Read '%s' for %d HRU(s); trimmed %d spin-up steps",
                 variable, len(result), spinup_steps)
    return result


def read_basin_output(
    output_dir: str | Path,
    file_prefix: str,
    variable: str,
    spinup_days: int = 365,
    time_step_hours: float = 1.0,
) -> "pd.Series":
    """Read a basin-aggregated SUMMA variable (e.g., averageRoutedRunoff).

    Returns a pd.Series with DatetimeIndex and spin-up trimmed.
    """
    import pandas as pd

    hru_data = read_hru_output(output_dir, file_prefix, variable, spinup_days, time_step_hours)
    if not hru_data:
        raise ValueError(f"No data for variable '{variable}'")
    series_list = list(hru_data.values())
    if len(series_list) == 1:
        return series_list[0]
    # Average across HRUs / GRUs on common index
    df = pd.concat(series_list, axis=1)
    return df.mean(axis=1)


def setup_trial_run_dir(
    base_settings_dir: str | Path,
    base_output_dir: str | Path,
    trial_id: str,
    trial_params_nc: str | Path,
    param_nc_filename: str = "trialParams.nc",
) -> tuple[Path, Path]:
    """Create an isolated working directory for a single trial run.

    Copies SUMMA settings and overwrites trialParams.nc with the trial values.

    Parameters
    ----------
    base_settings_dir:
        Source SUMMA settings directory (fileManager, modelDecisions, etc.).
    base_output_dir:
        Root directory where per-trial output folders are created.
    trial_id:
        Short identifier for this trial (e.g., "trial_042").
    trial_params_nc:
        Path to the trial-specific trialParams.nc to inject.
    param_nc_filename:
        Filename of the trialParams file expected by SUMMA.

    Returns
    -------
    (settings_dir, output_dir)
        Paths to the trial's settings and output directories.
    """
    base_settings_dir = Path(base_settings_dir)
    base_output_dir = Path(base_output_dir)
    trial_params_nc = Path(trial_params_nc)

    settings_dir = base_output_dir / trial_id / "settings"
    output_dir = base_output_dir / trial_id / "output"

    # Copy entire settings directory
    if settings_dir.exists():
        shutil.rmtree(settings_dir)
    shutil.copytree(base_settings_dir, settings_dir)

    # Overwrite trialParams.nc with the trial-specific version
    dest_params = settings_dir / param_nc_filename
    shutil.copy2(trial_params_nc, dest_params)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Trial '%s': settings=%s  output=%s", trial_id, settings_dir, output_dir)
    return settings_dir, output_dir


def patch_file_manager(
    settings_dir: Path,
    output_dir: Path,
    file_manager_name: str = "fileManager.txt",
    sim_start: Optional[str] = None,
    sim_end: Optional[str] = None,
) -> Path:
    """Update paths and optionally simulation times in fileManager.txt.

    Parameters
    ----------
    settings_dir, output_dir:
        Trial-specific paths to inject.
    sim_start, sim_end:
        If provided, override simStartTime/simEndTime in fileManager.txt.
        Format: 'YYYY-MM-DD HH:MM' (SUMMA expects quoted strings in the file).

    Returns the path to the patched fileManager.txt.
    """
    fm_path = settings_dir / file_manager_name
    if not fm_path.exists():
        raise FileNotFoundError(f"fileManager.txt not found at {fm_path}")

    lines = fm_path.read_text().splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("outputPath"):
            key_part = line[: line.index("outputPath") + len("outputPath")]
            new_lines.append(f"{key_part}    '{output_dir}/'")
        elif stripped.startswith("settingsPath"):
            key_part = line[: line.index("settingsPath") + len("settingsPath")]
            new_lines.append(f"{key_part}    '{settings_dir}/'")
        elif stripped.startswith("simStartTime") and sim_start is not None:
            key_part = line[: line.index("simStartTime") + len("simStartTime")]
            new_lines.append(f"{key_part}    '{sim_start}'")
        elif stripped.startswith("simEndTime") and sim_end is not None:
            key_part = line[: line.index("simEndTime") + len("simEndTime")]
            new_lines.append(f"{key_part}    '{sim_end}'")
        else:
            new_lines.append(line)

    fm_path.write_text("\n".join(new_lines) + "\n")
    logger.debug("Patched fileManager.txt → outputPath=%s  sim=%s→%s",
                 output_dir, sim_start, sim_end)
    return fm_path
