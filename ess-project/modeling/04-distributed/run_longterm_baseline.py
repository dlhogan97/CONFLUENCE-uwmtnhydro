#!/usr/bin/env python3
"""Run the long-term baseline SUMMA simulation for the distributed East River domain.

Uses the calibrated trialParams.nc from the most recent staged optimization run.
Copies the base settings directory into a dedicated output folder, patches
fileManager.txt, and executes SUMMA.

Usage
-----
    python run_longterm_baseline.py                          # defaults below
    python run_longterm_baseline.py --start 1999-10-01 --end 2024-09-30
    python run_longterm_baseline.py --trial-params /path/to/trialParams.nc
    python run_longterm_baseline.py --dry-run   # print plan, don't run SUMMA

Forcing adjustments (edit defaults below or pass via CLI):
    python run_longterm_baseline.py \\
        --precip-mult 1.5,1.5,1.9,1.9,2.3,2.3,1.7,1.7 \\
        --lw-mult     1.4,1.5,1.7,1.8,1.25,1.25,1.35,1.35
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults — edit here or pass as CLI args
# ---------------------------------------------------------------------------
BASIN = "East_River"  # "East_River" or "East_River"
SETTINGS_DIR = Path(
    f"/scratch/dlhogan/ess-project-data/domain_{BASIN}_distributed_elevTPI/settings/SUMMA"
)
TRIAL_PARAMS_NC = Path(
    f"/scratch/dlhogan/ess-project-data/domain_{BASIN}_distributed_elevTPI/simulations/evaluation/bigBuckt_evaluation/settings/SUMMA/trialParams.nc"
)
OUTPUT_BASE = Path(
    f"/scratch/dlhogan/ess-project-data/domain_{BASIN}_distributed_elevTPI/simulations"
)
RUN_NAME = "baseline_longterm"
SIM_START = "1999-10-01 00:00"
SIM_END   = "2021-09-30 23:00"   # full water-year end
OUT_PREFIX = "bigBuckt_distributed_elevTPI_best"
SUMMA_EXE = "summa"
GROUNDWATER_OPTION = None   # None = keep whatever is in modelDecisions.txt

# Per-HRU forcing adjustments.
aspects_per_band = [1, 5, 5, 5, 5, 3, 1, 1]  # bands 0→7 (3681m → 1396m), 26 total
# Base frozenPrecipMultip per elevation band (elev_class 1-5, low to high).
# Set to scalar 1.0 to use TPI multipliers directly, or override with calibrated
# per-band values from a prior elevation-only run.


def expand_per_band(per_band, n_per_band):
    return np.array(sum([[v] * n for v, n in zip(per_band, n_per_band)], []))

if BASIN == "Tuolumne_River":
    PRECIP_MULTIPLIER_ELEV = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])  # shape (5,), ordered by elev_class 1-5
    LW_MULTIPLIER_ELEV = np.array([1.3, 1.3, 1.15, 1.1, 1., 1.0, 1.0, 1.0])
    # PRECIP_MULTIPLIER_ELEV = np.array([1.5, 1.5, 1.9, 1.9, 2.3, 2.3, 1.7, 1.7])
    # LW_MULTIPLIER_ELEV = np.array([1.3, 1.3, 1.3, 1.3, 1.15, 1.0, 1.0, 1.0])
    elev_class = "elevClass"
    tpi_class = "tpiClass"
else:
    PRECIP_MULTIPLIER_ELEV = np.array([1.2, 1.2, 1.2, 1.0, 1.0]) #BB
    LW_MULTIPLIER_ELEV     =np.array([1.0, 1.0, 1.0, 1.0, 1.0]) #BB
    # PRECIP_MULTIPLIER_ELEV = np.array([1.25,1.4,1.4,1.0,1.0])
    # LW_MULTIPLIER_ELEV     =np.array([1.25,1.20,1.20,1.,1.0]) #BB
    elev_class = "elev_class"
    tpi_class = "tpi_class" 

def expand_per_band(per_band, n_per_band):
    return np.array(sum([[v] * n for v, n in zip(per_band, n_per_band)], []))

tpi=True
if tpi == True:
    TPI_MULTIPLIERS_CSV = Path(f'/scratch/dlhogan/ess-project-data/domain_{BASIN}_distributed_elevTPI/settings/SUMMA/tpi_swe_multipliers.csv')
    TRIAL_PARAM_FILE_PATH = Path(f'/scratch/dlhogan/ess-project-data/domain_{BASIN}_distributed_elevTPI/settings/SUMMA/trialParams.nc')
    _tpi_df = pd.read_csv(TPI_MULTIPLIERS_CSV).sort_values('HRU_ID').reset_index(drop=True)
    tpi_per_band = _tpi_df.groupby([elev_class])[tpi_class].count().values[::-1]   # shape (5,), ordered by elev_class 1-5
    PRECIP_MULTIPLIER_HRU = expand_per_band(PRECIP_MULTIPLIER_ELEV, tpi_per_band)
    LW_MULTIPLIER_HRU     = expand_per_band(LW_MULTIPLIER_ELEV, tpi_per_band)
# Set to None for no adjustment, or provide a list with one value per HRU.
# These can also be overridden at runtime with --precip-mult / --lw-mult / --temp-offset.
PRECIP_MULTIPLIER: list[float] | None = PRECIP_MULTIPLIER_HRU
LW_MULTIPLIER:     list[float] | None = LW_MULTIPLIER_HRU
TEMP_OFFSET_K:     list[float] | None = None
APPLY_LW_DILLEY_OBRIEN: bool = True  # replace LWRadAtm with Dilley-O'Brien before applying LW_MULTIPLIER
 
# ---------------------------------------------------------------------------
# Resolve forcing_adjuster from the staged_calibration sibling directory
# ---------------------------------------------------------------------------
_STAGED_CAL_DIR = Path(__file__).resolve().parent.parent / "staged_calibration"
if str(_STAGED_CAL_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGED_CAL_DIR))

from forcing_adjuster import write_adjusted_forcing  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_model_decisions(md_path: Path, overrides: dict[str, str]) -> None:
    """Replace decision values in modelDecisions.txt for the given keys."""
    lines = md_path.read_text().splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        replaced = False
        for key, value in overrides.items():
            if stripped.startswith(key) and (
                len(stripped) == len(key) or not stripped[len(key)].isalnum()
            ):
                parts = line.split("!")
                comment = "!" + parts[1] if len(parts) > 1 else ""
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{key:<15}{value:<26}{comment}".rstrip())
                replaced = True
                break
        if not replaced:
            out.append(line)
    md_path.write_text("\n".join(out) + "\n")


def _patch_filemanager(fm_path: Path, patches: dict[str, str]) -> None:
    """Replace quoted values in fileManager.txt for the given keys."""
    lines = fm_path.read_text().splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        replaced = False
        for key, value in patches.items():
            if stripped.startswith(key):
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{key:<25} '{value}'")
                replaced = True
                break
        if not replaced:
            out.append(line)
    fm_path.write_text("\n".join(out) + "\n")


def _read_forcing_path_from_fm(fm_path: Path) -> Path:
    """Extract the forcingPath value from fileManager.txt."""
    for line in fm_path.read_text().splitlines():
        if line.strip().startswith("forcingPath"):
            m = re.search(r"'([^']+)'", line)
            if m:
                return Path(m.group(1))
    raise ValueError(f"forcingPath not found in {fm_path}")


def _infer_n_hru(forcing_dir: Path) -> int:
    """Return the number of HRUs from the first netCDF in forcing_dir."""
    import xarray as xr
    nc_files = sorted(forcing_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files in {forcing_dir}")
    with xr.open_dataset(nc_files[0]) as ds:
        for dim in ("hru", "gru"):
            if dim in ds.sizes:
                return ds.sizes[dim]
    raise ValueError(f"Could not determine HRU count from {nc_files[0]}")


def _parse_mult_arg(raw: str | None, label: str) -> list[float] | None:
    """Parse a comma-separated float string into a list; return None if empty."""
    if not raw:
        return None
    try:
        return [float(x.strip()) for x in raw.split(",")]
    except ValueError:
        sys.exit(f"ERROR: --{label} must be comma-separated floats, got: {raw!r}")


def _all_close(values: list[float], target: float) -> bool:
    return all(abs(v - target) < 1e-9 for v in values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run distributed East River long-term baseline")
    p.add_argument("--settings-dir",   type=Path, default=SETTINGS_DIR)
    p.add_argument("--trial-params",   type=Path, default=TRIAL_PARAMS_NC,
                   help="Calibrated trialParams.nc to use")
    p.add_argument("--output-base",    type=Path, default=OUTPUT_BASE)
    p.add_argument("--run-name",       default=RUN_NAME)
    p.add_argument("--start",          default=SIM_START, dest="sim_start")
    p.add_argument("--end",            default=SIM_END,   dest="sim_end")
    p.add_argument("--out-prefix",     default=OUT_PREFIX)
    p.add_argument("--summa-exe",      default=SUMMA_EXE)
    p.add_argument("--groundwater",    default=GROUNDWATER_OPTION,
                   help="Override groundwater decision (e.g. bigBuckt, bigBuckt). "
                        "Shorthand for --decision groundwatr=<value>.")
    p.add_argument("--decision",       metavar="KEY=VALUE", action="append", default=[],
                   help="Override any modelDecisions.txt entry. May be repeated: "
                        "--decision stomResist=Jarvis --decision groundwatr=bigBuckt")
    p.add_argument("--forcing-path",   type=Path, default=None,
                   help="Override forcingPath in fileManager.txt (use for full-period runs "
                        "when the source settings only cover a subset)")
    p.add_argument("--forcing-list",   type=Path, default=None,
                   help="Replace forcingFileList.txt with this file (e.g. the 303-file "
                        "full-period list from the base settings dir)")
    p.add_argument("--precip-mult",    default=None,
                   help="Comma-separated per-HRU precipitation multipliers "
                        "(e.g. 1.5,1.5,1.9,1.9,2.3,2.3,1.7,1.7). "
                        "Overrides PRECIP_MULTIPLIER default.")
    p.add_argument("--lw-mult",        default=None,
                   help="Comma-separated per-HRU LW radiation multipliers. "
                        "Overrides LW_MULTIPLIER default.")
    p.add_argument("--temp-offset",    default=None,
                   help="Comma-separated per-HRU additive temperature offsets in K. "
                        "Overrides TEMP_OFFSET_K default.")
    p.add_argument("--no-dilley-obrien", action="store_true",
                   help="Skip Dilley-O'Brien LW replacement (use raw ERA5 LWRadAtm).")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print plan without running SUMMA")
    args = p.parse_args()

    # ── Resolve forcing multipliers (CLI > script defaults) ─────────────────
    precip_mult      = _parse_mult_arg(args.precip_mult, "precip-mult") or PRECIP_MULTIPLIER
    lw_mult          = _parse_mult_arg(args.lw_mult,     "lw-mult")     or LW_MULTIPLIER
    temp_offset      = _parse_mult_arg(args.temp_offset, "temp-offset") or TEMP_OFFSET_K
    apply_dilley_obrien = APPLY_LW_DILLEY_OBRIEN and not args.no_dilley_obrien

    apply_precip = precip_mult is not None and not _all_close(precip_mult, 1.0)
    apply_lw     = lw_mult     is not None and not _all_close(lw_mult,     1.0)
    apply_temp   = temp_offset is not None and not _all_close(temp_offset, 0.0)
    apply_forcing_adj = apply_precip or apply_lw or apply_temp or apply_dilley_obrien

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not args.settings_dir.exists():
        sys.exit(f"ERROR: settings dir not found: {args.settings_dir}")
    if not args.trial_params.exists():
        sys.exit(f"ERROR: trial params not found: {args.trial_params}")
    if args.forcing_list and not args.forcing_list.exists():
        sys.exit(f"ERROR: forcing list not found: {args.forcing_list}")

    # ── Build run directory ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_base / f"{args.run_name}_{timestamp}"
    settings_copy = run_dir / "settings"
    output_dir    = run_dir / "output"

    print(f"Run directory : {run_dir}")
    print(f"Settings from : {args.settings_dir}")
    print(f"trialParams   : {args.trial_params}")
    print(f"Period        : {args.sim_start}  →  {args.sim_end}")
    print(f"Output prefix : {args.out_prefix}")
    if args.groundwater:
        print(f"groundwatr    : {args.groundwater} (override)")
    for d in args.decision:
        print(f"decision      : {d} (override)")
    if args.forcing_path:
        print(f"forcingPath   : {args.forcing_path} (override)")
    if args.forcing_list:
        print(f"forcingList   : {args.forcing_list} (override)")
    print(f"Dilley-O'Brien: {'yes' if apply_dilley_obrien else 'no (--no-dilley-obrien)'}")
    if apply_precip:
        print(f"precip mult   : {precip_mult}")
    if apply_lw:
        print(f"LW mult       : {lw_mult}")
    if apply_temp:
        print(f"temp offset K : {temp_offset}")

    if args.dry_run:
        print("\n[dry-run] No files written, SUMMA not launched.")
        return

    # ── Copy settings + swap in calibrated params ────────────────────────────
    print("\nCopying settings directory ...")
    shutil.copytree(args.settings_dir, settings_copy)

    print("Installing calibrated trialParams.nc ...")
    shutil.copy2(args.trial_params, settings_copy / "trialParams.nc")

    decision_overrides = {}
    if args.groundwater:
        decision_overrides["groundwatr"] = args.groundwater
    for item in args.decision:
        if "=" not in item:
            sys.exit(f"ERROR: --decision must be 'KEY=VALUE': {item!r}")
        k, v = item.split("=", 1)
        decision_overrides[k.strip()] = v.strip()
    if decision_overrides:
        md_path = settings_copy / "modelDecisions.txt"
        _patch_model_decisions(md_path, decision_overrides)
        for k, v in decision_overrides.items():
            print(f"Patched modelDecisions.txt → {k} = {v}")

    if args.forcing_list:
        shutil.copy2(args.forcing_list, settings_copy / "forcingFileList.txt")
        print(f"Installed forcingFileList.txt from {args.forcing_list}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Apply forcing adjustments if requested ───────────────────────────────
    fm_path = settings_copy / "fileManager.txt"
    fm_patches: dict[str, str] = {
        "simStartTime":  args.sim_start,
        "simEndTime":    args.sim_end,
        "outFilePrefix": args.out_prefix,
        "outputPath":    str(output_dir) + "/",
        "settingsPath":  str(settings_copy) + "/",
    }

    if args.forcing_path:
        fm_patches["forcingPath"] = str(args.forcing_path) + "/"

    if apply_forcing_adj:
        base_forcing_dir = args.forcing_path or _read_forcing_path_from_fm(fm_path)
        n_hru = _infer_n_hru(Path(base_forcing_dir))
        adj_forcing_dir = run_dir / "forcing_adj"

        print(f"\nApplying forcing adjustments ({n_hru} HRUs) → {adj_forcing_dir}")
        write_adjusted_forcing(
            base_forcing_dir=base_forcing_dir,
            trial_forcing_dir=adj_forcing_dir,
            sim_start=args.sim_start,
            sim_end=args.sim_end,
            n_hru=n_hru,
            hru_precip_multipliers=dict(enumerate(precip_mult)) if apply_precip else None,
            hru_lw_multipliers=dict(enumerate(lw_mult))         if apply_lw     else None,
            hru_temp_deltas_K=dict(enumerate(temp_offset))      if apply_temp   else None,
            dilley_obrien_lw=apply_dilley_obrien,
        )
        fm_patches["forcingPath"] = str(adj_forcing_dir) + "/"
        print(f"Adjusted forcing written.  forcingPath → {adj_forcing_dir}/")

    # ── Patch fileManager.txt ────────────────────────────────────────────────
    _patch_filemanager(fm_path, fm_patches)
    print(f"Patched fileManager.txt → output to {output_dir}/")

    # ── Run SUMMA ────────────────────────────────────────────────────────────
    cmd = [args.summa_exe, "-m", str(fm_path)]
    log_path = run_dir / "summa.log"
    print(f"\nLaunching: {' '.join(cmd)}")
    print(f"Log        : {log_path}")

    with log_path.open("w") as log_fh:
        result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        print(f"\nSUMMA finished successfully.  Output in: {output_dir}")
    else:
        print(f"\nERROR: SUMMA exited with code {result.returncode}.  Check {log_path}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
