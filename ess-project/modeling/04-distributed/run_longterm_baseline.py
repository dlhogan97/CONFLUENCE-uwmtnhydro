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
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults — edit here or pass as CLI args
# ---------------------------------------------------------------------------
SETTINGS_DIR = Path(
    "/scratch/dlhogan/ess-project-data/domain_East_River_distributed/settings/SUMMA"
)
TRIAL_PARAMS_NC = Path(
    "/scratch/dlhogan/ess-project-data/domain_East_River_distributed"
    "/optimization/staged_results/final_trialParams.nc"
)
OUTPUT_BASE = Path(
    "/scratch/dlhogan/ess-project-data/domain_East_River_distributed/simulations"
)
RUN_NAME = "baseline_longterm"
SIM_START = "1999-10-01 00:00"
SIM_END   = "2024-09-30 23:00"   # full water-year end
OUT_PREFIX = "bigBuckt_distributed_baseline_20260413"
SUMMA_EXE = "summa"


# ---------------------------------------------------------------------------

def _patch_filemanager(fm_path: Path, patches: dict[str, str]) -> None:
    """Replace quoted values in fileManager.txt for the given keys."""
    lines = fm_path.read_text().splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        replaced = False
        for key, value in patches.items():
            if stripped.startswith(key):
                # Preserve leading whitespace; replace everything after the key
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{key:<25} '{value}'")
                replaced = True
                break
        if not replaced:
            out.append(line)
    fm_path.write_text("\n".join(out) + "\n")


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
    p.add_argument("--dry-run",        action="store_true",
                   help="Print plan without running SUMMA")
    args = p.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not args.settings_dir.exists():
        sys.exit(f"ERROR: settings dir not found: {args.settings_dir}")
    if not args.trial_params.exists():
        sys.exit(f"ERROR: trial params not found: {args.trial_params}")

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

    if args.dry_run:
        print("\n[dry-run] No files written, SUMMA not launched.")
        return

    # ── Copy settings + swap in calibrated params ────────────────────────────
    print("\nCopying settings directory ...")
    shutil.copytree(args.settings_dir, settings_copy)

    print("Installing calibrated trialParams.nc ...")
    shutil.copy2(args.trial_params, settings_copy / "trialParams.nc")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Patch fileManager.txt ────────────────────────────────────────────────
    fm_path = settings_copy / "fileManager.txt"
    _patch_filemanager(fm_path, {
        "simStartTime":  args.sim_start,
        "simEndTime":    args.sim_end,
        "outFilePrefix": args.out_prefix,
        "outputPath":    str(output_dir) + "/",
        "settingsPath":  str(settings_copy) + "/",
    })
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
