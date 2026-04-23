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
    "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/settings/SUMMA"
)
TRIAL_PARAMS_NC = Path(
    "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/lightweight_distributed/run_20260422_152322/settings/SUMMA/trialParams.nc"
)
OUTPUT_BASE = Path(
    "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations"
)
RUN_NAME = "baseline_longterm"
SIM_START = "1999-10-01 00:00"
SIM_END   = "2021-09-30 23:00"   # full water-year end
OUT_PREFIX = "qTopmodl_distributed_elevAspect_best"
SUMMA_EXE = "summa"
GROUNDWATER_OPTION = None   # None = keep whatever is in modelDecisions.txt


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
    p.add_argument("--groundwater",    default=GROUNDWATER_OPTION,
                   help="Override groundwater decision (e.g. noXplict, bigBuckt). "
                        "Shorthand for --decision groundwatr=<value>.")
    p.add_argument("--decision",       metavar="KEY=VALUE", action="append", default=[],
                   help="Override any modelDecisions.txt entry. May be repeated: "
                        "--decision stomResist=Jarvis --decision groundwatr=noXplict")
    p.add_argument("--forcing-path",   type=Path, default=None,
                   help="Override forcingPath in fileManager.txt (use for full-period runs "
                        "when the source settings only cover a subset)")
    p.add_argument("--forcing-list",   type=Path, default=None,
                   help="Replace forcingFileList.txt with this file (e.g. the 303-file "
                        "full-period list from the base settings dir)")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print plan without running SUMMA")
    args = p.parse_args()

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

    # ── Patch fileManager.txt ────────────────────────────────────────────────
    fm_path = settings_copy / "fileManager.txt"
    fm_patches = {
        "simStartTime":  args.sim_start,
        "simEndTime":    args.sim_end,
        "outFilePrefix": args.out_prefix,
        "outputPath":    str(output_dir) + "/",
        "settingsPath":  str(settings_copy) + "/",
    }
    if args.forcing_path:
        fm_patches["forcingPath"] = str(args.forcing_path) + "/"
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
