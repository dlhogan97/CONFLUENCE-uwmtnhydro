"""synthetic_extended_runs.py

Synthetic multi-decadal SUMMA experiments using alternating wet/dry year forcing.

Wet year donor : WY2011 (Oct 2010 – Sep 2011)
Dry year donor : WY2012 (Oct 2011 – Sep 2012)

Forcing patterns (each year = one full water year, Oct–Sep):
  alt_1w1d     W-D × 5              →  10 years
  alt_2w2d     (WW-DD) × 3          →  12 years
  alt_3w3d     (WWW-DDD) × 2        →  12 years
  alt_4w4d     WWWW-DDDD            →   8 years
  alt_5w5d_wf  WWWWW-DDDDD          →  10 years (wet first)
  alt_5w5d_df  DDDDD-WWWWW          →  10 years (dry first)
  wet10_dry10  W×10 – D×10          →  20 years
  dry10_wet10  D×10 – W×10          →  20 years

All runs start from the WY2021 warm state (Aug 31 2020 23:00) already produced
by the seasonal ensemble experiment.

Usage
-----
  python synthetic_extended_runs.py setup --domain East_River_distributed [--pattern alt_1w1d]
  python synthetic_extended_runs.py run   --domain East_River_distributed --model bigBuckt [--pattern alt_1w1d] [--workers N]
  python synthetic_extended_runs.py all   --domain East_River_distributed
  python synthetic_extended_runs.py list
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import threading
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  synthetic_extended — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("synthetic_extended")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_BASE     = Path("/scratch/dlhogan/ess-project-data")
SYNTH_SUBDIR     = "synthetic_extended_runs"
SEASONAL_SUBDIR  = "seasonal_ensemble_simulations"

# ── Donor years ───────────────────────────────────────────────────────────────
WET_WY = 2011   # water year Oct 2010 – Sep 2011
DRY_WY = 2012   # water year Oct 2011 – Sep 2012

# ── Simulation time window ────────────────────────────────────────────────────
SIM_START = "2020-09-01 01:00"    # matches existing warm state date
SUMMA_EXE = "summa"

# ── Forcing patterns ──────────────────────────────────────────────────────────
# Each element is 'W' (wet) or 'D' (dry); each element = one full water year.
PATTERNS: Dict[str, List[str]] = {
    "alt_1w1d"    : ["W", "D"] * 5,
    "alt_2w2d"    : (["W"] * 2 + ["D"] * 2) * 3,
    "alt_3w3d"    : (["W"] * 3 + ["D"] * 3) * 2,
    "alt_4w4d"    : ["W"] * 4 + ["D"] * 4,
    "alt_5w5d_wf" : ["W"] * 5 + ["D"] * 5,
    "alt_5w5d_df" : ["D"] * 5 + ["W"] * 5,
    "wet10_dry10" : ["W"] * 10 + ["D"] * 10,
    "dry10_wet10" : ["D"] * 10 + ["W"] * 10,
}


def pattern_sim_end(pattern_name: str) -> str:
    n_years = len(PATTERNS[pattern_name])
    end_year = 2020 + n_years
    return f"{end_year}-09-30 23:00"


# ── Domain / model registry ───────────────────────────────────────────────────
DOMAIN_REGISTRY: Dict[str, dict] = {
    "East_River_distributed": {
        "data_dir": "domain_East_River_distributed",
        "models": {
            "bigBuckt": "bigBuckt_elevation_best_Jarvis_20260414_141650",
            "noXplict": "noXplict_elevation_best_20260414_144029",
        },
    },
    "East_River_distributed_elevTPI": {
        "data_dir": "domain_East_River_distributed_elevTPI",
        "models": {
            "bigBuckt": "bigBuckt_best",
            "noXplict": "noXplict_best",
        },
    },
    "Tuolumne_River_distributed_elev": {
        "data_dir": "domain_Tuolumne_River_distributed_elev",
        "models": {
            "qTopmodl": "qTopmodl_best",
            "noXplict": "noXplict_best",
        },
    },
    "Tuolumne_River_distributed_elevTPI": {
        "data_dir": "domain_Tuolumne_River_distributed_elevTPI",
        "models": {
            "qTopmodl": "qTopmodl_best",
            "noXplict": "noXplict_best",
        },
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_synth_dir(domain: str) -> Path:
    cfg = DOMAIN_REGISTRY[domain]
    return SCRATCH_BASE / cfg["data_dir"] / "simulations" / SYNTH_SUBDIR


def get_seasonal_base_settings(domain: str, model: str) -> Path:
    """Return the base_settings dir from the seasonal ensemble (has warmState.nc)."""
    cfg = DOMAIN_REGISTRY[domain]
    return (SCRATCH_BASE / cfg["data_dir"] / "simulations"
            / SEASONAL_SUBDIR / model / "base_settings")


def find_fm(model_dir: Path) -> Path:
    for sub in ("fileManager.txt", "settings/fileManager.txt",
                "settings/SUMMA/fileManager.txt"):
        p = model_dir / sub
        if p.exists():
            return p
    raise FileNotFoundError(f"fileManager.txt not found under {model_dir}")


def get_forcing_dir(domain: str) -> Path:
    cfg = DOMAIN_REGISTRY[domain]
    return SCRATCH_BASE / cfg["data_dir"] / "forcing" / "SUMMA_input"


def get_forcing_prefix(domain: str) -> str:
    """Detect forcing file prefix; prefer coverage of both WY2011 and WY2012."""
    forcing_dir = get_forcing_dir(domain)
    nc_files = sorted(forcing_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files in {forcing_dir}")
    prefix_dates: dict = {}
    for f in nc_files:
        m = re.match(r"^(.+_)(\d{6})$", f.stem)
        if m:
            prefix_dates.setdefault(m.group(1), []).append(int(m.group(2)))
    if not prefix_dates:
        raise ValueError(f"No YYYYMM-patterned forcing in {forcing_dir}")
    if len(prefix_dates) == 1:
        return next(iter(prefix_dates))
    # Prefer prefix covering both donor years
    need_min = (WET_WY - 1) * 100 + 10   # Oct of wet-1 (e.g. 201010)
    need_max = DRY_WY * 100 + 9           # Sep of dry year (e.g. 201209)
    for prefix, dates in sorted(prefix_dates.items(),
                                key=lambda kv: len(kv[1]), reverse=True):
        if min(dates) <= need_min and max(dates) >= need_max:
            return prefix
    return max(prefix_dates, key=lambda p: len(prefix_dates[p]))


def build_month_sequence(pattern: List[str]) -> List[Tuple[int, int, int, int]]:
    """Return list of (tgt_year, tgt_month, src_year, src_month) for the run.

    Sequence starts with Sep 2020 (lead-in using first year's donor) then
    covers n_years full water years (Oct–Sep) starting WY2021.
    """
    months: List[Tuple[int, int, int, int]] = []

    def donor_wy(c: str) -> int:
        return WET_WY if c == "W" else DRY_WY

    # Lead-in: September 2020 — use the Sep that precedes the first water year's donor
    first_donor = donor_wy(pattern[0])
    # Sep before WY start: calendar Sep of (donor_wy - 1) year
    months.append((2020, 9, first_donor - 1, 9))

    for i, year_type in enumerate(pattern):
        dwy = donor_wy(year_type)
        target_wy = 2021 + i            # WY2021 is first pattern year

        for month_offset in range(12):  # Oct, Nov, ..., Sep
            cal_month = (month_offset + 9) % 12 + 1   # 10,11,12,1,...,9
            # Calendar year within the water year
            src_cal_year = (dwy - 1) if cal_month >= 10 else dwy
            tgt_cal_year = (target_wy - 1) if cal_month >= 10 else target_wy
            months.append((tgt_cal_year, cal_month, src_cal_year, cal_month))

    return months


# ── Forcing builder ───────────────────────────────────────────────────────────

def _relabel_time(src_path: Path, dst_path: Path,
                  target_year: int, target_month: int) -> None:
    """Copy src forcing file, relabelling all timestamps to target year/month."""
    EPOCH = np.datetime64("1990-01-01T00:00:00", "s")

    with nc4.Dataset(src_path, "r") as src_nc, \
         nc4.Dataset(dst_path, "w") as dst_nc:
        src_nc.set_auto_mask(False)

        # Copy global attributes
        dst_nc.setncatts({k: src_nc.getncattr(k) for k in src_nc.ncattrs()})

        # Time axis
        src_times = src_nc.variables["time"][:]
        src_dt_s  = float(src_times[1] - src_times[0]) if len(src_times) > 1 else 3600.0
        n_src     = len(src_times)
        n_days_tgt = monthrange(target_year, target_month)[1]
        n_steps    = int(n_days_tgt * 86400 / src_dt_s)

        t0_tgt = (np.datetime64(f"{target_year:04d}-{target_month:02d}-01T01:00:00", "s")
                  - EPOCH) / np.timedelta64(1, "s")
        new_times = t0_tgt + np.arange(n_steps) * src_dt_s

        # Dimensions — use n_steps for time so leap-year padding is reflected in file size
        for name, dim in src_nc.dimensions.items():
            if name == "time":
                dst_nc.createDimension(name, None if dim.isunlimited() else n_steps)
            else:
                dst_nc.createDimension(name, None if dim.isunlimited() else len(dim))

        # Copy variables
        for name, var in src_nc.variables.items():
            fv = var._FillValue if hasattr(var, "_FillValue") else False
            out = dst_nc.createVariable(name, var.datatype, var.dimensions,
                                        zlib=True, complevel=1, fill_value=fv)
            attrs = {k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"}
            out.setncatts(attrs)

            if name == "time":
                out[:] = new_times
            elif name == "data_step":
                # Always write the actual timestep rather than copying (avoids 0-value bug)
                out[:] = int(src_dt_s)
            else:
                data = var[:]
                is_time = var.dimensions and var.dimensions[0] == "time"
                if is_time:
                    if data.shape[0] < n_steps:
                        # Pad by repeating last timestep — handles Feb with 28-day donor
                        # but 29-day target (leap year). Uses prior day's forcing.
                        pad = n_steps - data.shape[0]
                        data = np.concatenate([data, np.repeat(data[-1:], pad, axis=0)], axis=0)
                    out[:] = data[:n_steps]
                else:
                    out[:] = data


class ForcingBuilder:
    """Builds the synthetic monthly forcing files for one domain × pattern."""

    def __init__(self, domain: str, pattern_name: str, progress: "ProgressLogger"):
        self.domain       = domain
        self.pattern_name = pattern_name
        self.progress     = progress
        self.prefix       = get_forcing_prefix(domain)
        self.src_dir      = get_forcing_dir(domain)
        self.out_dir      = get_synth_dir(domain) / "forcing" / pattern_name
        self.month_seq    = build_month_sequence(PATTERNS[pattern_name])

    def build(self) -> None:
        key = f"FORCING:{self.pattern_name}"
        if self.progress.is_done(key):
            logger.info("  forcing %s: already done", self.pattern_name)
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("  Building forcing: %s / %s (%d months)",
                    self.domain, self.pattern_name, len(self.month_seq))

        for tgt_yr, tgt_mo, src_yr, src_mo in self.month_seq:
            src_name = f"{self.prefix}{src_yr:04d}{src_mo:02d}.nc"
            src_path = self.src_dir / src_name
            if not src_path.exists():
                raise FileNotFoundError(f"Source forcing not found: {src_path}")

            dst_name = f"synthetic_{tgt_yr:04d}{tgt_mo:02d}.nc"
            dst_path = self.out_dir / dst_name
            if not dst_path.exists():
                _relabel_time(src_path, dst_path, tgt_yr, tgt_mo)

        self.progress.mark_done(key)
        logger.info("  [progress] FORCING:%s → DONE", self.pattern_name)

    def write_forcing_list(self, run_dir: Path) -> None:
        """Write forcingFileList.txt into run_dir listing the synthetic files."""
        lines = [f"synthetic_{tgt_yr:04d}{tgt_mo:02d}.nc\n"
                 for tgt_yr, tgt_mo, *_ in self.month_seq]
        (run_dir / "forcingFileList.txt").write_text("".join(lines))


# ── Run setup ─────────────────────────────────────────────────────────────────

SETTINGS_FILES = [
    "attributes.nc", "trialParams.nc", "modelDecisions.txt",
    "outputControl.txt", "localParamInfo.txt", "basinParamInfo.txt",
    "TBL_VEGPARM.TBL", "TBL_SOILPARM.TBL", "TBL_GENPARM.TBL", "TBL_MPTABLE.TBL",
]


class RunSetupManager:
    """Creates the settings directory for one domain × model × pattern run."""

    def __init__(self, domain: str, model: str, pattern_name: str,
                 progress: "ProgressLogger"):
        self.domain   = domain
        self.model    = model
        self.pattern  = pattern_name
        self.progress = progress

        synth_dir      = get_synth_dir(domain)
        self.run_dir   = synth_dir / model / pattern_name
        self.force_dir = synth_dir / "forcing" / pattern_name

        # Source settings from best_simulations
        cfg = DOMAIN_REGISTRY[domain]
        best = SCRATCH_BASE / cfg["data_dir"] / "simulations" / "best_simulations"
        model_dir = best / cfg["models"][model]
        # Settings may live directly in model_dir or in a settings/ subdirectory
        self.src_settings = (model_dir / "settings"
                             if (model_dir / "settings" / "attributes.nc").exists()
                             else model_dir)

        # Warm state from seasonal ensemble base_settings
        self.warm_state = get_seasonal_base_settings(domain, model) / "warmState.nc"

    def setup(self) -> None:
        key = f"RUNDIR:{self.domain}:{self.model}:{self.pattern}"
        if self.progress.is_done(key):
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        out_dir = self.run_dir / "output"
        out_dir.mkdir(exist_ok=True)

        # Symlink settings files from source
        for fname in SETTINGS_FILES:
            src = self.src_settings / fname
            dst = self.run_dir / fname
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if src.exists():
                dst.symlink_to(src.resolve())
            else:
                logger.warning("  setup: %s not found in %s", fname, self.src_settings)

        # Symlink warm state
        ws_link = self.run_dir / "warmState.nc"
        if ws_link.exists() or ws_link.is_symlink():
            ws_link.unlink()
        ws_link.symlink_to(self.warm_state.resolve())

        # Write forcingFileList.txt
        fb = ForcingBuilder(self.domain, self.pattern,
                            self.progress)  # progress already loaded
        fb.write_forcing_list(self.run_dir)

        # Write fileManager.txt
        sim_end = pattern_sim_end(self.pattern)
        prefix  = f"synthetic_{self.domain}_{self.model}_{self.pattern}"
        fm_txt  = (
            f"controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n"
            f"simStartTime         '{SIM_START}'\n"
            f"simEndTime           '{sim_end}'\n"
            f"tmZoneInfo           'utcTime'\n"
            f"outFilePrefix        '{prefix}'\n"
            f"settingsPath         '{self.run_dir}/'\n"
            f"forcingPath          '{self.force_dir}/'\n"
            f"outputPath           '{out_dir}/'\n"
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
        (self.run_dir / "fileManager.txt").write_text(fm_txt)

        self.progress.mark_done(key)
        logger.info("  [progress] RUNDIR:%s:%s:%s → DONE",
                    self.domain, self.model, self.pattern)


# ── SUMMA runner ──────────────────────────────────────────────────────────────

class SynthRunner:
    """Runs SUMMA for one domain × model, across patterns."""

    def __init__(self, domain: str, model: str, progress: "ProgressLogger"):
        self.domain   = domain
        self.model    = model
        self.progress = progress

    def _run_one(self, pattern: str, dry_run: bool) -> None:
        run_key = f"RUN:{self.domain}:{self.model}:{pattern}"
        if self.progress.is_done(run_key):
            logger.info("  %s: already done", pattern)
            return

        synth_dir = get_synth_dir(self.domain)
        run_dir   = synth_dir / self.model / pattern
        fm_path   = run_dir / "fileManager.txt"

        if not fm_path.exists():
            self.progress.mark_failed(run_key, "missing fileManager.txt")
            logger.error("  Missing fileManager: %s", fm_path)
            return

        logger.info("  Running %s / %s / %s …", self.domain, self.model, pattern)

        if dry_run:
            logger.info("  [dry-run] would run: %s -m %s", SUMMA_EXE, fm_path)
            self.progress.mark_done(run_key)
            return

        log_path = run_dir / "summa.log"
        result = subprocess.run(
            [SUMMA_EXE, "-m", str(fm_path)],
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
            cwd=str(run_dir),
        )
        if result.returncode == 0:
            self.progress.mark_done(run_key)
            logger.info("  [progress] %s → DONE", run_key)
        else:
            self.progress.mark_failed(run_key, f"rc={result.returncode}")
            logger.error("  SUMMA failed (rc=%d): %s", result.returncode, run_key)

    def run_all(self, patterns: List[str], dry_run: bool = False,
                workers: int = 1) -> None:
        if workers <= 1:
            for p in patterns:
                self._run_one(p, dry_run)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._run_one, p, dry_run): p
                           for p in patterns}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        logger.error("  Thread error %s: %s", futures[fut], exc)


# ── Progress logger ───────────────────────────────────────────────────────────

class ProgressLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._lock    = threading.Lock()
        self._done: set = set()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if "=DONE" in line:
                    self._done.add(line.split("=DONE")[0].strip())

    def is_done(self, key: str) -> bool:
        with self._lock:
            return key in self._done

    def mark_done(self, key: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._done.add(key)
            with open(self.log_path, "a") as fh:
                fh.write(f"{key}=DONE  # {ts}\n")

    def mark_failed(self, key: str, reason: str = "") -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with open(self.log_path, "a") as fh:
                fh.write(f"{key}=FAILED  # {ts}  {reason}\n")


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_setup(domain: str, patterns: List[str]) -> None:
    synth_dir = get_synth_dir(domain)
    synth_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(synth_dir / "progress.log")
    cfg = DOMAIN_REGISTRY[domain]

    for pattern_name in patterns:
        logger.info("=== Setup forcing: %s / %s ===", domain, pattern_name)
        fb = ForcingBuilder(domain, pattern_name, progress)
        fb.build()

    for model in cfg["models"]:
        for pattern_name in patterns:
            logger.info("=== Setup run dir: %s / %s / %s ===",
                        domain, model, pattern_name)
            rm = RunSetupManager(domain, model, pattern_name, progress)
            try:
                rm.setup()
            except Exception as exc:
                logger.error("  RunSetup failed: %s", exc)


def step_run(domain: str, model: str, patterns: List[str],
             dry_run: bool = False, workers: int = 1) -> None:
    synth_dir = get_synth_dir(domain)
    progress  = ProgressLogger(synth_dir / "progress.log")
    runner    = SynthRunner(domain, model, progress)
    runner.run_all(patterns, dry_run=dry_run, workers=workers)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic multi-year SUMMA experiment (wet/dry alternating forcing)"
    )
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="Print all domains, models, and patterns")

    # setup
    p_setup = sub.add_parser("setup", help="Build forcing files and run directories")
    p_setup.add_argument("--domain", required=True, choices=list(DOMAIN_REGISTRY))
    p_setup.add_argument("--pattern", default=None,
                         help="Specific pattern (default: all)")

    # run
    p_run = sub.add_parser("run", help="Run SUMMA for domain/model")
    p_run.add_argument("--domain",  required=True, choices=list(DOMAIN_REGISTRY))
    p_run.add_argument("--model",   required=True)
    p_run.add_argument("--pattern", default=None)
    p_run.add_argument("--workers", type=int, default=1)
    p_run.add_argument("--dry-run", action="store_true")

    # all
    p_all = sub.add_parser("all", help="Setup + run all models for a domain")
    p_all.add_argument("--domain",  required=True, choices=list(DOMAIN_REGISTRY))
    p_all.add_argument("--workers", type=int, default=1)
    p_all.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "list":
        for domain, cfg in DOMAIN_REGISTRY.items():
            print(f"\n{domain}")
            for model in cfg["models"]:
                print(f"  {model}")
        print("\nPatterns:")
        for name, seq in PATTERNS.items():
            print(f"  {name:20s}  {len(seq)} years  {seq[:6]}{'...' if len(seq)>6 else ''}")
        return

    if args.command == "setup":
        patterns = [args.pattern] if args.pattern else list(PATTERNS)
        step_setup(args.domain, patterns)

    elif args.command == "run":
        if args.model not in DOMAIN_REGISTRY[args.domain]["models"]:
            parser.error(f"Model '{args.model}' not registered for {args.domain}")
        patterns = [args.pattern] if args.pattern else list(PATTERNS)
        step_run(args.domain, args.model, patterns,
                 dry_run=args.dry_run, workers=args.workers)

    elif args.command == "all":
        cfg = DOMAIN_REGISTRY[args.domain]
        step_setup(args.domain, list(PATTERNS))
        for model in cfg["models"]:
            step_run(args.domain, model, list(PATTERNS),
                     dry_run=args.dry_run, workers=args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
