#!/usr/bin/env python3
"""
prepare_observations.py — Format raw East River observations for staged optimization.

Reads source files from the lumped domain observations directory and writes
formatted CSVs to the optimization/observations/ directory.

All output files have two columns:
    date  (YYYY-MM-DD)
    value (SI units)

Units after formatting:
    snotel_swe.csv      — daily SWE [mm]
    streamflow_obs.csv  — daily mean discharge [m³/s]
    et_open_et.csv      — monthly ET [mm/month]

Usage
-----
    python optimization/prepare_observations.py

    # Or for a distributed/elevAspect domain (downloads fresh ET first):
    python optimization/prepare_observations.py --domain distributed_elevAspect
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH_ROOT = Path("/scratch/dlhogan/ess-project-data")

LUMPED_OBS_DIR = SCRATCH_ROOT / "domain_East_River_lumped" / "observations"

SNOW_RAW = LUMPED_OBS_DIR / "snow" / "380_Butte_sntl_obs.csv"
STREAMFLOW_RAW = LUMPED_OBS_DIR / "streamflow" / "preprocessed" / "East_River_lumped_streamflow_processed.csv"
ET_LUMPED_RAW = LUMPED_OBS_DIR / "et" / "openet_et_ensemble_East_monthly.csv"

# Domain-specific output observation directories (formatted files written here)
DOMAIN_OBS_DIRS = {
    "lumped":                 SCRATCH_ROOT / "domain_East_River_lumped"                 / "observations" / "formatted",
    "distributed":            SCRATCH_ROOT / "domain_East_River_distributed"            / "observations" / "formatted",
    "distributed_elevAspect": SCRATCH_ROOT / "domain_East_River_distributed_elevAspect" / "observations" / "formatted",
}

# Raw OpenET downloads (written by download_distributed_et.py)
ET_RAW = {
    "lumped":                 ET_LUMPED_RAW,
    "distributed":            SCRATCH_ROOT / "domain_East_River_distributed"            / "observations" / "et" / "openet_et_ensemble_East_distributed_monthly.csv",
    "distributed_elevAspect": SCRATCH_ROOT / "domain_East_River_distributed_elevAspect" / "observations" / "et" / "openet_et_ensemble_East_elevAspect_monthly.csv",
}

INCHES_TO_MM = 25.4


# ---------------------------------------------------------------------------
# Snow — SNOTEL Butte (station 380), SWE in inches → mm, daily
# ---------------------------------------------------------------------------

def prepare_snow(out_path: Path) -> None:
    """Read Butte SNOTEL file, convert SWE inches→mm, write daily CSV."""
    print(f"Reading snow: {SNOW_RAW}")
    df = pd.read_csv(SNOW_RAW, parse_dates=["datetime"])

    # Strip timezone, floor to date (SNOTEL readings are at 08:00 UTC)
    df["date"] = df["datetime"].dt.tz_localize(None).dt.normalize()

    # Confirm units and convert
    if df["SWE_units"].iloc[0] != "in":
        raise ValueError(f"Unexpected SWE units: {df['SWE_units'].unique()}")
    df["value"] = df["SWE"].astype(float) * INCHES_TO_MM

    # Daily: one reading per day already (SNOTEL is daily), but deduplicate just in case
    daily = (
        df[["date", "value"]]
        .dropna(subset=["value"])
        .groupby("date", as_index=False)
        .mean()
    )
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    print(f"  Wrote {len(daily)} daily rows → {out_path}")
    print(f"  Date range: {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}")
    print(f"  SWE range:  {daily['value'].min():.1f} – {daily['value'].max():.1f} mm")


# ---------------------------------------------------------------------------
# Streamflow — hourly m³/s → daily mean m³/s
# ---------------------------------------------------------------------------

def prepare_streamflow(out_path: Path) -> None:
    """Resample hourly streamflow to daily mean and write CSV."""
    print(f"Reading streamflow: {STREAMFLOW_RAW}")
    df = pd.read_csv(STREAMFLOW_RAW, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "date", "discharge_cms": "value"})
    df = df.set_index("date").sort_index()

    # Resample to daily mean; require at least 18 of 24 hourly values
    daily = df["value"].resample("1D").mean()
    # Mark days with too many missing hours as NaN
    count = df["value"].resample("1D").count()
    daily[count < 18] = float("nan")
    daily = daily.dropna().reset_index()
    daily.columns = ["date", "value"]
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    print(f"  Wrote {len(daily)} daily rows → {out_path}")
    print(f"  Date range: {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}")
    print(f"  Q range:    {daily['value'].min():.3f} – {daily['value'].max():.1f} m³/s")


# ---------------------------------------------------------------------------
# ET — OpenET ensemble monthly mm/month → date, value
# ---------------------------------------------------------------------------

OPENET_FILL_VALUE = 37.642    # API returns this for months outside coverage
OPENET_START_DATE = "2016-01-01"  # OpenET ensemble reliable coverage begins ~2016

# Per-HRU raw file locations (written by download_distributed_et.py --per-hru)
ET_PER_HRU_RAW = {
    "distributed":            SCRATCH_ROOT / "domain_East_River_distributed"            / "observations" / "et" / "openet_et_per_hru_distributed_monthly.csv",
    "distributed_elevAspect": SCRATCH_ROOT / "domain_East_River_distributed_elevAspect" / "observations" / "et" / "openet_et_per_hru_distributed_elevAspect_monthly.csv",
}


def prepare_et(raw_path: Path, out_path: Path) -> None:
    """Rename OpenET ensemble CSV columns to date/value, strip fill rows, and write.

    OpenET ensemble coverage begins around 2016; earlier months are returned as
    a flat fill value (37.642 mm).  Both a date floor and a value check are
    applied so that fill rows are excluded regardless of which condition triggers.
    """
    if not raw_path.exists():
        print(f"  ET source not found: {raw_path}")
        print("  Run download_distributed_et.py first (see below).")
        return

    print(f"Reading ET: {raw_path}")
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df = df.rename(columns={"et": "value"})
    df = df[["date", "value"]].dropna()

    n_raw = len(df)

    # Drop fill rows: value == fill sentinel OR date before reliable coverage
    fill_mask = (df["value"] == OPENET_FILL_VALUE) | (df["date"] < OPENET_START_DATE)
    n_fill = fill_mask.sum()
    df = df[~fill_mask].copy()

    if n_fill:
        print(f"  Dropped {n_fill}/{n_raw} fill-value rows (pre-{OPENET_START_DATE} or value={OPENET_FILL_VALUE})")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Wrote {len(df)} monthly rows → {out_path}")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"  ET range:   {df['value'].min():.1f} – {df['value'].max():.1f} mm/month")


def prepare_et_per_hru(raw_path: Path, out_path: Path) -> None:
    """Format the wide per-HRU ET CSV: strip fill rows, write clean wide CSV.

    Input:  date, hru_1, hru_2, ..., hru_N  (from download_distributed_et.py --per-hru)
    Output: same shape, fill rows removed, date formatted as YYYY-MM-DD.

    The optimizer reads this file for per-HRU ET anchor comparisons in stage 2.
    """
    if not raw_path.exists():
        print(f"  Per-HRU ET source not found: {raw_path}")
        print("  Run: python optimization/download_distributed_et.py --domain <domain> --per-hru")
        return

    print(f"Reading per-HRU ET: {raw_path}")
    df = pd.read_csv(raw_path, parse_dates=["date"])

    n_raw = len(df)
    hru_cols = [c for c in df.columns if c != "date"]

    # A row is fill if ALL HRU columns equal the fill value or date is pre-coverage
    all_fill = (df[hru_cols] == OPENET_FILL_VALUE).all(axis=1)
    pre_coverage = df["date"] < OPENET_START_DATE
    fill_mask = all_fill | pre_coverage
    n_fill = fill_mask.sum()
    df = df[~fill_mask].copy()

    if n_fill:
        print(f"  Dropped {n_fill}/{n_raw} fill rows (pre-{OPENET_START_DATE} or all-fill)")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"  Wrote {len(df)} rows × {len(hru_cols)} HRUs → {out_path}")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    # Show per-HRU mean ET for a quick sanity check
    means = df[hru_cols].mean().round(1)
    print(f"  Annual mean ET by HRU [mm/month]:\n    {means.to_dict()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare observation files for staged optimizer")
    parser.add_argument(
        "--domain",
        default="distributed_elevAspect",
        choices=list(DOMAIN_OBS_DIRS.keys()),
        help="Target domain — sets output directory under /scratch (default: distributed_elevAspect)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default: domain scratch observations/formatted/)",
    )
    parser.add_argument(
        "--per-hru",
        action="store_true",
        help="Also format the per-HRU ET file (requires --per-hru download to have run first)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DOMAIN_OBS_DIRS[args.domain]

    print("=" * 60)
    print(f"Preparing East River observation files — domain: {args.domain}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # Snow and streamflow are the same for all distributed domain variants
    print("\n[1/3] Snow SWE (SNOTEL Butte 380)")
    prepare_snow(out_dir / "snotel_swe.csv")

    print("\n[2/3] Streamflow (daily mean)")
    prepare_streamflow(out_dir / "streamflow_obs.csv")

    # Basin-dissolved ET
    et_raw = ET_RAW[args.domain]
    print(f"\n[3/3] ET — basin dissolved (OpenET ensemble monthly)")
    if not et_raw.exists():
        _print_et_download_hint(args.domain)
    else:
        prepare_et(et_raw, out_dir / "et_open_et.csv")

    # Per-HRU ET (optional)
    if args.per_hru:
        per_hru_raw = ET_PER_HRU_RAW.get(args.domain)
        if per_hru_raw:
            print(f"\n[4/4] ET — per HRU (OpenET ensemble monthly)")
            prepare_et_per_hru(per_hru_raw, out_dir / "et_per_hru.csv")

    print(f"\nDone.  Formatted files in: {out_dir}")


def _print_et_download_hint(domain: str) -> None:
    print(f"""
  To download OpenET ET for the {domain} domain, run:
    python optimization/download_distributed_et.py --domain {domain}
  (requires OpenET API key in .env)
""")


if __name__ == "__main__":
    main()
