#!/usr/bin/env python3
"""
download_distributed_et.py — Download OpenET ET for distributed East River domains.

Two modes:
  basin  (default) — dissolves all HRUs into one polygon; single timeseries.
  per-hru          — separate request per HRU polygon; wide CSV keyed by HRU_ID.
                     Useful for per-HRU ET comparison in the staged optimizer.

Usage
-----
    # Basin-dissolved (fast, single timeseries)
    python optimization/download_distributed_et.py --domain distributed_elevAspect

    # Per-HRU (25 requests × 2 time chunks = 50 API calls; ~60–90 min)
    python optimization/download_distributed_et.py --domain distributed_elevAspect --per-hru

    # Custom date range
    python optimization/download_distributed_et.py --domain distributed_elevAspect \\
        --start-date 2016-01-01 --end-date 2022-09-30 --per-hru
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add repo root so openet_utils is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.data.openet_utils import download_openet_data, OpenETClient  # type: ignore

# ---------------------------------------------------------------------------
# Domain configurations
# ---------------------------------------------------------------------------
DOMAIN_CONFIGS = {
    "distributed": {
        "shapefile": "/scratch/dlhogan/ess-project-data/domain_East_River_distributed/shapefiles/catchment/East_River_distributed_HRUs_elevation.shp",
        "output_dir": "/scratch/dlhogan/ess-project-data/domain_East_River_distributed/observations/et",
        "output_filename": "openet_et_ensemble_East_distributed_monthly.csv",
        "label": "East River distributed (elevation bands)",
    },
    "distributed_elevAspect": {
        "shapefile": "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/shapefiles/catchment/East_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
        "output_dir": "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/observations/et",
        "output_filename": "openet_et_ensemble_East_elevAspect_monthly.csv",
        "label": "East River elevation × aspect",
    },
}

DEFAULT_START_DATE = "2007-10-01"
DEFAULT_END_DATE = "2022-09-30"
ENV_FILE = REPO_ROOT / ".env"


MAX_YEARS_PER_REQUEST = 9  # OpenET API limit is 10 years; use 9 for safety


def _date_chunks(start_date: str, end_date: str, max_years: int = MAX_YEARS_PER_REQUEST):
    """Yield (chunk_start, chunk_end) pairs that each span ≤ max_years."""
    from datetime import date
    from dateutil.relativedelta import relativedelta

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + relativedelta(years=max_years) - relativedelta(days=1), end)
        yield chunk_start.isoformat(), chunk_end.isoformat()
        chunk_start = chunk_end + relativedelta(days=1)


def download_et(domain: str, start_date: str, end_date: str) -> None:
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {list(DOMAIN_CONFIGS)}")

    cfg = DOMAIN_CONFIGS[domain]
    shapefile = Path(cfg["shapefile"])

    if not shapefile.exists():
        print(f"Shapefile not found: {shapefile}")
        print("Run the distributed workflow preprocessing steps first:")
        print("  python ess-project/modeling/04-distributed/run_east_river_distributed_workflow_light.py")
        sys.exit(1)

    if not ENV_FILE.exists():
        print(f"OpenET API key not found. Copy .env.template → .env and add your key.")
        sys.exit(1)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["output_filename"]

    print(f"Downloading OpenET ensemble ET (monthly) for {cfg['label']}")
    print(f"  Shapefile:  {shapefile}")
    print(f"  Output:     {out_path}")
    print(f"  Period:     {start_date} → {end_date}")

    chunks = list(_date_chunks(start_date, end_date))
    print(f"  Splitting into {len(chunks)} chunk(s) (API limit: {MAX_YEARS_PER_REQUEST} years)")

    all_dfs = []
    for i, (cs, ce) in enumerate(chunks, 1):
        print(f"  Chunk {i}/{len(chunks)}: {cs} → {ce}")
        df_chunk = download_openet_data(
            shapefile_path=str(shapefile),
            output_dir=str(out_dir),
            start_date=cs,
            end_date=ce,
            env_file=str(ENV_FILE),
            variable="et",
            models=["ensemble"],
            interval="monthly",
            units="mm",
            output_filename=f"_chunk_{i}_{cfg['output_filename']}",
        )
        all_dfs.append(df_chunk)

    df = pd.concat(all_dfs, ignore_index=True).sort_values("date").drop_duplicates("date")
    df.to_csv(out_path, index=False)

    # Remove temporary chunk files
    for i in range(1, len(chunks) + 1):
        tmp = out_dir / f"_chunk_{i}_{cfg['output_filename']}"
        tmp.unlink(missing_ok=True)

    print(f"Downloaded {len(df)} monthly records → {out_path}")
    print(df.head())
    print(f"\nNext: run prepare_observations.py --domain {domain}")


def download_et_per_hru(domain: str, start_date: str, end_date: str) -> None:
    """Download OpenET timeseries separately for each HRU polygon.

    Output: wide CSV with columns [date, hru_<HRU_ID>, hru_<HRU_ID>, ...]
    Filename: openet_et_per_hru_<domain>_monthly.csv

    Note: OpenET reliability starts ~2016; earlier months will be fill values
    and are stripped by prepare_observations.py.  Pass --start-date 2016-01-01
    to avoid wasting API calls on fill-value months.
    """
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {list(DOMAIN_CONFIGS)}")

    cfg = DOMAIN_CONFIGS[domain]
    shapefile = Path(cfg["shapefile"])

    if not shapefile.exists():
        print(f"Shapefile not found: {shapefile}")
        sys.exit(1)

    if not ENV_FILE.exists():
        print("OpenET API key not found. Copy .env.template → .env and add your key.")
        sys.exit(1)

    import geopandas as gpd
    gdf = gpd.read_file(shapefile)
    n_hru = len(gdf)
    chunks = list(_date_chunks(start_date, end_date))
    print(f"Per-HRU download: {n_hru} HRUs × {len(chunks)} time chunk(s) = {n_hru * len(chunks)} API calls")
    print(f"  Domain:  {cfg['label']}")
    print(f"  Period:  {start_date} → {end_date}")
    print(f"  Estimated time: {n_hru * len(chunks) * 1.5:.0f}–{n_hru * len(chunks) * 2.5:.0f} min")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    per_hru_dir = out_dir / "per_hru_chunks"
    per_hru_dir.mkdir(exist_ok=True)

    client = OpenETClient(env_file=str(ENV_FILE))

    # Collect all results: dict of HRU_ID -> list of DataFrames (one per chunk)
    hru_chunks: dict = {}

    for chunk_idx, (cs, ce) in enumerate(chunks, 1):
        print(f"\n--- Time chunk {chunk_idx}/{len(chunks)}: {cs} → {ce} ---")
        chunk_df = client.get_timeseries_per_polygon(
            shapefile_path=str(shapefile),
            start_date=cs,
            end_date=ce,
            variable="et",
            model="ensemble",
            interval="monthly",
            units="mm",
            id_column="HRU_ID",
            output_dir=per_hru_dir,
            combine=True,
            delay_between_requests=1.5,
        )

        if chunk_df.empty:
            print(f"  WARNING: no data returned for chunk {chunk_idx}")
            continue

        # chunk_df columns: date, et, polygon_id
        for hru_id, hru_df in chunk_df.groupby("polygon_id"):
            hru_key = str(hru_id)
            if hru_key not in hru_chunks:
                hru_chunks[hru_key] = []
            hru_chunks[hru_key].append(hru_df[["date", "et"]].copy())

    if not hru_chunks:
        print("No data retrieved for any HRU.")
        return

    # Concatenate chunks per HRU and pivot to wide format
    print("\nMerging chunks and pivoting to wide format...")
    per_hru_series = {}
    for hru_id, dfs in hru_chunks.items():
        combined = pd.concat(dfs).sort_values("date").drop_duplicates("date")
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.set_index("date")["et"].rename(f"hru_{hru_id}")
        per_hru_series[hru_id] = combined

    wide = pd.concat(per_hru_series.values(), axis=1).sort_index().reset_index()
    wide.columns.name = None

    out_filename = f"openet_et_per_hru_{domain}_monthly.csv"
    out_path = out_dir / out_filename
    wide.to_csv(out_path, index=False)

    # Clean up per-HRU chunk files
    import shutil
    shutil.rmtree(per_hru_dir, ignore_errors=True)

    print(f"Written: {out_path}  ({len(wide)} rows × {len(wide.columns)-1} HRUs)")
    print(wide.head(3).to_string())
    print(f"\nNext: run prepare_observations.py --domain {domain} --per-hru")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenET ET for distributed East River domain")
    parser.add_argument(
        "--domain",
        required=True,
        choices=list(DOMAIN_CONFIGS.keys()),
        help="Which distributed domain to download ET for",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--per-hru",
        action="store_true",
        help="Download separately for each HRU polygon (25 HRUs × chunks; ~60-90 min)",
    )
    args = parser.parse_args()

    if args.per_hru:
        download_et_per_hru(args.domain, args.start_date, args.end_date)
    else:
        download_et(args.domain, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
