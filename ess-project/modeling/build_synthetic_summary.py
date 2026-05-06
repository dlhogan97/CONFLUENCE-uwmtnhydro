"""build_synthetic_summary.py

Aggregate all synthetic extended run outputs into a per-water-year tidy CSV.

Each run is a 10–20 year continuous simulation driven by alternating wet (WY2011)
and dry (WY2012) forcing. This script slices each output file into individual
water years, computes basin-scale metrics, and records the forcing year_type
(W/D) from the pattern definition.

Output columns:
    domain, model, pattern, water_year, year_idx, year_type,
    total_q_mm, total_p_mm, runoff_ratio,
    total_et_mm, peak_swe_mm, peak_swe_dowy, days_with_snow,
    mean_soil_liq_mm, min_aquifer_mm,
    peak_q_dowy, com_q_dowy

Usage:
    python build_synthetic_summary.py [--out PATH]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Registry (mirrors synthetic_extended_runs.py)
# ---------------------------------------------------------------------------
SCRATCH_BASE = Path("/scratch/dlhogan/ess-project-data")
SYNTH_SUBDIR = "synthetic_extended_runs"

PATTERNS: dict[str, list[str]] = {
    "alt_1w1d"    : ["W", "D"] * 5,
    "alt_2w2d"    : (["W"] * 2 + ["D"] * 2) * 3,
    "alt_3w3d"    : (["W"] * 3 + ["D"] * 3) * 2,
    "alt_4w4d"    : ["W"] * 4 + ["D"] * 4,
    "alt_5w5d_wf" : ["W"] * 5 + ["D"] * 5,
    "alt_5w5d_df" : ["D"] * 5 + ["W"] * 5,
    "wet10_dry10" : ["W"] * 10 + ["D"] * 10,
    "dry10_wet10" : ["D"] * 10 + ["W"] * 10,
}

DOMAIN_REGISTRY: dict[str, dict] = {
    "East_River_distributed": {
        "data_dir": "domain_East_River_distributed",
        "models": ["bigBuckt", "noXplict"],
    },
    "East_River_distributed_elevTPI": {
        "data_dir": "domain_East_River_distributed_elevTPI",
        "models": ["bigBuckt", "noXplict"],
    },
    "Tuolumne_River_distributed_elev": {
        "data_dir": "domain_Tuolumne_River_distributed_elev",
        "models": ["noXplict", "qTopmodl"],
    },
    "Tuolumne_River_distributed_elevTPI": {
        "data_dir": "domain_Tuolumne_River_distributed_elevTPI",
        "models": ["noXplict", "qTopmodl"],
    },
}

SWE_THRESHOLD_MM = 1.0   # mm; days above this count as a snow day
FIRST_WY = 2021          # WY2021 = Oct 2020 – Sep 2021 is pattern year_idx 0


# ---------------------------------------------------------------------------
# HRU area weights
# ---------------------------------------------------------------------------

def load_hru_weights(data_dir: Path) -> np.ndarray | None:
    attr_path = data_dir / "settings" / "SUMMA" / "attributes.nc"
    if not attr_path.exists():
        return None
    try:
        with xr.open_dataset(attr_path) as ds:
            for var in ("HRUarea", "hruArea", "area"):
                if var in ds:
                    areas = ds[var].values.ravel().astype(float)
                    return areas / areas.sum()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Spatial aggregation
# ---------------------------------------------------------------------------

def _hru_mean(da: xr.DataArray, weights: np.ndarray | None) -> np.ndarray:
    """Area-weighted mean across HRUs → 1-D time series."""
    if "hru" not in da.dims:
        return da.values.ravel() if da.ndim == 1 else da.mean(dim=[d for d in da.dims if d != "time"]).values
    arr = da.values  # (time, hru[, ...])
    if weights is not None and arr.ndim == 2:
        return (arr * weights[None, :]).sum(axis=1)
    return arr.mean(axis=tuple(range(1, arr.ndim)))


def _gru_mean(da: xr.DataArray) -> np.ndarray:
    """Simple mean across GRU (typically just 1 GRU for lumped basin output)."""
    sp = [d for d in da.dims if d != "time"]
    return da.mean(dim=sp).values if sp else da.values


# ---------------------------------------------------------------------------
# Per-water-year metric computation
# ---------------------------------------------------------------------------

def _dowy(timestamps) -> np.ndarray:
    """Day-of-water-year (1 = Oct 1) for an array of timestamps."""
    ts = pd.DatetimeIndex(timestamps)
    yr = ts[0].year if ts[0].month >= 10 else ts[0].year - 1
    oct1 = pd.Timestamp(f"{yr}-10-01")
    return ((ts - oct1).days + 1).values


def compute_wy_metrics(ds_wy: xr.Dataset, weights: np.ndarray | None) -> dict:
    """Compute all metrics from a dataset already sliced to one water year."""
    t = pd.DatetimeIndex(ds_wy.time.values)
    dt_s = (t[1] - t[0]).total_seconds() if len(t) > 1 else 3600.0

    # -- Routed runoff (m s-1 → mm yr-1) ------------------------------------
    q_var = next((v for v in ("averageRoutedRunoff", "basin__TotalRunoff",
                               "averageInstantRunoff") if v in ds_wy), None)
    q_ts = _gru_mean(ds_wy[q_var]) if q_var else np.zeros(len(t))
    total_q_mm = float(np.nansum(q_ts) * dt_s * 1e3)

    # -- Precipitation (kg m-2 s-1 → mm yr-1, area-weighted) ----------------
    ppt_ts = _hru_mean(ds_wy["pptrate"], weights) / 1000.0  # → m s-1 equivalent
    total_p_mm = float(np.nansum(ppt_ts) * dt_s * 1e3)
    runoff_ratio = total_q_mm / total_p_mm if total_p_mm > 0 else np.nan

    # -- ET (kg m-2 s-1 → mm yr-1, area-weighted) ---------------------------
    # kg m-2 s-1 × dt_s [s] = kg m-2 = mm (water density 1000 kg m-3 implicit)
    et_ts = np.zeros(len(t))
    for v in ("scalarTotalET", "scalarCanopySublimation", "scalarSnowSublimation"):
        if v in ds_wy:
            et_ts += _hru_mean(ds_wy[v], weights)
    total_et_mm = float(np.nansum(np.abs(et_ts)) * dt_s)

    # -- SWE (kg m-2 = mm, area-weighted) -----------------------------------
    peak_swe_mm = np.nan
    peak_swe_dowy = np.nan
    days_with_snow = np.nan
    if "scalarSWE" in ds_wy:
        swe_ts = _hru_mean(ds_wy["scalarSWE"], weights)  # kg m-2 = mm
        peak_swe_mm = float(np.nanmax(swe_ts))
        peak_swe_dowy = float(_dowy(t)[np.nanargmax(swe_ts)])
        swe_daily = pd.Series(swe_ts, index=t).resample("D").max()
        days_with_snow = int((swe_daily > SWE_THRESHOLD_MM).sum())

    # -- Soil moisture (kg m-2 = mm, area-weighted) -------------------------
    mean_soil_liq_mm = np.nan
    if "scalarTotalSoilLiq" in ds_wy:
        mean_soil_liq_mm = float(np.nanmean(_hru_mean(ds_wy["scalarTotalSoilLiq"], weights)))

    # -- Aquifer storage (GRU-level, bigBuckt only) -------------------------
    min_aquifer_mm = np.nan
    for aq in ("scalarAquiferStorage_mean", "scalarAquiferStorage"):
        if aq in ds_wy:
            min_aquifer_mm = float(np.nanmin(_gru_mean(ds_wy[aq])))
            break

    # -- July SWE (mean over July, area-weighted) ----------------------------
    july_swe_mm = np.nan
    if "scalarSWE" in ds_wy:
        july_mask = t.month == 7
        if july_mask.any():
            swe_ts = _hru_mean(ds_wy["scalarSWE"], weights)
            july_swe_mm = float(np.nanmean(swe_ts[july_mask]))

    # -- Runoff timing -------------------------------------------------------
    dowy = _dowy(t)
    peak_q_dowy = float(dowy[np.nanargmax(q_ts)]) if q_ts.size else np.nan
    total_q = np.nansum(q_ts)
    com_q_dowy = float(np.nansum(q_ts * dowy) / total_q) if total_q > 0 else np.nan

    return {
        "total_q_mm"      : round(total_q_mm,       2),
        "total_p_mm"      : round(total_p_mm,       2),
        "runoff_ratio"    : round(runoff_ratio,      4),
        "total_et_mm"     : round(total_et_mm,      2),
        "peak_swe_mm"     : round(peak_swe_mm,       2),
        "peak_swe_dowy"   : round(peak_swe_dowy,     1),
        "july_swe_mm"     : round(july_swe_mm,       2),
        "days_with_snow"  : days_with_snow,
        "mean_soil_liq_mm": round(mean_soil_liq_mm,  2),
        "min_aquifer_mm"  : round(min_aquifer_mm,    4),
        "peak_q_dowy"     : round(peak_q_dowy,       1),
        "com_q_dowy"      : round(com_q_dowy,        1),
    }


# ---------------------------------------------------------------------------
# Main aggregation loop
# ---------------------------------------------------------------------------

def process_run(domain: str, model: str, pattern: str,
                data_dir: Path, weights: np.ndarray | None) -> list[dict]:
    nc_dir = data_dir / "simulations" / SYNTH_SUBDIR / model / pattern / "output"
    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        print(f"  SKIP  {domain}/{model}/{pattern}: no output file")
        return []

    year_types = PATTERNS[pattern]
    rows = []
    try:
        ds = xr.open_dataset(nc_files[0])
        for year_idx, ytype in enumerate(year_types):
            wy = FIRST_WY + year_idx
            t_start = pd.Timestamp(f"{wy - 1}-10-01")
            t_end   = pd.Timestamp(f"{wy}-09-30 23:59:59")
            ds_wy = ds.sel(time=slice(t_start, t_end))
            if len(ds_wy.time) == 0:
                print(f"  WARN  {domain}/{model}/{pattern} WY{wy}: empty slice")
                continue
            metrics = compute_wy_metrics(ds_wy, weights)
            rows.append({
                "domain"    : domain,
                "model"     : model,
                "pattern"   : pattern,
                "water_year": wy,
                "year_idx"  : year_idx,
                "year_type" : ytype,
                **metrics,
            })
        ds.close()
    except Exception as exc:
        print(f"  ERROR {domain}/{model}/{pattern}: {exc}")
    return rows


def build_summary(out_path: Path) -> None:
    all_rows: list[dict] = []

    for domain, cfg in DOMAIN_REGISTRY.items():
        data_dir = SCRATCH_BASE / cfg["data_dir"]
        weights  = load_hru_weights(data_dir)
        print(f"\n{domain}  (HRU weights: {'loaded' if weights is not None else 'unavailable'})")

        for model in cfg["models"]:
            for pattern in PATTERNS:
                print(f"  {model}/{pattern} ...", end=" ", flush=True)
                rows = process_run(domain, model, pattern, data_dir, weights)
                all_rows.extend(rows)
                print(f"{len(rows)} WYs")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build synthetic run per-WY summary CSV")
    parser.add_argument(
        "--out", type=Path,
        default=SCRATCH_BASE / "synthetic_summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    build_summary(args.out)
