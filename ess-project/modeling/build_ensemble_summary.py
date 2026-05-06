"""
build_ensemble_summary.py
=========================
Aggregate all seasonal ensemble results into a single tidy CSV.

Output index:  (domain, model, season)
Output columns per metric: {metric}_mean, _std, _min, _max, _baseline

Metrics
-------
  total_q_mm            – water-year total routed runoff
  total_p_mm            – water-year total precipitation (area-weighted across HRUs)
  runoff_ratio          – total_q / total_p (WY-wide)
  total_et_sublim_mm    – ET + sublimation (area-weighted)
  min_soil_liq          – minimum total soil liquid water (area-weighted spatial mean, WY min)
  min_aquifer_storage   – minimum aquifer storage (bigBuckt only; NaN otherwise)
  peak_swe_mm           – peak SWE during water year (area-weighted spatial mean)
  days_with_snow        – days with area-weighted SWE > 1 mm
  peak_q_dowy           – day-of-water-year (1 = Oct 1) of peak routed runoff
  center_of_mass_dowy   – flow-weighted mean day-of-water-year
  seasonal_q_mm         – total runoff within the perturbed season only
  seasonal_p_mm         – total precip within the perturbed season only (area-weighted)
  seasonal_runoff_ratio – seasonal_q / seasonal_p (within-season water balance)

Baseline = WY2021 slice of the longterm best-simulation NC (unperturbed reference).

Note on area-weighting: pptrate and other HRU-level variables (SWE, soil moisture, ET)
are averaged across HRUs using fractional area weights loaded from attributes.nc.
basin__RoutedRunoff is already a basin-average flux (m/s), so no further weighting needed.

Usage
-----
  python build_ensemble_summary.py [--out PATH]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Domain / model registry (mirrors DOMAIN_REGISTRY in seasonal_ensemble_experiment.py)
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/scratch/dlhogan/ess-project-data")

REGISTRY = {
    "East_River_distributed": {
        "data_dir": "domain_East_River_distributed",
        "models": ["bigBuckt", "noXplict", "qTopmodl"],
        "longterm_nc": {
            "bigBuckt": "distributed_elevation_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevation_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elevation_qTopmodl_best_longterm.nc",
        },
    },
    "East_River_distributed_elevAspect": {
        "data_dir": "domain_East_River_distributed_elevAspect",
        "models": ["bigBuckt", "noXplict", "qTopmodl"],
        "longterm_nc": {
            "bigBuckt": "distributed_elevAspect_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevAspect_best_longterm.nc",
            "qTopmodl": "distributed_elevAspect_qTopmodl_best_longterm.nc",
        },
    },
    "East_River_distributed_elevTPI": {
        "data_dir": "domain_East_River_distributed_elevTPI",
        "models": ["bigBuckt", "noXplict", "qTopmodl"],
        "longterm_nc": {
            "bigBuckt": "distributed_elevTPI_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevTPI_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elevTPI_qTopmodl_best_longterm.nc",
        },
    },
    "Tuolumne_River_distributed_elev": {
        "data_dir": "domain_Tuolumne_River_distributed_elev",
        "models": ["bigBuckt", "noXplict", "qTopmodl"],
        "longterm_nc": {
            "bigBuckt": "distributed_elev_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elev_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elev_qTopmodl_best_longterm.nc",
        },
    },
    "Tuolumne_River_distributed_elevAspect": {
        "data_dir": "domain_Tuolumne_River_distributed_elevAspect",
        "models": ["bigBuckt", "noXplict", "qTopmodl"],
        "longterm_nc": {
            "bigBuckt": "distributed_elevAspect_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevAspect_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elevAspect_qTopmodl_best_longterm.nc",
        },
    },
    "Tuolumne_River_distributed_elevTPI": {
        "data_dir": "domain_Tuolumne_River_distributed_elevTPI",
        "models": ["bigBuckt", "noXplict", "qTopmodl"],
        "longterm_nc": {
            "bigBuckt": "distributed_elevTPI_bigBuckt_best_longterm.nc",
            "noXplict": "distributed_elevTPI_noXplict_best_longterm.nc",
            "qTopmodl": "distributed_elevTPI_qTopmodl_best_longterm.nc",
        },
    },
}

SEASONS = ["fall", "winter", "spring", "summer"]

WY_START = pd.Timestamp("2020-10-01")
WY_END   = pd.Timestamp("2021-09-30 23:59:59")

# Date ranges for each perturbed season within WY2021 (Oct 2020 – Sep 2021).
# Fall starts Oct 1 (Sep is warmup before the WY window).
SEASON_SLICES = {
    "fall":   (pd.Timestamp("2020-10-01"), pd.Timestamp("2020-11-30 23:59:59")),
    "winter": (pd.Timestamp("2020-12-01"), pd.Timestamp("2021-03-31 23:59:59")),
    "spring": (pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30 23:59:59")),
    "summer": (pd.Timestamp("2021-07-01"), pd.Timestamp("2021-08-31 23:59:59")),
}

SWE_THRESHOLD_MM = 1.0  # mm; days above this count as "snow day"


# ---------------------------------------------------------------------------
# HRU area weights
# ---------------------------------------------------------------------------

def load_hru_weights(data_dir: Path) -> np.ndarray | None:
    """Return fractional HRU area weights from attributes.nc, or None on failure."""
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
# Spatial averaging (area-weighted when weights are provided)
# ---------------------------------------------------------------------------

def _spatial_mean(da: xr.DataArray, weights: np.ndarray | None = None) -> np.ndarray:
    """Collapse all non-time dimensions to a 1-D time series.

    If `weights` is provided and the DataArray has an 'hru' dimension,
    apply weighted average across HRUs instead of a simple mean.
    """
    sp = [d for d in da.dims if d != "time"]
    if not sp:
        return da.values
    if weights is not None and "hru" in da.dims:
        # da shape: (time, hru[, ...]) — weight only the hru axis
        arr = da.values
        if arr.ndim == 2:
            return (arr * weights[None, :]).sum(axis=1)
        # fallback for higher dims
        return da.mean(dim=sp).values
    return da.mean(dim=sp).values


def _dowy(timestamps) -> np.ndarray:
    """Day-of-water-year (1 = Oct 1) for an array of timestamps."""
    ts = pd.DatetimeIndex(timestamps)
    oct1 = pd.Timestamp(f"{ts[0].year if ts[0].month >= 10 else ts[0].year - 1}-10-01")
    return ((ts - oct1).days + 1).values


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(ds_wy: xr.Dataset, season: str,
                    hru_weights: np.ndarray | None = None) -> dict:
    """Compute all metrics from a WY-sliced dataset.

    Parameters
    ----------
    ds_wy : xr.Dataset
        Dataset already sliced to the water year (Oct – Sep).
    season : str
        One of fall / winter / spring / summer — used to compute seasonal metrics.
    hru_weights : np.ndarray | None
        Fractional HRU area weights.  None falls back to simple mean.
    """
    dt_s = (
        (pd.Timestamp(ds_wy.time.values[1]) - pd.Timestamp(ds_wy.time.values[0])).total_seconds()
        if len(ds_wy.time) > 1 else 3600.0
    )

    def _wmean(da):
        return _spatial_mean(da, hru_weights)

    # Routed runoff (m/s → mm over WY); basin__RoutedRunoff is already area-normalised
    q_var = "basin__RoutedRunoff" if "basin__RoutedRunoff" in ds_wy else "averageInstantRunoff"
    q_ms  = _spatial_mean(ds_wy[q_var])          # no area-weight needed (GRU-level flux)
    total_q_mm = float(np.nansum(q_ms)) * dt_s * 1e3

    # Precipitation: kg/m²/s = mm/s; area-weight across HRUs
    ppt_ms = _wmean(ds_wy["pptrate"]) / 1000.0   # kg/m²/s → m/s depth
    total_p_mm = float(np.nansum(ppt_ms)) * dt_s * 1e3
    runoff_ratio = total_q_mm / total_p_mm if total_p_mm > 0 else np.nan

    # ET + sublimation (area-weighted)
    et = np.zeros(len(ds_wy.time))
    for v in ("scalarTotalET", "scalarCanopySublimation", "scalarSnowSublimation"):
        if v in ds_wy:
            et += _wmean(ds_wy[v])
    total_et_mm = float(np.nansum(et)) * dt_s * 1e3

    # Minimum soil liquid water (area-weighted spatial mean, then WY min)
    min_soil = np.nan
    if "scalarTotalSoilLiq" in ds_wy:
        min_soil = float(_wmean(ds_wy["scalarTotalSoilLiq"]).min())

    # Minimum aquifer storage (bigBuckt only; GRU-level so no area weight needed)
    min_aq = np.nan
    for aq_var in ("scalarAquiferStorage_mean", "scalarAquiferStorage"):
        if aq_var in ds_wy:
            min_aq = float(_spatial_mean(ds_wy[aq_var]).min())
            break

    # Peak SWE (mm, area-weighted spatial mean)
    peak_swe = np.nan
    if "scalarSWE" in ds_wy:
        swe_ts = _wmean(ds_wy["scalarSWE"]) * 1e3  # m → mm
        peak_swe = float(np.nanmax(swe_ts))

    # Days with snow (area-weighted SWE > threshold)
    days_snow = np.nan
    if "scalarSWE" in ds_wy:
        swe_ts = _wmean(ds_wy["scalarSWE"]) * 1e3
        swe_daily = (
            pd.Series(swe_ts, index=pd.DatetimeIndex(ds_wy.time.values))
            .resample("D").max()
        )
        days_snow = int((swe_daily > SWE_THRESHOLD_MM).sum())

    # Peak-Q day-of-water-year
    peak_q_dowy = np.nan
    if len(q_ms) > 0 and not np.all(np.isnan(q_ms)):
        dowy = _dowy(ds_wy.time.values)
        peak_q_dowy = float(dowy[np.nanargmax(q_ms)])

    # Center-of-mass timing (flow-weighted mean DOWY)
    com_dowy = np.nan
    if len(q_ms) > 0:
        total = np.nansum(q_ms)
        if total > 0:
            dowy = _dowy(ds_wy.time.values)
            com_dowy = float(np.nansum(q_ms * dowy) / total)

    # ------------------------------------------------------------------
    # Seasonal metrics: Q and P within the perturbed season window only
    # ------------------------------------------------------------------
    s_start, s_end = SEASON_SLICES[season]
    ds_seas = ds_wy.sel(time=slice(s_start, s_end))
    seasonal_q_mm = np.nan
    seasonal_p_mm = np.nan
    seasonal_rr    = np.nan
    if len(ds_seas.time) > 0:
        sq_ms = _spatial_mean(ds_seas[q_var])
        seasonal_q_mm = float(np.nansum(sq_ms)) * dt_s * 1e3
        sp_ms = _wmean(ds_seas["pptrate"]) / 1000.0
        seasonal_p_mm = float(np.nansum(sp_ms)) * dt_s * 1e3
        if seasonal_p_mm > 0:
            seasonal_rr = seasonal_q_mm / seasonal_p_mm

    return {
        "total_q_mm"          : round(total_q_mm,       2),
        "total_p_mm"          : round(total_p_mm,       2),
        "runoff_ratio"        : round(runoff_ratio,      4),
        "total_et_sublim_mm"  : round(total_et_mm,      2),
        "min_soil_liq"        : round(min_soil,          4),
        "min_aquifer_storage" : round(min_aq,            4),
        "peak_swe_mm"         : round(peak_swe,          2),
        "days_with_snow"      : days_snow,
        "peak_q_dowy"         : round(peak_q_dowy,       1),
        "center_of_mass_dowy" : round(com_dowy,          1),
        "seasonal_q_mm"       : round(seasonal_q_mm,    2),
        "seasonal_p_mm"       : round(seasonal_p_mm,    2),
        "seasonal_runoff_ratio": round(seasonal_rr,     4),
    }


# ---------------------------------------------------------------------------
# Per-member metrics from ensemble NC files
# ---------------------------------------------------------------------------

def collect_ensemble_metrics(domain: str, cfg: dict) -> pd.DataFrame:
    data_dir = DATA_ROOT / cfg["data_dir"]
    hru_weights = load_hru_weights(data_dir)
    ens_base = data_dir / "simulations" / "seasonal_ensemble_simulations"
    rows = []
    for model in cfg["models"]:
        for season in SEASONS:
            sim_dir = ens_base / model / f"{season}_simulations"
            if not sim_dir.is_dir():
                continue
            for donor_dir in sorted(sim_dir.iterdir()):
                if not donor_dir.is_dir() or not donor_dir.name.startswith("donor_"):
                    continue
                try:
                    donor_wy = int(donor_dir.name.replace("donor_", ""))
                except ValueError:
                    continue
                out_files = sorted((donor_dir / "output").glob("*.nc"))
                if not out_files:
                    continue
                try:
                    ds = xr.open_dataset(out_files[0])
                    ds_wy = ds.sel(time=slice(WY_START, WY_END))
                    if len(ds_wy.time) == 0:
                        ds.close()
                        continue
                    metrics = compute_metrics(ds_wy, season, hru_weights)
                    ds.close()
                    rows.append({
                        "domain": domain, "model": model,
                        "season": season, "donor_wy": donor_wy,
                        **metrics,
                    })
                except Exception as exc:
                    print(f"  WARN {domain}/{model}/{season}/donor_{donor_wy}: {exc}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline from longterm NC (WY2021 slice)
# ---------------------------------------------------------------------------

def compute_baseline(domain: str, cfg: dict) -> pd.DataFrame:
    data_dir = DATA_ROOT / cfg["data_dir"]
    hru_weights = load_hru_weights(data_dir)
    best_dir = data_dir / "simulations" / "best_simulations"
    rows = []
    for model in cfg["models"]:
        nc_name = cfg["longterm_nc"].get(model)
        if not nc_name:
            continue
        matches = sorted(best_dir.glob(nc_name))
        if not matches:
            print(f"  WARN baseline NC not found: {best_dir / nc_name}")
            continue
        nc_path = matches[0]
        try:
            ds = xr.open_dataset(nc_path)
            ds_wy = ds.sel(time=slice(WY_START, WY_END))
            if len(ds_wy.time) == 0:
                print(f"  WARN baseline WY2021 slice empty: {nc_path.name}")
                ds.close()
                continue
            for season in SEASONS:
                metrics = compute_metrics(ds_wy, season, hru_weights)
                rows.append({"domain": domain, "model": model, "season": season, **metrics})
            ds.close()
        except Exception as exc:
            print(f"  WARN baseline {domain}/{model}: {exc}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate and write
# ---------------------------------------------------------------------------

METRICS = [
    "total_q_mm", "total_p_mm", "runoff_ratio",
    "total_et_sublim_mm", "min_soil_liq", "min_aquifer_storage",
    "peak_swe_mm", "days_with_snow", "peak_q_dowy", "center_of_mass_dowy",
    "seasonal_q_mm", "seasonal_p_mm", "seasonal_runoff_ratio",
]


def build_summary(out_path: Path):
    all_ens   = []
    all_base  = []

    for domain, cfg in REGISTRY.items():
        print(f"Processing {domain} ...")
        ens_df  = collect_ensemble_metrics(domain, cfg)
        base_df = compute_baseline(domain, cfg)
        all_ens.append(ens_df)
        all_base.append(base_df)
        print(f"  ensemble rows: {len(ens_df)}  baseline rows: {len(base_df)}")

    ens_df  = pd.concat(all_ens,  ignore_index=True)
    base_df = pd.concat(all_base, ignore_index=True)

    # Aggregate ensemble → mean/std/min/max per (domain, model, season)
    idx = ["domain", "model", "season"]
    agg = (
        ens_df.groupby(idx)[METRICS]
        .agg(["mean", "std", "min", "max"])
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()

    # Merge baseline (one row per domain/model/season)
    base_cols = idx + list(METRICS)
    base_rename = {m: f"{m}_baseline" for m in METRICS}
    base_df = base_df[base_cols].rename(columns=base_rename)
    result = agg.merge(base_df, on=idx, how="left")

    # Final column order: index cols, then per-metric blocks
    col_order = idx.copy()
    for m in METRICS:
        for stat in ["mean", "std", "min", "max", "baseline"]:
            c = f"{m}_{stat}"
            if c in result.columns:
                col_order.append(c)
    result = result[col_order]

    result.to_csv(out_path, index=False)
    print(f"\nWrote {len(result)} rows → {out_path}")
    print(f"Columns: {list(result.columns)}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", type=Path,
        default=Path("/scratch/dlhogan/ess-project-data/ensemble_summary.csv"),
        help="Output CSV path (default: /scratch/dlhogan/ess-project-data/ensemble_summary.csv)",
    )
    args = parser.parse_args()
    build_summary(args.out)
