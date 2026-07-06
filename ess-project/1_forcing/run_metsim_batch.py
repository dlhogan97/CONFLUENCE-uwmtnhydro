#!/usr/bin/env python3
"""Batch MetSim runner for East River monthly forcing production.

This script parallelizes monthly MetSim runs using existing monthly daily-input files,
normalizes units before MetSim, converts outputs to SUMMA-style schema, writes
flat monthly files (`metsim_YYYYMM.nc`), and cleans temporary files.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
import re
import signal
import shutil
import subprocess
import zipfile
from ftplib import FTP
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import xarray as xr

try:
    import cdsapi

    HAS_CDSAPI = True
except Exception:
    HAS_CDSAPI = False

try:
    import pydaymet as daymet

    HAS_PYDAYMET = True
except Exception:
    HAS_PYDAYMET = False

try:
    from pykrige.ok import OrdinaryKriging

    HAS_PYKRIGE = True
except Exception:
    HAS_PYKRIGE = False


@dataclass(frozen=True)
class BatchConfig:
    basin_root: Path
    out_dir: Path
    tmp_root: Path
    reference_path: Path
    dem_path: Path
    metsim_exe: Path
    workers: int
    keep_tmp: bool
    apply_mask: bool
    build_missing_daily: bool
    allow_download: bool
    day_start_hour: int
    catchment_shp: Path
    prism_ftp_host: str


def _setup_root_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("metsim_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def _month_range(start_month: str, end_month: str) -> List[str]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end-month must be >= start-month")
    out = []
    cur = start
    while cur <= end:
        out.append(str(cur))
        cur += 1
    return out


def _interpolate_to_target(x_obs, y_obs, z_obs, x_tgt, y_tgt):
    if HAS_PYKRIGE and len(z_obs) >= 8:
        try:
            ok = OrdinaryKriging(
                x_obs,
                y_obs,
                z_obs,
                variogram_model="spherical",
                verbose=False,
                enable_plotting=False,
            )
            z_tgt, _ = ok.execute("points", x_tgt, y_tgt)
            return np.asarray(z_tgt), "ordinary_kriging"
        except Exception:
            pass

    try:
        from scipy.interpolate import griddata
    except ImportError as exc:
        raise ImportError("scipy is required for fallback spatial interpolation") from exc

    z_tgt = griddata((x_obs, y_obs), z_obs, (x_tgt, y_tgt), method="linear")
    if np.any(np.isnan(z_tgt)):
        z_near = griddata((x_obs, y_obs), z_obs, (x_tgt, y_tgt), method="nearest")
        z_tgt = np.where(np.isnan(z_tgt), z_near, z_tgt)
    return z_tgt, "scipy_griddata"


def _get_target_grid(cfg: BatchConfig):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required to derive target basin grid") from exc

    catchment = gpd.read_file(cfg.catchment_shp).to_crs("EPSG:4326")
    bounds = catchment.total_bounds
    dx = 0.0416667
    dy = 0.0416667
    lons = np.arange(bounds[0] - dx, bounds[2] + dx, dx)
    lats = np.arange(bounds[1] - dy, bounds[3] + dy, dy)
    lon2d, lat2d = np.meshgrid(lons, lats)
    grid_points = gpd.GeoDataFrame(
        {"lon": lon2d.ravel(), "lat": lat2d.ravel()},
        geometry=gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()),
        crs="EPSG:4326",
    )
    selected = gpd.sjoin(grid_points, catchment[["geometry"]], how="inner", predicate="within")
    selected = selected.drop(columns=["index_right"]).reset_index(drop=True)
    return catchment, selected, bounds


def _extract_yyyymmdd(text: str):
    m = re.search(r"(\d{8})", text)
    return m.group(1) if m else None


def _prism_var_dir(var: str, year: str):
    return f"/time_series/us/an/4km/{var}/daily/{year}"


def _download_prism_monthly(var: str, month: str, dest_dir: Path, host: str) -> list[Path]:
    year = month[:4]
    mm = month[5:7]
    month_tag = f"{year}{mm}"
    out = []
    with FTP(host) as ftp:
        ftp.login()
        ftp.cwd(_prism_var_dir(var, year))
        files = sorted([f for f in ftp.nlst() if month_tag in f and f.endswith(".zip")])
        for fname in files:
            zpath = dest_dir / var / fname
            zpath.parent.mkdir(parents=True, exist_ok=True)
            if not zpath.exists():
                with open(zpath, "wb") as f:
                    ftp.retrbinary(f"RETR {fname}", f.write)
            out.append(zpath)
    return out


def _find_raster_for_date(search_dir: Path, yyyymmdd: str):
    for p in sorted(search_dir.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        if yyyymmdd not in name:
            continue
        if name.endswith(".bil") or name.endswith(".tif") or name.endswith(".tiff"):
            return p
    return None


def _sample_raster_to_selected(raster_path: Path, selected_df):
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError("rasterio and pyproj are required for PRISM sampling") from exc

    with rasterio.open(raster_path) as src:
        lons = selected_df["lon"].to_numpy()
        lats = selected_df["lat"].to_numpy()
        if src.crs is not None and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)
            pts = list(zip(xs, ys))
        else:
            pts = list(zip(lons, lats))
        vals = np.array([v[0] for v in src.sample(pts)], dtype=float)
        if src.nodata is not None:
            vals = np.where(vals == src.nodata, np.nan, vals)
        return vals


def _build_prism_monthly_to_nc(var: str, month: str, selected_df, out_path: Path, download_root: Path, ftp_host: str):
    if out_path.exists():
        return

    zips = _download_prism_monthly(var, month, download_root, ftp_host)
    if not zips:
        raise FileNotFoundError(f"No PRISM ZIPs found for {var} {month}")

    lat_vals = np.sort(selected_df["lat"].unique())
    lon_vals = np.sort(selected_df["lon"].unique())
    lat_to_i = {v: i for i, v in enumerate(lat_vals)}
    lon_to_j = {v: j for j, v in enumerate(lon_vals)}

    day_datasets = []
    tmp_extract = download_root / "tmp_extract" / var
    tmp_extract.mkdir(parents=True, exist_ok=True)

    try:
        for zpath in zips:
            yyyymmdd = _extract_yyyymmdd(zpath.name)
            if yyyymmdd is None:
                continue
            extract_dir = tmp_extract / zpath.stem
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(extract_dir)
            raster = _find_raster_for_date(extract_dir, yyyymmdd)
            if raster is None:
                continue
            sampled = _sample_raster_to_selected(raster, selected_df)
            grid = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)
            for k, val in enumerate(sampled):
                i = lat_to_i.get(selected_df.iloc[k]["lat"])
                j = lon_to_j.get(selected_df.iloc[k]["lon"])
                if i is not None and j is not None:
                    grid[i, j] = val
            out_name = {
                "tmin": "prism_tmin",
                "tmax": "prism_tmax",
                "ppt": "prism_precip",
                "soltotal": "prism_soltotal",
                "vpdmin": "prism_vpdmin",
                "vpdmax": "prism_vpdmax",
            }[var]
            ds_day = xr.Dataset(
                {out_name: (("time", "latitude", "longitude"), grid[np.newaxis, :, :])},
                coords={
                    "time": [pd.to_datetime(yyyymmdd, format="%Y%m%d")],
                    "latitude": lat_vals,
                    "longitude": lon_vals,
                },
            )
            day_datasets.append(ds_day)
    finally:
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)

    if not day_datasets:
        raise RuntimeError(f"Failed to build PRISM monthly grid for {var} {month}")

    ds_out = xr.concat(day_datasets, dim="time").sortby("time")
    units = {
        "tmin": "degC",
        "tmax": "degC",
        "ppt": "mm",
        "soltotal": "MJ m-2 day-1",
        "vpdmin": "hPa",
        "vpdmax": "hPa",
    }[var]
    out_name = list(ds_out.data_vars)[0]
    ds_out[out_name].attrs["units"] = units
    ds_out.to_netcdf(out_path)


def _download_era5_if_missing(month: str, bounds, out_path: Path):
    if out_path.exists():
        try:
            with xr.open_dataset(out_path, engine="netcdf4") as ds_exist:
                has_u = "u10" in ds_exist.data_vars or "10m_u_component_of_wind" in ds_exist.data_vars
                has_v = "v10" in ds_exist.data_vars or "10m_v_component_of_wind" in ds_exist.data_vars
            if has_u and has_v:
                return out_path
        except Exception:
            pass
        if out_path.exists():
            out_path.unlink()
    if not HAS_CDSAPI:
        raise ImportError("cdsapi is required to download ERA5-Land when missing")

    logger = logging.getLogger("metsim_batch")

    period = pd.Period(month, freq="M")
    year = f"{period.start_time.year:04d}"
    mm = f"{period.start_time.month:02d}"
    n_days = calendar.monthrange(period.start_time.year, period.start_time.month)[1]
    request = {
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": year,
        "month": mm,
        "day": [f"{d:02d}" for d in range(1, n_days + 1)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "format": "netcdf",
        "area": [float(bounds[3]), float(bounds[0]), float(bounds[1]), float(bounds[2])],
    }
    timeout_s = int(os.environ.get("ERA5_TIMEOUT_SECONDS", "900"))
    max_retries = int(os.environ.get("ERA5_MAX_RETRIES", "3"))
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("[%s] ERA5 download attempt %d/%d (timeout=%ss)", month, attempt, max_retries, timeout_s)
            c = cdsapi.Client(timeout=timeout_s, quiet=True)
            c.retrieve("reanalysis-era5-land", request, str(out_path))
            break
        except Exception as exc:
            last_err = exc
            logger.warning("[%s] ERA5 download attempt %d failed: %s", month, attempt, exc)
            if out_path.exists():
                out_path.unlink()
            if attempt == max_retries:
                raise RuntimeError(f"[{month}] ERA5 download failed after {max_retries} attempts") from last_err
    if zipfile.is_zipfile(out_path):
        unzip_target = out_path.with_name(out_path.stem + "_unzipped.nc")
        with zipfile.ZipFile(out_path, "r") as zf:
            nc_members = [m for m in zf.namelist() if m.lower().endswith(".nc")]
            if not nc_members:
                raise RuntimeError(f"No .nc found in downloaded ERA5 archive: {out_path}")
            with zf.open(nc_members[0]) as src, open(unzip_target, "wb") as dst:
                dst.write(src.read())
        out_path.unlink()
        return unzip_target
    return out_path


def _build_daymet_if_missing(month: str, catchment_gdf, selected_df, out_path: Path):
    if out_path.exists():
        return
    if not HAS_PYDAYMET:
        raise ImportError("pydaymet is required to download Daymet vapor pressure when missing")

    from shapely.geometry import Polygon

    logger = logging.getLogger("metsim_batch")

    def _timeout_handler(signum, frame):
        raise TimeoutError("Daymet request timed out")

    def _call_with_timeout(func, timeout_s: int):
        if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
            return func()
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
        try:
            return func()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    period = pd.Period(month, freq="M")
    start = period.start_time
    end = period.end_time
    polygon = Polygon(catchment_gdf.geometry.iloc[0].exterior.coords)
    timeout_s = int(os.environ.get("DAYMET_TIMEOUT_SECONDS", "900"))
    max_retries = int(os.environ.get("DAYMET_MAX_RETRIES", "3"))
    ds_daymet = None
    last_err = None
    coords = list(zip(selected_df["lon"].to_numpy(), selected_df["lat"].to_numpy()))
    for attempt in range(1, max_retries + 1):
        logger.info("[%s] Daymet by-coordinates attempt %d/%d (timeout=%ss)", month, attempt, max_retries, timeout_s)
        try:
            ds_daymet = _call_with_timeout(
                lambda: daymet.get_bycoords(
                    coords=coords,
                    dates=(start.date().isoformat(), end.date().isoformat()),
                    variables=["vp"],
                    time_scale="daily",
                    to_xarray=True,
                ),
                timeout_s,
            )
            break
        except Exception as coords_exc:
            last_err = coords_exc
            logger.warning("[%s] Daymet by-coordinates attempt %d failed: %s", month, attempt, coords_exc)

    if ds_daymet is None:
        raise RuntimeError(f"[{month}] Daymet download failed after {max_retries} attempts") from last_err

    if "id" in ds_daymet.dims:
        lon_vals = selected_df["lon"].to_numpy()
        lat_vals = selected_df["lat"].to_numpy()
        ds_daymet = ds_daymet.assign_coords(longitude=("id", lon_vals), latitude=("id", lat_vals))
        ds_daymet = ds_daymet.set_index(id=["latitude", "longitude"]).unstack("id")
        ds_daymet = ds_daymet.transpose("time", "latitude", "longitude")

    ds_daymet = ds_daymet.sortby("latitude").sortby("longitude")
    ds_daymet.to_netcdf(out_path)


def _build_missing_daily_input(month: str, cfg: BatchConfig, force_rebuild: bool = False):
    period = pd.Period(month, freq="M")
    start = period.start_time
    end = period.end_time

    monthly_root = cfg.basin_root / "forcing" / "monthly_workflow"
    raw_dir = monthly_root / "daily_sources" / month
    metsim_dir = monthly_root / "metsim" / month
    raw_dir.mkdir(parents=True, exist_ok=True)
    metsim_dir.mkdir(parents=True, exist_ok=True)

    daily_out = metsim_dir / f"metsim_daily_input_{month}.nc"
    if daily_out.exists() and not force_rebuild:
        return daily_out
    if daily_out.exists() and force_rebuild:
        daily_out.unlink()

    catchment, selected, bounds = _get_target_grid(cfg)

    prism_tmin = raw_dir / f"prism_tmin_{month}.nc"
    prism_tmax = raw_dir / f"prism_tmax_{month}.nc"
    prism_ppt = raw_dir / f"prism_precip_{month}.nc"
    prism_swrad = raw_dir / f"prism_soltotal_{month}.nc"
    prism_vpdmin = raw_dir / f"prism_vpdmin_{month}.nc"
    prism_vpdmax = raw_dir / f"prism_vpdmax_{month}.nc"
    era5_uv = raw_dir / f"era5land_u_v_{month}.nc"

    if cfg.allow_download:
        _build_prism_monthly_to_nc("tmin", month, selected, prism_tmin, raw_dir / "prism_zips", cfg.prism_ftp_host)
        _build_prism_monthly_to_nc("tmax", month, selected, prism_tmax, raw_dir / "prism_zips", cfg.prism_ftp_host)
        _build_prism_monthly_to_nc("ppt", month, selected, prism_ppt, raw_dir / "prism_zips", cfg.prism_ftp_host)
        _build_prism_monthly_to_nc("soltotal", month, selected, prism_swrad, raw_dir / "prism_zips", cfg.prism_ftp_host)
        _build_prism_monthly_to_nc("vpdmin", month, selected, prism_vpdmin, raw_dir / "prism_zips", cfg.prism_ftp_host)
        _build_prism_monthly_to_nc("vpdmax", month, selected, prism_vpdmax, raw_dir / "prism_zips", cfg.prism_ftp_host)
        era5_uv = _download_era5_if_missing(month, bounds, era5_uv)

    for p in [prism_tmin, prism_tmax, prism_ppt, prism_swrad, prism_vpdmin, prism_vpdmax, era5_uv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required source for {month}: {p}")

    try:
        with xr.open_dataset(prism_tmin, engine="netcdf4") as ds_tmin, xr.open_dataset(prism_tmax, engine="netcdf4") as ds_tmax, xr.open_dataset(
            prism_ppt, engine="netcdf4"
        ) as ds_ppt, xr.open_dataset(prism_swrad, engine="netcdf4") as ds_swrad, xr.open_dataset(
            prism_vpdmin, engine="netcdf4"
        ) as ds_vpdmin, xr.open_dataset(prism_vpdmax, engine="netcdf4") as ds_vpdmax, xr.open_dataset(era5_uv, engine="netcdf4") as ds_era5:
            target_lat = np.sort(selected["lat"].unique())
            target_lon = np.sort(selected["lon"].unique())

            era5_lat = "latitude" if "latitude" in ds_era5.coords else "lat"
            era5_lon = "longitude" if "longitude" in ds_era5.coords else "lon"
            if float(ds_era5[era5_lon].max()) > 180.0:
                ds_era5 = ds_era5.assign_coords({era5_lon: (((ds_era5[era5_lon] + 180) % 360) - 180)}).sortby(era5_lon)
            era5_time = "time" if "time" in ds_era5.coords else "valid_time"
            if era5_time != "time":
                ds_era5 = ds_era5.rename({era5_time: "time"})

            u_name = "u10" if "u10" in ds_era5.data_vars else "10m_u_component_of_wind"
            v_name = "v10" if "v10" in ds_era5.data_vars else "10m_v_component_of_wind"
            da_ws = np.hypot(ds_era5[u_name], ds_era5[v_name])

            ws_parts = []
            for t in da_ws["time"].values:
                da2d = da_ws.sel(time=t)
                src_lat = da2d[era5_lat].values
                src_lon = da2d[era5_lon].values
                src_vals = da2d.values
                src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)
                x_obs = src_lon_2d.ravel()
                y_obs = src_lat_2d.ravel()
                z_obs = src_vals.ravel()
                valid = np.isfinite(z_obs)
                x_tgt_2d, y_tgt_2d = np.meshgrid(target_lon, target_lat)
                z_tgt, _ = _interpolate_to_target(
                    x_obs[valid],
                    y_obs[valid],
                    z_obs[valid],
                    x_tgt_2d.ravel(),
                    y_tgt_2d.ravel(),
                )
                z_grid = np.asarray(z_tgt).reshape(len(target_lat), len(target_lon))
                ws_parts.append(
                    xr.DataArray(
                        z_grid,
                        dims=("latitude", "longitude"),
                        coords={"latitude": target_lat, "longitude": target_lon},
                    ).expand_dims(time=[pd.Timestamp(t)])
                )

            da_ws_daily = xr.concat(ws_parts, dim="time").sortby("time").resample(time="1D").mean()

            month_slice = slice(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
            tmin = ds_tmin["prism_tmin"].sel(time=month_slice)
            tmax = ds_tmax["prism_tmax"].sel(time=month_slice)
            precip = ds_ppt["prism_precip"].sel(time=month_slice)
            shortwave = ds_swrad["prism_soltotal"].sel(time=month_slice) * (1.0e6 / 86400.0)
            # Vapor pressure from PRISM VPD via Magnus formula: ea = es(Tmean) - VPD_mean
            # PRISM vpdmin/vpdmax are in hPa; multiply by 100 to get Pa.
            tmean = (tmin + tmax) / 2.0
            es_hpa = 6.112 * np.exp(17.67 * tmean / (tmean + 243.5))
            vpdmean_hpa = (
                ds_vpdmin["prism_vpdmin"].sel(time=month_slice)
                + ds_vpdmax["prism_vpdmax"].sel(time=month_slice)
            ) / 2.0
            vapor_pressure = (es_hpa - vpdmean_hpa).clip(min=0.01) * 100.0
            da_ws_daily = da_ws_daily.sel(time=month_slice)

            tmin, tmax, precip, shortwave, vapor_pressure, da_ws_daily = xr.align(
                tmin,
                tmax,
                precip,
                shortwave,
                vapor_pressure,
                da_ws_daily,
                join="inner",
            )

            ds_daily = xr.Dataset(
                {
                    "t_min": tmin,
                    "t_max": tmax,
                    "precip": precip,
                    "shortwave": shortwave,
                    "vapor_pressure": vapor_pressure,
                    "wind": da_ws_daily,
                },
                coords={
                    "time": tmin["time"],
                    "latitude": tmin["latitude"],
                    "longitude": tmin["longitude"],
                },
            )

            # Ensure the output contains every day in the target month, even if one
            # source product is missing an endpoint day.
            expected_days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
            ds_daily = ds_daily.assign_coords(time=pd.DatetimeIndex(ds_daily["time"].values).normalize())
            ds_daily = ds_daily.sortby("time")
            ds_daily = ds_daily.sel(time=~ds_daily.get_index("time").duplicated())
            ds_daily = ds_daily.reindex(time=expected_days)
        ds_daily = ds_daily.ffill("time").bfill("time")

        shifted = pd.DatetimeIndex(ds_daily["time"].values).normalize() + pd.Timedelta(hours=cfg.day_start_hour)
        ds_daily = ds_daily.assign_coords(time=shifted).sortby("time")
        ds_daily["t_min"].attrs["units"] = "C"
        ds_daily["t_max"].attrs["units"] = "C"
        ds_daily["precip"].attrs["units"] = "mm day-1"
        ds_daily["shortwave"].attrs["units"] = "W m-2"
        ds_daily["vapor_pressure"].attrs["units"] = "Pa"
        ds_daily["wind"].attrs["units"] = "m s-1"
        ds_daily.to_netcdf(daily_out)
    finally:
        pass

    return daily_out


def _normalize_daily_for_metsim(ds: xr.Dataset, month: str) -> xr.Dataset:
    """Ensure MetSim daily input is in MetSim physical units.

    Required output units:
    - t_min/t_max: C
    - precip: mm day-1
    - shortwave: W m-2
    - vapor_pressure: Pa
    - wind: m s-1
    """
    ds = ds.copy()

    # Temperature
    for name in ["t_min", "t_max"]:
        u = str(ds[name].attrs.get("units", "")).lower()
        med = float(np.nanmedian(ds[name].values))
        if "k" in u or med > 170.0:
            ds[name] = ds[name] - 273.15
        ds[name].attrs["units"] = "C"

    # Precipitation
    pu = str(ds["precip"].attrs.get("units", "")).lower()
    p = ds["precip"]
    if "mm day" in pu or "mm/day" in pu or pu == "mm":
        pass
    elif "mm s" in pu or "mm/s" in pu or "kg m-2 s-1" in pu or "kg m**-2 s**-1" in pu:
        p = p * 86400.0
    elif "mm hr" in pu or "mm/hr" in pu or "mm h-1" in pu:
        p = p * 24.0
    ds["precip"] = p
    ds["precip"].attrs["units"] = "mm day-1"

    # Shortwave
    su = str(ds["shortwave"].attrs.get("units", "")).lower()
    s = ds["shortwave"]
    if "w m-2" in su or "w/m2" in su or "w m**-2" in su:
        pass
    elif "mj m-2 day-1" in su or "mj/m2/day" in su:
        s = s * (1.0e6 / 86400.0)
    elif "kj m-2 day-1" in su or "kj/m2/day" in su:
        s = s * (1.0e3 / 86400.0)
    ds["shortwave"] = s
    ds["shortwave"].attrs["units"] = "W m-2"

    # Vapor pressure and wind are assumed to already be physical units
    ds["vapor_pressure"].attrs["units"] = "Pa"
    ds["wind"].attrs["units"] = "m s-1"

    # Normalize time to 08:00 month days for consistency with existing workflow
    period = pd.Period(month, freq="M")
    shifted = pd.date_range(period.start_time.normalize(), period.end_time.normalize(), freq="D") + pd.Timedelta(hours=8)
    ds = ds.sel(time=slice(shifted[0], shifted[-1]))
    ds = ds.assign_coords(time=shifted[: ds.sizes["time"]])

    return ds


def _build_domain_from_daily(ds_daily: xr.Dataset, dem_path: Path, domain_path: Path) -> None:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to build a MetSim domain from DEM when no precomputed domain is available"
        ) from exc

    target_lat = np.asarray(ds_daily["latitude"].values)
    target_lon = np.asarray(ds_daily["longitude"].values)
    lon2d, lat2d = np.meshgrid(target_lon, target_lat)

    with rasterio.open(dem_path) as dem_src:
        pts = list(zip(lon2d.ravel(), lat2d.ravel()))
        elev = np.array([v[0] for v in dem_src.sample(pts)], dtype=float).reshape(lat2d.shape)
        if dem_src.nodata is not None:
            elev = np.where(elev == dem_src.nodata, np.nan, elev)

    mask = np.isfinite(ds_daily["t_min"].isel(time=0)).astype(np.int8).values
    elev = np.where(mask > 0, elev, np.nan)

    ds_domain = xr.Dataset(
        {
            "mask": (("latitude", "longitude"), mask),
            "elev": (("latitude", "longitude"), elev.astype(np.float32)),
            "lat": (("latitude", "longitude"), np.broadcast_to(lat2d, mask.shape).astype(np.float32)),
            "lon": (("latitude", "longitude"), np.broadcast_to(lon2d, mask.shape).astype(np.float32)),
        },
        coords={"latitude": target_lat, "longitude": target_lon},
    )
    ds_domain["mask"].attrs["units"] = "1"
    ds_domain["elev"].attrs["units"] = "m"
    ds_domain["lat"].attrs["units"] = "degrees_north"
    ds_domain["lon"].attrs["units"] = "degrees_east"
    ds_domain.to_netcdf(domain_path)


def _build_state_from_daily(ds_daily: xr.Dataset, state_path: Path, month: str) -> None:
    start = pd.Period(month, freq="M").start_time.normalize()
    state_dates = pd.date_range(end=start - pd.Timedelta(days=1), periods=90, freq="D")

    seed_tmin = ds_daily["t_min"].isel(time=0).astype(np.float32)
    seed_tmax = ds_daily["t_max"].isel(time=0).astype(np.float32)
    seed_prec = ds_daily["precip"].isel(time=0).astype(np.float32)

    state_tmin = xr.concat([seed_tmin] * len(state_dates), dim="time").assign_coords(time=state_dates)
    state_tmax = xr.concat([seed_tmax] * len(state_dates), dim="time").assign_coords(time=state_dates)
    state_prec = xr.concat([seed_prec] * len(state_dates), dim="time").assign_coords(time=state_dates)

    ds_state = xr.Dataset(
        {
            "t_min": state_tmin,
            "t_max": state_tmax,
            "precip": state_prec,
        },
        coords={
            "time": state_dates,
            "latitude": ds_daily["latitude"],
            "longitude": ds_daily["longitude"],
        },
    )
    ds_state["t_min"].attrs["units"] = "C"
    ds_state["t_max"].attrs["units"] = "C"
    ds_state["precip"].attrs["units"] = "mm day-1"
    ds_state.to_netcdf(state_path)


def _write_metsim_ini(month: str, daily_path: Path, state_path: Path, domain_path: Path, out_dir: Path, ini_path: Path) -> None:
    period = pd.Period(month, freq="M")
    start = period.start_time.normalize()
    stop = period.end_time.normalize()

    lines = [
        "[MetSim]",
        f"start = {start:%Y-%m-%d}",
        f"stop = {stop:%Y-%m-%d}",
        "time_step = 60",
        f"forcing = {daily_path}",
        f"state = {state_path}",
        f"domain = {domain_path}",
        f"out_dir = {out_dir}",
        "forcing_fmt = netcdf",
        "lw_type = prata",
        f"out_prefix = metsim_{month.replace('-', '')}",
        "",
        "[forcing_vars]",
        "t_min = t_min",
        "t_max = t_max",
        "prec = precip",
        "wind = wind",
        "shortwave = shortwave",
        "vapor_pressure = vapor_pressure",
        "",
        "[out_vars]",
        "temp = airtemp",
        "prec = precip",
        "shortwave = SWRadAtm",
        "longwave = LWRadAtm",
        "vapor_pressure = vapor_pressure",
        "wind = wind",
        "air_pressure = airpres",
        "spec_humid = spechum",
        "",
        "[domain_vars]",
        "mask = mask",
        "elev = elev",
        "lat = lat",
        "lon = lon",
        "",
        "[state_vars]",
        "t_min = t_min",
        "t_max = t_max",
        "prec = precip",
        "",
        "[chunks]",
        "",
    ]
    ini_path.write_text("\n".join(lines), encoding="utf-8")


def _convert_hourly_to_summa(hourly_path: Path, out_path: Path, reference_path: Path, domain_path: Path, apply_mask: bool) -> None:
    with xr.open_dataset(hourly_path, engine="netcdf4", cache=False) as ds_in:
        ds = ds_in.load()
    with xr.open_dataset(reference_path, engine="netcdf4", cache=False) as ds_ref_in:
        ds_ref = ds_ref_in.load()

    def pick(cands: Iterable[str], label: str) -> str:
        for c in cands:
            if c in ds.data_vars:
                return c
        raise KeyError(f"Missing {label}: candidates={list(cands)}; vars={list(ds.data_vars)}")

    air = pick(["airtemp", "tair", "temp"], "airtemp")
    ppt = pick(["precip", "pptrate", "prec"], "precip")
    sw = pick(["SWRadAtm", "shortwave"], "shortwave")
    lw = pick(["LWRadAtm", "longwave"], "longwave")
    ap = pick(["airpres", "air_pressure"], "air pressure")
    sh = pick(["spechum", "spec_humid"], "specific humidity")
    ws = pick(["windspd", "wind"], "wind")

    out = xr.Dataset()

    t = ds[air]
    if str(t.attrs.get("units", "")).lower().startswith("c") or float(np.nanmedian(t.values)) < 170.0:
        t = t + 273.15
    t.attrs["units"] = "K"
    out["airtemp"] = t

    p = ds[ppt]
    pu = str(p.attrs.get("units", "")).lower()
    if "mm timestep" in pu or "mm/hr" in pu or "mm h-1" in pu:
        p = p / 3600.0
    elif "mm/day" in pu or "mm day-1" in pu:
        p = p / 86400.0
    p.attrs["units"] = "kg m**-2 s**-1"
    out["pptrate"] = p

    out["SWRadAtm"] = ds[sw]
    out["SWRadAtm"].attrs["units"] = "W m**-2"

    a = ds[ap]
    if "kpa" in str(a.attrs.get("units", "")).lower():
        a = a * 1000.0
    a.attrs["units"] = "Pa"
    out["airpres"] = a

    q = ds[sh]
    if float(np.nanmedian(np.abs(q.values))) > 1.0:
        q = q / 1000.0
    q.attrs["units"] = "kg kg**-1"
    out["spechum"] = q

    # Dilley and O'Brien (1998) empirical downwelling LW from T, P, q.
    # Replaces MetSim's internal prata estimate for consistency with the
    # forcing pipeline used elsewhere in this project.
    _p_kpa = a / 1000.0
    _e0 = (q * _p_kpa) / (0.622 + q * 0.378)  # actual VP (kPa)
    out["LWRadAtm"] = 59.38 + 113.7 * (t / 273.16) ** 6 + 96.96 * np.sqrt(4650.0 * _e0 / (2.5 * t))
    out["LWRadAtm"].attrs["units"] = "W m**-2"

    w = ds[ws]
    w.attrs["units"] = "((m s**-1)**2 + (m s**-1)**2)**0.5"
    out["windspd"] = w

    if apply_mask and domain_path.exists():
        with xr.open_dataset(domain_path, engine="netcdf4", cache=False) as dm:
            mask = dm["mask"].load()
        out = out.where(mask > 0)

    # Sanity checks
    lw_min, lw_max = float(np.nanmin(out["LWRadAtm"].values)), float(np.nanmax(out["LWRadAtm"].values))
    if lw_min < 0 or lw_max > 1000:
        raise ValueError(f"LWRadAtm out of plausible range (W m-2): min={lw_min}, max={lw_max}")

    for c in ["time", "latitude", "longitude"]:
        if c in out.coords and c in ds_ref.coords:
            out[c].attrs = dict(ds_ref[c].attrs)

    ordered = [v for v in ds_ref.data_vars if v in out.data_vars]
    out = out[ordered].transpose("time", "latitude", "longitude")
    out.attrs["Conventions"] = ds_ref.attrs.get("Conventions", "CF-1.6")

    tmp = out_path.with_suffix(".tmp.nc")
    if tmp.exists():
        tmp.unlink()
    out.to_netcdf(tmp)
    os.replace(tmp, out_path)


def _month_worker(month: str, cfg: BatchConfig) -> str:
    month_tag = month.replace("-", "")
    month_dash = month
    final_out = cfg.out_dir / f"metsim_{month_tag}.nc"

    logger = logging.getLogger("metsim_batch")
    logger.info("[%s] Starting month", month)

    if final_out.exists():
        logger.info("[%s] Output exists, skipping overwrite: %s", month, final_out)
        return month

    period = pd.Period(month, freq="M")
    expected_days = calendar.monthrange(period.start_time.year, period.start_time.month)[1]

    # Source daily input expected from existing monthly workflow structure.
    daily_src = cfg.basin_root / "forcing" / "monthly_workflow" / "metsim" / month_dash / f"metsim_daily_input_{month_dash}.nc"
    if not daily_src.exists() and cfg.build_missing_daily:
        logger.info("[%s] Daily input missing; building source datasets and MetSim daily input", month)
        daily_src = _build_missing_daily_input(month, cfg)
    if daily_src.exists():
        with xr.open_dataset(daily_src, engine="netcdf4", cache=False) as ds_chk:
            if "time" not in ds_chk.coords:
                raise KeyError(f"[{month}] Daily input has no time coordinate: {daily_src}")
            n_days = pd.DatetimeIndex(ds_chk["time"].values).normalize().nunique()
        if n_days < expected_days:
            if cfg.build_missing_daily:
                logger.warning(
                    "[%s] Daily input has %d/%d days; rebuilding source datasets and daily input",
                    month,
                    n_days,
                    expected_days,
                )
                daily_src = _build_missing_daily_input(month, cfg, force_rebuild=True)
            else:
                raise RuntimeError(
                    f"[{month}] Daily input has {n_days}/{expected_days} days ({daily_src}). "
                    "Rerun with --build-missing-daily (and --allow-download if needed) to auto-rebuild."
                )
    if not daily_src.exists():
        raise FileNotFoundError(
            f"[{month}] Missing daily source: {daily_src}. "
            "Enable --build-missing-daily and --allow-download, or precreate monthly daily inputs."
        )
    month_dir = cfg.basin_root / "forcing" / "monthly_workflow" / "metsim" / month_dash
    domain_src = month_dir / f"metsim_domain_{month_dash}.nc"
    if not domain_src.exists():
        alt = month_dir / f"metsim_domain_mask_{month_dash}.nc"
        if alt.exists():
            domain_src = alt

    work_dir = cfg.tmp_root / month_tag
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        daily_norm = work_dir / f"metsim_daily_{month_tag}.nc"
        domain_nc = work_dir / f"metsim_domain_{month_tag}.nc"
        state_nc = work_dir / f"metsim_state_{month_tag}.nc"
        ini_path = work_dir / f"metsim_config_{month_tag}.ini"

        logger.info("[%s] Loading and normalizing daily input", month)
        with xr.open_dataset(daily_src, engine="netcdf4", cache=False) as ds_in:
            ds_daily = _normalize_daily_for_metsim(ds_in.load(), month)
        ds_daily.to_netcdf(daily_norm)

        if domain_src.exists():
            logger.info("[%s] Reusing precomputed domain file", month)
            shutil.copy2(domain_src, domain_nc)
        else:
            logger.info("[%s] Building domain file from DEM", month)
            _build_domain_from_daily(ds_daily, cfg.dem_path, domain_nc)

        logger.info("[%s] Building state file", month)
        _build_state_from_daily(ds_daily, state_nc, month)

        logger.info("[%s] Writing MetSim config", month)
        _write_metsim_ini(month, daily_norm, state_nc, domain_nc, work_dir, ini_path)

        logger.info("[%s] Running MetSim", month)
        proc = subprocess.run([str(cfg.metsim_exe), str(ini_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"[{month}] MetSim failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

        hourly_candidates = sorted(work_dir.glob(f"metsim_{month_tag}*.nc"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not hourly_candidates:
            raise FileNotFoundError(f"[{month}] No hourly MetSim output in {work_dir}")
        hourly_out = hourly_candidates[0]

        logger.info("[%s] Converting to SUMMA schema -> %s", month, final_out)
        _convert_hourly_to_summa(hourly_out, final_out, cfg.reference_path, domain_nc, cfg.apply_mask)

        logger.info("[%s] Completed successfully", month)
        return month
    finally:
        if not cfg.keep_tmp and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.info("[%s] Cleaned temp workspace: %s", month, work_dir)


def _prepare_daily_inputs(months: list[str], cfg: BatchConfig) -> None:
    """Build month-level daily inputs serially before parallel MetSim execution."""
    if not cfg.build_missing_daily:
        return

    logger = logging.getLogger("metsim_batch")
    for month in months:
        month_dir = cfg.basin_root / "forcing" / "monthly_workflow" / "metsim" / month
        daily_src = month_dir / f"metsim_daily_input_{month}.nc"
        if daily_src.exists():
            continue
        logger.info("[%s] Prebuilding daily input serially", month)
        _build_missing_daily_input(month, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MetSim monthly batch with multiprocessing and cleanup")
    parser.add_argument("--start-month", required=True, help="Start month YYYY-MM")
    parser.add_argument("--end-month", required=True, help="End month YYYY-MM")
    parser.add_argument("--basin-root", default="/scratch/dlhogan/ess-project-data/domain_East_River_lumped")
    parser.add_argument("--output-dir", default="/scratch/dlhogan/ess-project-data/domain_East_River_lumped/forcing/metsim_outputs")
    parser.add_argument("--tmp-root", default="/scratch/dlhogan/ess-project-data/domain_East_River_lumped/forcing/metsim_batch_tmp")
    parser.add_argument("--reference", default="/scratch/dlhogan/ess-project-data/domain_East_River_lumped/forcing/merged_data/ERA5_merged_201512.nc")
    parser.add_argument("--dem", default="/scratch/dlhogan/ess-project-data/domain_East_River_lumped/attributes/elevation/dem/domain_East_River_lumped_elv.tif")
    parser.add_argument("--metsim-exe", default="/home/dlhogan/miniforge3/envs/metsim-run/bin/ms")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--no-mask", action="store_true", help="Disable mask application in final conversion")
    parser.add_argument("--build-missing-daily", action="store_true", help="Build missing monthly MetSim daily input inside each worker")
    parser.add_argument(
        "--serial-prebuild",
        action="store_true",
        help="Prebuild missing daily inputs serially before parallel month workers",
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow PRISM/Daymet/ERA5 API downloads when building missing daily input")
    parser.add_argument(
        "--catchment-shp",
        default="/scratch/dlhogan/ess-project-data/domain_East_River_lumped/shapefiles/catchment/East_River_lumped_HRUs_GRUs.shp",
    )
    parser.add_argument("--day-start-hour", type=int, default=8)
    parser.add_argument("--prism-ftp-host", default="prism.oregonstate.edu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    months = _month_range(args.start_month, args.end_month)

    cfg = BatchConfig(
        basin_root=Path(args.basin_root),
        out_dir=Path(args.output_dir),
        tmp_root=Path(args.tmp_root),
        reference_path=Path(args.reference),
        dem_path=Path(args.dem),
        metsim_exe=Path(args.metsim_exe),
        workers=max(1, args.workers),
        keep_tmp=bool(args.keep_tmp),
        apply_mask=not args.no_mask,
        build_missing_daily=bool(args.build_missing_daily),
        allow_download=bool(args.allow_download),
        day_start_hour=int(args.day_start_hour),
        catchment_shp=Path(args.catchment_shp),
        prism_ftp_host=str(args.prism_ftp_host),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_root.mkdir(parents=True, exist_ok=True)

    log_file = cfg.out_dir / f"metsim_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = _setup_root_logger(log_file)

    logger.info("Starting batch run")
    logger.info("Months: %s", ", ".join(months))
    logger.info("Workers: %d", cfg.workers)
    logger.info("Output dir: %s", cfg.out_dir)
    logger.info("Temp root: %s", cfg.tmp_root)
    logger.info("Build missing daily inputs: %s", cfg.build_missing_daily)
    logger.info("Serial prebuild phase: %s", bool(args.serial_prebuild))
    logger.info("Allow downloads: %s", cfg.allow_download)

    if args.serial_prebuild:
        _prepare_daily_inputs(months, cfg)

    successes = []
    failures = []

    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(_month_worker, m, cfg): m for m in months}
        for fut in as_completed(futures):
            month = futures[fut]
            try:
                fut.result()
                successes.append(month)
                logger.info("[%s] SUCCESS", month)
            except Exception as exc:
                failures.append((month, str(exc)))
                logger.error("[%s] FAILED: %s", month, exc)

    logger.info("Finished batch run: %d success, %d failed", len(successes), len(failures))
    if failures:
        logger.error("Failure summary:")
        for month, err in failures:
            logger.error("  %s -> %s", month, err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
