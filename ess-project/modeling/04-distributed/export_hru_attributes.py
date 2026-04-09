#!/usr/bin/env python3
"""Export and slice HRU attributes for distributed runs.

Outputs a joined HRU table from catchment shapefile + SUMMA attributes.nc,
then optional elevation-band and aspect-class slices.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _resolve_project_dir(cfg: dict) -> Path:
    return Path(cfg["CONFLUENCE_DATA_DIR"]) / f"domain_{cfg['DOMAIN_NAME']}"


def _resolve_catchment_shp(cfg: dict, project_dir: Path) -> Path:
    catchment_path = cfg.get("CATCHMENT_PATH", "default")
    if catchment_path == "default" or not catchment_path:
        catchment_dir = project_dir / "shapefiles" / "catchment"
    else:
        catchment_dir = Path(catchment_path)

    catchment_name = cfg.get("CATCHMENT_SHP_NAME", "default")
    if catchment_name == "default" or not catchment_name:
        method = str(cfg.get("DOMAIN_DISCRETIZATION", "GRUs")).replace(",", "_")
        catchment_name = f"{cfg['DOMAIN_NAME']}_HRUs_{method}.shp"

    shp = catchment_dir / catchment_name
    if not shp.exists():
        raise FileNotFoundError(f"Catchment shapefile not found: {shp}")
    return shp


def _resolve_attributes_nc(cfg: dict, project_dir: Path) -> Path:
    settings_path = cfg.get("SETTINGS_SUMMA_PATH", "default")
    if settings_path == "default" or not settings_path:
        settings_dir = project_dir / "settings" / "SUMMA"
    else:
        settings_dir = Path(settings_path)

    attributes_name = cfg.get("SETTINGS_SUMMA_ATTRIBUTES", "attributes.nc")
    nc_path = settings_dir / attributes_name
    if not nc_path.exists():
        raise FileNotFoundError(f"SUMMA attributes file not found: {nc_path}")
    return nc_path


def _read_attributes(nc_path: Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    with xr.open_dataset(nc_path) as ds:
        data_vars = [name for name in ds.data_vars if "hru" in ds[name].dims]
        keep = list(columns) if columns else data_vars
        keep = [name for name in keep if name in ds.data_vars and "hru" in ds[name].dims]

        frame = pd.DataFrame({
            name: ds[name].values for name in keep
        })
        frame["hruId"] = ds["hruId"].values.astype(int)
    return frame


def _aspect_to_class(aspect_deg: pd.Series) -> pd.Series:
    bins = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    wrapped = np.mod(aspect_deg, 360.0)
    idx = np.digitize(wrapped, bins, right=False) - 1
    idx = np.where(idx == 8, 0, idx)
    return pd.Series([labels[i] for i in idx], index=aspect_deg.index)


def _parse_bands(raw: str) -> List[float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) < 2:
        raise ValueError("At least two band edges are required")
    return sorted(values)


def export_hru_tables(
    config_path: Path,
    variables: str = "",
    elevation_bands: str = "",
    aspect_classes: bool = False,
    outdir: str = "",
) -> tuple[Path, Path]:
    """Create full and summary HRU attribute CSV files.

    This function is intentionally straightforward so it can be called from scripts
    and notebooks without pulling in more workflow complexity.
    """
    cfg = _load_config(config_path)
    project_dir = _resolve_project_dir(cfg)

    output_dir = Path(outdir).expanduser().resolve() if outdir else project_dir / "diagnostics" / "hru_attributes"
    output_dir.mkdir(parents=True, exist_ok=True)

    catchment_shp = _resolve_catchment_shp(cfg, project_dir)
    attrs_nc = _resolve_attributes_nc(cfg, project_dir)

    hru_id_col = cfg.get("CATCHMENT_SHP_HRUID", "HRU_ID")
    gdf = gpd.read_file(catchment_shp)
    if hru_id_col not in gdf.columns:
        raise KeyError(f"HRU id column {hru_id_col} missing in {catchment_shp}")

    selected_vars = [v.strip() for v in variables.split(",") if v.strip()] if variables else None
    attrs_df = _read_attributes(attrs_nc, selected_vars)

    # Join shapefile class labels (elevClass/soilClass/landClass) with SUMMA attributes.
    # Geometry is dropped here because these outputs are tabular diagnostics.
    gdf[hru_id_col] = gdf[hru_id_col].astype(int)
    merged = gdf.drop(columns="geometry").merge(attrs_df, left_on=hru_id_col, right_on="hruId", how="left")

    if aspect_classes and "aspect" in merged.columns:
        merged["aspect_class"] = _aspect_to_class(merged["aspect"].astype(float))

    # Optional continuous elevation band labels for downstream summaries.
    if elevation_bands:
        if "elev_mean" in merged.columns:
            elev_source = merged["elev_mean"].astype(float)
        elif "elevation" in merged.columns:
            elev_source = merged["elevation"].astype(float)
        else:
            raise KeyError("No elevation column found. Expected elev_mean or elevation")
        bands = _parse_bands(elevation_bands)
        labels = [f"{int(bands[i])}_{int(bands[i + 1])}" for i in range(len(bands) - 1)]
        merged["elevation_band"] = pd.cut(elev_source, bins=bands, labels=labels, include_lowest=True)

    full_csv = output_dir / "hru_attributes_full.csv"
    merged.to_csv(full_csv, index=False)

    # Keep summary compact but include class columns needed for quick plotting.
    summary_cols = [hru_id_col]
    for col in [
        "hruId",
        "GRU_ID",
        "elevClass",
        "soilClass",
        "landClass",
        "elevation_band",
        "aspect_class",
        "soilTypeIndex",
        "vegTypeIndex",
        "elev_mean",
        "elevation",
        "aspect",
    ]:
        if col in merged.columns and col not in summary_cols:
            summary_cols.append(col)

    summary_csv = output_dir / "hru_attributes_summary.csv"
    merged[summary_cols].to_csv(summary_csv, index=False)

    return full_csv, summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and slice HRU attributes")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--variables",
        default="",
        help="Comma-separated attributes.nc variables to include (default: all HRU-dimension vars)",
    )
    parser.add_argument(
        "--elevation-bands",
        default="",
        help="Comma-separated elevation band edges, e.g., 2500,2800,3100,3400",
    )
    parser.add_argument(
        "--aspect-classes",
        action="store_true",
        help="Add aspect_class labels (N,NE,E,SE,S,SW,W,NW) from attributes aspect",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Output directory (default: project_dir/diagnostics/hru_attributes)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    full_csv, summary_csv = export_hru_tables(
        config_path=config_path,
        variables=args.variables,
        elevation_bands=args.elevation_bands,
        aspect_classes=bool(args.aspect_classes),
        outdir=args.outdir,
    )

    print(f"Wrote full table: {full_csv}")
    print(f"Wrote summary table: {summary_csv}")


if __name__ == "__main__":
    main()
