"""
Within-HRU-2 SWE depth distributions from the on-disk ASO 50 m flights.

Gives two things the N-bin build needs:
  1. the SHAPE of the subgrid depth distribution (-> bin area weights)
  2. a directly observed SCA-vs-mean-SWE depletion relation, independent of
     any Landsat NDSI retrieval
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

ASO_DIR = Path("/scratch/dlhogan/ess-project-data/geospatial-data/ASO/Tuolumne")
SHP = Path("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_"
           "elevAspect/shapefiles/catchment/"
           "Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp")
OUT = Path("/tmp/claude-1008/-home-dlhogan-projects-forked-repos-CONFLUENCE-"
           "uwmtnhydro/6a72c66c-be14-42e5-9f00-c8e8394a47f3/scratchpad")
HRU = 2
QS = [.05, .10, .25, .50, .75, .90, .95, .99]


def hru_geom(hru_id=HRU, crs="EPSG:32611"):
    g = gpd.read_file(SHP)
    return g[g.HRU_ID == hru_id].to_crs(crs)


def flight_date(p: Path):
    return pd.Timestamp(re.search(r"(\d{8})\.tif$", p.name).group(1))


def extract(path: Path, geoms):
    with rasterio.open(path) as src:
        try:
            arr, _ = rio_mask(src, geoms, crop=True, filled=True,
                              nodata=np.nan, all_touched=False)
        except ValueError:                       # polygon outside raster
            return None
        a = arr[0].astype("float64")
        nod = src.nodata
    if nod is not None and not np.isnan(nod):
        a[a == nod] = np.nan
    a[a < -100] = np.nan
    return a


def main():
    h2 = hru_geom()
    geoms = list(h2.geometry)
    files = sorted(ASO_DIR.glob("*.tif"), key=flight_date)

    rows = []
    for f in files:
        a = extract(f, geoms)
        if a is None:
            rows.append(dict(date=flight_date(f), file=f.name, status="no overlap"))
            continue
        valid = a[np.isfinite(a)]
        n_cell = a.size
        if valid.size == 0:
            rows.append(dict(date=flight_date(f), file=f.name, status="all nodata"))
            continue
        cov = valid.size / n_cell
        snow = valid[valid > 0.01]               # >1 cm SWE counts as covered
        r = dict(
            date=flight_date(f), file=f.name, status="ok",
            valid_frac=round(cov, 3), n_valid=int(valid.size),
            sca=round(snow.size / valid.size, 4),
            mean_all=round(float(valid.mean()), 4),          # over valid cells
            mean_snow=round(float(snow.mean()), 4) if snow.size else 0.0,
            max=round(float(valid.max()), 3),
        )
        if snow.size:
            for q, v in zip(QS, np.quantile(snow, QS)):
                r[f"q{int(q*100):02d}"] = round(float(v), 3)
            r["cv_snow"] = round(float(snow.std() / snow.mean()), 3)
        rows.append(r)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    ok = df[df.status == "ok"].copy()
    print("=== flights with HRU-2 coverage ===")
    print(ok[["date", "valid_frac", "n_valid", "sca", "mean_all", "mean_snow",
              "max", "cv_snow", "q05", "q25", "q50", "q75", "q95"]].to_string(index=False))

    bad = df[df.status != "ok"]
    if len(bad):
        print("\n=== skipped ===")
        print(bad[["date", "file", "status"]].to_string(index=False))

    # units sanity: ASO 50M SWE products are metres
    print(f"\nvalue range across flights: {ok['max'].min():.2f} - {ok['max'].max():.2f}")
    print("(ASO 50M SWE is in METRES -> these are plausible peak depths)")

    ok.to_csv(OUT / "aso_hru2_stats.csv", index=False)
    print(f"\nwrote {OUT/'aso_hru2_stats.csv'}")

    # late-season subset = closest thing to an EOS depletion observation
    late = ok[ok.date.dt.month >= 6]
    if len(late):
        print("\n=== late-season flights (Jun onward) ===")
        print(late[["date", "sca", "mean_all", "mean_snow", "cv_snow",
                    "q25", "q50", "q75", "q95"]].to_string(index=False))


if __name__ == "__main__":
    main()
