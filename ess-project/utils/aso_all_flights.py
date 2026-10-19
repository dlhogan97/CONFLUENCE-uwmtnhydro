"""
ASO snow-covered fraction for ALL flights (not just late season), per elevation
band and basin-wide, at several SWE thresholds — so it can be compared with the
Landsat NDSI retrieval and with the fitted recession curve.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

import landsat_fsca as L
from aso_hru2 import ASO_DIR, flight_date

D = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
BANDS = [8, 7, 6, 5]
THRESH = [0.01, 0.10, 0.25, 0.50]


def main():
    g = gpd.read_file(L.SHP).to_crs("EPSG:32611")
    BA = (g.groupby("elevClass").HRU_area.sum() / 1e6).to_dict()
    geoms = {b: list(g[g.elevClass == b].dissolve().geometry) for b in BANDS}
    geoms["basin"] = list(g.dissolve().geometry)
    AREA = dict(BA); AREA["basin"] = float(g.HRU_area.sum() / 1e6)

    rows = []
    for f in sorted(ASO_DIR.glob("*.tif"), key=flight_date):
        d = flight_date(f)
        rec = dict(date=d, year=d.year, doy=d.dayofyear)
        ok = True
        for key in list(BANDS) + ["basin"]:
            with rasterio.open(f) as src:
                try:
                    arr, _ = rio_mask(src, geoms[key], crop=True, filled=True,
                                      nodata=np.nan, all_touched=False)
                except ValueError:
                    ok = False; break
                px = abs(src.res[0] * src.res[1]) / 1e6
                nod = src.nodata
            a = arr[0].astype("float64")
            if nod is not None and not np.isnan(nod):
                a[a == nod] = np.nan
            a[a < -100] = np.nan
            v = a[np.isfinite(a)]
            if v.size < 200:
                ok = False; break
            cov = v.size * px / AREA[key]
            rec[f"cov_{key}"] = round(cov, 3)
            for t in THRESH:
                # fSCA over the OBSERVED part, then applied to the full band
                rec[f"f_{key}_t{t}"] = round(float((v > t).mean()), 5)
                rec[f"A_{key}_t{t}"] = round(float((v > t).mean() * AREA[key]), 4)
        if ok:
            rows.append(rec)

    aso = pd.DataFrame(rows)
    aso.to_csv(D / "aso_all_flights.csv", index=False)
    print(f"{len(aso)} flights with full band coverage "
          f"({aso.date.min().date()} - {aso.date.max().date()})")
    print(f"months present: {sorted(aso.date.dt.month.unique())}")

    # match to Landsat scenes
    ls = pd.read_csv(D / "scenes_by_band3.csv", parse_dates=["date"])
    ls = ls[ls.status == "ok"].copy()
    ls["f_b8_ls"] = ls.fsca_b8_sh
    ls["A_basin_ls"] = sum(ls[f"fsca_b{b}_sh"].fillna(0) * BA[b]
                           for b in [8, 7, 6, 5, 4, 3, 2])
    pairs = []
    for _, a in aso.iterrows():
        dt = (ls.date - a.date).abs()
        if dt.min() > pd.Timedelta("6D"):
            continue
        m = ls.loc[dt.idxmin()]
        pairs.append(dict(aso_date=a.date, ls_date=m.date,
                          lag=int((m.date - a.date).days), doy=a.doy,
                          aso_f_b8_t025=a["f_8_t0.25"], aso_f_b8_t05=a["f_8_t0.5"],
                          aso_f_b8_t001=a["f_8_t0.01"],
                          ls_f_b8=round(float(m.f_b8_ls), 5),
                          aso_A_basin_t025=a["A_basin_t0.25"],
                          ls_A_basin=round(float(m.A_basin_ls), 3)))
    p = pd.DataFrame(pairs)
    p.to_csv(D / "aso_landsat_pairs_all.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n{len(p)} near-coincident ASO/Landsat pairs (|lag| <= 6 d)")
    print(p[["aso_date", "ls_date", "lag", "doy", "aso_f_b8_t025", "ls_f_b8",
             "aso_A_basin_t025", "ls_A_basin"]].to_string(index=False))
    if len(p) >= 3:
        for c, lab in [("aso_f_b8_t001", "SWE>0.01"), ("aso_f_b8_t025", "SWE>0.25"),
                       ("aso_f_b8_t05", "SWE>0.50")]:
            r = np.corrcoef(p[c], p.ls_f_b8)[0, 1]
            k = (p.ls_f_b8 * p[c]).sum() / (p[c] ** 2).sum()
            print(f"  band-8 fSCA, ASO {lab}: r={r:+.3f}  Landsat/ASO slope={k:.2f}")


if __name__ == "__main__":
    main()
