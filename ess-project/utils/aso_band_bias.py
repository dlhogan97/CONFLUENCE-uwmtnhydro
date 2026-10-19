"""
Band-resolved ASO vs Landsat, to derive a bias factor in the regime that
matters (deep, pixel-dominant snow in the top bands) rather than one
dominated by thin low-elevation snow that NDSI structurally cannot see.

Also reports the factor as a function of the ASO SWE threshold, which
separates "Landsat is biased" from "the two products define snow differently".
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from pathlib import Path

import landsat_fsca as L
from aso_hru2 import ASO_DIR, flight_date

OUT = Path(__file__).parent
BANDS = [8, 7, 6, 5]
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.50]


def main():
    g = gpd.read_file(L.SHP).to_crs("EPSG:32611")
    BA = (g.groupby("elevClass").HRU_area.sum() / 1e6).to_dict()
    band_geom = {b: list(g[g.elevClass == b].dissolve().geometry) for b in BANDS}

    rows = []
    for f in sorted(ASO_DIR.glob("*.tif"), key=flight_date):
        d = flight_date(f)
        if d.month < 6:
            continue
        rec = dict(date=d)
        okflag = True
        for b in BANDS:
            with rasterio.open(f) as src:
                try:
                    arr, _ = rio_mask(src, band_geom[b], crop=True, filled=True,
                                      nodata=np.nan, all_touched=False)
                except ValueError:
                    okflag = False; break
                px = abs(src.res[0] * src.res[1]) / 1e6
                nod = src.nodata
            a = arr[0].astype("float64")
            if nod is not None and not np.isnan(nod):
                a[a == nod] = np.nan
            a[a < -100] = np.nan
            v = a[np.isfinite(a)]
            if v.size < 200:
                okflag = False; break
            rec[f"cov_b{b}"] = v.size * px / BA[b]
            for t in THRESHOLDS:
                rec[f"aso_b{b}_t{t}"] = float((v > t).sum() * px)
        if okflag:
            rows.append(rec)
    aso = pd.DataFrame(rows)
    aso.to_csv(OUT / "aso_band_area.csv", index=False)
    print(f"{len(aso)} ASO flights (Jun+) with band coverage\n")

    ls = pd.read_csv(OUT / "scenes_by_band.csv", parse_dates=["date"])
    ls = ls[ls.status == "ok"].copy()
    for b in BANDS:
        ls[f"ls_b{b}"] = ls[f"fsca_b{b}"].fillna(0) * BA[b]

    pairs = []
    for _, a in aso.iterrows():
        dt = (ls.date - a.date).abs()
        if dt.min() > pd.Timedelta("6D"):
            continue
        m = ls.loc[dt.idxmin()]
        r = dict(aso_date=a.date.date(), ls_date=m.date.date(),
                 lag=int((m.date - a.date).days), ls_valid=m.valid_frac)
        for b in BANDS:
            r[f"ls_b{b}"] = round(float(m[f"ls_b{b}"]), 3)
            for t in THRESHOLDS:
                r[f"aso_b{b}_t{t}"] = round(float(a[f"aso_b{b}_t{t}"]), 3)
        pairs.append(r)
    p = pd.DataFrame(pairs)
    pd.set_option("display.width", 250)
    print("=== matched pairs, snow area km2 ===")
    show = ["aso_date", "ls_date", "lag", "ls_valid"]
    for b in [8, 7]:
        show += [f"ls_b{b}", f"aso_b{b}_t0.01", f"aso_b{b}_t0.25"]
    print(p[show].to_string(index=False))

    print(f"\n=== multiplicative factor (ASO / Landsat), by band and ASO SWE "
          f"threshold ===")
    print(f"{'band':>5} " + "".join(f"{'t='+str(t):>10}" for t in THRESHOLDS))
    for b in BANDS:
        x = p[f"ls_b{b}"].values
        line = f"{b:>5} "
        for t in THRESHOLDS:
            y = p[f"aso_b{b}_t{t}"].values
            k = (x * y).sum() / (x * x).sum() if (x * x).sum() > 0 else np.nan
            line += f"{k:10.2f}"
        print(line)
    print("\n(a factor near 1 means the two products agree; a large factor means")
    print(" ASO is counting thin snow that NDSI cannot resolve, not Landsat bias)")
    p.to_csv(OUT / "aso_band_pairs.csv", index=False)


if __name__ == "__main__":
    main()
