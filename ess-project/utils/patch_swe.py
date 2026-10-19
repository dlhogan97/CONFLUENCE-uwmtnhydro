"""Observed peak-SWE index for the Tuolumne: mean of the CDEC site z-scores.

Sites are standardised before averaging so a deep site does not dominate the index, and a
water year is kept only when the site reported >=150 days and a peak above 50 mm.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

SNOTEL = Path("/scratch/dlhogan/ess-project-data/meteorological-data/SNOTEL")
IN2MM = 25.4


def peak_swe_index(directory=SNOTEL, min_days=150, min_peak=50.0, min_years=20):
    out = {}
    for f in sorted(glob.glob(str(Path(directory)/"*_cdec_obs.csv"))):
        d = pd.read_csv(f, usecols=lambda c: c in ("datetime", "SWE"))
        if "SWE" not in d.columns:
            continue
        d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        d = d.dropna().set_index("datetime").sort_index()
        swe = pd.to_numeric(d["SWE"], errors="coerce")*IN2MM
        swe = swe[(swe >= 0) & (swe < 3000)]
        wy = swe.index.year + (swe.index.month >= 10)
        pk, n = swe.groupby(wy).max(), swe.groupby(wy).size()
        pk = pk[(n >= min_days) & (pk > min_peak)]
        if len(pk) >= min_years:
            out[Path(f).stem.split("_")[0]] = (pk - pk.mean())/pk.std()
    return pd.DataFrame(out).mean(axis=1).dropna()
