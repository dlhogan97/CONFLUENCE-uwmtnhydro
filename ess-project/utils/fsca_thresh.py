"""
NDSI-threshold sensitivity of the end-of-season snow record (same scenes, grid, masks and window as fsca_band2.py / fsca_final.py).

Classifications, all computed from one read per scene:
  t40         NDSI > 0.40 and NIR > 0.11                         -- the current record (reproduces scenes_by_band3.csv)
  sensor      NDSI > 0.08 (OLI: Landsat 8/9) or > 0.11 (TM/ETM+: Landsat 4/5/7), and NIR > 0.11
  sensor_raw  the same sensor thresholds, no NIR guard           -- the prior paper's rule as stated
Output: scenes_by_band_thresh.csv (per scene, per band, both maskings) in the snowpatch_fsca data folder.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

import landsat_fsca as L
import landsat_grid as G
from fsca_by_band import basin_grid, MIN_COVER, MIN_VALID, PX_KM2

DATA = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
SENSOR_THRESH = {"landsat-8": 0.08, "landsat-9": 0.08, "landsat-4": 0.11, "landsat-5": 0.11, "landsat-7": 0.11}


def scene_stats_thresh(it, grid, hru_ids):
    m, lab = grid["mask"], grid["labels"]
    try:
        qa = G.read_on_grid(it.assets["qa_pixel"].href, grid).astype("uint32")
        g = G.read_on_grid(it.assets["green"].href, grid) * L.SR_SCALE + L.SR_OFF
        sw = G.read_on_grid(it.assets["swir16"].href, grid) * L.SR_SCALE + L.SR_OFF
        n = G.read_on_grid(it.assets["nir08"].href, grid) * L.SR_SCALE + L.SR_OFF
    except Exception as e:                         # noqa: BLE001
        return dict(status=f"read fail {type(e).__name__}")

    ins = m & (qa != 0) & ((qa & 1) == 0)
    cover = ins.sum() / m.sum()
    if cover < MIN_COVER:
        return dict(status="partial", cover=round(float(cover), 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi = (g - sw) / (g + sw)
    th = SENSOR_THRESH[it.properties.get("platform")]
    classes = {"t40": (ndsi > L.NDSI_THRESH) & (n > L.NIR_THRESH),
               "sensor": (ndsi > th) & (n > L.NIR_THRESH),
               "sensor_raw": ndsi > th}

    out = dict(status="ok", cover=round(float(cover), 3), ndsi_thresh=th)
    for tag, bits in (("", (1, 2, 3, 4)), ("_sh", (1, 2, 3))):
        bad = np.zeros(qa.shape, dtype=bool)
        for b in bits:
            bad |= ((qa >> b) & 1).astype(bool)
        valid = ins & ~bad & (g > 0) & (sw > 0)
        if tag == "_sh":
            valid &= g > 0.05                       # deep shadow -> NDSI noise
        vfrac = valid.sum() / m.sum()
        out[f"valid{tag}"] = round(float(vfrac), 3)
        if vfrac < MIN_VALID:
            continue
        nv = np.bincount(lab[valid], minlength=30).astype(float)
        nvb = {}
        for h in hru_ids:
            nvb[grid["band_of_hru"][h]] = nvb.get(grid["band_of_hru"][h], 0.0) + nv[h]
        for b_, v_ in nvb.items():
            out[f"nv_b{b_}{tag}"] = v_
        for c, snow in classes.items():
            ns = np.bincount(lab[valid & snow], minlength=30).astype(float)
            nsb = {}
            for h in hru_ids:
                nsb[grid["band_of_hru"][h]] = nsb.get(grid["band_of_hru"][h], 0.0) + ns[h]
            for b_, s_ in nsb.items():
                out[f"fsca_{c}_b{b_}{tag}"] = (s_ / nvb[b_]) if nvb[b_] else np.nan
    if "valid" not in out or out["valid"] < MIN_VALID:
        out["status"] = "too cloudy" if out.get("valid_sh", 0) < MIN_VALID else "ok"
    return out


def main():
    t0 = time.time()
    grid, gdf = basin_grid()
    hru_ids = sorted(gdf.HRU_ID)
    items = [i for i in L.search(1985, 2025, mon_lo=(7, 1), mon_hi=(10, 10), cloud_lt=35)
             if i.properties.get("landsat:wrs_path") == "042" and i.properties.get("landsat:wrs_row") == "034"]
    print(f"{len(items)} scenes; grid N={grid['n_hru']} px", flush=True)

    def job(it):
        r = scene_stats_thresh(it, grid, hru_ids)
        r.update(date=pd.Timestamp(it.datetime).tz_localize(None).normalize(),
                 platform=it.properties.get("platform"), scene_cloud=it.properties.get("eo:cloud_cover"))
        return r

    with ThreadPoolExecutor(8) as ex:
        df = pd.DataFrame(list(ex.map(job, items))).sort_values("date")
    df["year"] = df.date.dt.year
    df.to_csv(DATA / "scenes_by_band_thresh.csv", index=False)
    print(f"usable {int((df.status == 'ok').sum())}/{len(df)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
