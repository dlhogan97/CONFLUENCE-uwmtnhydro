"""
Basin-wide pass v2: compute BOTH maskings per band, so the zero years can be
interrogated (is the snow hiding in shadow-masked pixels, or truly absent?).
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

OUT = Path(__file__).parent


def scene_stats2(it, grid, hru_ids):
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
    base_snow = (ndsi > L.NDSI_THRESH) & (n > L.NIR_THRESH)

    out = dict(status="ok", cover=round(float(cover), 3))
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
            out[f"area_basin{tag}"] = np.nan
            continue
        snow = valid & base_snow
        nv = np.bincount(lab[valid], minlength=30).astype(float)
        ns = np.bincount(lab[snow], minlength=30).astype(float)
        bands = {}
        for h in hru_ids:
            b_ = grid["band_of_hru"][h]
            bands.setdefault(b_, [0.0, 0.0])
            bands[b_][0] += nv[h]; bands[b_][1] += ns[h]
        for b_, (v_, s_) in bands.items():
            out[f"area_b{b_}{tag}"] = s_ * PX_KM2
            out[f"nv_b{b_}{tag}"] = v_          # denominators, for extrapolation
            out[f"ns_b{b_}{tag}"] = s_
            out[f"fsca_b{b_}{tag}"] = (s_ / v_) if v_ else np.nan
        out[f"area_basin{tag}"] = ns[1:].sum() * PX_KM2
        out[f"npx_snow{tag}"] = int(ns[1:].sum())
    out["status"] = "ok" if np.isfinite(out.get("area_basin", np.nan)) \
        else "too cloudy"
    return out


def main():
    t0 = time.time()
    grid, gdf = basin_grid()
    hru_ids = sorted(gdf.HRU_ID)
    items = [i for i in L.search(1985, 2025, mon_lo=(7, 1), mon_hi=(10, 10),
                                 cloud_lt=35)
             if i.properties.get("landsat:wrs_path") == "042"
             and i.properties.get("landsat:wrs_row") == "034"]
    print(f"{len(items)} scenes; grid N={grid['n_hru']} px")

    def job(it):
        r = scene_stats2(it, grid, hru_ids)
        r.update(date=pd.Timestamp(it.datetime).tz_localize(None).normalize(),
                 platform=it.properties.get("platform"),
                 scene_cloud=it.properties.get("eo:cloud_cover"))
        return r

    with ThreadPoolExecutor(8) as ex:
        df = pd.DataFrame(list(ex.map(job, items))).sort_values("date")
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df.to_csv(OUT / "scenes_by_band3.csv", index=False)
    print(f"usable {int((df.status=='ok').sum())}/{len(df)}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
