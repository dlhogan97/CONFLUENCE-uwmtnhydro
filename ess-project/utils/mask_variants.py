"""Compare several QA maskings against near-coincident ASO SCA truth."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

import landsat_fsca as L
import landsat_grid as G
from aso_hru2 import hru_geom, extract, flight_date, ASO_DIR

OUT = Path(__file__).parent
GREEN_MIN = 0.05

# bit sets to treat as unusable
VARIANTS = {
    "b1234": (1, 2, 3, 4),      # dilated, cirrus, cloud, shadow  (the strict one)
    "b123":  (1, 2, 3),         # keep terrain shadow
    "b234":  (2, 3, 4),         # drop dilation (it grows around misflagged snow)
    "b23":   (2, 3),            # cloud + cirrus only
    "b3":    (3,),              # high-confidence cloud only
}


def classify(qa, g, sw, n, m, bits, guard=True):
    ins = m & (qa != 0) & ((qa & 1) == 0)
    bad = np.zeros(qa.shape, dtype=bool)
    for b in bits:
        bad |= ((qa >> b) & 1).astype(bool)
    valid = ins & ~bad & (g > 0) & (sw > 0)
    if guard:
        valid &= g > GREEN_MIN
    if valid.sum() == 0:
        return np.nan, 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi = (g - sw) / (g + sw)
    snow = valid & (ndsi > L.NDSI_THRESH) & (n > L.NIR_THRESH)
    return float(snow.sum() / valid.sum()), float(valid.sum() / m.sum())


def main():
    grid = G.reference_grid(); m = grid["mask"]
    h2 = hru_geom(); geoms = list(h2.geometry)

    aso = []
    for f in sorted(ASO_DIR.glob("*.tif"), key=flight_date):
        d = flight_date(f)
        if d.month < 6:
            continue
        a = extract(f, geoms)
        if a is None:
            continue
        v = a[np.isfinite(a)]
        if v.size < 100:
            continue
        aso.append(dict(aso_date=d, aso_sca=float((v > 0.01).mean())))
    aso = pd.DataFrame(aso)

    items = [i for i in L.search(2013, 2023, mon_lo=(6, 1), mon_hi=(9, 15),
                                 cloud_lt=60)
             if i.properties.get("landsat:wrs_path") == "042"
             and i.properties.get("landsat:wrs_row") == "034"]

    def job(it):
        try:
            qa = G.read_on_grid(it.assets["qa_pixel"].href, grid).astype("uint32")
            g = G.read_on_grid(it.assets["green"].href, grid) * L.SR_SCALE + L.SR_OFF
            sw = G.read_on_grid(it.assets["swir16"].href, grid) * L.SR_SCALE + L.SR_OFF
            n = G.read_on_grid(it.assets["nir08"].href, grid) * L.SR_SCALE + L.SR_OFF
        except Exception:                          # noqa: BLE001
            return None
        r = dict(date=pd.Timestamp(it.datetime).tz_localize(None).normalize(),
                 cloud=it.properties.get("eo:cloud_cover"))
        for name, bits in VARIANTS.items():
            f_, v_ = classify(qa, g, sw, n, m, bits)
            r[f"f_{name}"], r[f"v_{name}"] = f_, v_
        return r

    with ThreadPoolExecutor(8) as ex:
        ls = pd.DataFrame([r for r in ex.map(job, items) if r]).sort_values("date")

    rows = []
    for _, a in aso.iterrows():
        dt = (ls.date - a.aso_date).abs()
        if dt.min() > pd.Timedelta("5D"):
            continue
        b = ls.loc[dt.idxmin()]
        r = dict(aso_date=a.aso_date.date(), ls_date=b.date.date(),
                 aso=round(a.aso_sca, 3), cloud=round(b.cloud, 1))
        for name in VARIANTS:
            r[name] = round(b[f"f_{name}"], 3)
            r[f"v_{name}"] = round(b[f"v_{name}"], 2)
        rows.append(r)
    cmp = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print("=== fSCA by masking variant vs ASO ===")
    print(cmp[["aso_date", "ls_date", "aso", "cloud"] + list(VARIANTS)].to_string(index=False))
    print("\n=== valid fraction of HRU by variant ===")
    print(cmp[["aso_date", "cloud"] + [f"v_{k}" for k in VARIANTS]].to_string(index=False))

    print("\n=== skill (pairs with a finite retrieval) ===")
    print(f"{'variant':8s} {'n':>3s} {'bias':>7s} {'MAE':>6s} {'r':>6s} {'medValid':>9s}")
    for name in VARIANTS:
        s = cmp[["aso", name, f"v_{name}"]].dropna()
        if len(s) < 3:
            print(f"{name:8s} {len(s):3d}   too few"); continue
        e = s[name] - s.aso
        r = np.corrcoef(s[name], s.aso)[0, 1]
        print(f"{name:8s} {len(s):3d} {e.mean():+7.3f} {e.abs().mean():6.3f} "
              f"{r:+6.3f} {s[f'v_{name}'].median():9.2f}")

    cmp.to_csv(OUT / "mask_variants.csv", index=False)
    print(f"\nwrote {OUT/'mask_variants.csv'}")


if __name__ == "__main__":
    main()
