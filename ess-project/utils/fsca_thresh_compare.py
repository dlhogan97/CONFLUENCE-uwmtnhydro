"""Annual end-of-season snow area under each NDSI rule (fsca_thresh.py), vs the current record and Hawkins et al. (2026)."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import geopandas as gpd
import landsat_fsca as L
from fsca_thresh import DATA, MIN_VALID

BANDS = [8, 7, 6, 5, 4, 3, 2]                     # same constants as fsca_final.py (not imported: it needs the uncopied fsca_figs module)


def band_areas():
    g = gpd.read_file(L.SHP)
    return (g.groupby("elevClass").HRU_area.sum() / 1e6).to_dict(), g.groupby("elevClass").elev_mean.mean().to_dict()

CLASSES = ("t40", "sensor", "sensor_raw")


def eos_by_class():
    BA, _ = band_areas()
    d = pd.read_csv(DATA / "scenes_by_band_thresh.csv", parse_dates=["date"])
    # same scene set as scenes_by_band3.csv: a scene counts only if the STRICT mask clears MIN_VALID (6 cloudy scenes pass the shadow mask alone)
    ok = d[(d.status == "ok") & (d.valid >= MIN_VALID) & d.apply(lambda r: (r.date.month, r.date.day) >= (8, 1), axis=1)].copy()
    rows = {}
    for c in CLASSES:
        for tier, tag, vcol in (("strict", "", "valid"), ("shadow", "_sh", "valid_sh")):
            cols = [f"fsca_{c}_b{b}{tag}" for b in BANDS]
            good = (ok[vcol] >= MIN_VALID) & ok[[f"fsca_{c}_b{b}{tag}" for b in (8, 7)]].notna().all(axis=1)
            A = np.where(good, sum(ok[col].fillna(0) * BA[b] for col, b in zip(cols, BANDS)), np.nan)
            s = pd.Series(A, index=ok.index).dropna()
            pick = s.groupby(ok.loc[s.index, "year"]).idxmin()
            rows[(c, tier)] = pd.Series(s[pick.values].values, index=pick.index)
            if tier == "shadow":
                rows[(c, "platform")] = ok.loc[pick.values, "platform"].set_axis(pick.index)
    return pd.DataFrame(rows)


def main():
    e = eos_by_class()
    old = pd.read_csv(DATA / "eos_final.csv", index_col="year")
    pub = pd.read_csv(DATA / "published_glacier_snowpatch_area.csv", comment="#").set_index("year")
    print("reproduction check (t40 vs eos_final.csv): max |diff| strict "
          f"{(e[('t40','strict')]-old.A_strict).abs().max():.4f}, shadow {(e[('t40','shadow')]-old.A_shadow).abs().max():.4f} km2")
    tab = pd.DataFrame({"platform": e[("sensor", "platform")]})
    for c in CLASSES:
        for t in ("strict", "shadow"):
            tab[f"{c}_{t}"] = e[(c, t)]
    tab["hawkins_snowpatch"] = pub.snowpatch_km2; tab["hawkins_combined"] = pub.combined_km2
    pd.set_option("display.width", 250)
    print("\nend-of-season snow area (km2), Aug 1 - Oct 10 annual minimum"); print(tab.round(3).to_string())

    print("\nratio to the current (t40) record, by sensor group (median over years)")
    for t in ("strict", "shadow"):
        for c in ("sensor", "sensor_raw"):
            r = tab[f"{c}_{t}"]/tab[f"t40_{t}"].where(tab[f"t40_{t}"] > 0.01)
            g = r.groupby(tab.platform.isin(["landsat-8", "landsat-9"]).map({True: "OLI (L8/9)", False: "TM/ETM+ (L4/5/7)"}))
            print(f"  {c:10s} {t:6s}: " + "   ".join(f"{k} x{v.median():.2f} (IQR {v.quantile(.25):.2f}-{v.quantile(.75):.2f}, n={v.count()})" for k, v in g))

    print("\nvs Hawkins et al. (2026), overlapping years")
    for ref in ("hawkins_snowpatch", "hawkins_combined"):
        for c in CLASSES:
            for t in ("strict", "shadow"):
              for sub, yrs in (("all", None), ("no 2001-02", (2001, 2002))):
                j = tab[[f"{c}_{t}", ref]].dropna()
                if yrs: j = j.drop(list(yrs), errors="ignore")   # user: Hawkins 2001-02 climatologically implausible
                x, y = j.iloc[:, 0], j.iloc[:, 1]
                print(f"  {ref:18s} {c:10s} {t:6s} {sub:10s} n={len(j):2d}  mean ours {x.mean():.3f} vs {y.mean():.3f} (ratio {x.mean()/y.mean():.2f})"
                      f"  r={np.corrcoef(x, y)[0,1]:+.2f}  rho={spearmanr(x, y)[0]:+.2f}  MAE={np.abs(x-y).mean():.3f}")
    tab.to_csv(DATA / "eos_thresh_compare.csv")
    print("\nwrote", DATA / "eos_thresh_compare.csv")


if __name__ == "__main__":
    main()
