"""
ASO vs SUMMA SWE on the elevation x aspect radial (Tuolumne), split into EARLY (Mar-Apr) and LATE (May-Jul) flights,
for the no-storage control and the July baseline. Same flight screening, HRU means and radial layout as
notebooks/evaluation/ASO_SUMMA_SWE_comparison.ipynb (cells 7-8).

Row 1 = early flights, row 2 = late flights; columns = ASO, control, control - ASO, July baseline - ASO.
A third figure shows the fraction of early-spring SWE still present on the late flights (late / early, same years),
which is the melt-order test: it should rise steadily with elevation and be higher on north than south aspects.
usage: python aso_radial_control.py [run output .nc] [label] [figure tag]
"""
from __future__ import annotations
import re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, xarray as xr
import rasterio
from rasterio.mask import mask as rio_mask
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
warnings.filterwarnings("ignore")

D = Path("/scratch/dlhogan/ess-project-data")
FIGDIR = Path(__file__).resolve().parents[1]/"figures"/"draft"
SHP = D/"domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp"
ASO_DIR = D/"geospatial-data/ASO/Tuolumne"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "control"
TAG = sys.argv[3] if len(sys.argv) > 3 else "control"
RUNS = {LABEL: Path(sys.argv[1]) if len(sys.argv) > 1 else
        D/"calibration/runs/Tuolumne_River_control_20261020_154016/best_full/keep/out/best_timestep.nc",
        "July baseline": D/"calibration/bestruns/Tuolumne_River_baseline_wbias10_1981_2025/run/out/baseline_timestep.nc"}
SWE_MAX_M, MIN_COVER = 5.0, 0.10
INK, GRAY = "#333", "#dddddd"
ASP_CENTER = {1: 45, 2: 135, 3: 225, 4: 315}


def daily_swe(path):
    ds = xr.open_dataset(path); v = "scalarSWE" if "scalarSWE" in ds else "scalarSWE_mean"     # timestep or daily-mean output
    swe = ds[v].resample(time="1D").mean().to_pandas()
    swe.columns = np.asarray(ds["hruId"].values).ravel(); ds.close(); return swe


def load(g):
    sims = {k: daily_swe(p) for k, p in RUNS.items()}
    t0 = max(s.index[0] for s in sims.values()); t1 = min(s.index[-1] for s in sims.values())
    exp_px = g["HRU_area"].values/2500.0; rows = []
    for f in sorted(ASO_DIR.glob("*.tif"), key=lambda p: re.search(r"(\d{8})", p.name).group(1)):
        date = pd.Timestamp(re.search(r"(\d{8})", f.name).group(1))
        if not (t0 <= date <= t1): continue
        with rasterio.open(f) as r:
            gg = g.to_crs(r.crs); aso, cov = [], []
            for i, (_, row) in enumerate(gg.iterrows()):
                out, _ = rio_mask(r, [row.geometry], crop=True, filled=True, nodata=np.nan)
                a = out[0].astype(float); a[a > SWE_MAX_M] = np.nan
                if r.nodata is not None and not np.isnan(r.nodata): a[a == r.nodata] = np.nan
                n = int((~np.isnan(a)).sum()); cov.append(min(n/exp_px[i], 1.0)); aso.append(np.nanmean(a)*1000.0 if n else np.nan)
        if np.nanmean(cov) < 0.5: continue
        for i, hid in enumerate(g["HRU_ID"].values):
            rec = dict(date=date, HRU_ID=hid, aso=aso[i], cover=cov[i])
            for k, s in sims.items(): rec[k] = float(s.loc[date, hid]) if hid in s.columns else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows); df.loc[df.cover < MIN_COVER, ["aso", *RUNS]] = np.nan
    df["season"] = np.where(df.date.dt.month <= 4, "early", np.where(df.date.dt.month <= 7, "late", "other"))
    return df[df.season != "other"].dropna(subset=["aso", *RUNS])


def band_edges(means):
    e = {}
    for i in range(len(means)):
        lo = (means[i-1]+means[i])/2 if i > 0 else means[i]-(means[i+1]-means[i])/2
        hi = (means[i]+means[i+1])/2 if i < len(means)-1 else means[i]+(means[i]-means[i-1])/2
        e[i] = (lo, hi)
    return e


def cellv(gg, band, asp, col):
    s = gg[(gg.elevClass == band) & (gg.aspectClas == asp)]
    if s.empty or s[col].isna().all(): return np.nan
    return float(np.nansum(s[col]*s["HRU_area"])/np.nansum(s["HRU_area"]))


def radial(ax, gg, col, cmap, norm, fmt="{:.0f}"):
    bands = sorted(gg.loc[gg.aspectClas > 0, "elevClass"].unique())
    means = np.array([np.average(gg.loc[gg.elevClass == b, "elev_mean"], weights=gg.loc[gg.elevClass == b, "HRU_area"]) for b in bands])
    e = band_edges(means); ehi = max(h for _, h in e.values()); elo = min(l for l, _ in e.values())
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    for bi, b in enumerate(bands):
        lo, hi = e[bi]
        for c in (1, 2, 3, 4):
            v = cellv(gg, b, c, col)
            ax.bar(np.deg2rad(ASP_CENTER[c]), hi-lo, width=np.deg2rad(90), bottom=ehi-hi,
                   color=plt.get_cmap(cmap)(norm(v)) if np.isfinite(v) else GRAY, edgecolor="white", linewidth=1.3, zorder=3, align="center")
            if np.isfinite(v):
                ax.text(np.deg2rad(ASP_CENTER[c]), ehi-(lo+hi)/2, fmt.format(v), ha="center", va="center", fontsize=7, color=INK, zorder=7,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.0))
    ax.set_rlim(0, ehi-elo); ax.set_yticklabels([]); ax.grid(False)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])); ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=8)
    for bi, mm in enumerate(means):
        lo, hi = e[bi]
        ax.text(np.deg2rad(-3), ehi-(lo+hi)/2, f"{int(mm)} m", ha="center", va="center", fontsize=6.5, color=INK, zorder=8,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))


def panel(fig, pos, g, col, title, cmap, norm, label, fmt="{:.0f}"):
    ax = fig.add_subplot(*pos, projection="polar"); radial(ax, g, col, cmap, norm, fmt); ax.set_title(title, fontsize=11, pad=14, color=INK)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.11, shrink=0.7, extend="both"); cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)


def main():
    g0 = gpd.read_file(SHP); df = load(g0)
    for k in RUNS: df[f"d_{k}"] = df[k]-df.aso
    fig = plt.figure(figsize=(21, 11))
    for r, season in enumerate(("early", "late")):
        s = df[df.season == season]; m = s.groupby("HRU_ID")[["aso", *RUNS, *[f"d_{k}" for k in RUNS]]].mean()
        g = g0.copy()
        for c in m.columns: g[c] = g.HRU_ID.map(m[c])
        hi = 50*np.ceil(np.nanmax([cellv(g, b, a, c) for b in g.elevClass.unique() for a in (1, 2, 3, 4) for c in ("aso", LABEL)])/50)
        lab = f"{season} ({'Mar-Apr' if season == 'early' else 'May-Jul'}, {s.date.nunique()} flights)"
        panel(fig, (2, 4, 4*r+1), g, "aso", f"ASO, {lab}", "Blues", Normalize(0, hi), "SWE (mm)")
        panel(fig, (2, 4, 4*r+2), g, LABEL, LABEL, "Blues", Normalize(0, hi), "SWE (mm)")
        panel(fig, (2, 4, 4*r+3), g, f"d_{LABEL}", f"{LABEL} - ASO", "RdBu", Normalize(-250, 250), "mm")
        panel(fig, (2, 4, 4*r+4), g, "d_July baseline", "July baseline - ASO", "RdBu", Normalize(-250, 250), "mm")
    fig.suptitle(f"Tuolumne SWE vs ASO by elevation band (radius, highest at centre) and aspect (wedge), "
                 f"WY{df.date.min().year}-{df.date.max().year}", fontsize=12.5, y=1.0, color=INK)
    fig.tight_layout(); fig.savefig(FIGDIR/f"aso_radial_{TAG}_early_late.png", dpi=120, bbox_inches="tight", facecolor="white")

    # fraction of early SWE remaining on the late flights, years with both
    yrs = sorted(set(df[df.season == "early"].date.dt.year) & set(df[df.season == "late"].date.dt.year))
    x = df[df.date.dt.year.isin(yrs)].groupby(["season", "HRU_ID"])[["aso", *RUNS]].mean()
    frac = (x.loc["late"]/x.loc["early"].where(x.loc["early"] > 25)).clip(upper=1.5)*100
    g = g0.copy()
    for c in frac.columns: g[f"f_{c}"] = g.HRU_ID.map(frac[c])
    fig = plt.figure(figsize=(16, 5.6))
    for i, (c, t) in enumerate((("aso", "ASO"), (LABEL, LABEL), ("July baseline", "July baseline"))):
        panel(fig, (1, 3, i+1), g, f"f_{c}", t, "viridis", Normalize(0, 100), "% of early-spring SWE left")
    fig.suptitle(f"Melt order: % of early-spring SWE still present on late flights (same {len(yrs)} years: {yrs[0]}-{yrs[-1]})",
                 fontsize=12, y=1.02, color=INK)
    fig.tight_layout(); fig.savefig(FIGDIR/f"aso_radial_{TAG}_meltorder.png", dpi=130, bbox_inches="tight", facecolor="white")

    a = g0.set_index("HRU_ID")
    tab = frac.join(a[["elevClass", "aspectClas"]]).groupby(["elevClass", "aspectClas"]).mean().round(0)
    pd.set_option("display.width", 200); print("% of early SWE left on late flights, by band x aspect (1 NE, 2 SE, 3 SW, 4 NW, 0 flat)"); print(tab.to_string())
    print("wrote", FIGDIR/f"aso_radial_{TAG}_early_late.png", FIGDIR/f"aso_radial_{TAG}_meltorder.png")


if __name__ == "__main__":
    main()
