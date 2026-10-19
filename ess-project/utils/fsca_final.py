"""
Definitive end-of-season snow record for the Tuolumne snowpatch experiment.

Three tiers, because the terrain-shadow decision is the dominant uncertainty:
  strict     -- mask cloud + cloud-shadow (bits 1,2,3,4).  LOWER BOUND: it
                removes the shaded pockets where late-season snow survives.
  shadow     -- keep terrain shadow (bits 1,2,3) + a g>0.05 radiance guard.
                PRIMARY for an end-of-season record.
  aso_corr   -- shadow, times the band-resolved ASO factor on bands 7-8 only
                (1.19 / 1.77, matched at ASO SWE>0.25 m -- snow that constitutes
                a STORE, not 10 mm dusting). Bands 5-6 are NOT corrected: their
                huge factors are a threshold-definition mismatch, not bias.

Snow area is always EXTRAPOLATED: A = SUM_b fSCA_b * Area_b.
Window Aug 1 - Oct 10; annual value = minimum (immune to cloud and to
transient early-October snowfall).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import landsat_fsca as L
from fsca_figs import FIGDIR, HERE, C_LS, C_ASO, INK, INK2, GRID

A_HRU2 = 8.611236
BANDS = [8, 7, 6, 5, 4, 3, 2]
ASO_FACTOR = {8: 1.19, 7: 1.77}        # ASO SWE>0.25 m: a *store*, not 10 mm
MIN_VALID = 0.35                       # per-tier validity, NOT fillna(0)
ALL_YEARS = list(range(1985, 2026))
LONGTERM = ("/scratch/dlhogan/ess-project-data/calibration/bestruns/"
            "Tuolumne_River_baseline_wbias10_1981_2025/run/out/"
            "baseline_timestep.nc")


def band_areas():
    g = gpd.read_file(L.SHP)
    return ((g.groupby("elevClass").HRU_area.sum() / 1e6).to_dict(),
            g.groupby("elevClass").elev_mean.mean().to_dict())


def peak_swe():
    ds = xr.open_dataset(LONGTERM)
    s = ds["scalarSWE"].isel(hru=1).to_series()
    wy = s.index.year + (s.index.month >= 10)
    return s.groupby(wy).max().rename("peak_mm")


def build():
    BA, EL = band_areas()
    d = pd.read_csv(HERE / "scenes_by_band3.csv", parse_dates=["date"])
    ok = d[d.status == "ok"].copy()
    ok = ok[ok.apply(lambda r: (r.date.month, r.date.day) >= (8, 1), axis=1)]

    # A tier is usable only if THAT tier cleared the valid-fraction bar.
    # fillna(0) would silently read "no retrieval" as "no snow" -- that bug
    # made 2023 (valid_sh 0.343) win the annual minimum with a spurious 0.
    ok["A_strict"] = np.where(ok.valid >= MIN_VALID,
                              sum(ok[f"fsca_b{b}"].fillna(0) * BA[b] for b in BANDS),
                              np.nan)
    sh_ok = (ok.valid_sh >= MIN_VALID) & ok[[f"fsca_b{b}_sh" for b in (8, 7)]].notna().all(axis=1)
    ok["A_shadow"] = np.where(sh_ok,
                              sum(ok[f"fsca_b{b}_sh"].fillna(0) * BA[b] for b in BANDS),
                              np.nan)
    ok["A_aso"] = np.where(sh_ok,
                           sum(ok[f"fsca_b{b}_sh"].fillna(0) * BA[b]
                               * ASO_FACTOR.get(b, 1.0) for b in BANDS),
                           np.nan)
    # each tier picks its own annual minimum among the scenes valid FOR IT
    parts = {}
    for t in ("strict", "shadow", "aso"):
        v = ok.dropna(subset=[f"A_{t}"])
        parts[t] = v.loc[v.groupby("year")[f"A_{t}"].idxmin()].set_index("year")
    eos = parts["shadow"].copy()
    for t in ("strict", "aso"):
        eos[f"A_{t}"] = parts[t][f"A_{t}"]
    eos["n_scenes"] = ok.dropna(subset=["A_shadow"]).groupby("year").size()
    for t in ("strict", "shadow", "aso"):
        eos[f"f_{t}"] = (eos[f"A_{t}"] / A_HRU2).clip(upper=1.0)
    return eos, ok, BA, EL


def main():
    eos, ok, BA, EL = build()
    pk = peak_swe()
    print(f"years: {len(eos)}/41   missing {sorted(set(ALL_YEARS)-set(eos.index))}")
    for t, lab in [("strict", "strict (lower bound)"),
                   ("shadow", "shadow-incl (PRIMARY)"),
                   ("aso", "ASO-corrected (upper)")]:
        a, f = eos[f"A_{t}"], eos[f"f_{t}"]
        print(f"  {lab:24s} area mean {a.mean():5.3f}  range {a.min():5.3f}-"
              f"{a.max():6.3f} km2   fSCA_eff mean {f.mean():.3f} max {f.max():.3f}"
              f"   zero-yrs {(a<0.01).sum()}   capped {(eos[f'A_{t}']>A_HRU2).sum()}")

    j = eos.join(pk, how="inner").dropna(subset=["peak_mm"])
    x = j.peak_mm.values / 1000
    print(f"\n=== recession vs modelled hru-2 peak SWE (n={len(j)}) ===")
    fits = {}
    for t in ("strict", "shadow", "aso"):
        y = j[f"f_{t}"].values
        r = np.corrcoef(x, y)[0, 1]
        rs = pd.Series(x).corr(pd.Series(y), method="spearman")
        b, a_ = np.polyfit(x, y, 1)
        fits[t] = (b, a_, r, rs)
        print(f"  {t:8s} r={r:+.3f} rho={rs:+.3f} slope={b:.4f}/m "
              f"({b*A_HRU2:.2f} km2/m) intercept {a_:+.4f}  "
              f"zero-crossing {(-a_/b)*1000:.0f} mm")

    ser = eos.reindex(ALL_YEARS)
    missing = sorted(set(ALL_YEARS) - set(eos.index))

    # =============================================================== FIGURE 1
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(13.5, 9.0), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.4, 1],
                                                   hspace=0.15))
    for yv in missing:
        for a_ in (ax, axb):
            a_.axvspan(yv-.5, yv+.5, color="#9e9e9e", alpha=.30, lw=0, zorder=0)

    ax.fill_between(ser.index, ser.f_strict, ser.f_aso, color=C_LS, alpha=.15,
                    lw=0, label="retrieval uncertainty (strict ↔ ASO-corrected)")
    ax.plot(ser.index, ser.f_shadow, color=C_LS, lw=2.1, marker="o", ms=5.5,
            label="effective fSCA of snowpatch HRU (shadow-inclusive)", zorder=4)
    ax.plot(ser.index, ser.f_strict, color=C_LS, lw=1.0, ls=":", alpha=.75,
            label="strict mask (lower bound)")
    ax.axhline(1.0, color="#c0392b", lw=1.2, ls="--", label="fully snow covered")
    for yv in (1993, 1995, 2011, 2017, 2019, 2023):
        if yv in eos.index:
            ax.annotate(str(yv), (yv, eos.loc[yv, "f_aso"]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=8, color=INK2, ha="center")
    ax.set_ylabel("fSCA of HRU 2\n(basin snow area / HRU 2 area)")
    ax.set_ylim(0, 1.08)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="#9e9e9e", alpha=.30)); l.append("no usable scene")
    ax.legend(h, l, frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.set_title("Snowpatch HRU effective snow cover — basin persistent snow "
                 "gathered into HRU 2", loc="left", fontsize=12.5, color=INK,
                 pad=22)
    ax.text(0, 1.02, "Landsat p042r034, Aug 1 – Oct 10 annual minimum; snow area "
            "extrapolated over masked pixels; HRU 2 = 8.61 km²",
            transform=ax.transAxes, fontsize=8.8, color=INK2)

    axb.fill_between(ser.index, ser.A_strict, ser.A_aso, color="#7b5aa6",
                     alpha=.16, lw=0)
    axb.plot(ser.index, ser.A_shadow, color="#7b5aa6", lw=1.9, marker="s", ms=4.5)
    axb.axhline(A_HRU2, color="#c0392b", lw=1.2, ls="--")
    axb.annotate("HRU 2 area = 8.6 km²", (1986, A_HRU2),
                 textcoords="offset points", xytext=(0, 4), fontsize=8,
                 color="#c0392b")
    axb.set_ylabel("basin persistent\nsnow area (km²)")
    axb.set_xlabel("water year")
    axb.set_title("the underlying quantity", loc="left", fontsize=10.5, color=INK)

    for a_ in (ax, axb):
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            a_.spines[s_].set_color("#b0b0b0")
        a_.grid(True, color=GRID, lw=.6, alpha=.9); a_.set_axisbelow(True)
        a_.tick_params(colors=INK2, labelsize=9); a_.set_xlim(1984, 2026)
    fig.savefig(FIGDIR / "fsca_final_timeseries.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    print("\nwrote", FIGDIR / "fsca_final_timeseries.png")

    # =============================================================== FIGURE 2
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
    axx = axes[0]
    for t, col, lab, sz in [("strict", "#9ec9e2", "strict (lower)", 34),
                            ("aso", "#a88bc9", "ASO-corrected (upper)", 34),
                            ("shadow", C_LS, "shadow-incl (primary)", 72)]:
        axx.scatter(x, j[f"f_{t}"].values, s=sz, color=col,
                    edgecolor="white" if t == "shadow" else "none",
                    lw=.9, zorder=3 if t == "shadow" else 2, label=lab)
    b, a_, r, rs = fits["shadow"]
    xf = np.linspace(x.min(), x.max(), 50)
    axx.plot(xf, a_ + b*xf, color=C_LS, lw=2.0, ls="--", zorder=4)
    for yr in (1995, 2011, 2017, 2023, 2015, 2022):
        if yr in j.index:
            axx.annotate(str(yr), (j.loc[yr, "peak_mm"]/1000, j.loc[yr, "f_shadow"]),
                         textcoords="offset points", xytext=(6, 3),
                         fontsize=7.5, color=INK2)
    axx.annotate(f"primary: r = {r:+.2f}   ρ = {rs:+.2f}\n"
                 f"slope = {b:.3f} /m  ({b*A_HRU2:.2f} km² per m)",
                 (0.03, 0.97), xycoords="axes fraction", va="top",
                 fontsize=9.5, color=INK)
    axx.set_xlabel("modelled HRU-2 peak SWE (m)")
    axx.set_ylabel("effective fSCA of HRU 2")
    axx.set_title(f"Recession curve  (n = {len(j)})", loc="left", fontsize=11.5)
    axx.legend(frameon=False, fontsize=8.5, loc="lower right")

    axc = axes[1]
    snow = [eos[f"fsca_b{b_}_sh"].mean()*BA[b_] for b_ in BANDS]
    tot = float(np.sum(snow))
    yy = np.arange(len(BANDS))
    axc.barh(yy, snow, color="#7b5aa6", alpha=.85, height=.72,
             edgecolor="white", lw=.8)
    for i, s_ in enumerate(snow):
        axc.annotate(f"{s_:.3f} km²  ({100*s_/tot:.0f}%)", (s_, i),
                     textcoords="offset points", xytext=(6, -3), fontsize=8,
                     color=INK2)
    axc.set_yticks(yy)
    axc.set_yticklabels([f"band {b_}\n{EL[b_]:.0f} m · {BA[b_]:.0f} km²"
                         for b_ in BANDS], fontsize=8)
    axc.set_xlabel("mean end-of-season snow area (km²)")
    axc.set_title("Where the persistent snow is (shadow-inclusive)",
                  loc="left", fontsize=11.5)
    axc.set_xlim(0, max(snow)*1.5)

    for a_ in axes:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            a_.spines[s_].set_color("#b0b0b0")
        a_.grid(True, color=GRID, lw=.6, alpha=.9, axis="x")
        a_.set_axisbelow(True); a_.tick_params(colors=INK2, labelsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fsca_final_recession.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    print("wrote", FIGDIR / "fsca_final_recession.png")

    print("\n=== band shares (shadow-inclusive, extrapolated) ===")
    for b_, s_ in zip(BANDS, snow):
        print(f"  band {b_} ({EL[b_]:.0f} m, {BA[b_]:5.1f} km2): "
              f"{s_:.3f} km2  {100*s_/tot:5.1f}%")

    cols = ["date", "A_strict", "A_shadow", "A_aso", "f_strict", "f_shadow",
            "f_aso", "valid", "valid_sh", "cover", "platform", "n_scenes"]
    eos[cols].to_csv(HERE / "eos_final.csv")
    print(f"\nwrote {HERE/'eos_final.csv'}")


if __name__ == "__main__":
    main()
