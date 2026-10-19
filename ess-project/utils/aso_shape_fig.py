"""ASO-derived precipitation shape function by elevation, both basins.

f(e) = mean ASO SWE in an elevation band / area-weighted basin-mean ASO SWE. The area-weighted
mean of f is 1.0 by construction, so redistributing precipitation by f conserves basin total.
Only the BLEND is drawn -- what was actually applied to the forcing:  f_blend = alpha*f + (1-alpha), alpha=0.5,
i.e. half the ASO gradient and half the original flat field.

TUOLUMNE ONLY. East's shape function was computed but never applied -- there is no
basin_averaged_data_gradient_corrected/ for East. Its curve is drawn for comparison and the
legend says so; do not describe it as a correction the East runs received.

Values live in geospatial-data/aso_precip_shape_function.csv (provenance in its header).
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro")
CSV = Path("/scratch/dlhogan/ess-project-data/geospatial-data/aso_precip_shape_function.csv")
FIG = REPO/"ess-project/figures/draft"
ALPHA = 0.5
HUE = {"East": "#0072B2", "Tuolumne": "#D55E00"}


PRECIP_CSV = Path("/scratch/dlhogan/ess-project-data/geospatial-data"
                  "/tuolumne_decmar_precip_bands.csv")


def main(alpha=ALPHA, basin="Tuolumne"):
    d = pd.read_csv(CSV, comment="#")
    d = d[d.basin == basin].sort_values("elev_m").copy()
    d["f_blend"] = alpha*d["f"] + (1-alpha)
    # Share of BASIN precipitation landing in each band after correction. Because the
    # area-weighted mean of f is 1 by construction and the blend preserves that
    # (a*1 + (1-a)*1 = 1), these shares sum to 100% -- the correction only moves water
    # between bands, it never creates or destroys any.
    w = d.area_km2/d.area_km2.sum()
    d["pct"] = 100*w*d.f_blend
    assert abs(d.pct.sum() - 100) < 0.5, d.pct.sum()

    # Panel B: measured Dec-Mar band means from the forcing itself, before and after.
    # NB "after" is the gradient correction ONLY -- the DJFM x1.10 winter bias lives in a
    # separate forcing directory and is deliberately not included here.
    pb = pd.read_csv(PRECIP_CSV).set_index("elevClass")
    d = d.join(pb[["before_mm", "after_mm"]], on="band")
    mu = float((d.before_mm*w).sum())
    # kept for the printed table: ASO SWE rescaled to the precip basin mean, not plotted
    d["aso_mm"] = d.aso_swe_mm * mu/float((d.aso_swe_mm*w).sum())

    plt.style.use("seaborn-v0_8-poster")
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.2), sharey=True)
    c = HUE[basin]

    a = ax[0]
    a.axvline(1.0, color="0.55", lw=1.6, ls=":", zorder=1)
    a.plot(d.f_blend, d.elev_m, color=c, lw=3.4, marker="o", ms=12, mfc=c, mew=0, zorder=3)
    for _, r in d.iterrows():
        a.annotate(f"{r.pct:.0f}%", (r.f_blend, r.elev_m), textcoords="offset points",
                   xytext=(13, -6), fontsize=14, color=c, fontweight="bold")
    a.text(0.99, 0.02, "% = share of total basin precipitation (sums to 100%)",
           transform=a.transAxes, ha="right", va="bottom", fontsize=13, color="0.35")
    a.set_xlabel("precipitation multiplier", fontsize=17)
    a.set_ylabel("elevation band mean (m)", fontsize=17)
    a.set_xlim(d.f_blend.min()-0.10, d.f_blend.max()+0.24)
    a.set_title("A   Multiplier applied to each band", loc="left", fontsize=16)

    b = ax[1]
    # VOLUME, not depth: depth x band area. Bars sit at true elevation (shared y with A), so
    # this reads hypsometrically -- the mid-elevation bands dominate because they are large,
    # not because their multiplier is large. Panel A's biggest multipliers move the least water.
    d["vol_before"] = d.before_mm*d.area_km2/1e3          # mm x km2 -> Mm3 (1e6 m3)
    d["vol_after"] = d.after_mm*d.area_km2/1e3
    h = 118.0                                              # bar half-height in metres
    b.barh(d.elev_m + h/2, d.vol_before, height=h, color="0.80", edgecolor="0.45",
           lw=1.2, zorder=3, label="unadjusted")
    b.barh(d.elev_m - h/2, d.vol_after, height=h, color=c, edgecolor="none",
           zorder=3, label="ASO-adjusted")
    b.set_xlabel("Dec–Mar volume (10$^6$ m$^3$)", fontsize=17)
    b.set_title("B   Band volume, unadjusted vs adjusted", loc="left", fontsize=16)
    b.text(0.985, 0.36, f"basin total {d.vol_before.sum():.0f} → {d.vol_after.sum():.0f} "
           "$\\times$10$^6$ m$^3$\n(unchanged: water is moved, not added)",
           transform=b.transAxes, ha="right", va="bottom", fontsize=13, color="0.35")
    b.legend(frameon=False, fontsize=14, loc="lower right")

    for a_ in ax:
        a_.tick_params(labelsize=15)
        a_.grid(alpha=0.18, lw=0.7); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    out = FIG/"aso_precip_shape_function.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("wrote", out)
    print(d[["band", "elev_m", "f_blend", "pct", "area_km2", "before_mm", "after_mm",
             "vol_before", "vol_after"]]
          .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("basin mean  before %.1f  after %.1f  ASO(scaled) %.1f mm"
          % ((d.before_mm*w).sum(), (d.after_mm*w).sum(), (d.aso_mm*w).sum()))
    return fig


if __name__ == "__main__":
    main()
