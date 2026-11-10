"""
Runoff-timing diagnosis of the full-basin G35 run (2026-10-27), against the ref albedo run and observations.
  fig 1  SWE hypsometry: area-weighted band-mean model SWE vs ASO, model sampled on the ASO flight days, averaged over all flights WY2013-2020
         and split by flight month (Mar-Apr, May, Jun-Jul)
  fig 2  monthly mean streamflow climatology (WY2013-2020), obs vs ref vs G35, and model - obs
usage: python g35_timing_figs.py
"""
import numpy as np, pandas as pd, netCDF4 as nc, xarray as xr, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run_calibration_snow as S
import full_basin_albedo as F

FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
RUNS = {"ref": ("#8c8c8c", "ref (Dana-only albedo)"), "G35": ("#2a78d6", "G35 (alpine albedo push)")}
r = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG)


def load(name):
    d = nc.Dataset(F.OUT/name/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    swe = pd.DataFrame(np.array(d["scalarSWE_mean"][:]).astype(float), index=t, columns=r.hruid.astype(int)).where(lambda z: np.abs(z) < 1e8)
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:, 0]), index=t)*r.area; d.close(); return swe, q


at = xr.open_dataset(r.settings/"attributes.nc"); el = pd.Series(at.elevation.values.ravel(), index=at.hruId.values.astype(int)); area = pd.Series(at.HRUarea.values.ravel(), index=el.index)
z = el.sort_values(); band = (z.diff() > 100).cumsum(); band = band.reindex(el.index)                 # elevation bands = HRUs within 100 m (the aspect splits)
bz = (el*area).groupby(band).sum()/area.groupby(band).sum()
M = {n: load(n) for n in RUNS}
aso = r.aso[r.aso.date.isin(M["G35"][0].index)].copy()
for n in RUNS: aso[n] = [M[n][0].at[d, h] for d, h in zip(aso.date, aso.hru)]
aso["band"] = aso.hru.map(band); aso["a"] = aso.hru.map(area); aso["m"] = aso.date.dt.month


def profile(df):
    g = df.groupby(["date", "band"]); out = {}
    for c in ["aso_mm", *RUNS]: out[c] = (df[c]*df.a).groupby([df.date, df.band]).sum()/df.a.groupby([df.date, df.band]).sum()
    return pd.DataFrame(out).groupby("band").mean()


sets = [("all flights", aso), ("Mar-Apr flights", aso[aso.m.isin([3, 4])]), ("May flights", aso[aso.m == 5]), ("Jun-Jul flights", aso[aso.m.isin([6, 7])])]
fig, ax = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
for a, (ttl, df) in zip(ax, sets):
    p = profile(df); y = bz.reindex(p.index)
    a.plot(p.aso_mm, y, "-o", color="black", lw=2.2, ms=8, label="ASO", zorder=5)
    for n, (c, lab) in RUNS.items(): a.plot(p[n], y, "-o", color=c, lw=2, ms=8, label=lab)
    a.set_title(f"{ttl} (n={df.date.nunique()})", loc="left"); a.set_xlabel("band-mean SWE (mm)"); a.grid(alpha=.3); a.set_xlim(left=0)
    for b in p.index: a.annotate(f"{p.at[b, 'G35']/p.at[b, 'aso_mm']:.2f}" if p.at[b, "aso_mm"] > 20 else "", (p.at[b, "G35"], y[b]), xytext=(6, -12), textcoords="offset points", fontsize=8, color="#2a78d6")
ax[0].set_ylabel("band mean elevation (m)"); ax[0].legend(fontsize=9, loc="lower right")
fig.suptitle("Tuolumne SWE hypsometry on ASO flight days, WY2013-2020 (blue numbers: G35 / ASO)", x=0.01, ha="left"); fig.tight_layout(); fig.savefig(FIG + "tuo_G35_swe_hypsometry_aso.png", dpi=130)
P = {k: profile(df) for k, df in sets}; print(pd.concat(P, axis=1).assign(elev=bz).round(0).to_string())

# fig 2 - monthly flow climatology
o = r.obs[r.analysis_start:r.sim_end]; order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]; lab = "OND J F M A M J J A S"
C = {"observed": o}; C.update({n: M[n][1][r.analysis_start:r.sim_end] for n in RUNS})
J = pd.concat(C, axis=1).dropna(); clim = J.groupby(J.index.month).mean().reindex(order)
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1])); x = np.arange(12)
ax[0].plot(x, clim.observed, "-o", color="black", lw=2.2, ms=8, label="observed", zorder=5)
for n, (c, l) in RUNS.items():
    ax[0].plot(x, clim[n], "-o", color=c, lw=2, ms=8, label=l); ax[1].bar(x + (-.2 if n == "ref" else .2), clim[n] - clim.observed, .38, color=c, label=l)
ax[1].axhline(0, color="black", lw=.8); ax[0].set_ylabel("mean flow (m3/s)"); ax[1].set_ylabel("model - observed (m3/s)")
ax[1].set_xticks(x); ax[1].set_xticklabels(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"])
for a in ax: a.grid(alpha=.3)
ax[0].legend(); ax[0].set_title("Tuolumne monthly mean streamflow, WY2013-2020", loc="left"); fig.tight_layout(); fig.savefig(FIG + "tuo_G35_monthly_flow_climatology.png", dpi=130)
vol = clim*clim.index.map(lambda m: pd.Timestamp(2001, m, 1).days_in_month).values[:, None]*86400/1e6
print(clim.round(1).to_string()); print("annual volume (Mm3):", vol.sum().round(0).to_dict()); print("share of annual volume by month (%):"); print((100*vol/vol.sum()).round(1).to_string())

# fig 3 - SWE volume by elevation band (band-mean SWE x band area), same flight subsets as fig 1
barea = area.groupby(band).sum()
fig, ax = plt.subplots(1, 4, figsize=(17, 6), sharey=True); hgt = 55
for a, (ttl, df) in zip(ax, sets):
    p = profile(df); y = bz.reindex(p.index); v = p.mul(barea.reindex(p.index), axis=0)*1e-3/1e6          # mm x m2 -> Mm3
    a.barh(y + hgt, v.aso_mm, hgt, color="black", label=f"ASO ({v.aso_mm.sum():.0f} Mm3)")
    for k, (n, (c, lab)) in enumerate(RUNS.items()): a.barh(y - k*hgt, v[n], hgt, color=c, label=f"{lab} ({v[n].sum():.0f} Mm3)")
    a.set_title(f"{ttl} (n={df.date.nunique()})", loc="left"); a.set_xlabel("SWE volume (Mm3)"); a.grid(alpha=.3, axis="x"); a.legend(fontsize=8, loc="upper right")
ax[0].set_ylabel("band mean elevation (m)"); ax[0].set_yticks(bz.values); ax[0].set_yticklabels([f"{z:.0f}\n{barea[b]/1e6:.0f} km2" for b, z in bz.items()], fontsize=8)
fig.suptitle("Tuolumne SWE volume by elevation band on ASO flight days, WY2013-2020 (band-mean SWE x band area)", x=0.01, ha="left"); fig.tight_layout()
fig.savefig(FIG + "tuo_G35_swe_volume_by_band.png", dpi=130)
VV = {k: profile(df).mul(barea, axis=0)*1e-9 for k, df in sets}; T = pd.concat(VV, axis=1).assign(elev=bz.round(0), area_km2=(barea/1e6).round(0))
print("\nSWE volume (Mm3)"); print(T.round(1).to_string()); print(T.drop(columns=["elev", "area_km2"]).sum().round(0).to_string())
