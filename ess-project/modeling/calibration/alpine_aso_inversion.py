"""
Backward energy check at the alpine HRUs against ASO (2026-10-27). Rerun of full_basin_albedo variant G (compressed albedo + decay ramp, forest kept)
with daily energy-balance output. For every HRU and every water year with an early (Mar-Apr) and a late (May 20-Jul) ASO flight:
  observed melt rate = (SWE_early - SWE_late)/days            (ASO; snow loss incl. sublimation)
  model melt rate    = model SWE loss over the same interval / days the model still had snow   (so an early model melt-out is not hidden)
  R = L_f*(obs rate - model rate)/86400   [W/m2]    (negative = the model has MORE energy than the observed melt needed)
Model energy terms (absorbed SW, net LW, H, LE, albedo) are averaged over the model's snow days in each interval, for attribution.
usage: python alpine_aso_inversion.py [--score-only]
"""
import sys, uuid, shutil, json, numpy as np, pandas as pd, netCDF4 as nc, xarray as xr
from pathlib import Path
import run_calibration_snow as S
import full_basin_albedo as F

LF = 3.34e5; OUT = F.OUT/"alpine_inversion"; OUT.mkdir(exist_ok=True)
F.OUTV = ["averageRoutedRunoff", "scalarSWE", "scalarSnowfall", "scalarRainfall", "scalarSnowSublimation", "scalarSnowAlbedo", "scalarGroundAbsorbedSolar",
          "scalarLWNetGround", "scalarSenHeatGround", "scalarLatHeatGround", "scalarGroundNetNrgFlux", "scalarSWRadAtm" if False else "scalarCanopySublimation"]


def run():
    r = F.Runner(work_root=str(OUT/f"G_{uuid.uuid4().hex[:6]}"), **S.CFG); r.variant = F.VARIANTS["G"]; w = Path(r.work_root)/"keep"
    params, proc = r.run_summa(np.zeros(len(r.specs)), w); shutil.rmtree(OUT/"G_energy", ignore_errors=True); shutil.copytree(w/"out", OUT/"G_energy")
    (OUT/"G_energy"/"forcing_list.json").write_text(json.dumps(sorted(str(f) for f in (w/"forcing").glob("*.nc"))))
    shutil.copytree(w/"forcing", OUT/"G_forcing", dirs_exist_ok=True); shutil.rmtree(r.work_root, ignore_errors=True); print("run rc", proc.returncode, proc.stderr[-300:])


def score():
    r = S.RunnerSnow(work_root=str(OUT/"score"), **S.CFG)
    d = nc.Dataset(OUT/"G_energy"/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    V = {v: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float), index=t, columns=r.hruid.astype(int)).where(lambda z: np.abs(z) < 1e8) for v in F.OUTV if f"{v}_mean" in d.variables and v != "averageRoutedRunoff"}; d.close()
    fx = xr.open_mfdataset(sorted((OUT/"G_forcing").glob("*.nc")), combine="by_coords"); SWin = pd.DataFrame(fx["SWRadAtm"].values, index=pd.DatetimeIndex(fx["time"].values), columns=r.hruid.astype(int)).resample("D").mean()
    at = xr.open_dataset(r.settings/"attributes.nc"); ids = at.hruId.values.astype(int)
    meta = pd.DataFrame(dict(elev=at.elevation.values.ravel(), aspect=at.aspect.values.ravel(), slope=np.degrees(np.arctan(np.abs(at.tan_slope.values.ravel()))), veg=at.vegTypeIndex.values.ravel()), index=ids)
    meta["northness"] = np.cos(np.radians(meta.aspect))*np.sin(np.radians(meta.slope))
    a = r.aso.copy(); a["wy"] = a.date.dt.year; early = a[a.date.dt.month.isin([3, 4])]; late = a[((a.date.dt.month == 5) & (a.date.dt.day >= 20)) | a.date.dt.month.isin([6, 7])]
    rows = []
    for (wy, h), e in early.groupby(["wy", "hru"]):
        l = late[(late.wy == wy) & (late.hru == h)]
        if l.empty: continue
        e = e.sort_values("date").iloc[-1]; l = l.sort_values("date").iloc[0]
        if e.aso_mm < 100: continue
        days = (l.date - e.date).days; sw = V["scalarSWE"][h]; win = sw[e.date:l.date]; snow = win > 1.0
        m_loss = float(sw.get(e.date, np.nan) - sw.get(l.date, np.nan)); m_days = max(int(snow.sum()), 1)
        o_rate = (e.aso_mm - l.aso_mm)/days; m_rate = m_loss/m_days; sel = snow[snow].index
        en = {k: float(V[v][h].reindex(sel).mean()) for k, v in (("absSW", "scalarGroundAbsorbedSolar"), ("LWnet", "scalarLWNetGround"), ("H", "scalarSenHeatGround"),
                                                                 ("LE", "scalarLatHeatGround"), ("albedo", "scalarSnowAlbedo"))}
        en["SWin"] = float(SWin[h].reindex(sel).mean())
        rows.append(dict(wy=wy, hru=h, early=e.date.date(), late=l.date.date(), days=days, aso_early=e.aso_mm, aso_late=l.aso_mm, mdl_early=float(sw.get(e.date, np.nan)), mdl_late=float(sw.get(l.date, np.nan)),
                         model_snow_days=m_days, obs_rate=o_rate, mdl_rate=m_rate, R=LF*(o_rate - m_rate)/86400.0, **en))
    D = pd.DataFrame(rows).join(meta, on="hru"); D.to_csv(OUT/"aso_interval_residuals.csv", index=False)
    pd.set_option("display.width", 250)
    g = D.groupby("hru").agg(n=("R", "size"), elev=("elev", "first"), northness=("northness", "first"), veg=("veg", "first"), obs_rate=("obs_rate", "mean"), mdl_rate=("mdl_rate", "mean"),
                             R=("R", "mean"), albedo=("albedo", "mean"), SWin=("SWin", "mean"), absSW=("absSW", "mean"), LWnet=("LWnet", "mean"), H=("H", "mean"), LE=("LE", "mean"), mdl_snow_days=("model_snow_days", "mean"), days=("days", "mean"))
    g["albedo_needed_if_SW"] = g.albedo - g.R/g.SWin
    print("per HRU, mean over ASO early->late intervals (rates mm/d; energy W/m2; R<0 = model has excess energy)"); print(g.sort_values("elev", ascending=False).round(2).to_string())
    for band, lo, hi in (("alpine >3300 m", 3300, 4000), ("3040-3071 m", 3000, 3100), ("<2800 m", 0, 2800)):
        s = D[(D.elev >= lo) & (D.elev < hi)]; print(f"{band}: mean R {s.R.mean():.1f} W/m2 (n {len(s)}), obs rate {s.obs_rate.mean():.1f} vs model {s.mdl_rate.mean():.1f} mm/d")
    print("\nSpearman of R with HRU properties / energy terms (all intervals):", {k: round(D[["R", k]].corr(method="spearman").iloc[0, 1], 2) for k in ("elev", "northness", "albedo", "SWin", "absSW", "LWnet", "H", "LE")})


if __name__ == "__main__":
    if "--score-only" not in sys.argv: run()
    score()
