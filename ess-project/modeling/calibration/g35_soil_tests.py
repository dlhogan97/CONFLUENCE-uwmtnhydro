"""
Tuolumne spring-runoff deficit: soil-storage sensitivity on the full-basin G35 setup (2026-10-28). ET parameters untouched.
g35_downslope_slug.py showed the Mar-May deficit is vertical fill-and-spill: each column absorbs ~240 mm of melt before saturation-excess
runoff starts (98% of runoff), and downslope connectivity does not matter. Variants:
  ref          G35 as is
  thin         soil layers thinned (coldState mLayerDepth / iLayerHeight) as far as the roots allow: never below rootingDepth + 5 cm,
               since a column thinner than its roots gives NaN and rootingDepth (ET) is left alone. Roots reach ~80% of the column, so
               the cut is only 12-18% (alpine HRUs 1-2 unchanged)
  macro        k_macropore 1e-5 -> 1e-3 m/s: fast drainage through unsaturated soil
  retention    theta_sat 0.45 -> 0.40, fieldCapacity 0.20 -> 0.12: less water held per metre (theta_sat must stay above
               critSoilTranspire 0.383, an ET parameter left untouched)
  thin_macro   thin + macro
Round 2 (2026-10-28), lateral drainage and precipitation phase (qTopmodl drains only water above fieldCapacity, at a rate scaled by
kAnisotropic, which was calibrated to 0.022, below its documented 0.1-10 range):
  phase        tempCritRain 273.15 -> 275.15 K. NOTE (2026-10-29): this was meant to undo the -2 C forcing offset for phase, but SUMMA makes rain
               ABOVE tempCritRain (derivforce.f90:297-304), so raising it made MORE snow; the undo is 271.15 K. Its null result says nothing about rain.
  tm_lat       thin_macro + kAnisotropic 10 (top of the documented range)
  tm_lat_fc    tm_lat + fieldCapacity 0.12 (lateral drainage starts at lower moisture)
  tm_lat_lin   tm_lat_fc + zScale_TOPMODEL 1.0 (outflow linear in drainable storage -> exponential recession)
  all          tm_lat_lin + phase
  all_direct   all + every HRU below 2800 m drains straight to the stream (downHRUindex 0)
Scored on WY2013-2020: monthly runoff vs observed, KGE, Mar-May and annual volume, soil-storage swing, runoff split, ASO SWE (should not move).
usage: python g35_soil_tests.py [--score-only]
"""
import os, sys, uuid, shutil, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import deepstore_build as DB
import run_calibration_snow as S
import full_basin_albedo as F
import full_basin_G35 as G

OUT = F.OUT/"soil_tests"; OUT.mkdir(exist_ok=True)
VARIANTS = {"ref": {}, "thin": dict(depth=0.5), "macro": dict(k_macropore=1e-3), "retention": dict(theta_sat=0.40, fieldCapacity=0.12),
            "thin_macro": dict(depth=0.5, k_macropore=1e-3)}
_TL = dict(depth=0.5, k_macropore=1e-3, kAnisotropic=10.0)
VARIANTS.update({"phase": dict(tempCritRain=275.15), "tm_lat": dict(_TL), "tm_lat_fc": dict(_TL, fieldCapacity=0.12),
                 "tm_lat_lin": dict(_TL, fieldCapacity=0.12, zScale_TOPMODEL=1.0),
                 "all": dict(_TL, fieldCapacity=0.12, zScale_TOPMODEL=1.0, tempCritRain=275.15),
                 "all_direct": dict(_TL, fieldCapacity=0.12, zScale_TOPMODEL=1.0, tempCritRain=275.15, direct_below=2800.0)})
HRUV = ["scalarRainPlusMelt", "scalarSurfaceRunoff", "scalarTotalET", "scalarTotalSoilWat", "scalarSWE"]
BASV = ["averageRoutedRunoff", "basin__SurfaceRunoff", "basin__ColumnOutflow"]


class Runner(G.Runner):
    soil = {}
    def _apply_params(self, work, params):
        super()._apply_params(work, params); n = len(self.hruid)
        for k in ("k_macropore", "theta_sat", "fieldCapacity", "kAnisotropic", "zScale_TOPMODEL", "tempCritRain"):
            if k in self.soil: DB.set_trial_param(work, k, [self.soil[k]]*n)
    def _subset(self, work):
        super()._subset(work)
        zd = self.soil.get("direct_below")
        if zd:
            with nc.Dataset(work/"attributes.nc", "a") as a:
                dn = np.array(a["downHRUindex"][:]); a["downHRUindex"][:] = np.where(np.array(a["elevation"][:]) < zd, 0, dn)
        f = self.soil.get("depth")
        if f:                                                        # nSnow = 0 in the cold state, so every layer is soil
            with nc.Dataset(work/"trialParams.nc") as tp: root = np.array(tp["rootingDepth"][:]).ravel()
            with nc.Dataset(work/"coldState.nc", "a") as c:
                z = np.array(c["iLayerHeight"][:]); fh = np.minimum(1.0, np.maximum(f, (root + 0.05)/z[-1]))         # per-HRU factor
                c["mLayerDepth"][:] = np.array(c["mLayerDepth"][:])*fh[None, :]; c["iLayerHeight"][:] = z*fh[None, :]
    def _write_filemanager(self, work, prefix, forcing=None):
        F.Runner._write_filemanager(self, work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n" + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in BASV + HRUV))


def run(name):
    r = Runner(work_root=str(OUT/f"{name}_{uuid.uuid4().hex[:6]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.soil = VARIANTS[name]; w = Path(r.work_root)/"keep"
    params, proc = r.run_summa(np.zeros(len(r.specs)), w)
    keep = OUT/name; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    return name, proc.returncode, proc.stderr[-300:]


def main():
    if "--score-only" not in sys.argv:
        todo = os.environ.get("SOIL_VARIANTS", ",".join(VARIANTS)).split(",")
        with ProcessPoolExecutor(len(todo)) as ex: print(list(ex.map(run, todo)), flush=True)
    r = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG); wt = r.hru_wt; sel = slice(r.analysis_start, r.sim_end)
    order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]; nd = np.array([31, 30, 31, 31, 28.25, 31, 30, 31, 30, 31, 31, 30])
    mon = lambda s: s.groupby(s.index.month).mean().reindex(order)*nd
    obs = r.obs[sel]/r.area*86400*1000; M = {"observed": mon(obs)}; rows = []
    for name in VARIANTS:
        d = nc.Dataset(OUT/name/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
        H = {v: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float), index=t).where(lambda z: np.abs(z) < 1e8) for v in HRUV}
        Bv = {v: pd.Series(np.array(d[f"{v}_mean"][:]).astype(float).ravel()[:len(t)], index=t) for v in BASV}; d.close()
        q = (Bv["averageRoutedRunoff"]*86400*1000)[sel]; M[name] = mon(q)
        soil = (H["scalarTotalSoilWat"]*wt).sum(1)[sel]; sclim = soil.groupby(soil.index.month).mean()
        j = pd.concat([q.rename("s"), obs.rename("o")], axis=1).dropna()
        kge = 1 - np.sqrt((np.corrcoef(j.s, j.o)[0, 1] - 1)**2 + (j.s.std()/j.o.std() - 1)**2 + (j.s.mean()/j.o.mean() - 1)**2)
        surf = (Bv["basin__SurfaceRunoff"]*86400*1000)[sel].sum()
        parts, diag = S.aso_objective(H["scalarSWE"].set_axis(r.hruid.astype(int), axis=1), r.aso, r.w_hru)
        rows.append(dict(variant=name, KGE=kge, MarMay_mm=M[name].loc[[3, 4, 5]].sum(), annual_mm=M[name].sum(), soil_swing_mm=sclim.max() - sclim.min(),
                         soil_min_mm=sclim.min(), surface_share=surf/q.sum(), ET_mm=float(-(H["scalarTotalET"]*wt).sum(1)[sel].mean()*86400*365.25),
                         aso_swe_nmae=diag["swe_nmae"]))
    T = pd.DataFrame(rows); T.loc[len(T)] = dict(variant="observed", MarMay_mm=M["observed"].loc[[3, 4, 5]].sum(), annual_mm=M["observed"].sum())
    pd.set_option("display.width", 220); print(T.round(3).to_string(index=False))
    MM = pd.DataFrame(M); print("\nmonthly runoff (mm/month), WY2013-2020"); print(MM.round(1).to_string())
    T.to_csv(OUT/"summary.csv", index=False); MM.to_csv(OUT/"monthly.csv")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(12)
    cols = {"observed": "black", "ref": "#8c8c8c", "thin": "#2a78d6", "macro": "#eb6834", "retention": "#1baf7a", "thin_macro": "#6250d6", "all": "#e34948", "all_direct": "#eda100"}
    for k, c in cols.items(): ax.plot(x, MM[k], "-o", color=c, lw=2.4 if k == "observed" else 1.8, ms=5, label=k)
    ax.set_xticks(x); ax.set_xticklabels(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]); ax.set_ylabel("runoff (mm/month)")
    ax.grid(alpha=.3); ax.legend(); ax.set_title("Tuolumne G35 soil tests, WY2013-2020", loc="left"); fig.tight_layout()
    fig.savefig("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/tuo_G35_soil_tests_monthly.png", dpi=130)


if __name__ == "__main__":
    main()
