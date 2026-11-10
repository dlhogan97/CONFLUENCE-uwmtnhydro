"""
Is the Tuolumne Mar-May runoff deficit a snow problem or a subsurface/routing delay? (2026-10-27)
Reruns the full-basin G35 setup (full_basin_G35.py) with water-balance outputs and compares, as WY2013-2020 daily and monthly climatologies:
  surface water input (rain + melt reaching the soil, scalarRainPlusMelt), unrouted runoff (scalarTotalRunoff and its surface / soil-baseflow /
  aquifer parts), routed runoff (averageRoutedRunoff), observed streamflow, ET, and the change in soil + aquifer storage.
usage: python g35_melt_vs_runoff.py [--score-only]
"""
import sys, uuid, shutil, numpy as np, pandas as pd, netCDF4 as nc, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
import run_calibration_snow as S
import full_basin_albedo as F
import full_basin_G35 as G

FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
KEEP = F.OUT/"G35_wb"
SUMV = ["scalarRainPlusMelt", "scalarTotalRunoff", "scalarSurfaceRunoff", "scalarSoilBaseflow", "scalarAquiferBaseflow", "scalarTotalET", "scalarRainfall", "scalarSnowfall"]
STATEV = ["scalarSWE", "scalarTotalSoilWat", "scalarAquiferStorage", "scalarCanopyWat"]


class Runner(G.Runner):
    def _write_filemanager(self, work, prefix, forcing=None):
        F.Runner._write_filemanager(self, work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\naverageRoutedRunoff | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n"
                                              + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in SUMV + STATEV))


def run():
    r = Runner(work_root=str(F.OUT/f"G35wb_{uuid.uuid4().hex[:6]}"), **S.CFG); r.variant = F.VARIANTS["G"]; w = Path(r.work_root)/"keep"
    params, proc = r.run_summa(np.zeros(len(r.specs)), w); print("rc", proc.returncode, proc.stderr[-300:], flush=True)
    shutil.rmtree(KEEP, ignore_errors=True); shutil.copytree(w/"out", KEEP); shutil.rmtree(r.work_root, ignore_errors=True)


def analyse():
    r = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG); wt = r.hru_wt; d = nc.Dataset(KEEP/"cal_day.nc")
    t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    B = lambda v: pd.Series((np.array(d[f"{v}_mean"][:]).astype(float)*wt[None, :]).sum(1), index=t)                 # basin area-weighted mean
    mm = {}                                                                                                         # fluxes in mm/day
    for v in SUMV: mm[v] = B(v)*86400*(1000 if v in ("scalarTotalRunoff", "scalarSurfaceRunoff", "scalarSoilBaseflow", "scalarAquiferBaseflow", "scalarRainPlusMelt") else 1)
    mm["routed"] = pd.Series(np.array(d["averageRoutedRunoff_mean"][:, 0]), index=t)*86400*1000
    mm["observed"] = r.obs.reindex(t)/r.area*86400*1000
    stor = dict(SWE=B("scalarSWE"), soil=B("scalarTotalSoilWat"), aquifer=B("scalarAquiferStorage")*1000, canopy=B("scalarCanopyWat")); d.close()
    D = pd.DataFrame(mm)[r.analysis_start:r.sim_end]; St = pd.DataFrame(stor)[r.analysis_start:r.sim_end]
    print("units check, WY means (mm/yr):"); wy = D.index.year + (D.index.month >= 10); print((D.groupby(wy).sum().mean()).round(0).to_string())
    order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]; ndays = np.array([31, 30, 31, 31, 28.25, 31, 30, 31, 30, 31, 31, 30])
    Mo = D.groupby(D.index.month).mean().reindex(order).mul(ndays, axis=0)                                         # mm/month
    Sm = St.groupby(St.index.month).mean().reindex(order)
    print("\nmonthly climatology (mm/month; storages mm, monthly mean)"); print(pd.concat([Mo, Sm], axis=1).round(1).to_string())
    doy = (D.index - pd.to_datetime((D.index.year - (D.index.month < 10)).astype(str) + "-10-01")).days
    Dc = D.groupby(doy).mean().rolling(7, center=True, min_periods=1).mean(); Sc = St.groupby(doy).mean()
    cum = lambda s: s.loc[150:300].cumsum()                                                                         # Mar-Jul cumulative
    c = {k: (Dc[k].loc[150:300].cumsum()) for k in ("scalarRainPlusMelt", "scalarTotalRunoff", "routed", "observed")}
    half = {k: int((v >= v.iloc[-1]/2).idxmax()) for k, v in c.items()}; print("\nday of WY when half of the Mar-Jul (day 150-300) total is reached:", half)
    lag = {}
    for k in ("scalarTotalRunoff", "routed"):
        x, y = Dc["scalarRainPlusMelt"].loc[120:330], Dc[k].loc[120:330]; lag[k] = int(max(range(0, 45), key=lambda L: np.corrcoef(x.values[:len(x) - L], y.values[L:])[0, 1]))
    print("best lag of runoff behind rain+melt (days):", lag)
    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True); x = Dc.index
    for k, c_, lab, lw in (("scalarRainPlusMelt", "#1baf7a", "rain + melt reaching soil (model)", 2), ("scalarTotalRunoff", "#eda100", "unrouted runoff (model)", 2),
                           ("routed", "#2a78d6", "routed runoff (model)", 2), ("observed", "black", "observed streamflow", 2.4), ("scalarTotalET", "#e34948", "ET (model)", 1.5)):
        ax[0].plot(x, Dc[k], color=c_, lw=lw, label=lab)
    ax[0].set_ylabel("mm/day (7-d mean climatology)"); ax[0].legend(fontsize=9); ax[0].set_title("Tuolumne G35, WY2013-2020: water input vs runoff", loc="left")
    for k, c_ in (("scalarSurfaceRunoff", "#eb6834"), ("scalarSoilBaseflow", "#6250d6"), ("scalarAquiferBaseflow", "#008300")): ax[1].plot(x, Dc[k], color=c_, lw=2, label=k.replace("scalar", ""))
    ax[1].plot(x, Dc["observed"], color="black", lw=2.4, label="observed"); ax[1].set_ylabel("mm/day"); ax[1].legend(fontsize=9); ax[1].set_title("unrouted runoff components", loc="left")
    for k, c_ in (("SWE", "#2a78d6"), ("soil", "#a0522d"), ("aquifer", "#008300"), ("canopy", "#8c8c8c")): ax[2].plot(x, Sc[k] - (Sc[k].min() if k != "SWE" else 0), color=c_, lw=2, label=k + ("" if k == "SWE" else " (minus seasonal min)"))
    ax[2].set_ylabel("storage (mm)"); ax[2].legend(fontsize=9); ax[2].set_title("basin storages", loc="left")
    ticks = [0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335]; ax[2].set_xticks(ticks); ax[2].set_xticklabels(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"])
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG + "tuo_G35_melt_vs_runoff.png", dpi=130); Mo.to_csv(F.OUT/"G35_wb_monthly.csv")


if __name__ == "__main__":
    if "--score-only" not in sys.argv: run()
    analyse()
