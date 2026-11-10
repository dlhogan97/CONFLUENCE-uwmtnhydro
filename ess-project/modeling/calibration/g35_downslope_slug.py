"""
Does downslope soil-water transfer delay Tuolumne runoff from Apr-Jun to May-Jun? (2026-10-28)
qTopmodl: each HRU's lateral column outflow enters the soil of its downHRUindex HRU (run_oneGRU.f90:228-242); only outlet HRUs (downHRUindex 0)
deliver column outflow to the stream; inflow beyond what a column can pass on exfiltrates and is added to its outflow (groundwatr.f90:456-476).
Chains: 1-6-11-16-21-26, 2-7-12-17-22, 3-8-13-18, 4-9-14-19-23(?) ... (read from attributes.nc).
Two full-basin G35 runs (full_basin_G35.py setup), per-HRU water-balance output:
  connected     as calibrated
  disconnected  downHRUindex = 0 everywhere: every HRU's lateral outflow goes straight to the stream (counterfactual)
Tests: (1) timing of local input, lateral inflow, surface runoff and soil storage by chain position; (2) basin runoff by chain position and by
source (surface runoff vs outlet column outflow); (3) connected vs disconnected monthly runoff against observations.
usage: python g35_downslope_slug.py [--score-only]
"""
import sys, uuid, shutil, numpy as np, pandas as pd, netCDF4 as nc, xarray as xr, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import run_calibration_snow as S
import full_basin_albedo as F
import full_basin_G35 as G

FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
OUT = F.OUT/"downslope_slug"; OUT.mkdir(exist_ok=True)
HRUV = ["scalarRainPlusMelt", "scalarSurfaceRunoff", "scalarExfiltration", "scalarSoilBaseflow", "scalarInfiltration", "scalarTotalET", "scalarTotalSoilWat", "scalarSWE",
        "mLayerColumnInflow", "mLayerColumnOutflow"]
BASV = ["averageRoutedRunoff", "basin__SurfaceRunoff", "basin__ColumnOutflow", "basin__TotalRunoff"]


class Runner(G.Runner):
    connected = True
    def _subset(self, work):
        super()._subset(work)
        if not self.connected:
            with nc.Dataset(work/"attributes.nc", "a") as a: a["downHRUindex"][:] = 0
    def _write_filemanager(self, work, prefix, forcing=None):
        F.Runner._write_filemanager(self, work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n" + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in BASV + HRUV))


def run(name):
    r = Runner(work_root=str(OUT/f"{name}_{uuid.uuid4().hex[:6]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.connected = name == "connected"; w = Path(r.work_root)/"keep"
    params, proc = r.run_summa(np.zeros(len(r.specs)), w)
    keep = OUT/name; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    return name, proc.returncode, proc.stderr[-300:]


def load(name, r):
    d = nc.Dataset(OUT/name/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    ids = r.hruid.astype(int); area = pd.Series(r.hru_area, index=ids) if hasattr(r, "hru_area") else None; H = {}
    for v in HRUV:
        a = np.array(d[f"{v}_mean"][:]).astype(float); a[np.abs(a) > 1e8] = np.nan
        if a.ndim == 3: a = np.nansum(a, axis=1)                                                         # (time, midSoil, hru): sum over soil layers
        H[v] = pd.DataFrame(a, index=t, columns=ids)
    B = {v: pd.Series(np.array(d[f"{v}_mean"][:]).astype(float).ravel()[:len(t)], index=t) for v in BASV}; d.close(); return H, B


def wyday(i): return (i - pd.to_datetime((i.year - (i.month < 10)).astype(str) + "-10-01")).days


def main():
    if "--score-only" not in sys.argv:
        with ProcessPoolExecutor(2) as ex: print(list(ex.map(run, ["connected", "disconnected"])), flush=True)
    r = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG)
    at = xr.open_dataset(r.settings/"attributes.nc"); ids = at.hruId.values.astype(int)
    down = pd.Series(at.downHRUindex.values.ravel().astype(int), index=ids); area = pd.Series(at.HRUarea.values.ravel(), index=ids); el = pd.Series(at.elevation.values.ravel(), index=ids)
    pos = {}                                                                                              # chain position: 0 = top (no upslope HRU)
    ups = {h: [u for u in ids if down[u] == h] for h in ids}
    def depth(h): return 0 if not ups[h] else 1 + max(depth(u) for u in ups[h])
    for h in ids: pos[h] = depth(h)
    pos = pd.Series(pos); outlet = down == 0; wt = area/area.sum()
    print("chain position by HRU:", pos.to_dict()); print("outlets:", list(down[outlet].index))
    sel = lambda x: x[r.analysis_start:r.sim_end]; R = {}
    for name in ("connected", "disconnected"):
        H, B = load(name, r); H = {k: sel(v) for k, v in H.items()}; B = {k: sel(v) for k, v in B.items()}
        mm = lambda v: v*86400*1000                                                                       # m/s -> mm/day
        lat_in = H["mLayerColumnInflow"].div(area)*86400*1000; lat_out = H["mLayerColumnOutflow"].div(area)*86400*1000
        R[name] = dict(input=mm(H["scalarRainPlusMelt"]), surf=mm(H["scalarSurfaceRunoff"]), exf=mm(H["scalarExfiltration"]), lat_in=lat_in, lat_out=lat_out,
                       soil=H["scalarTotalSoilWat"], et=H["scalarTotalET"]*86400, swe=H["scalarSWE"],
                       routed=B["averageRoutedRunoff"]*86400*1000, b_surf=B["basin__SurfaceRunoff"]*86400*1000, b_col=B["basin__ColumnOutflow"]/area.sum()*86400*1000)
    obs = sel(r.obs)/r.area*86400*1000
    c = R["connected"]; idx = c["input"].index; wd = wyday(idx); order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]; nd = np.array([31, 30, 31, 31, 28.25, 31, 30, 31, 30, 31, 31, 30])
    mon = lambda s: s.groupby(s.index.month).mean().reindex(order)*nd
    # (1) timing by chain position (area-weighted within position), Mar-Jul half-volume day and storage peak day
    rows = []
    for p in sorted(pos.unique()):
        hs = pos[pos == p].index; w = area[hs]/area[hs].sum()
        agg = {k: (c[k][hs]*w).sum(1) for k in ("input", "lat_in", "surf", "exf", "lat_out", "soil")}
        clim = {k: v.groupby(wd).mean() for k, v in agg.items()}
        half = lambda s: int((s.loc[150:300].cumsum() >= s.loc[150:300].sum()/2).idxmax()) if s.loc[150:300].sum() > 0 else np.nan
        rows.append(dict(position=p, n_hru=len(hs), mean_elev=float((el[hs]*w).sum()), area_km2=area[hs].sum()/1e6,
                         input_mm=agg["input"].groupby(agg["input"].index.year).sum().mean(), lat_in_mm=agg["lat_in"].groupby(agg["lat_in"].index.year).sum().mean(),
                         surf_mm=agg["surf"].groupby(agg["surf"].index.year).sum().mean(), lat_out_mm=agg["lat_out"].groupby(agg["lat_out"].index.year).sum().mean(),
                         half_input=half(clim["input"]), half_lat_in=half(clim["lat_in"]), half_surf=half(clim["surf"]), half_lat_out=half(clim["lat_out"]),
                         soil_peak_day=int(clim["soil"].loc[100:330].idxmax()), surf_share_when_inflow_gt_input=float(agg["surf"][agg["lat_in"] > agg["input"]].sum()/max(agg["surf"].sum(), 1e-9))))
    T = pd.DataFrame(rows); pd.set_option("display.width", 250); print("\n(1) by chain position (0 = top). half_* = day of WY when half the Mar-Jul (day 150-300) total is reached"); print(T.round(1).to_string(index=False))
    # (2) basin runoff by chain position (surface runoff, area-weighted to basin mm) + outlet column outflow, monthly
    M = pd.DataFrame({f"surf pos{p}": mon((c["surf"][pos[pos == p].index]*wt[pos[pos == p].index]).sum(1)) for p in sorted(pos.unique())})
    M["outlet column outflow"] = mon(c["b_col"]); M["routed total"] = mon(c["routed"]); M["observed"] = mon(obs)
    print("\n(2) connected run: basin runoff (mm/month) by source"); print(M.round(1).to_string())
    # (3) connected vs disconnected
    K = pd.DataFrame({"observed": mon(obs), "connected": mon(R["connected"]["routed"]), "disconnected": mon(R["disconnected"]["routed"]),
                      "disc surface": mon(R["disconnected"]["b_surf"]), "disc column outflow": mon(R["disconnected"]["b_col"]),
                      "conn soil (mm)": (R["connected"]["soil"]*wt).sum(1).groupby(idx.month).mean().reindex(order),
                      "disc soil (mm)": (R["disconnected"]["soil"]*wt).sum(1).groupby(idx.month).mean().reindex(order),
                      "conn ET": mon(-(R["connected"]["et"]*wt).sum(1)), "disc ET": mon(-(R["disconnected"]["et"]*wt).sum(1))})
    print("\n(3) connected vs disconnected (mm/month)"); print(K.round(1).to_string()); print("annual:", K.drop(columns=["conn soil (mm)", "disc soil (mm)"]).sum().round(0).to_dict())
    T.to_csv(OUT/"timing_by_position.csv", index=False); M.to_csv(OUT/"runoff_by_position.csv"); K.to_csv(OUT/"connected_vs_disconnected.csv")
    # figure: (a) per-position climatology of input / lateral inflow / surface runoff for one chain; (b) runoff by position stacked; (c) connected vs disconnected vs obs
    fig, ax = plt.subplots(3, 1, figsize=(12, 13)); x = np.arange(12); labs = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    cols = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e34948", "#6250d6", "#008300"]
    for p, col in zip(sorted(pos.unique()), cols):
        hs = pos[pos == p].index; w = area[hs]/area[hs].sum(); s = (c["lat_in"][hs]*w).sum(1).groupby(wd).mean().rolling(7, center=True, min_periods=1).mean()
        sr = (c["surf"][hs]*w).sum(1).groupby(wd).mean().rolling(7, center=True, min_periods=1).mean()
        ax[0].plot(s.index, s, color=col, lw=1.4, ls="--"); ax[0].plot(sr.index, sr, color=col, lw=2, label=f"position {p} (~{T.mean_elev[T.position == p].iloc[0]:.0f} m): surface runoff (solid), lateral inflow (dashed)")
    ax[0].set_ylabel("mm/day (7-d mean)"); ax[0].legend(fontsize=8); ax[0].set_title("connected run: surface runoff and lateral inflow by chain position (0 = top)", loc="left")
    ax[0].set_xticks([0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335]); ax[0].set_xticklabels(labs)
    bot = np.zeros(12)
    for k, col in zip([m for m in M.columns if m.startswith("surf")] + ["outlet column outflow"], cols):
        ax[1].bar(x, M[k], bottom=bot, color=col, label=k, edgecolor="white", lw=.5); bot += M[k].values
    ax[1].plot(x, M.observed, "-o", color="black", lw=2.2, label="observed"); ax[1].set_ylabel("mm/month"); ax[1].legend(fontsize=8); ax[1].set_title("connected run: basin runoff by source", loc="left")
    ax[2].plot(x, K.observed, "-o", color="black", lw=2.2, label="observed"); ax[2].plot(x, K.connected, "-o", color="#2a78d6", lw=2, label="connected (as calibrated)")
    ax[2].plot(x, K.disconnected, "-o", color="#eb6834", lw=2, label="disconnected (all HRUs drain to stream)"); ax[2].set_ylabel("mm/month"); ax[2].legend(); ax[2].set_title("routed runoff, WY2013-2020", loc="left")
    for a in ax[1:]: a.set_xticks(x); a.set_xticklabels(labs)
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG + "tuo_G35_downslope_slug.png", dpi=130); print("figure written")


if __name__ == "__main__":
    main()
