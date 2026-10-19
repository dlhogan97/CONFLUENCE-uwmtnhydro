"""Hydrograph-FEATURE scorecard across models/runs. One number per feature (signed error + a normalised error), same window for all.
Features: overall bias; monthly-volume error; fall low flow (Aug-Oct 7-d min); winter low flow (Dec-Mar 7-d min); rising-limb timing (day of 10% / 25% of Mar-Jul volume);
peak flow (annual max 7-d); peak timing; centre-of-mass timing (50% of water-year volume); receding limb (days from peak to 25% of peak, 7-d flow); plus daily KGE/NSE/logNSE."""
import sys, warnings, glob, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr
L = "/scratch/dlhogan/ess-project-data/calibration/"; EO = "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/"
SD = "/scratch/dlhogan/ess-project-data/streamflow-data/TuolumneRiver/"
OBS = {"East": "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/observations/streamflow/preprocessed/East_River_USGS09112500_dailyQ_WY1970_2025.csv",
       "Tuolumne": "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/preprocessed/Tuolumne_River_merged_dailyQ_WY1971_2025.csv"}
def nc_series(d, f):
    ds = xr.open_dataset(f"{d}/out/{f}" if "out" not in f and (d.endswith("run") or "best_full" in d) else f"{d}/{f}"); t = pd.DatetimeIndex(ds["time"].values)
    A = xr.open_dataset(f"{d}/attributes.nc")["HRUarea"].values.sum(); return (pd.Series(ds["averageRoutedRunoff"].isel(gru=0).values, index=t)*A).resample("D").mean()
def csv_series(p, col=None):
    d = pd.read_csv(p, index_col=0, parse_dates=True); return d[col or d.columns[-1]] if col or len(d.columns) > 1 else d.iloc[:, 0]
def tsv(p, name):  # streamflow-data style: time,station_id,discharge
    d = pd.read_csv(p, parse_dates=["time"]).set_index("time"); return d["discharge"]
RUNS = {
 "East": {"old best (SUMMA, v1 obj)": nc_series(L+"bestruns/East_River_baseline_1981_2025/run", "baseline_timestep.nc"),
          "v2 round-1 best (SUMMA)": nc_series(L+"runs/East_River_v2_20261014_185148/best_full/rank0", "best_timestep.nc"),
          "v2 round-2 best (SUMMA)": nc_series(L+"runs/East_River_r2_20261015_110603/best_full/rank0", "best_timestep.nc"),
          "M0 aquifer-refit (SUMMA)": csv_series(EO+"overnight/summa_daily_M0_full.csv"), "M1 deep store (SUMMA)": csv_series(EO+"overnight/summa_daily_M1_full.csv"),
          "M0 v2-fit (emulator)": csv_series(EO+"overnight_v2/daily_M0_full.csv"), "M1 v2-fit (emulator)": csv_series(EO+"overnight_v2/daily_M1_full.csv")},
 "Tuolumne": {"old best (SUMMA, v1 obj)": nc_series(L+"bestruns/Tuolumne_River_baseline_1981_2025/run", "baseline_timestep.nc"),
              "v2 round-1 best (SUMMA)": nc_series(L+"runs/Tuolumne_River_v2_20261014_185148/best_full/rank0", "best_timestep.nc"),
              "v2 round-2 best (SUMMA)": nc_series(L+"runs/Tuolumne_River_r2_20261015_110603/best_full/rank0", "best_timestep.nc"),
              "old best + snowpatch (anchor m080)": tsv(SD+"TuolumneRiver_SUMMA_snowpatch_anchorm080_dailyDischarge_1981_2025.csv", ""),
              "old best + snowpatch (m090q089)": tsv(SD+"TuolumneRiver_SUMMA_snowpatch_m090q089_dailyDischarge_1981_2025.csv", "")}}
def wy(i): return i.year + (i.month >= 10)
def roll7(q): return q.rolling(7, center=True, min_periods=7).mean()
def per_wy(q, f):
    out = {}
    for y, g in q.groupby(wy(q.index)):
        if len(g) >= 360: out[y] = f(g, y)
    return pd.Series(out)
def doy_frac(g, y, months, frac):
    s = g[g.index.month.isin(months)]; c = s.cumsum()/s.sum(); return float((c.index[np.argmax(c.values >= frac)] - pd.Timestamp(int(y)-1, 10, 1)).days)
def peak(g, y): r = roll7(g).dropna(); t = r.idxmax(); return (float(r.max()), float((t - pd.Timestamp(int(y)-1, 10, 1)).days))
def recess(g, y):
    r = roll7(g).dropna(); t = r.idxmax(); post = r.loc[t:]; below = post[post <= 0.25*r.max()]
    return float((below.index[0]-t).days) if len(below) else np.nan
def min7(g, months): r = roll7(g.where(g.index.month.isin(months))); return float(r.min())
def features(sim, obs, a, b):
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna(); j = j[(j.index >= a) & (j.index <= b)]; s, o = j.s.clip(lower=0.01), j.o.clip(lower=0.01); F = {}
    F["overall bias (%)"] = 100*(s.sum()/o.sum()-1)
    mm = (s.groupby(s.index.month).mean()-o.groupby(o.index.month).mean()); F["monthly-volume error (% of obs, sum|bias|)"] = 100*mm.abs().sum()/o.groupby(o.index.month).mean().sum()
    r = lambda f, nm, rel=True: None
    fall = (per_wy(s, lambda g, y: min7(g, [8, 9, 10]))/per_wy(o, lambda g, y: min7(g, [8, 9, 10]))).dropna(); F["fall low flow: Aug-Oct 7-d min, model/obs (median)"] = float(fall.median())
    win = (per_wy(s, lambda g, y: min7(g, [12, 1, 2, 3]))/per_wy(o, lambda g, y: min7(g, [12, 1, 2, 3]))).dropna(); F["winter low flow: Dec-Mar 7-d min, model/obs (median)"] = float(win.median())
    for fr in (0.10, 0.25):
        d = (per_wy(s, lambda g, y: doy_frac(g, y, [3, 4, 5, 6, 7], fr)) - per_wy(o, lambda g, y: doy_frac(g, y, [3, 4, 5, 6, 7], fr))).dropna()
        F[f"rising limb: day of {int(100*fr)}% Mar-Jul volume, model-obs (d, median)"] = float(d.median()); F[f"   |error| (d, mean)" if fr == .10 else "   |error| (d, mean) "] = float(d.abs().mean())
    ps, po = per_wy(s, peak).dropna(), per_wy(o, peak).dropna(); c = ps.index.intersection(po.index)
    pf = pd.Series({y: ps[y][0]/po[y][0]-1 for y in c}); pt = pd.Series({y: ps[y][1]-po[y][1] for y in c})
    F["peak flow: annual max 7-d, bias (%, median)"] = 100*float(pf.median()); F["   |error| (%, mean)"] = 100*float(pf.abs().mean())
    F["peak timing: model-obs (d, median)"] = float(pt.median()); F["   |error| (d, mean)  "] = float(pt.abs().mean())
    com = (per_wy(s, lambda g, y: doy_frac(g, y, list(range(1, 13)), 0.5)) - per_wy(o, lambda g, y: doy_frac(g, y, list(range(1, 13)), 0.5))).dropna(); F["centre-of-mass timing: model-obs (d, median)"] = float(com.median())
    rc = (per_wy(s, recess)-per_wy(o, recess)).dropna(); F["receding limb: days peak->25% of peak, model-obs (median)"] = float(rc.median())
    ls, lo = np.log10(s), np.log10(o); F["daily KGE"] = 1-np.sqrt((np.corrcoef(o, s)[0, 1]-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)
    F["daily NSE"] = 1-((o-s)**2).sum()/((o-o.mean())**2).sum(); F["daily logNSE"] = 1-((lo-ls)**2).sum()/((lo-lo.mean())**2).sum(); return F
if __name__ == "__main__":
    win = {"WY2009-2020": ("2008-10-01", "2020-09-30"), "WY2022-2025 (held out)": ("2021-10-01", "2025-09-30")}
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    allr = []
    for basin in RUNS:
        o = pd.read_csv(OBS[basin], parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
        for wn, (a, b) in win.items():
            tab = pd.DataFrame({nm: features(s, o, a, b) for nm, s in RUNS[basin].items() if len(s[(s.index >= a) & (s.index <= b)]) > 300})
            print(f"\n######## {basin} | {wn} ########"); print(tab.to_string()); tab.assign(basin=basin, window=wn).to_csv(f"/tmp/claude-1008/-home-dlhogan-projects-forked-repos-CONFLUENCE-uwmtnhydro/c0858c32-d024-48f4-b236-80c8fd0350ee/scratchpad/scorecard_{basin}_{wn[:8]}.csv")
