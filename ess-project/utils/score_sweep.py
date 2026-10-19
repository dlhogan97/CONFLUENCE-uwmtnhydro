"""Score every sweep member. Run once the sweep finishes.

East      : low-flow lag-k ACF (Sep-Nov window, YE-NOV bins) vs observed.
Tuolumne  : same for its window (Aug-Oct, YS-OCT), PLUS fSCA vs the Landsat record
            scored at 30 Sep with a 0.05 m snow-covered threshold (per today's call:
            keep the Sep-30 date, use the threshold to kill the early-autumn veneer
            that put WY1989 at fSCA 0.80 on a 0.03 m store).
"""
from pathlib import Path
import glob, re
import numpy as np, pandas as pd, netCDF4 as nc4, xarray as xr
from scipy.stats import rankdata, pearsonr

SC = Path("/scratch/dlhogan/ess-project-data")
EAST_SW = SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/sweep_sf_20260812"
TUO_SW  = SC/"domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_sweep_20260812"
D = SC/"geospatial-data/snowpatch_fsca"
SWE_TH = 0.05
YR0, YR1 = 1970, 2024


def spm(a, b):
    ra = rankdata(a)-(len(a)+1)/2.; rb = rankdata(b)-(len(b)+1)/2.
    d = np.sqrt((ra*ra).sum()*(rb*rb).sum()); return (ra*rb).sum()/d if d > 0 else np.nan


def acf(s, nlag=3, nperm=4000, seed=0):
    s = s.dropna().astype(float); v = s.values
    pos = {int(y): i for i, y in enumerate(s.index)}; rng = np.random.default_rng(seed)
    rho, pv = [], []
    for k in range(1, nlag+1):
        ip = np.array([pos[int(y)-k] for y in s.index if int(y)-k in pos], int)
        ic = np.array([i for i, y in enumerate(s.index) if int(y)-k in pos], int)
        if ip.size <= 2: rho.append(np.nan); pv.append(np.nan); continue
        r = spm(v[ip], v[ic]); rho.append(r)
        perm = [abs(spm(p[ip], p[ic])) for p in (rng.permutation(v) for _ in range(nperm))]
        pv.append(float(np.mean(np.array(perm) >= abs(r))))
    return np.array(rho), np.array(pv)


def low(d, months, rs):
    s = d.where(d.index.month.isin(months)).rolling(7, center=True, min_periods=7).mean().resample(rs).min()
    s[s < 0.01] = np.nan; s.index = s.index.year
    return s[(s.index >= YR0) & (s.index <= YR1)]


def qdaily(run):
    run = Path(run)
    at = nc4.Dataset(run/"attributes.nc"); a = float(np.array(at["HRUarea"][:]).sum()); at.close()
    f = (glob.glob(str(run/"output"/"*.nc")) or glob.glob(str(run/"out"/"*.nc")))[0]
    ds = nc4.Dataset(f); t = ds["time"]
    tt = nc4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
    q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a; ds.close()
    return pd.Series(q, index=pd.DatetimeIndex(tt)).resample("1D").mean()


def tuo_fsca(run):
    run = Path(run)
    b = pd.read_csv(run/"bin_spec.csv"); w = b.set_index("bin").area_frac
    ds = xr.open_dataset(glob.glob(str(run/"output"/"*.nc"))[0])
    hid = ds.hruId.values.ravel(); t = pd.DatetimeIndex(ds.time.values)
    sw = pd.DataFrame({int(r.bin): np.asarray(
            ds["scalarSWE"].isel(hru=int(np.where(hid == r.hruId)[0][0])).values).ravel()/1000.0
        for r in b.itertuples()}, index=t)
    m = (t.month == 9) & (t.day == 30)
    out = {}
    for y in sorted(set(t[m].year)):
        v = sw[m & (t.year == y)]
        if len(v): out[y] = float(w.values[(v.iloc[-1].values) > SWE_TH].sum())
    return pd.Series(out), float((sw*w).sum(axis=1).iloc[-1]/(sw*w).sum(axis=1).iloc[0])


def main():
    # ---------- East ----------
    o = pd.read_csv(SC/"domain_East_River_distributed_elevAspect/observations/streamflow/raw_data/"
                    "EastRiver_dailyDischarge_WY1970_2025.csv",
                    parse_dates=["time"]).set_index("time")["discharge"]
    ob = low(o.groupby(o.index.normalize()).mean(), [9, 10, 11], "YE-NOV")
    r_ob, p_ob = acf(ob)
    print(f"EAST observed: rho1={r_ob[0]:+.2f}(p={p_ob[0]:.3f}) rho2={r_ob[1]:+.2f} rho3={r_ob[2]:+.2f}\n")
    rows = []
    for d in sorted(glob.glob(str(EAST_SW/"sf*"))):
        if not glob.glob(d+"/output/*.nc"): continue
        sf = float(re.search(r"sf(\d+)p(\d+)", Path(d).name).expand(r"\1.\2"))
        r, p = acf(low(qdaily(d), [9, 10, 11], "YE-NOV"))
        rows.append(dict(sf=sf, rho1=r[0], p1=p[0], rho2=r[1], p2=p[1], rho3=r[2],
                         mae=np.nanmean(np.abs(r-r_ob))))
    if rows:
        t = pd.DataFrame(rows).sort_values("sf")
        print("=== EAST sweep ===")
        print(t.round(3).to_string(index=False))
        print(f"\nsignificant rho1: {(t.p1 < 0.05).sum()}/{len(t)}")
        print(f"best shape (MAE): sf={t.loc[t.mae.idxmin(), 'sf']:.0f}")
        t.to_csv(EAST_SW/"scores.csv", index=False)

    # ---------- Tuolumne ----------
    to = pd.read_csv(SC/"domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/"
                     "preprocessed/Tuolumne_River_lumped_streamflow_processed.csv",
                     parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
    tob = low(to.groupby(to.index).mean(), [8, 9, 10], "YS-OCT")
    tr_ob, tp_ob = acf(tob)
    eos = pd.read_csv(D/"eos_final.csv", index_col="year").f_shadow
    print(f"\nTUOLUMNE observed: rho1={tr_ob[0]:+.2f} rho2={tr_ob[1]:+.2f} rho3={tr_ob[2]:+.2f}"
          f" | Landsat mean fSCA {eos.mean():.3f}")
    rows = []
    for d in sorted(glob.glob(str(TUO_SW/"m*"))):
        if not glob.glob(d+"/output/*.nc"): continue
        nm = Path(d).name
        f, growth = tuo_fsca(d)
        j = pd.DataFrame({"m": f, "o": eos.reindex(f.index)}).dropna()
        r, p = acf(low(qdaily(d), [8, 9, 10], "YS-OCT"))
        rows.append(dict(member=nm, fsca=j.m.mean(), obs=j.o.mean(), ratio=j.m.mean()/j.o.mean(),
                         r_fsca=pearsonr(j.m, j.o)[0], store_growth=growth,
                         rho1=r[0], rho2=r[1], rho3=r[2],
                         mae=np.nanmean(np.abs(r-tr_ob))))
    if rows:
        t = pd.DataFrame(rows).sort_values("member")
        print("\n=== TUOLUMNE sweep ===")
        print(t.round(3).to_string(index=False))
        print(f"\nclosest fSCA (ratio->1): {t.loc[(t.ratio-1).abs().idxmin(), 'member']}")
        print(f"best lag shape (MAE):    {t.loc[t.mae.idxmin(), 'member']}")
        t.to_csv(TUO_SW/"scores.csv", index=False)


if __name__ == "__main__":
    main()
