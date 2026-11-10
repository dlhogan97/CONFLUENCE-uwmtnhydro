"""
Snowmelt run "backwards" at the Dana Meadows and Tuolumne Meadows pillows (2026-10-27).

Step 1  energy residual. Each water year is run alone (Oct 1 start) for the flat-meadow HRUs that match the pillow sites (HRU 8 at 3040 m for Dana;
        HRUs 13 (2730 m) and 18 (2409 m) interpolated to 2591 m for Tuolumne Meadows). Parameters: the Dana-only melt best (uniform -2 C with its
        Dilley-O'Brien longwave increment). On days when BOTH model and pillow carry > 50 mm SWE and neither gains snow, the energy the model is
        missing is   R = L_f * (dSWE_model - dSWE_obs) / 86400   [W/m2]   (positive = the model melts too slowly and needs more energy).
        Rain and sublimation enter both sides almost equally and cancel to first order.
Step 2  attribution. R is compared with what the forcing and the snow surface know on that day: clear-sky index (SW / 95th-percentile SW for
        the day of year), air temperature, relative humidity, wind, model snow albedo, days since snowfall, and the CDEC station air temperature
        minus the forcing air temperature (DAN / TUM).
usage: python melt_energy_inversion.py
"""
import os, json, uuid, shutil, numpy as np, pandas as pd, xarray as xr, netCDF4 as nc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import run_calibration_melt2 as M2
from run_calibration import seed_vector

LF = 3.34e5; SN = "/scratch/dlhogan/ess-project-data/meteorological-data/SNOTEL/"
OUT = Path("/scratch/dlhogan/ess-project-data/calibration/diagnostics/melt_energy_inversion"); OUT.mkdir(parents=True, exist_ok=True)
HRUS = [8, 13, 18]; W_TUM = (2591.0 - 2409.0)/(2730.0 - 2409.0)
YEARS = {"Dana": [1996, 1998, 2000, 2001, 2024, 2014, 2015, 2016, 2017, 2020], "TUM": [1996, 1998, 2000, 2001, 2024, 2017, 2018, 2019, 2020]}
PARAMS = dict(M2.SEED)                                          # Dana-only best, tilt 0
OUTV = ["scalarSWE", "scalarSnowfall", "scalarRainfall", "scalarSnowSublimation", "scalarSnowAlbedo", "scalarGroundAbsorbedSolar", "scalarLWNetGround",
        "scalarSenHeatGround", "scalarLatHeatGround", "scalarGroundNetNrgFlux"]
M2.HRUS = HRUS


class Runner(M2.RunnerMelt2):
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n" + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in OUTV))


def run(y):
    r = Runner(work_root=f"/tmp/claude-1008/meltinv/{y}_{uuid.uuid4().hex[:6]}", **M2.CFG); x = seed_vector(r, dict(PARAMS)); w = Path(r.work_root)/"w"
    r.run_start, r.run_end = f"{y-1}-10-01", f"{y}-09-30"; params, proc = r.run_summa(x, w)
    d = nc.Dataset(w/"out"/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    out = {v: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float), index=t, columns=HRUS) for v in OUTV}; d.close()
    f = xr.open_mfdataset(sorted((w/"forcing").glob("*.nc")), combine="by_coords")
    for v in ("airtemp", "SWRadAtm", "LWRadAtm", "windspd", "spechum", "airpres"):
        out["F_"+v] = pd.DataFrame(f[v].values, index=pd.DatetimeIndex(f["time"].values), columns=HRUS).resample("D").mean()
    f.close(); shutil.rmtree(r.work_root, ignore_errors=True)
    return y, {k: v.loc[f"{y-1}-10-01":f"{y}-09-30"] for k, v in out.items()}


def site_series(o, site):
    if site == "Dana": return {k: v[8] for k, v in o.items()}
    return {k: W_TUM*v[13] + (1 - W_TUM)*v[18] for k, v in o.items()}


def station_temp(code):
    fn = {"Dana": "DAN_DANA_MEADOWS_cdec_obs.csv", "TUM": "TUM_TUOLUMNE_MEADOWS_cdec_obs.csv"}[code]
    d = pd.read_csv(SN + fn, usecols=["datetime", "AIR TEMP", "AIR TEMP_units"]); d.index = pd.to_datetime(d.datetime, utc=True).dt.tz_convert(None).dt.normalize()
    t = pd.to_numeric(d["AIR TEMP"], errors="coerce"); u = str(d["AIR TEMP_units"].dropna().iloc[0]).upper() if d["AIR TEMP_units"].notna().any() else "DEG F"
    t = (t - 32)*5/9 if "F" in u else t; t = t.where(t.between(-40, 40)); return t.groupby(t.index).mean()


def main():
    yrs = sorted(set(YEARS["Dana"]) | set(YEARS["TUM"]))
    cache = OUT/"runs.pkl"
    if cache.exists(): res = pd.read_pickle(cache)
    else:
        with ProcessPoolExecutor(min(16, len(yrs))) as ex: res = dict(ex.map(run, yrs))
        pd.to_pickle(res, cache)
    PIL = {k: pd.read_csv(v[1], index_col=0, parse_dates=True).value.asfreq("D") for k, v in M2.PIL.items()}
    rows = []
    for site in ("Dana", "TUM"):
        st_t = station_temp(site)
        for y in YEARS[site]:
            s = site_series(res[y], site); swe_m = s["scalarSWE"]; swe_o = PIL[site].reindex(swe_m.index).interpolate(limit=3)
            so = swe_o.rolling(3, center=True, min_periods=2).mean(); sm = swe_m.rolling(3, center=True, min_periods=2).mean()
            dm, do = sm.diff(), so.diff(); sf = s["scalarSnowfall"]*86400
            ok = (sm > 50) & (so > 50) & (dm <= 1) & (do <= 1) & (sf < 1) & (swe_m.index.month.isin([3, 4, 5, 6, 7]))
            Rr = LF*(dm - do)/86400.0                                        # dSWE in kg/m2/day
            sw = s["F_SWRadAtm"]; doy = sw.index.dayofyear
            dsn = (sf > 1).astype(int).groupby(((sf > 1).astype(int)).cumsum()).cumcount()
            Tk = s["F_airtemp"]; e = s["F_spechum"]*s["F_airpres"]/(0.622 + 0.378*s["F_spechum"]); esat = 611.2*np.exp(17.67*(Tk - 273.15)/(Tk - 29.65))
            df = pd.DataFrame(dict(site=site, wy=y, R=Rr, melt_obs=-do, melt_mod=-dm, sw=sw, lw=s["F_LWRadAtm"], tair=Tk - 273.15, rh=(e/esat).clip(0, 1.2),
                                   wind=s["F_windspd"], albedo=s["scalarSnowAlbedo"], days_since_snow=dsn, absSW=s["scalarGroundAbsorbedSolar"],
                                   LWnet=s["scalarLWNetGround"], H=s["scalarSenHeatGround"], LE=s["scalarLatHeatGround"],
                                   t_station_minus_forcing=st_t.reindex(sw.index) - (Tk - 273.15))).reindex(swe_m.index)[ok]
            rows.append(df)
    D = pd.concat(rows); doymax = D.groupby(D.index.dayofyear).sw.quantile(0.95)
    D["clear_sky_index"] = (D.sw/D.index.dayofyear.map(doymax).values).clip(0, 1.3)
    D.to_csv(OUT/"melt_energy_residual_daily.csv")
    pd.set_option("display.width", 250)
    print("STEP 1: missing melt energy R (W/m2; + = model needs more energy), melt-season days with snow on both")
    g = D.groupby(["site", "wy"]).agg(n_days=("R", "size"), R_mean=("R", "mean"), R_median=("R", "median"), melt_obs_mm_d=("melt_obs", "mean"), melt_mod_mm_d=("melt_mod", "mean"))
    print(g.round(1).to_string()); print(D.groupby("site").R.describe().round(1).to_string())
    print("\nSTEP 2: Spearman correlation of R with each predictor (per site)")
    preds = ["clear_sky_index", "sw", "lw", "tair", "rh", "wind", "albedo", "days_since_snow", "absSW", "LWnet", "H", "LE", "t_station_minus_forcing"]
    print(D.groupby("site").apply(lambda g: pd.Series({p: g[["R", p]].corr(method="spearman").iloc[0, 1] for p in preds})).round(2).T.to_string())
    print("\nR (W/m2) by clear-sky tercile and by air-temperature tercile")
    for p in ("clear_sky_index", "tair", "wind", "albedo"):
        D[p+"_bin"] = D.groupby("site")[p].transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"]))
        print(p, D.groupby(["site", p+"_bin"]).R.mean().unstack().round(1).to_string())
    print("\nstation minus forcing air temperature on these days (C):", D.groupby("site").t_station_minus_forcing.describe()[["count", "mean", "50%"]].round(2).to_dict())


if __name__ == "__main__":
    main()
