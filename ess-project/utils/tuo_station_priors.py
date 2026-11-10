"""
Station-based priors for the Tuolumne forcing corrections used by run_calibration_snow.py (2026-10-21).

Compares CDEC stations inside the basin or within ~20 km of it with the BASE forcing (forcing/SUMMA_input, before any
calibration adjustment). The forcing is evaluated at each station's elevation by a daily linear fit on the 28 HRU elevations.

  temperature   daily mean T (sensor 30), WY2009-2020. Bias (station - forcing) per station and month is fitted as
                dT_m(z) = a_m + b*(z - Z_REF)/1000   (one elevation slope b, a monthly intercept a_m)
                -> the PRIOR of the structured correction  dT_i = a_m + da + (b + db)*(z_i - Z_REF)/1000 + c*northness_i,
                in which only da, db, c are calibrated.
  vapour press. hourly T (sensor 4) and RH (sensor 12), WY2013-2020. Ratio of daily-mean vapour pressure (station / forcing),
                fitted per month as  ln r_m(z) = alpha_m + beta*(z - Z_REF)/1000   (one elevation slope).
                Applied as a FIXED multiplier on spechum (capped at saturation); not calibrated.

Writes observations/tuolumne_forcing_priors.json (+ the per-station tables as csv).
usage: python tuo_station_priors.py
"""
from __future__ import annotations
import glob, io, json, re
from pathlib import Path
import numpy as np, pandas as pd, netCDF4 as nc, requests

FORC = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/forcing/SUMMA_input"
ATTR = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/attributes.nc"
OBS = Path(__file__).resolve().parents[1]/"modeling/calibration/observations"
# CDEC stations with daily mean T inside the basin or within ~20 km of it (located 2026-10-21; elevations from CDEC metadata)
STATIONS = {"DAN": ("Dana Meadows", 2975, True), "TUM": ("Tuolumne Meadows", 2591, True), "SLI": ("Slide Canyon", 2804, True),
            "TES": ("Tioga Pass Entrance", 3031, False), "TNY": ("Tenaya Lake", 2460, False), "WHW": ("White Wolf", 2408, False),
            "PDS": ("Paradise Meadow", 2332, False), "HRS": ("Horse Meadow", 2560, False), "GIN": ("Gin Flat", 2149, False)}
VP_STATIONS = ("DAN", "TUM", "SLI", "TES", "TNY", "WHW", "GIN")          # the ones with hourly RH
Z_REF = 2750.0      # m, reference elevation of both corrections (mid-station; basin area-weighted mean ~2900 m)


def cdec(sta, sensor, dur, t0, t1):
    url = f"https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={sta}&SensorNums={sensor}&dur_code={dur}&Start={t0}&End={t1}"
    d = pd.read_csv(io.StringIO(requests.get(url, timeout=180).text))
    v = pd.to_numeric(d["VALUE"], errors="coerce")
    v.index = pd.to_datetime(d["DATE TIME"], format="%Y%m%d %H%M")
    return v


def degf_to_c(v):
    c = (v.where((v > -40) & (v < 100)) - 32)*5/9
    return c.where((c - c.rolling(7, center=True, min_periods=3).median()).abs() < 15)    # spikes / stuck values


def esat_kpa(T):
    return 0.6112*np.exp(17.67*T/(T + 243.5))                                           # over water, T in C (as CDEC RH)


def forcing_daily(var, y0m0, y1m1):
    elev = np.array(nc.Dataset(ATTR)["elevation"][:], dtype=float); out = []
    for f in sorted(glob.glob(FORC+"/*.nc")):
        ym = re.search(r"(\d{6})\.nc$", f).group(1)
        if not (y0m0 <= ym <= y1m1): continue
        d = nc.Dataset(f); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
        if var == "T":
            x = np.array(d["airtemp"][:]) - 273.15
        else:
            q, p = np.array(d["spechum"][:]), np.array(d["airpres"][:]); x = q*p/(0.622 + 0.378*q)/1000.0     # kPa
        out.append(pd.DataFrame(x, index=t).resample("D").mean())
    X = pd.concat(out); A = np.vstack([np.ones_like(elev), elev]).T
    coef = np.linalg.lstsq(A, X.values.T, rcond=None)[0]
    return lambda z: pd.Series(coef[0] + coef[1]*z, index=X.index)


def fit_common_slope(df, ycol, wcol="n"):
    """y = a_m + b*(z - Z_REF)/1000 with one b for all months: weighted least squares on [month dummies, dz]."""
    m = df.month.values; dz = (df.elev_m.values - Z_REF)/1000; w = np.sqrt(df[wcol].values)
    X = np.column_stack([(m == k).astype(float) for k in range(1, 13)] + [dz])
    beta = np.linalg.lstsq(X*w[:, None], df[ycol].values*w, rcond=None)[0]
    res = df[ycol].values - X @ beta
    return beta[:12], beta[12], float(np.sqrt(np.average(res**2, weights=df[wcol].values)))


def main():
    # ---------------- temperature ----------------
    fT = forcing_daily("T", "200810", "202009"); rows = []
    for s, (name, z, inside) in STATIONS.items():
        o = degf_to_c(cdec(s, 30, "D", "2008-10-01", "2020-09-30")); o.index = o.index.normalize()
        j = pd.concat([o.rename("obs"), fT(z).rename("forc")], axis=1).dropna()
        for m, g in j.groupby(j.index.month):
            rows.append(dict(sta=s, name=name, elev_m=z, in_basin=inside, month=m, n=len(g), obs=g.obs.mean(), forc=g.forc.mean(), bias=(g.obs - g.forc).mean()))
    TB = pd.DataFrame(rows); TB.to_csv(OBS/"tuolumne_station_tbias_monthly.csv", index=False)
    a_m, b, sdT = fit_common_slope(TB, "bias")
    print("TEMPERATURE  station - base forcing (C), WY2009-2020")
    print(TB.pivot_table(index=["sta", "elev_m"], columns="month", values="bias").sort_index(level=1).round(1).to_string())
    print(f"  fit: b = {b:+.2f} C/km, a_m at {Z_REF:.0f} m = {np.round(a_m, 2).tolist()}  (weighted residual sd {sdT:.2f} C)")

    # ---------------- vapour pressure ----------------
    fE = forcing_daily("e", "201210", "202009"); rows = []
    for s in VP_STATIONS:
        name, z, inside = STATIONS[s]
        T = degf_to_c(cdec(s, 4, "H", "2012-10-01", "2020-09-30")); RH = cdec(s, 12, "H", "2012-10-01", "2020-09-30").where(lambda x: (x > 0) & (x <= 100))
        e = (RH/100*esat_kpa(T)).dropna().resample("D").mean()
        j = pd.concat([e.rename("obs"), fE(z).rename("forc")], axis=1).dropna()
        for m, g in j.groupby(j.index.month):
            rows.append(dict(sta=s, name=name, elev_m=z, month=m, n=len(g), lnratio=np.log(g.obs.mean()/g.forc.mean())))
    VB = pd.DataFrame(rows); VB.to_csv(OBS/"tuolumne_station_vp_ratio_monthly.csv", index=False)
    al_m, be, sdV = fit_common_slope(VB, "lnratio")
    print("\nVAPOUR PRESSURE  station / base forcing, WY2013-2020")
    print(np.exp(VB.pivot_table(index=["sta", "elev_m"], columns="month", values="lnratio")).sort_index(level=1).round(2).to_string())
    print(f"  fit: ln r = alpha_m + {be:+.3f}*(z-{Z_REF:.0f})/1000; ratio at {Z_REF:.0f} m by month = {np.round(np.exp(al_m), 2).tolist()}  (resid sd {sdV:.2f} in ln)")

    pri = dict(z_ref=Z_REF, temp=dict(a_month=list(map(float, a_m)), b_per_km=float(b), resid_sd=sdT, stations=list(STATIONS), period="WY2009-2020"),
               vp=dict(ln_alpha_month=list(map(float, al_m)), beta_per_km=float(be), resid_sd_ln=sdV, stations=list(VP_STATIONS), period="WY2013-2020"),
               base_forcing=FORC, note="station - forcing; see ess-project/utils/tuo_station_priors.py")
    (OBS/"tuolumne_forcing_priors.json").write_text(json.dumps(pri, indent=1)); print("\nwrote", OBS/"tuolumne_forcing_priors.json")


if __name__ == "__main__":
    main()
