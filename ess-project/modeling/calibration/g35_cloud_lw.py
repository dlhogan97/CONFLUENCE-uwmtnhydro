"""
Cloud-based incoming longwave correction for the Tuolumne forcing (2026-11-06, set with the user).
Why: the energy-balance run (g35_suite5_energy.py) showed the snowpack is far too cold. The forcing sky emissivity (LWRadAtm / sigma Ta^4) is ~0.73 on dry days but
only 0.79-0.87 on precipitating days (overcast storm sky should be ~0.95-1.0). The forcing LW is roughly clear-sky Dilley-O'Brien + 27 W/m2 with a weak
cloud response. The user rejected a flat LW shift ("would nuke the snowpack"): the correction must add LW only when it is cloudy.
Method (Crawford & Duchon 1999 all-sky emissivity):
  clear-sky SW  per HRU and day of year = 95th percentile of the raw forcing's DAILY-mean SW within +-10 days, all years 1981-2025 (cached .npy)
  cloud c       = 1 - min(1, BASIN-MEAN daily SW / basin-mean clear-sky SW), one value per day for every HRU and hour (nights take their day's value);
                  hours with precipitation at that HRU get c = 1. Basin mean because per-HRU clearness on terrain-shaded north slopes is noisy
                  (HRU 23 dry days came out c = 0.25, +11 W/m2 at night).
  emissivity    eps = eps_DO + s*c*(1 - eps_DO), eps_DO = Dilley-O'Brien clear-sky LW / sigma Ta^4 from the STAGED (adjusted) T, p, q
  LW            = max(staged LWRadAtm, eps*sigma*Ta^4): it never removes LW, so clear nights keep the forcing value
s = strength (1 = full Crawford & Duchon). Clearness uses the RAW forcing SW (same file name), so swScale does not change c.
Only real files in the staged forcing dir are edited (out-of-period months are symlinks to the originals and are left alone).
"""
import numpy as np, xarray as xr
from pathlib import Path
from calibration_runner import _lw_dilley_obrien

SIG = 5.670374e-8
PPT_WET = 0.1/3600.0           # kg m-2 s-1 (0.1 mm/h): hour counts as overcast
CACHE = Path("/scratch/dlhogan/ess-project-data/calibration/diagnostics/full_basin_albedo/suite5_cal/cloud_lw_clearsw.npy")


def clear_sw(raw_dir):
    """(366, nHRU) clear-sky daily-mean SW from the raw forcing."""
    if CACHE.exists(): return np.load(CACHE)
    days = []
    for f in sorted(Path(raw_dir).glob("*.nc")):
        with xr.open_dataset(f) as d: days.append(d["SWRadAtm"].resample(time="D").mean().to_pandas())
    import pandas as pd
    D = pd.concat(days); doy = np.minimum(D.index.dayofyear.values, 366); out = np.zeros((366, D.shape[1]))
    for k in range(1, 367):
        dist = np.abs(doy - k); dist = np.minimum(dist, 366 - dist); m = dist <= 10
        out[k - 1] = np.nanpercentile(D.values[m], 95, axis=0)
    np.save(CACHE, out); return out


def new_lw(ds, raw_sw, clr, s):
    """Corrected LW (time, hru) for one staged month (xarray Dataset) given the raw-forcing SW (time, hru)."""
    t = ds["time"].to_index(); T = ds["airtemp"].values.astype(float); p = ds["airpres"].values.astype(float); q = ds["spechum"].values.astype(float)
    day = t.normalize(); u, inv = np.unique(day, return_inverse=True); u = __import__("pandas").DatetimeIndex(u)
    sw_day = np.vstack([raw_sw[inv == i].mean(0) for i in range(len(u))])                       # (ndays, hru)
    cs = clr[np.minimum(u.dayofyear.values, 366) - 1]
    c = 1.0 - np.clip(sw_day.mean(1)/np.maximum(cs.mean(1), 1.0), 0.0, 1.0)              # (ndays,) basin-mean clearness
    c = np.repeat(c[inv][:, None], T.shape[1], axis=1); c = np.where(ds["pptrate"].values >= PPT_WET, 1.0, c)
    eps_clr = _lw_dilley_obrien(T, p, q)/(SIG*T**4)
    eps = eps_clr + s*c*(1.0 - eps_clr)
    return np.maximum(ds["LWRadAtm"].values.astype(float), eps*SIG*T**4), c


def apply(fdir, raw_dir, s):
    clr = clear_sw(raw_dir); n = 0
    for f in sorted(Path(fdir).glob("*.nc")):
        if f.is_symlink(): continue
        with xr.open_dataset(raw_dir/f.name) as r: raw_sw = r["SWRadAtm"].values.astype(float)
        with xr.open_dataset(f) as d: ds = d.load()
        lw, _ = new_lw(ds, raw_sw, clr, s); ds["LWRadAtm"].values[:] = lw.astype(ds["LWRadAtm"].dtype)
        tmp = f.with_suffix(".tmp.nc"); ds.to_netcdf(tmp); ds.close(); tmp.replace(f); n += 1
    return n
