"""
Daily snow emulator for the 28 Tuolumne HRUs (2026-11-03). Purpose: search melt timing in seconds instead of SUMMA's ~50 min per generation, then carry the
answer back to SUMMA knobs. Driven by the forcing SUMMA actually saw (g35_suite3_wb.py output), with SUMMA's own rain/snow split, so only melt is emulated.

Per HRU, per day (v2, simplified energy balance; v1 temperature-index could not mimic SUMMA's alpine pack: SWE nMAE 0.18, 3682 m band 0.17 left vs 0.79):
  albedo   a = a_min(z) + (a_max - a_min(z)) exp(-age/tau_a), a_min(z) = a_min + a_min_z (z - 2700 m)/1000 m, age reset by >= 3 mm snowfall
  energy   Q = tc (1 - a) SW + lw_f (LW - 315.6) + kh T + dq_shrub  [W/m2]  (315.6 = sigma 273.15^4, a melting surface; tc = canopy SW transmission, forest only;
           dq_shrub on open-shrub HRUs (veg 7) only: in SUMMA they melt out ~30 d before barren HRUs at the same elevation)
  cold     cold content CC (mm melt-equivalent) grows by cc_k |T| on days below 0 C (capped at CC_CAP * SWE) and must be paid before melt
  melt     min(max(Q, 0) * 86400 / L_f - CC paid, SWE);  sublimation: SUMMA's per-HRU monthly rate on snow days, applied while the emulator has snow
  SWI      rain + melt  (rain passes through the pack)
Melt-order target (user, 2026-11-03: elevation-dependent, no Landsat): ASO fraction of early (Mar-Apr) SWE left on late (May 20-Jul) flights, per
ELEVATION BAND (area-weighted), bands with >= 100 mm early SWE (2414-3682 m), WY2013-2023 years with both flight windows.
"""
import numpy as np, pandas as pd, netCDF4 as nc, xarray as xr
from numba import njit
from pathlib import Path

ATTR = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/attributes.nc"
OBSD = Path(__file__).parent/"observations"
BAND_EDGES = [1900, 2250, 2600, 2900, 3200, 3550]
FOREST, SHRUB = 1, 7                                          # MODIFIED_IGBP evergreen needleleaf, open shrubland
PARAMS = [("a_max", 0.75, 0.95, False), ("a_min", 0.40, 0.80, False), ("a_min_z", -0.1, 0.25, False), ("tau_a", 1.0, 60.0, True),
          ("tc", 0.1, 1.0, False), ("lw_f", 0.3, 1.5, False), ("kh", 0.0, 15.0, False), ("cc_k", 0.0, 3.0, False), ("dq_shrub", -20.0, 40.0, False)]
CC_CAP = 0.1


def load(path):
    """Daily per-HRU forcing and SUMMA water fluxes (mm/d, C, W/m2) from a g35_suite3_wb.py cal_day.nc."""
    d = nc.Dataset(path); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    ids = np.array(d["hruId"][:]).astype(int)
    V = lambda v, f=1.0: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float)*f, index=t, columns=ids)
    X = dict(T=V("airtemp") - 273.15, SW=V("SWRadAtm"), LW=V("LWRadAtm"), snow=V("scalarSnowfall", 86400), rain=V("scalarRainfall", 86400),
             subl=V("scalarSnowSublimation", 86400), swe=V("scalarSWE"), swi=V("scalarRainPlusMelt", 86400*1000), et=V("scalarTotalET", -86400),
             runoff=V("scalarTotalRunoff", 86400*1000), q_routed=pd.Series(np.array(d["averageRoutedRunoff_mean"][:, 0]), index=t)*86400*1000,
             csubl=V("scalarCanopySublimation", 86400), cwat=V("scalarCanopyWat"))
    d.close()
    # water reaching the soil from SUMMA's mass balance; scalarRainPlusMelt does not close (rain on snow-free days is mostly missing from it)
    # SWE is a daily MEAN, so its day-to-day change is half a day out of step with the fluxes: negative days are carried forward (mass-conserving)
    raw = (X["rain"] + X["snow"] + X["subl"] + X["csubl"] - X["swe"].diff().fillna(0.0) - X["cwat"].diff().fillna(0.0)).values
    X["swi_mb"] = pd.DataFrame(_carry(raw), index=t, columns=ids)
    return X


@njit(cache=True)
def _carry(x):
    out = np.zeros_like(x)
    for h in range(x.shape[1]):
        debt = 0.0
        for i in range(x.shape[0]):
            v = x[i, h] + debt
            if v < 0.0: debt = v
            else: out[i, h] = v; debt = 0.0
    return out


def hrus():
    a = xr.open_dataset(ATTR)
    H = pd.DataFrame({"elev": a.elevation.values, "area": a.HRUarea.values, "veg": a.vegTypeIndex.values}, index=a.hruId.values.astype(int)); a.close()
    H["band"] = np.digitize(H.elev, BAND_EDGES); H["w"] = H.area/H.area.sum(); return H


def subl_rate(X):
    """SUMMA snow + canopy sublimation (mm/d, >= 0) per HRU and calendar month, on days SUMMA had > 10 mm SWE."""
    s = (-(X["subl"] + X["csubl"])).clip(lower=0).where(X["swe"] > 10)
    return s.groupby(s.index.month).mean().reindex(range(1, 13)).fillna(0.0)


@njit(cache=True)
def _run(T, SW, LW, snow, rain, subl, z, forest, shrub, a_max, a_min, a_min_z, tau_a, tc, lw_f, kh, cc_k, dq_shrub, swe_out, swi_out):
    nt, nh = T.shape
    for h in range(nh):
        swe = 0.0; cc = 0.0; age = 30.0
        amin = min(max(a_min + a_min_z*(z[h] - 2700.0)/1000.0, 0.3), a_max); trans = tc if forest[h] else 1.0; dq = dq_shrub if shrub[h] else 0.0
        for i in range(nt):
            swe += snow[i, h]
            age = 0.0 if snow[i, h] >= 3.0 else age + 1.0
            if swe > 0.0:
                a = amin + (a_max - amin)*np.exp(-age/tau_a)
                if T[i, h] < 0.0: cc = min(cc + cc_k*(-T[i, h]), CC_CAP*swe)
                q = trans*(1.0 - a)*SW[i, h] + lw_f*(LW[i, h] - 315.6) + kh*T[i, h] + dq
                pot = max(q, 0.0)*86400.0/3.34e5
                pay = min(cc, pot); cc -= pay; pot -= pay
                m = min(pot, swe); swe -= m
                sb = min(subl[i, h], swe); swe -= sb
                swi_out[i, h] = rain[i, h] + m
            else:
                cc = 0.0; swi_out[i, h] = rain[i, h]
            swe_out[i, h] = swe


def simulate(p, X, H, srate):
    T, SW, LW, sn, rn = (X[k].values for k in ("T", "SW", "LW", "snow", "rain"))
    sub = srate.loc[X["T"].index.month].values
    swe, swi = np.zeros_like(T), np.zeros_like(T)
    _run(T, SW, LW, sn, rn.clip(min=0), sub, H.elev.values, (H.veg.values == FOREST), (H.veg.values == SHRUB), *(float(p[k]) for k, *_ in PARAMS), swe, swi)
    return pd.DataFrame(swe, index=X["T"].index, columns=H.index), pd.DataFrame(swi, index=X["T"].index, columns=H.index)


def params_of(u):
    return {k: float(10**(np.log10(lo) + v*(np.log10(hi) - np.log10(lo))) if lg else lo + v*(hi - lo)) for (k, lo, hi, lg), v in zip(PARAMS, u)}


def aso_table(H, t0=None, t1=None):
    a = pd.read_csv(OBSD/"tuolumne_aso_hru_swe.csv", parse_dates=["date"]); a = a[a.covf >= 0.8]
    if t0 is not None: a = a[(a.date >= t0) & (a.date <= t1)]
    return a[["date", "hru", "aso_mm"]]


def band_melt_order(swe, aso, H, min_early=100.0):
    """Fraction of early SWE left on late flights per elevation band, ASO and model, years with both windows. Returns DataFrame band x (aso, mdl)."""
    a = aso[aso.date.isin(swe.index)].copy(); a["mdl"] = [swe.at[d, h] for d, h in zip(a.date, a.hru)]; a["band"] = a.hru.map(H.band); a["w"] = a.hru.map(H.w)
    b = a.groupby(["date", "band"]).apply(lambda g: pd.Series({"aso": np.sum(g.w*g.aso_mm)/g.w.sum(), "mdl": np.sum(g.w*g.mdl)/g.w.sum()})).reset_index()
    m = b.date.dt.month; early = b[m.isin([3, 4])]; late = b[((m == 5) & (b.date.dt.day >= 20)) | m.isin([6, 7])]
    yrs = sorted(set(early.date.dt.year) & set(late.date.dt.year))
    e = early[early.date.dt.year.isin(yrs)].groupby("band")[["aso", "mdl"]].mean(); l = late[late.date.dt.year.isin(yrs)].groupby("band")[["aso", "mdl"]].mean()
    ok = e.index[e.aso >= min_early]
    out = pd.DataFrame({"elev": H.groupby("band").elev.mean()[ok], "aso": (l.aso/e.aso)[ok], "mdl": (l.mdl/e.mdl.clip(lower=1.0))[ok].clip(upper=1.5)})
    out.attrs["years"] = yrs; return out


def aso_amount(swe, aso, H):
    """Area-weighted nMAE of HRU SWE over all ASO flights (same definition as run_calibration_snow.aso_objective)."""
    a = aso[aso.date.isin(swe.index)].copy(); a["mdl"] = [swe.at[d, h] for d, h in zip(a.date, a.hru)]; a["w"] = a.hru.map(H.w)
    return float(a.groupby("date").apply(lambda g: np.sum(g.w*(g.mdl - g.aso_mm).abs())/max(np.sum(g.w*g.aso_mm), 1.0)).mean())


class AsoScorer:
    """Precomputed-index version of aso_amount + band_melt_order for fitting loops (same definitions; checked equal in tuo_emulator_fit)."""
    def __init__(self, index, aso, H, min_early=100.0):
        a = aso[aso.date.isin(index)].reset_index(drop=True); col = {h: j for j, h in enumerate(H.index)}
        self.ti, self.hi = index.get_indexer(a.date), a.hru.map(col).values; self.obs = a.aso_mm.values; self.w = a.hru.map(H.w).values
        self.dates = a.date.factorize()[0]; self.nd = self.dates.max() + 1
        self.den = np.maximum(np.bincount(self.dates, self.w*self.obs), 1.0)
        b = a.hru.map(H.band).values; key = pd.Series(list(zip(a.date, b))); self.g, gk = key.factorize(); gk = pd.DataFrame(list(gk), columns=["date", "band"])
        self.gw = np.bincount(self.g, self.w); self.gobs = np.bincount(self.g, self.w*self.obs)/self.gw
        m = gk.date.dt.month; early = m.isin([3, 4]).values; late = (((m == 5) & (gk.date.dt.day >= 20)) | m.isin([6, 7])).values
        yrs = sorted(set(gk.date[early].dt.year) & set(gk.date[late].dt.year)); iy = gk.date.dt.year.isin(yrs).values
        self.bands = sorted(gk.band.unique()); self.E, self.L = {}, {}
        for bd in self.bands:
            self.E[bd] = np.where(early & iy & (gk.band == bd).values)[0]; self.L[bd] = np.where(late & iy & (gk.band == bd).values)[0]
        self.ok = [bd for bd in self.bands if len(self.E[bd]) and self.gobs[self.E[bd]].mean() >= min_early and len(self.L[bd])]
        self.fo = np.array([self.gobs[self.L[bd]].mean()/self.gobs[self.E[bd]].mean() for bd in self.ok]); self.elev = H.groupby("band").elev.mean()[self.ok].values

    def __call__(self, swe):
        m = swe[self.ti, self.hi]
        nmae = float(np.mean(np.bincount(self.dates, self.w*np.abs(m - self.obs), self.nd)/self.den))
        gm = np.bincount(self.g, self.w*m)/self.gw
        fm = np.array([min(gm[self.L[bd]].mean()/max(gm[self.E[bd]].mean(), 1.0), 1.5) for bd in self.ok])
        return nmae, float(np.abs(fm - self.fo).mean()), fm
