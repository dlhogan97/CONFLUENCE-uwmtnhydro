"""Offline emulator of East River aquifer structures, validated against SUMMA.

Recharge (net of aquifer transpiration), surface runoff and routing come from the calibrated baseline run
(calibration/bestruns/East_River_baseline_1981_2025). Local aquifers are independent power-law buckets
(identical to SUMMA bigAquifer). The single-basin extension adds a state-dependent leak from every local bucket
to one basin-wide store (SUMMA spatial_gw = singleBasin in /home/dlhogan/summa_deepstore).
The coupling is one-way, so with the upstream snow/soil parameters frozen this reproduces SUMMA to <0.4% (monthly flow).
"""
import numpy as np, pandas as pd, numba as nb, netCDF4 as nc
from pathlib import Path
from scipy.stats import gamma, spearmanr
from scipy.signal import fftconvolve

SC = Path("/scratch/dlhogan/ess-project-data")
BASE_RUN = SC/"calibration/bestruns/East_River_baseline_1981_2025/run"
OBS_CSV = SC/"streamflow-data/EastRiver/EastRiver_dailyDischarge_WY1970_2025.csv"
CACHE = Path(__file__).with_name("recharge_cache.npz")
RATE0, EXP0, SF0, SHAPE = 9.43986454e-06, 2.056814, 2.6577657921019333, 2.5

def prepare(force=False):
    """Extract hourly net recharge (= aquifer baseflow + dS/dt), surface runoff and HRU areas from the baseline run."""
    if CACHE.exists() and not force: return
    d = nc.Dataset(BASE_RUN/"out/baseline_timestep.nc"); t = d["time"]
    tt = pd.DatetimeIndex(nc.num2date(t[:], t.units, only_use_cftime_datetimes=False))
    A = np.array(nc.Dataset(BASE_RUN/"attributes.nc")["HRUarea"][:]).astype(float)
    S = np.array(d["scalarAquiferStorage_mean"][:]).astype(float); Qb = np.array(d["scalarAquiferBaseflow"][:]).astype(float)
    SR = np.array(d["scalarSurfaceRunoff"][:]).astype(float)
    dS = np.zeros_like(S); dS[1:-1] = (S[2:]-S[:-2])/7200.0; dS[0] = dS[1]; dS[-1] = dS[-2]
    np.savez(CACHE, S0=S[0], Rch=Qb+dS, SR=SR, A=A, time=tt.values.astype("datetime64[h]"))

prepare()
_Z = np.load(CACHE); Rch = _Z["Rch"]; SR = _Z["SR"]; A = _Z["A"]; S0 = _Z["S0"]; tt = pd.DatetimeIndex(_Z["time"])
NT, NH = Rch.shape; w = A/A.sum(); Atot = A.sum()

def load_obs():
    r = pd.read_csv(OBS_CSV, parse_dates=["time"]).set_index("time")["discharge"]
    return r.groupby(r.index.normalize()).mean()
obs = load_obs()

@nb.njit
def run_buckets(Rch, A, Sf, rate, ex, lamb, ml, Sfd, exd, rated, Sinit, Sdinit, dt, Qs, Qd, Sd):
    """HRU buckets (baseflow + leak outlets) feeding one basin store. lamb=0 -> independent buckets (baseline)."""
    NT, NH = Rch.shape; S = Sinit.copy(); sd = Sdinit; At = A.sum()
    for t in range(NT):
        leak = 0.0
        for i in range(NH):
            U = Rch[t, i]; kk = rate/Sf**ex; kl = lamb*rate/Sf**ml; s = S[i]; sn = s
            for it in range(30):
                f = sn-s-dt*U+dt*(kk*sn**ex+kl*sn**ml); fp = 1.0+dt*(kk*ex*sn**(ex-1.0)+kl*ml*sn**(ml-1.0)); step = f/fp; sn = sn-step
                if sn < 1e-12: sn = 1e-12
                if abs(step) < 1e-12*max(1.0, sn): break
            S[i] = sn; Qs[t, i] = kk*sn**ex; leak += kl*sn**ml*A[i]/At
        kd = rated/Sfd**exd; s = sd; sn = s
        for it in range(40):
            f = sn-s-dt*leak+dt*kd*sn**exd; fp = 1.0+dt*kd*exd*sn**(exd-1.0); step = f/fp; sn = sn-step
            if sn < 1e-12: sn = 1e-12
            if abs(step) < 1e-12*max(1.0, sn): break
        sd = sn; Qd[t] = kd*sn**exd; Sd[t] = sn
    return S, sd

def emulate(p, passes=3):
    """p: dict(Sf_h, exp_h, lam, m_leak, Sf_d, exp_d). Returns Qs (m/s per HRU), Qd (m/s basin), Sd (m), small-bucket end state, deep end state."""
    S = S0.copy(); Qs = np.zeros((NT, NH)); Qd = np.zeros(NT); Sd = np.zeros(NT); sdi = 0.5*p["Sf_d"]
    for _ in range(passes):
        S, sdi = run_buckets(Rch, A, p["Sf_h"], RATE0, p["exp_h"], p["lam"], p["m_leak"], p["Sf_d"], p["exp_d"], RATE0, S, sdi, 3600.0, Qs, Qd, Sd)
    return Qs, Qd, Sd, S, sdi

def unit_hydrograph(scale):
    cdf = gamma.cdf(np.arange(0, 1001)*3600.0, a=SHAPE, scale=scale); uh = np.diff(cdf); return uh/uh.sum()

def daily_Q(Qs, Qd, scale=51629.17):
    unr = ((SR+Qs)*w).sum(1)+Qd; routed = fftconvolve(unr, unit_hydrograph(scale))[:NT]
    return pd.Series(routed*Atot, index=tt).resample("1D").mean()

# ---------------- metrics (same definitions as the notebook analysis) ----------------
def kge(o, m):
    r = np.corrcoef(o, m)[0, 1]; return 1-np.sqrt((r-1)**2+(m.std()/o.std()-1)**2+(m.mean()/o.mean()-1)**2)
def nse(o, m): return 1-((m-o)**2).sum()/((o-o.mean())**2).sum()
def skill(q, per):
    j = pd.concat([obs, q], axis=1, join="inner").dropna().loc[per[0]:per[1]]; j.columns = ["o", "m"]
    eps = 0.01*j.o.mean(); lo = j[j.index.month.isin([8, 9, 10, 11])]
    return dict(KGE=kge(j.o, j.m), NSE=nse(j.o, j.m), logNSE=nse(np.log(j.o+eps), np.log(j.m+eps)),
                PB=100*(j.m.sum()-j.o.sum())/j.o.sum(), PB_AN=100*(lo.m.sum()-lo.o.sum())/lo.o.sum())
def series(q, y0, y1):
    mn, pk = {}, {}
    for y in range(y0, y1+1):
        a = q.loc[f"{y}-08-01":f"{y}-11-30"]
        if a.isna().mean() <= 0.1 and len(a) > 100:
            m = a.rolling(7, center=True, min_periods=7).mean().min()
            if np.isfinite(m) and m > 0.01: mn[y] = m
        wv = q.loc[f"{y}-04-01":f"{y}-07-31"]
        if wv.isna().mean() <= 0.05: pk[y] = wv.interpolate(limit=5).rolling(7, center=True, min_periods=7).mean().max()
    return np.log(pd.Series(mn)), np.log(pd.Series(pk))
def lag_r(s, l):
    d = pd.DataFrame({"x": s.reindex(s.index-l).values, "y": s.values}).dropna()
    return spearmanr(d.x, d.y)[0] if len(d) > 5 else np.nan
def resid(y, x):
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna(); b = np.polyfit(d.x, d.y, 1); return d.y-np.polyval(b, d.x)
def memory(q, y0, y1, Sdeep=None):
    lf, pk = series(q, y0, y1); r = resid(lf, pk)
    out = dict(n=len(lf), raw1=lag_r(lf, 1), res1=lag_r(r, 1), raw5=lag_r(lf, 5), res5=lag_r(r, 5))
    if Sdeep is not None:
        Sp = pd.Series({y: Sdeep.loc[f"{y-1}-07-25":f"{y-1}-08-05"].mean() for y in lf.index})
        d = pd.concat([lf, Sp, pk], axis=1).dropna(); d.columns = ["x", "y", "z"]
        rx = d.x-np.polyval(np.polyfit(d.z, d.x, 1), d.z); ry = d.y-np.polyval(np.polyfit(d.z, d.y, 1), d.z); out["Sprev_pk"] = spearmanr(rx, ry)[0]
    return out

WINDOWS = {"full": dict(skill=("1986-10-01", "2024-09-30"), mem=(1986, 2024)),
           "A": dict(skill=("1986-10-01", "2005-09-30"), mem=(1986, 2005)),
           "B": dict(skill=("2005-10-01", "2024-09-30"), mem=(2006, 2024))}
