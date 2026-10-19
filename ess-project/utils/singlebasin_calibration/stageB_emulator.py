"""Stage-B test bed: does fitting only the aquifer+routing (emulator, East) recover the low-flow / recession / monthly / bias features equally well from DIFFERENT stage-A (snow/energy) states?
A 'state' = the hourly net recharge, surface runoff, HRU areas and initial storage taken from one SUMMA run (old best, v2 round-1 best, v2 round-2 best). Window WY2009-2020 (spin-up from 2005-10-01).
Stage-B objective = mean of five normalised feature errors: fall low flow, winter low flow, receding limb, monthly-volume error, overall bias."""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, numba as nb, netCDF4 as nc
from pathlib import Path
from scipy.stats import gamma
from scipy.signal import fftconvolve
import emu_core as E
from emu_core import run_buckets

L = "/scratch/dlhogan/ess-project-data/calibration/"
OUT = Path("/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/stageB"); OUT.mkdir(parents=True, exist_ok=True)
T0, T1, SCORE0 = "2005-10-01", "2020-09-30 23:00", "2008-10-01"
SRC = {"old": (L+"bestruns/East_River_baseline_1981_2025/run", "baseline_timestep.nc"),
       "r1": (L+"runs/East_River_v2_20261014_185148/best_full/rank0", "best_timestep.nc"),
       "r2": (L+"runs/East_River_r2_20261015_110603/best_full/rank0", "best_timestep.nc")}
def build_state(name):
    fn = OUT/f"state_{name}.npz"
    if not fn.exists():
        d0, f = SRC[name]; d = nc.Dataset(f"{d0}/out/{f}"); t = d["time"]; tt = pd.DatetimeIndex(nc.num2date(t[:], t.units, only_use_cftime_datetimes=False))
        A = np.array(nc.Dataset(f"{d0}/attributes.nc")["HRUarea"][:]).astype(float).ravel()
        S = np.array(d["scalarAquiferStorage_mean"][:]).astype(float); Qb = np.array(d["scalarAquiferBaseflow"][:]).astype(float); SR = np.array(d["scalarSurfaceRunoff"][:]).astype(float)
        dS = np.zeros_like(S); dS[1:-1] = (S[2:]-S[:-2])/7200.0; dS[0] = dS[1]; dS[-1] = dS[-2]
        m = (tt >= T0) & (tt <= T1); i0 = int(np.argmax(m))
        np.savez(fn, S0=S[i0], Rch=(Qb+dS)[m], SR=SR[m], A=A, time=tt[m].values.astype("datetime64[h]"))
    z = np.load(fn); return dict(S0=z["S0"], Rch=z["Rch"], SR=z["SR"], A=z["A"], tt=pd.DatetimeIndex(z["time"]))
def emulate(st, p, passes=3):
    NT, NH = st["Rch"].shape; S = st["S0"].copy(); Qs = np.zeros((NT, NH)); Qd = np.zeros(NT); Sd = np.zeros(NT); sdi = 0.5*p["Sf_d"]; rate = p.get("rate", E.RATE0)
    for _ in range(passes): S, sdi = run_buckets(st["Rch"], st["A"], p["Sf_h"], rate, p["exp_h"], p["lam"], p["m_leak"], p["Sf_d"], p["exp_d"], rate, S, sdi, 3600.0, Qs, Qd, Sd)
    return Qs, Qd, Sd
def daily_Q(st, Qs, Qd, scale):
    w = st["A"]/st["A"].sum(); unr = ((st["SR"]+Qs)*w).sum(1)+Qd; routed = fftconvolve(unr, E.unit_hydrograph(scale))[:len(unr)]
    return pd.Series(routed*st["A"].sum(), index=st["tt"]).resample("1D").mean()
# ---------------- feature errors ----------------
def _wy(i): return i.year + (i.month >= 10)
def _roll7(q): return q.rolling(7, center=True, min_periods=7).mean()
def _per_wy(q, f):
    return pd.Series({y: f(g, y) for y, g in q.groupby(_wy(q.index)) if len(g) >= 360})
def _min7(g, months): return float(_roll7(g.where(g.index.month.isin(months))).min())
def _recess(g, y):
    r = _roll7(g).dropna(); t = r.idxmax(); post = r.loc[t:]; b = post[post <= 0.25*r.max()]; return float((b.index[0]-t).days) if len(b) else np.nan
OBSD = E.obs.loc[SCORE0:"2020-09-30"]
_OF = {"fall": _per_wy(OBSD, lambda g, y: _min7(g, [8, 9, 10])), "winter": _per_wy(OBSD, lambda g, y: _min7(g, [12, 1, 2, 3])), "rec": _per_wy(OBSD, _recess)}
_OM = OBSD.groupby(OBSD.index.month).mean()
def terms(q):
    s = q.loc[SCORE0:"2020-09-30"].clip(lower=0.01); o = OBSD.clip(lower=0.01); out = {}
    for k, mo in (("fall", [8, 9, 10]), ("winter", [12, 1, 2, 3])):
        sv = _per_wy(s, lambda g, y: _min7(g, mo)); c = sv.index.intersection(_OF[k].index)
        out[k] = float(np.clip(np.abs(np.log10(sv[c].clip(lower=0.01))-np.log10(_OF[k][c].clip(lower=0.01))).mean()/0.15, 0, 3))
    rv = _per_wy(s, _recess); c = rv.index.intersection(_OF["rec"].index); out["recession"] = float(np.clip((rv[c]-_OF["rec"][c]).abs().mean()/7.0, 0, 3))
    sm = s.groupby(s.index.month).mean(); out["monthly"] = float(np.clip((sm-_OM).abs().sum()/_OM.sum()/0.10, 0, 3)); out["bias"] = float(np.clip(abs(s.sum()/o.sum()-1)/0.05, 0, 3))
    return out
def J_of(q): t = terms(q); return float(np.mean(list(t.values()))), t
# ---------------- parameters ----------------
SPEC = {"M0": [("lSf_h", -0.5, 1.1), ("exp_h", 1.0, 8.0), ("lscale", 4.3, 5.2)],
        "M1": [("lSf_h", -0.5, 1.1), ("exp_h", 1.0, 8.0), ("llam", -2.3, 0.18), ("m_leak", 1.0, 6.0), ("lSf_d", -0.5, 1.9), ("exp_d", 1.0, 8.0), ("lscale", 4.3, 5.2)]}
def to_params(model, x):
    d = dict(zip([s[0] for s in SPEC[model]], x)); p = dict(Sf_h=10**d["lSf_h"], exp_h=d["exp_h"], scale=10**d["lscale"], lam=0.0, m_leak=1.0, Sf_d=1.0, exp_d=1.0)
    if model == "M1": p.update(lam=10**d["llam"], Sf_d=10**d["lSf_d"], m_leak=d["m_leak"], exp_d=d["exp_d"])
    return p
_ctx = {}
def cost(x, fixed=None):
    try:
        xx = np.array(x, float); st = _ctx["st"]; p = to_params(_ctx["model"], xx); Qs, Qd, Sd = emulate(st, p); J, _ = J_of(daily_Q(st, Qs, Qd, p["scale"])); return J if np.isfinite(J) else 10.0
    except Exception: return 10.0
