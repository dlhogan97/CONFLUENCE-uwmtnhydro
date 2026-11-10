"""
East River in-between deep store: two free parameters (tau, n), everything else tied to observations (2026-10-23).

Control = routing/timing control (buckets + routing frozen at run_calibration_east_routing best).
Treatment = leak fraction f of every bucket's outflow -> one basin store with outflow  Q = (S/tau) * (S/S_ref)^(n-1),
S_ref = f*b0*tau (the storage at which outflow equals the mean inflow f*b0), so tau is the residence time AT MEAN STORAGE and n sets the
nonlinearity (n = 1 linear reservoir). Mapped to SUMMA: basin__aquiferBaseflowExp = n, ScaleFactor = S_ref, HydCond = f*b0 (outflow at S_ref).
f is not free: for every (tau, n) it is the value that makes mean Dec-Feb flow equal the observed one (WY1987-2024), as in east_deepstore_build.py.
The (tau, n) grid is mapped on the emulator (validated against SUMMA). Memory is Spearman lag-1 of the residual of ln low flow on ln Apr-Jul peak:
  fall = Aug1-Nov30 7-d min (emu_core.memory), winter = Dec-Mar 30-d min, fullWY = 7-d min over the whole water year.
Split-sample: pick (tau, n) on WY1987-2005 and check it on WY2006-2024.
usage: python east_store_surface.py   (multiprocessing, ~20 workers)
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
sys.path.insert(0, "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration")
import numpy as np, pandas as pd, netCDF4 as nc
from multiprocessing import Pool
from scipy.optimize import brentq
import stageB_emulator as B, emu_core as E, signature_objective_plus as P
from run_calibration_v2 import best_trial

OUT = B.OUT; YR = 365.25*86400.0; RUNS = "/scratch/dlhogan/ess-project-data/calibration/runs/"
bt = best_trial(RUNS+"East_River_routing_20261022_153020")[0]["params"]
RATE, SF, EX, SCALE = bt["aquiferBaseflowRate"], bt["aquiferScaleFactor"], bt["aquiferBaseflowExp"], bt["routingGammaScale"]
st = B.build_state("routfull"); W = st["A"]/st["A"].sum(); obs = E.obs; NT, NH = st["Rch"].shape
Atot = float(np.array(nc.Dataset(f"{B.SRC['routfull'][0]}/attributes.nc")["HRUarea"][:]).sum())
A_, B_ = "1986-10-01", "2024-09-30"
WIN = {"A": (1987, 2005), "B": (2006, 2024), "full": (1987, 2024), "recent": (2009, 2024)}
TAUS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]; NS = [1.0, 1.5, 2.0, 2.5, 3.0]


ML = float(os.environ.get("LEAK_EXP", EX)); TAG = os.environ.get("TAG", "")      # leak exponent (default = bucket exponent, the first build); output-file tag


def _run(lam, tau_yr, n, ibar, passes):
    Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT); S = st["S0"].copy(); ts = tau_yr*YR
    if lam == 0: sfd, rated, sdi = 1.0, 1.0/ts, 0.0
    else: sfd, rated = ibar*ts, ibar; sdi = sfd
    for _ in range(passes):
        S, sdi = E.run_buckets(st["Rch"], st["A"], SF, RATE, EX, float(lam), ML, float(sfd), float(n), float(rated), S, float(sdi), 3600.0, Qs, Qd, Sd)
    return Qs, Qd, Sd


def simulate(lam, tau_yr, n, b0, passes=4):
    """S_ref = Ibar*tau where Ibar = mean leak inflow to the deep store (= lam*b0 when the leak exponent equals the bucket exponent; recomputed from the bucket state otherwise)."""
    out = _run(lam, tau_yr, n, lam*b0, passes)
    if lam > 0 and abs(ML - EX) > 1e-9:
        kk = RATE/SF**EX; leak = (lam*RATE/SF**ML*(out[0]/kk)**(ML/EX)*W).sum(1)
        out = _run(lam, tau_yr, n, float(leak.mean()), passes)
    return out


def flow(Qs, Qd): return B.daily_Q(st, Qs, Qd, SCALE)
def mo(q, m): return q[A_:B_][q[A_:B_].index.month.isin(m)].mean()
OBS_DJF = mo(obs, [12, 1, 2])


def winter_series(q, y0, y1):
    out = {}
    for y in range(y0, y1+1):
        w = q[f"{y-1}-12-01":f"{y}-03-31"].rolling(30, center=True, min_periods=20).mean().dropna()
        if len(w) > 60: out[y] = np.log(max(w.min(), 0.01))
    return pd.Series(out)


def fullwy_series(q, y0, y1):
    lf = {}
    for y in range(y0, y1+1):
        a = q.loc[f"{y-1}-10-01":f"{y}-09-30"]
        if a.isna().mean() <= 0.05 and len(a) > 350:
            m = a.rolling(7, center=True, min_periods=7).mean().min()
            if np.isfinite(m) and m > 0.01: lf[y] = m
    return np.log(pd.Series(lf))


def lags(q, y0, y1):
    lf, pk = E.series(q, y0, y1); fall = E.memory(q, y0, y1)["res1"]
    ws = winter_series(q, y0, y1); win = E.lag_r(E.resid(ws, pk.reindex(ws.index)), 1)
    fw = fullwy_series(q, y0, y1); full = E.lag_r(E.resid(fw, pk.reindex(fw.index)), 1)
    return fall, win, full


def metrics(q):
    out = {}
    for w, (y0, y1) in WIN.items():
        f, wi, fu = lags(q, y0, y1); out[f"fall_{w}"], out[f"win_{w}"], out[f"fullwy_{w}"] = f, wi, fu
    return out


OBS_M = metrics(obs)
b0 = None


def point(args):
    tau, n = args; global b0
    g = lambda lam: mo(flow(*simulate(lam, tau, n, b0)[:2]), [12, 1, 2]) - OBS_DJF
    lo, hi = 0.0, 0.25
    while g(hi) < 0 and hi < 16: lo, hi = hi, hi*2
    f = float(brentq(g, lo, hi, xtol=2e-3)) if g(hi) >= 0 else np.nan
    if not np.isfinite(f): return dict(tau=tau, n=n, f=np.nan)
    Qs, Qd, Sd = simulate(f, tau, n, b0); q = flow(Qs, Qd)
    unr = ((st["SR"]+Qs)*W).sum(1)+Qd; hr = pd.DataFrame(dict(Sd=Sd, Qd=Qd, unr=unr), index=st["tt"])[A_:B_]; ann = hr.Sd.groupby(hr.index.year + (hr.index.month >= 10)).mean()
    r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], obs["2008-10-01":"2020-09-30"])
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1)["2008-10-01":"2024-09-30"].dropna()
    return dict(tau=tau, n=n, f=f, S_ref=np.nan, obj=r.objective, KGE=r.diagnostics["KGE"], logKGE=E.kge(np.log(j.o.clip(lower=.05)), np.log(j.s.clip(lower=.05))),
                deep_mean_m=float(hr.Sd.mean()), deep_range_m=float(ann.max()-ann.min()), deep_share_pct=float(100*hr.Qd.mean()/hr.unr.mean()), **metrics(q))


if __name__ == "__main__":
    Q0s, Q0d, _ = simulate(0.0, 1.0, 1.0, 0.0); b0 = float((Q0s*W).sum(1).mean()); q0 = flow(Q0s, Q0d)
    r0 = P.signature_objective_routing(q0["2008-10-01":"2020-09-30"], obs["2008-10-01":"2020-09-30"]); ctl = dict(tau=np.nan, n=np.nan, f=0.0, obj=r0.objective, KGE=r0.diagnostics["KGE"], **metrics(q0))
    print("observed:", {k: round(v, 2) for k, v in OBS_M.items()}, flush=True); print("control:", {k: round(v, 2) for k, v in ctl.items() if k.startswith(("fall", "win", "full"))}, flush=True)
    grid = [(t, n) for t in TAUS for n in NS]
    with Pool(int(os.environ.get('NPROC', 20))) as p: rows = p.map(point, grid, chunksize=1)
    T = pd.DataFrame(rows); T.to_csv(OUT/f"store_surface_tau_n{TAG}.csv", index=False)
    json.dump(dict(observed=OBS_M, control=ctl, b0=b0, bucket=dict(rate=RATE, Sf=SF, exp=EX, scale=SCALE)), open(OUT/f"store_surface_tau_n{TAG}.json", "w"), indent=1, default=float)
    print("DONE", len(T), flush=True)
