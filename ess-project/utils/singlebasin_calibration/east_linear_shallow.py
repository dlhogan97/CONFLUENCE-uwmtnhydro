"""
East River option 2 (2026-10-27): LINEAR shallow bucket with tau fixed at the observed master-recession timescale (27 d, mrc_analysis on the
observed record), plus the deep store; nothing fitted to streamflow skill or memory.
  shallow bucket (per HRU):  Q_i = S_i/tau_s,  leak_i = f*S_i/tau_s  (linear leak, a fixed fraction f of the bucket outflow)
  deep store (basin):        Q_d = (S_d/tau_d)*(S_d/S_ref)^(n-1),   S_ref = mean leak inflow * tau_d
  f from the Dec-Feb volume balance (WY1987-2024), per (tau_d, n). Recharge and surface runoff = routing-control SUMMA state; routing unchanged.
Reports: routing objective + KGE/logKGE, MRC tau and Brutsaert-Nieber b of the simulated flow, low-flow memory at lags 1-5
(ln Aug1-Nov30 7-d min; raw, residual on same-year Apr-Jul peak, residual on same-year total flow), prior deep storage -> low flow, deep share at the fall minimum.
usage: python east_linear_shallow.py
"""
import os, sys, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
from scipy.signal import fftconvolve
from multiprocessing import Pool
import east_store_surface as X
sys.path.insert(0, "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/mrc_analysis")
from mrc.config import load_config
from mrc.cli import run_pipeline

E, B, st, W, P = X.E, X.B, X.st, X.W, X.P
TAU_S_D = 27.0; YR = X.YR; NT, NH = st["Rch"].shape; A = st["A"].sum(); UH = E.unit_hydrograph(X.SCALE)
OUT = X.OUT; obs = E.obs; MRC_CFG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/mrc_analysis/config/east_river.yaml"


def simulate(lam, tau_yr, n, passes=4):
    rate = 1.0/(TAU_S_D*86400.0)                     # Sf = 1 m, exponent 1 -> Q = S/tau_s
    def run(ibar):
        Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT); S = np.full(NH, 0.05); ts = tau_yr*YR
        if lam == 0: sfd, rated, sdi = 1.0, 1.0/ts, 0.0
        else: sfd, rated = ibar*ts, ibar; sdi = sfd
        for _ in range(passes): S, sdi = E.run_buckets(st["Rch"], st["A"], 1.0, rate, 1.0, float(lam), 1.0, float(sfd), float(n), float(rated), S, float(sdi), 3600.0, Qs, Qd, Sd)
        return Qs, Qd, Sd
    out = run(1e-9)
    if lam > 0: out = run(float((lam*out[0]*W).sum(1).mean()))       # mean leak inflow = f * mean bucket outflow
    return out


def routed(unr): return pd.Series(fftconvolve(unr, UH)[:len(unr)]*A, index=st["tt"]).resample("1D").mean()
def flows(Qs, Qd): qb = routed(((st["SR"] + Qs)*W).sum(1)); qd = routed(Qd); return qb + qd, qb, qd
OBS_DJF = X.OBS_DJF


def f_balance(tau, n):
    g = lambda lam: X.mo(flows(*simulate(lam, tau, n)[:2])[0], [12, 1, 2]) - OBS_DJF
    if g(0.0) >= 0: return 0.0
    lo, hi = 0.0, 0.25
    while g(hi) < 0 and hi < 16: lo, hi = hi, hi*2
    return float(brentq(g, lo, hi, xtol=2e-3)) if g(hi) >= 0 else np.nan


def mrc(q, tag):
    fn = f"/tmp/claude-1008/mrc_east/lin_{tag}.csv"; q = q["1983-10-01":"2024-09-30"]
    pd.DataFrame({"time": q.index.strftime("%Y-%m-%d"), "station_id": "x", "discharge": q.values}).to_csv(fn, index=False)
    cfg = load_config(MRC_CFG); cfg.io.streamflow.path = fn; cfg.output.write_tables = False; cfg.output.write_figures = False
    r = run_pipeline(cfg, outdir=__import__("pathlib").Path("/tmp/claude-1008/mrc_east")); return r["full_result"].tau, (r["bn"].overall.b if r["bn"] else np.nan)


def sp(a, b):
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna(); return spearmanr(d.a, d.b)[0] if len(d) > 5 else np.nan


def lagstats(q, S, y0, y1):
    lf, pk = E.series(q, y0-5, y1); tot = pd.Series({y: np.log(q[f"{y-1}-10-01":f"{y}-09-30"].mean()) for y in range(y0-5, y1+1)})
    ys = [y for y in lf.index if y0 <= y <= y1]; out = {}
    for nm, rr in (("raw", lf), ("rpk", E.resid(lf, pk.reindex(lf.index))), ("rtot", E.resid(lf, tot.reindex(lf.index)))):
        for k in (1, 2, 5):
            prev = rr.copy(); prev.index = prev.index + k; out[f"LF_{nm}_k{k}"] = sp(rr[ys], prev)
        if S is not None: out[f"S_{nm}_k1"] = sp(rr[ys], pd.Series({y: S.get(pd.Timestamp(f"{y-1}-09-30"), np.nan) for y in ys}))
    return out


def deep_share(qt, qd, y0, y1):
    sh = []
    for y in range(y0, y1+1):
        a = qt.loc[f"{y}-08-01":f"{y}-11-30"]; d = a.rolling(7, center=True, min_periods=7).mean().idxmin(); sl = slice(d - pd.Timedelta(days=3), d + pd.Timedelta(days=3))
        sh.append(qd[sl].mean()/qt[sl].mean())
    return float(np.mean(sh))


def point(args):
    tau, n = args
    f = 0.0 if tau is None else f_balance(tau, n)
    Qs, Qd, Sd = simulate(f, tau or 1.0, n or 1.0); qt, qb, qd = flows(Qs, Qd)
    r = P.signature_objective_routing(qt["2008-10-01":"2020-09-30"], obs["2008-10-01":"2020-09-30"])
    j = pd.concat([qt.rename("s"), obs.rename("o")], axis=1)["2008-10-01":"2024-09-30"].dropna()
    S = pd.Series(Sd, index=st["tt"]).resample("D").mean() if f > 0 else None
    tm, b = mrc(qt, f"{tau}_{n}")
    row = dict(model="linear shallow, no deep store" if tau is None else f"linear shallow + deep tau={tau:g}yr n={n:g}", f=f, obj=r.objective, KGE=r.diagnostics["KGE"],
               logKGE=E.kge(np.log(j.o.clip(lower=.05)), np.log(j.s.clip(lower=.05))), MRC_tau=tm, BN_b=b, DJF_ratio=X.mo(qt, [12, 1, 2])/OBS_DJF,
               fallmin_bias_pct=100*(np.exp(E.series(qt, 1987, 2024)[0]).mean()/np.exp(E.series(obs, 1987, 2024)[0]).mean() - 1),
               deep_share_fallmin=deep_share(qt, qd, 1987, 2024) if f > 0 else 0.0, **{f"{k}": v for k, v in r.parts.items()})
    for w, (y0, y1) in {"87": (1987, 2024), "09": (2009, 2024)}.items():
        row.update({f"{k}_{w}": v for k, v in lagstats(qt, S, y0, y1).items()})
    return row


if __name__ == "__main__":
    grid = [(None, None)] + [(t, n) for t in (1.0, 2.0, 5.0) for n in (1.0, 2.0)]
    with Pool(7) as p: rows = p.map(point, grid, chunksize=1)
    ob = {"model": "OBSERVED", "MRC_tau": 27.16, "BN_b": 1.17}
    for w, (y0, y1) in {"87": (1987, 2024), "09": (2009, 2024)}.items(): ob.update({f"{k}_{w}": v for k, v in lagstats(obs, None, y0, y1).items()})
    T = pd.DataFrame([ob] + rows); T.to_csv(OUT/"east_linear_shallow.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60)
    c1 = ["model", "f", "obj", "KGE", "logKGE", "MRC_tau", "BN_b", "DJF_ratio", "fallmin_bias_pct", "deep_share_fallmin", "peak_timing", "peak_volume", "falling_limb", "com_timing", "monthly_bias"]
    print(T[c1].round(2).to_string(index=False))
    for w in ("87", "09"):
        c = ["model"] + [f"LF_{nm}_k{k}_{w}" for nm in ("raw", "rpk", "rtot") for k in (1, 2, 5)] + [f"S_{nm}_k1_{w}" for nm in ("raw", "rpk", "rtot")]
        print(f"\nWY{'1987' if w=='87' else '2009'}-2024 lags (LF = ln Aug1-Nov30 7-d min; rpk/rtot = residual on same-year peak / total flow; S = prior Sep-30 deep storage)")
        print(T[c].round(2).to_string(index=False))
