"""
East River threshold shallow store, calibrated against a control on the SAME objective (2026-10-27). Follows east_threshold_store.py (sweep).
  treatment  Q_i = max(S_i - S_th, 0)/tau_s to the stream; leak_i = S_i/tau_l to ONE basin deep store, Q_d = S_d/tau_d (tau_d = 2 yr fixed)
             free: log10 tau_s, S_th, routing scale; tau_l set by the Dec-Feb volume balance (not free)
  control    the same threshold bucket, but the leak goes STRAIGHT to the stream (no deep store); log10 tau_l is free (one more free parameter
             than the treatment, so the comparison is conservative)
Objective (both) = east_ctl_treat_cal_v2 with REC_W = 1: mean of signature_objective_routing parts (WY2009-2020), the late-summer recession
term and |DJF/obs - 1|/0.10. No low-flow level or memory term; rho1 etc. are computed only for the final bests (east_threshold_cal_report.py).
usage: python east_threshold_cal.py {control|treatment} [--workers 6 --popsize 8 --maxiter 15]
"""
import os, sys, json, time, argparse, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, numba as nb
from scipy.optimize import brentq, differential_evolution
import east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V

E, st, W, P, obs = C.E, C.st, C.W, C.P, C.obs
NT, NH = st["Rch"].shape; YR = C.YR; DT = 3600.0; TAU_D = 2.0*YR; KGE_MIN, FAIL, MIN_GEN = C.KGE_MIN, C.FAIL, 6
BOUNDS = {"treatment": [("log_tau_s", np.log10(10.0), np.log10(90.0)), ("S_th", 0.0, 0.4), ("lscale", 4.3, 5.4)],
          "control": [("log_tau_s", np.log10(10.0), np.log10(90.0)), ("S_th", 0.0, 0.4), ("lscale", 4.3, 5.4), ("log_tau_l", np.log10(5.0), np.log10(20*365.0))]}
SEEDS = [dict(log_tau_s=np.log10(35.0), S_th=0.1, lscale=np.log10(48921.0), log_tau_l=np.log10(402.0)),       # sweep-selected case
         dict(log_tau_s=np.log10(20.0), S_th=0.05, lscale=np.log10(48921.0), log_tau_l=np.log10(300.0)),
         dict(log_tau_s=np.log10(50.0), S_th=0.2, lscale=np.log10(48921.0), log_tau_l=np.log10(800.0))]
OUTD = C.X.OUT/"threshold_cal"; OUTD.mkdir(exist_ok=True)


@nb.njit(cache=True)
def kernel(Rch, w, sth, tau_s, tau_l, tau_d, deep, S, sd, Qs, Qd, Sd):
    nt, nh = Rch.shape
    for t in range(nt):
        leak = 0.0
        for i in range(nh):
            s = S[i] + Rch[t, i]*DT
            if s < 0.0: s = 0.0
            q = max(s - sth, 0.0)/tau_s*DT; l = s/tau_l*DT
            if q + l > s: f = s/(q + l); q *= f; l *= f
            S[i] = s - q - l
            if deep: Qs[t, i] = q/DT; leak += w[i]*l
            else: Qs[t, i] = (q + l)/DT
        sd = sd + leak; qd = sd/tau_d*DT; sd -= qd; Qd[t] = qd/DT; Sd[t] = sd
    return S, sd


def simulate(sth, tau_s_d, tau_l_d, deep, passes=4):
    Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT); S = np.full(NH, sth); sd = 0.0
    for _ in range(passes): S, sd = kernel(st["Rch"], W, sth, tau_s_d*86400.0, tau_l_d*86400.0, TAU_D, deep, S, sd, Qs, Qd, Sd)
    return Qs, Qd, Sd


def flow(Qs, Qd, lscale): return C.B.daily_Q(st, Qs, Qd, 10**lscale)


def tau_l_balance(sth, tau_s, lscale):
    g = lambda lt: C.djf(flow(*simulate(sth, tau_s, 10**lt, True, passes=3)[:2], lscale)) - 1.0      # larger tau_l -> less winter flow
    lo, hi = np.log10(5.0), np.log10(20*365.0); glo, ghi = g(lo), g(hi)
    if glo < 0 or ghi > 0: return np.nan
    return float(10**brentq(g, lo, hi, xtol=0.02, rtol=1e-3, maxiter=14))


def run_model(p, model):
    tau_s = 10**p["log_tau_s"]
    tl = tau_l_balance(p["S_th"], tau_s, p["lscale"]) if model == "treatment" else 10**p["log_tau_l"]
    if not np.isfinite(tl): raise ValueError("winter balance unreachable")
    Qs, Qd, Sd = simulate(p["S_th"], tau_s, tl, model == "treatment"); return flow(Qs, Qd, p["lscale"]), tl, Qs, Qd, Sd


def evaluate(x, model):
    names = [b[0] for b in BOUNDS[model]]; p = dict(zip(names, x)); t0 = time.time()
    try:
        q, tl = run_model(p, model)[:2]
        r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], obs["2008-10-01":"2020-09-30"]); tm = V.recession_taus(q)
        j = pd.concat([tm.rename("m"), V.TAU_OBS.rename("o")], axis=1).fillna(5000.0).clip(upper=5000.0); dj = C.djf(q); parts = dict(r.parts)
        parts["recession"] = float(np.median(np.abs(np.log(j.m/j.o)))/np.log(1.25)); parts["winter_volume"] = float(abs(dj - 1.0)/0.10)
        obj = float(np.mean(list(parts.values()))); kge = r.diagnostics["KGE"]
        if kge < KGE_MIN: obj = FAIL + (KGE_MIN - kge)*10
        rec = dict(x=list(map(float, x)), params={**p, "tau_s_d": 10**p["log_tau_s"], "tau_l_d": tl}, objective=obj, parts=parts, KGE=kge,
                   tau_recess_median=float(tm.median()), DJF_ratio=dj, eval_s=time.time() - t0)
    except Exception as e:
        obj = FAIL; rec = dict(x=list(map(float, x)), objective=obj, reason=f"{type(e).__name__}:{e}")
    with open(OUTD/f"trials_{model}.jsonl", "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
    return obj


class _Obj:
    def __init__(self, model): self.model = model
    def __call__(self, x): return evaluate(x, self.model)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--popsize", type=int, default=8); ap.add_argument("--maxiter", type=int, default=15); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(); B = BOUNDS[a.model]; bounds = [(lo, hi) for _, lo, hi in B]; nd = len(bounds); rng = np.random.default_rng(a.seed)
    init = np.array([[lo + (hi - lo)*u for (lo, hi), u in zip(bounds, row)] for row in rng.random((a.popsize*nd, nd))])
    for i, s in enumerate(SEEDS): init[i] = np.clip([s[k] for k, _, _ in B], [b[0] for b in bounds], [b[1] for b in bounds])
    hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUTD/f"history_{a.model}.json").write_text(json.dumps(hist))
        print(f"[threshold {a.model}] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
        return len(hist) >= MIN_GEN and (hist[-4] - hist[-1]) < 0.005*abs(hist[-4])
    res = differential_evolution(_Obj(a.model), bounds=bounds, workers=a.workers, init=init, maxiter=a.maxiter, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUTD/f"result_{a.model}.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev), names=[b[0] for b in B])))
    print(f"[threshold {a.model}] DONE {res.fun:.4f}", flush=True)


if __name__ == "__main__":
    main()
