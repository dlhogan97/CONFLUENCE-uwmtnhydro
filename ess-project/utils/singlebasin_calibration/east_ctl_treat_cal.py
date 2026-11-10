"""
East River control vs deep-store treatment, calibrated on the SAME objective with the SAME free parameters (2026-10-27).

  control    one fast bucket per HRU (all subsurface processes lumped):  Q_i = rate*(S_i/Sf)^exp
  treatment  the same bucket + a LINEAR leak  leak_i = f*rate*(S_i/Sf)  to one basin deep store  Q_d = S_d/tau_d,  tau_d = 2 yr fixed
Free (both): log10 bucket rate, bucket size Sf, bucket exponent, routing gamma scale.  Treatment f:
  variant a  f set by the Dec-Feb volume balance (not free); BOTH models also carry a Dec-Feb volume term in the objective
  variant b  no winter information anywhere; treatment log10 f is a free 5th parameter
Objective = mean of: signature_objective_routing parts (peak timing, peak volume, falling limb, centre of mass, Apr-Oct monthly bias; WY2009-2020),
  MRC term |tau_MRC - 27.2 d| / (0.2*27.2) on total flow (mrc_analysis, WY1984-2024), and for variant a |DJF/obs - 1|/0.10 (WY1987-2024).
No low-flow level or memory term. KGE >= 0.80 guard (graded). Recharge / surface runoff = routing-control SUMMA state (emulator validated r = 1.0000).
usage: python east_ctl_treat_cal.py {control|treatment} {a|b} [--workers 2 --popsize 6 --maxiter 12]
"""
import os, sys, json, time, uuid, argparse, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
import east_linear_shallow as L

X, E, B, st, W, P = L.X, L.E, L.B, L.st, L.W, L.P
obs = E.obs; NT, NH = st["Rch"].shape; YR = X.YR; TAU_D_YR = 2.0
MRC_OBS = 27.16; KGE_MIN = 0.80; FAIL = 10.0
BOUNDS = [("lrate", -6.6, -4.3), ("Sf", 0.3, 4.0), ("exp", 1.0, 3.5), ("lscale", 4.3, 5.4)]
SEEDS = {"routing control": dict(lrate=np.log10(7.27e-6), Sf=1.668, exp=2.268, lscale=np.log10(48921.0)),
         "M1 buckets": dict(lrate=np.log10(9.44e-6), Sf=0.704, exp=2.795, lscale=np.log10(142731.0)),
         "linear 27 d": dict(lrate=np.log10(1.0/(27*86400.0)), Sf=1.0, exp=1.0, lscale=np.log10(48921.0))}
OUTD = X.OUT/"ctl_treat_cal"; OUTD.mkdir(exist_ok=True)


def simulate(p, lam, passes=4):
    Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT); S = st["S0"].copy(); sdi = 0.0
    for _ in range(passes):
        S, sdi = E.run_buckets(st["Rch"], st["A"], float(p["Sf"]), float(10**p["lrate"]), float(p["exp"]), float(lam), 1.0, 1.0, 1.0, 1.0/(TAU_D_YR*YR), S, float(sdi), 3600.0, Qs, Qd, Sd)
    return Qs, Qd, Sd


def flow(Qs, Qd, p): return B.daily_Q(st, Qs, Qd, 10**p["lscale"])
def djf(q): return X.mo(q, [12, 1, 2])/X.OBS_DJF


def f_balance(p):
    """f with DJF = observed. With a linear leak under a steep bucket, DJF(f) saturates fast, so solve in log10 f (1e-4 .. 3); nan if unreachable."""
    from scipy.optimize import brentq
    g = lambda lf: djf(flow(*simulate(p, 10**lf, passes=3)[:2], p)) - 1.0
    lo, hi = -4.0, np.log10(3.0); glo, ghi = g(lo), g(hi)
    if glo >= 0: return 10**lo
    if ghi < 0: return np.nan
    return float(10**brentq(g, lo, hi, xtol=0.03, rtol=1e-3, maxiter=12))


def evaluate(x, model, variant):
    p = dict(zip([b[0] for b in BOUNDS], x[:4])); t0 = time.time()
    try:
        if model == "control": f = 0.0
        elif variant == "a":
            f = f_balance(p)
            if not np.isfinite(f): raise ValueError("winter balance unreachable")
        else: f = float(10**x[4])
        Qs, Qd, Sd = simulate(p, f); q = flow(Qs, Qd, p)
        r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], obs["2008-10-01":"2020-09-30"])
        tm, b = L.mrc(q, f"{os.getpid()}_{uuid.uuid4().hex[:6]}")
        parts = dict(r.parts); parts["mrc_tau"] = float(min(abs(tm - MRC_OBS)/(0.2*MRC_OBS), 3.0))
        dj = djf(q)
        if variant == "a": parts["winter_volume"] = float(min(abs(dj - 1.0)/0.10, 3.0))
        obj = float(np.mean(list(parts.values()))); kge = r.diagnostics["KGE"]
        if kge < KGE_MIN: obj = FAIL + (KGE_MIN - kge)*10
        rec = dict(x=list(map(float, x)), params={**p, "f": f}, objective=obj, parts=parts, KGE=kge, MRC_tau=tm, BN_b=b, DJF_ratio=dj, eval_s=time.time() - t0)
    except Exception as e:
        obj = FAIL; rec = dict(x=list(map(float, x)), objective=obj, reason=f"{type(e).__name__}:{e}")
    with open(OUTD/f"trials_{model}_{variant}.jsonl", "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
    return obj


class _Obj:
    def __init__(self, model, variant): self.model, self.variant = model, variant
    def __call__(self, x): return evaluate(x, self.model, self.variant)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("variant"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--popsize", type=int, default=6); ap.add_argument("--maxiter", type=int, default=12); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(); tag = f"{a.model}_{a.variant}"
    bounds = [(lo, hi) for _, lo, hi in BOUNDS] + ([(-4.0, 0.5)] if (a.model == "treatment" and a.variant == "b") else [])
    nd = len(bounds); npop = a.popsize*nd; rng = np.random.default_rng(a.seed)
    init = np.array([[lo + (hi - lo)*u for (lo, hi), u in zip(bounds, row)] for row in rng.random((npop, nd))])
    for i, s in enumerate(SEEDS.values()):                                       # seed the population with the three known bucket shapes
        v = [s[b[0]] for b in BOUNDS] + ([-1.5] if nd == 5 else []); init[i] = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
    hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUTD/f"history_{tag}.json").write_text(json.dumps(hist))
        print(f"[{tag}] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
        return len(hist) >= 4 and (hist[-4] - hist[-1]) < 0.005*abs(hist[-4])      # early stop: <0.5% gain over 3 generations
    res = differential_evolution(_Obj(a.model, a.variant), bounds=bounds, workers=a.workers, init=init, maxiter=a.maxiter, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUTD/f"result_{tag}.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev), names=[b[0] for b in BOUNDS] + (["log10_f"] if nd == 5 else []))))
    print(f"[{tag}] DONE {res.fun:.4f}", flush=True)


if __name__ == "__main__":
    main()
