"""Fit the four aquifer structures on a calibration window with a common objective.
M0 baseline (independent buckets) | M1 single-basin full | M2 single-basin reduced (linear leak + linear deep store) | M3 = M1 + memory-informed objective.
Objective (all models): (1-KGE) + (1-logNSE) + |Aug-Nov bias|/100 + |total bias|/100   [+ |d res1| + |d raw1| vs OBSERVED in the same window, M3 only]
"""
import json, sys, time, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
import emu_core as E

OUT = Path("/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/overnight")
OUT.mkdir(parents=True, exist_ok=True)
# free parameters: (name, lo, hi) in the optimiser's coordinates; log10 for scale-like quantities
SPEC = {
 "M0": [("lSf_h", -0.5, 1.1), ("exp_h", 1.2, 8.0), ("lscale", 4.3, 5.2)],
 "M1": [("lSf_h", -0.5, 1.1), ("exp_h", 1.2, 8.0), ("llam", -2.3, 0.18), ("m_leak", 1.0, 6.0), ("lSf_d", -0.5, 1.9), ("exp_d", 1.0, 8.0), ("lscale", 4.3, 5.2)],
 "M2": [("lSf_h", -0.5, 1.1), ("exp_h", 1.2, 8.0), ("llam", -2.3, 0.18), ("lSf_d", -0.5, 1.9), ("lscale", 4.3, 5.2)],
}
SPEC["M3"] = SPEC["M1"]
def to_params(model, x):
    d = dict(zip([s[0] for s in SPEC[model]], x))
    p = dict(Sf_h=10**d["lSf_h"], exp_h=d["exp_h"], scale=10**d["lscale"], lam=0.0, m_leak=1.0, Sf_d=1.0, exp_d=1.0)
    if model != "M0":
        p["lam"] = 10**d["llam"]; p["Sf_d"] = 10**d["lSf_d"]
        p["m_leak"] = d.get("m_leak", 1.0); p["exp_d"] = d.get("exp_d", 1.0)
    return p
_ctx = {}
def cost(x):
    model, win, target = _ctx["model"], _ctx["win"], _ctx["target"]; W = E.WINDOWS[win]
    try:
        p = to_params(model, x); Qs, Qd, Sd, _, _ = E.emulate(p, passes=3); q = E.daily_Q(Qs, Qd, p["scale"]); s = E.skill(q, W["skill"])
        J = (1-s["KGE"])+(1-s["logNSE"])+abs(s["PB_AN"])/100+abs(s["PB"])/100
        if model == "M3":
            m = E.memory(q, *W["mem"]); J += abs(m["res1"]-target["res1"])+abs(m["raw1"]-target["raw1"])
        return J if np.isfinite(J) else 10.0
    except Exception:
        return 10.0
def evaluate_all(model, x):
    p = to_params(model, x); Qs, Qd, Sd, S, sd = E.emulate(p, passes=8); q = E.daily_Q(Qs, Qd, p["scale"]); Sdeep = pd.Series(Sd, index=E.tt); out = {}
    for wn, W in E.WINDOWS.items():
        out[wn] = dict(skill=E.skill(q, W["skill"]), memory=E.memory(q, *W["mem"], Sdeep=Sdeep))
    out["deep"] = dict(mean=float(Sd.mean()), cv=float(Sdeep.resample("YE").mean().std()/Sdeep.resample("YE").mean().mean()),
                       leak_share=float(Qd.mean()/(Qd.mean()+(Qs*E.w).sum(1).mean())))
    return p, out, S, sd
def fit(model, win, popsize, maxiter, workers, seed=21, tag=None):
    fn = OUT/f"fit_{model}_{win}{tag or ''}.json"
    if fn.exists(): print("exists", fn.name, flush=True); return json.load(open(fn))
    W = E.WINDOWS[win]; target = E.memory(E.obs, *W["mem"]); _ctx.update(model=model, win=win, target=target)
    t0 = time.time(); bnds = [(s[1], s[2]) for s in SPEC[model]]
    r = differential_evolution(cost, bnds, popsize=popsize, maxiter=maxiter, tol=1e-5, seed=seed, workers=workers, updating="deferred", polish=False)
    p, ev, S, sd = evaluate_all(model, r.x)
    res = dict(model=model, window=win, params=p, J=float(r.fun), nfev=int(r.nfev), secs=round(time.time()-t0), obs_target=target, eval=ev,
               x=list(map(float, r.x)), S_small=list(map(float, S)), S_deep=float(sd), popsize=popsize, maxiter=maxiter)
    json.dump(res, open(fn, "w"), default=float, indent=1); print(model, win, "J=%.3f"%r.fun, "secs", res["secs"], flush=True); return res
if __name__ == "__main__":
    model, win = sys.argv[1], sys.argv[2]; pop = int(sys.argv[3]); it = int(sys.argv[4]); nw = int(sys.argv[5]); tag = sys.argv[6] if len(sys.argv) > 6 else None
    fit(model, win, pop, it, nw, tag=tag)
