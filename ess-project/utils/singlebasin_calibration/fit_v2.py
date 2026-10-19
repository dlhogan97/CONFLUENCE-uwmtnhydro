"""Aquifer+routing refit of M0 (no deep store) and M1 (single-basin) under the v2 signature objective
(com_timing, monthly_volume, total_volume, peak_flow, peak_timing, lowflow_sepnov, recession x0.5). Snow/soil frozen => SWE terms constant, not used.
usage: python fit_v2.py M0|M1 full|A|B popsize maxiter workers [tag]"""
import json, sys, time, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
sys.path.insert(0, "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration")
import signature_objective_plus as P
import emu_core as E, fit_models as F
OUT = F.OUT.parent/"overnight_v2"; OUT.mkdir(parents=True, exist_ok=True)
TAU = 27.0; _ctx = {}
def v2(q, win):
    a, b = E.WINDOWS[win]["skill"]; s = q.loc[a:b]; o = E.obs.loc[a:b]
    return P.signature_objective_plus(s, o, TAU, area_m2=E.Atot)
def cost(x):
    try:
        p = F.to_params(_ctx["model"], x); Qs, Qd, Sd, _, _ = E.emulate(p, passes=3); q = E.daily_Q(Qs, Qd, p["scale"])
        J = v2(q, _ctx["win"]).objective; return J if np.isfinite(J) else 10.0
    except Exception: return 10.0
def evaluate_all(model, x):
    p = F.to_params(model, x); Qs, Qd, Sd, S, sd = E.emulate(p, passes=8); q = E.daily_Q(Qs, Qd, p["scale"]); Sdeep = pd.Series(Sd, index=E.tt); out = {}
    for wn, W in E.WINDOWS.items():
        r = v2(q, wn); out[wn] = dict(objective=r.objective, parts=r.parts, skill=E.skill(q, W["skill"]), memory=E.memory(q, *W["mem"], Sdeep=Sdeep))
    out["deep"] = dict(mean=float(Sd.mean()), leak_share=float(Qd.mean()/(Qd.mean()+(Qs*E.w).sum(1).mean())))
    return p, out, S, sd, q
def fit(model, win, pop, it, nw, tag=""):
    fn = OUT/f"fit_{model}_{win}{tag}.json"
    if fn.exists(): print("exists", fn.name); return
    _ctx.update(model=model, win=win); t0 = time.time()
    r = differential_evolution(cost, [(s[1], s[2]) for s in F.SPEC[model]], popsize=pop, maxiter=it, tol=1e-5, seed=21, workers=nw, updating="deferred", polish=False)
    p, ev, S, sd, q = evaluate_all(model, r.x)
    json.dump(dict(model=model, window=win, params=p, J=float(r.fun), nfev=int(r.nfev), secs=round(time.time()-t0), eval=ev, x=list(map(float, r.x)),
                   S_small=list(map(float, S)), S_deep=float(sd), popsize=pop, maxiter=it), open(fn, "w"), default=float, indent=1)
    q.to_csv(OUT/f"daily_{model}_{win}{tag}.csv", header=["q_cms"]); print(model, win, "J=%.3f"%r.fun, "secs", round(time.time()-t0), flush=True)
if __name__ == "__main__":
    fit(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else "")
