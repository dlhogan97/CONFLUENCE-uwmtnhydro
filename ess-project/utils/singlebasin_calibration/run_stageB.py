"""Fit M0 / M1 aquifer+routing on each stage-A state (old, r1, r2) with the stage-B feature objective. usage: run_stageB.py <state> <model> <popsize> <maxiter> <workers>"""
import sys, json, time, numpy as np, pandas as pd
from scipy.optimize import differential_evolution
import stageB_emulator as B, feature_scorecard as FS
st_name, model, pop, it, nw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
fn = B.OUT/f"fit_{st_name}_{model}.json"
if fn.exists(): print("exists", fn.name); sys.exit()
st = B.build_state(st_name); B._ctx.update(st=st, model=model); t0 = time.time()
r = differential_evolution(B.cost, [(s[1], s[2]) for s in B.SPEC[model]], popsize=pop, maxiter=it, tol=1e-6, seed=21, workers=nw, updating="deferred", polish=False)
p = B.to_params(model, r.x); Qs, Qd, Sd = B.emulate(st, p); q = B.daily_Q(st, Qs, Qd, p["scale"]); J, terms = B.J_of(q)
feat = FS.features(q, B.E.obs, "2008-10-01", "2020-09-30")
json.dump(dict(state=st_name, model=model, params=p, J=J, terms=terms, features=feat, nfev=int(r.nfev), secs=round(time.time()-t0), x=list(map(float, r.x)),
               deep_mean_m=float(Sd.mean()), deep_end_m=float(Sd[-1])), open(fn, "w"), default=float, indent=1)
q.to_csv(B.OUT/f"daily_{st_name}_{model}.csv", header=["q_cms"]); print(st_name, model, "J=%.3f" % J, "secs", round(time.time()-t0), flush=True)
