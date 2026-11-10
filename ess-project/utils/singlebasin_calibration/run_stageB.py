"""Fit M0 / M1 aquifer+routing on a stage-A state with the stage-B feature objective.
usage: run_stageB.py <state> <model> <popsize> <maxiter> <workers> [sfd=<m>]
  sfd=<m>  hold the deep-store scale factor Sf_d fixed at <m> metres (M1 only), to see what a store that must fill and
           drain between years costs in fit and buys in memory. Scoring window end: env STAGEB_END (stageB_emulator)."""
import sys, json, time, numpy as np, pandas as pd
from scipy.optimize import differential_evolution
import stageB_emulator as B, feature_scorecard as FS
sys.path.insert(0, "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration")
import signature_objective_plus as P
st_name, model, pop, it, nw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sfd = float(sys.argv[6].split("=")[1]) if len(sys.argv) > 6 else None
tag = f"{st_name}_{model}" + (f"_sfd{sfd:g}" if sfd is not None else "") + B.SUF
fn = B.OUT/f"fit_{tag}.json"
if fn.exists(): print("exists", fn.name); sys.exit()
spec = B.SPEC[model]
if sfd is not None:
    i_fix = [s[0] for s in spec].index("lSf_d"); spec = [s for s in spec if s[0] != "lSf_d"]


def full_x(x):
    return np.insert(np.asarray(x, float), i_fix, np.log10(sfd)) if sfd is not None else np.asarray(x, float)


def cost(x):
    return B.cost(full_x(x))


st = B.build_state(st_name); B._ctx.update(st=st, model=model); t0 = time.time()
r = differential_evolution(cost, [(s[1], s[2]) for s in spec], popsize=pop, maxiter=it, tol=1e-6, seed=21, workers=nw, updating="deferred", polish=False)
x = full_x(r.x); p = B.to_params(model, x); Qs, Qd, Sd = B.emulate(st, p); q = B.daily_Q(st, Qs, Qd, p["scale"]); J, terms = B.J_of(q)
feat = FS.features(q, B.E.obs, B.SCORE0, B.END)
lag1_sim, lag1_obs = P.lowflow_peak_residual_lag1(q.loc[B.SCORE0:B.END]), P.lowflow_peak_residual_lag1(B.E.obs.loc[B.SCORE0:B.END])
sd_ann = pd.Series(Sd, index=st["tt"]).groupby(st["tt"].year + (st["tt"].month >= 10)).mean()
json.dump(dict(state=st_name, model=model, sfd_fixed=sfd, end=B.END, params=p, J=J, terms=terms, features=feat, nfev=int(r.nfev), secs=round(time.time()-t0),
               x=list(map(float, x)), deep_mean_m=float(Sd.mean()), deep_end_m=float(Sd[-1]), deep_annual_range_m=float(sd_ann.max() - sd_ann.min()),
               resid_lag1_sim=lag1_sim, resid_lag1_obs=lag1_obs), open(fn, "w"), default=float, indent=1)
q.to_csv(B.OUT/f"daily_{tag}.csv", header=["q_cms"]); print(tag, "J=%.3f" % J, "lag1 %.2f (obs %.2f)" % (lag1_sim, lag1_obs), "secs", round(time.time()-t0), flush=True)
