"""
East River control vs deep-store treatment, round 2 (2026-10-27). Same structures, free parameters and winter-volume handling as variant (a) of
east_ctl_treat_cal.py; changes:
  * MRC term DROPPED: on the noisy gauge record the MRC sees only short upper-recession pieces (obs tau 27 d), on the smooth model whole recessions
    (tau 65 d); adding 5% daily noise to a model series moved its MRC tau 65 -> 15 d. It measures noise, not the recession.
  * NEW recession term, computed identically on observed and simulated flow and smoothed at low flows: per water year, the lower envelope
    (7-d rolling min, then 7-d centred mean) over Aug 1 - Sep 30, Theil-Sen slope of ln Q -> tau_y = -1/slope.
    term = median_y |ln(tau_model/tau_obs)| / ln(1.25)    (25% off -> 1), WY1984-2024. Observed median tau 68 d (IQR 58-97).
  * no cap on any term; winter-volume term |DJF/obs - 1|/0.10 as before (both models).
  * at least 6 generations before the early stop (<0.5% gain over 3 generations); max 12.
usage: [REC_W=3] python east_ctl_treat_cal_v2.py {control|treatment} [--workers 3]   (REC_W != 1 writes to ctl_treat_cal_v3_recw<W>)
"""
import os, sys, json, time, uuid, argparse, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy.stats import theilslopes
from scipy.optimize import differential_evolution
import east_ctl_treat_cal as C

E, P, obs = C.E, C.P, C.obs
KGE_MIN, FAIL, MIN_GEN = C.KGE_MIN, C.FAIL, 6
REC_W = float(os.environ.get("REC_W", "1"))                      # weight on the recession term (round 3: 3)
OUTD = C.X.OUT/("ctl_treat_cal_v2" if REC_W == 1 else f"ctl_treat_cal_v3_recw{REC_W:g}"); OUTD.mkdir(exist_ok=True)


def recession_taus(q, y0=1984, y1=2024):
    low = q.rolling(7, center=True, min_periods=5).min().rolling(7, center=True, min_periods=5).mean()
    out = {}
    for y in range(y0, y1 + 1):
        w = low[f"{y}-08-01":f"{y}-09-30"].dropna()
        if len(w) < 50: continue
        s = theilslopes(np.log(np.clip(w.values, 1e-4, None)), np.arange(len(w)))[0]
        out[y] = -1.0/s if s < 0 else np.nan
    return pd.Series(out)


TAU_OBS = recession_taus(obs)


def evaluate(x, model):
    p = dict(zip([b[0] for b in C.BOUNDS], x[:4])); t0 = time.time()
    try:
        f = 0.0 if model == "control" else C.f_balance(p)
        if not np.isfinite(f): raise ValueError("winter balance unreachable")
        Qs, Qd, Sd = C.simulate(p, f); q = C.flow(Qs, Qd, p)
        r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], obs["2008-10-01":"2020-09-30"])
        tm = recession_taus(q); j = pd.concat([tm.rename("m"), TAU_OBS.rename("o")], axis=1)
        j = j.fillna(5000.0).clip(upper=5000.0)                               # a year that does not recede counts as very slow
        parts = dict(r.parts); dj = C.djf(q)
        parts["recession"] = float(np.median(np.abs(np.log(j.m/j.o)))/np.log(1.25)); parts["winter_volume"] = float(abs(dj - 1.0)/0.10)
        obj = float((sum(parts.values()) + (REC_W - 1.0)*parts["recession"])/(len(parts) + REC_W - 1.0)); kge = r.diagnostics["KGE"]
        if kge < KGE_MIN: obj = FAIL + (KGE_MIN - kge)*10
        rec = dict(x=list(map(float, x)), params={**p, "f": f}, objective=obj, parts=parts, KGE=kge, tau_recess_median=float(tm.median()), DJF_ratio=dj, eval_s=time.time() - t0)
    except Exception as e:
        obj = FAIL; rec = dict(x=list(map(float, x)), objective=obj, reason=f"{type(e).__name__}:{e}")
    with open(OUTD/f"trials_{model}.jsonl", "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
    return obj


class _Obj:
    def __init__(self, model): self.model = model
    def __call__(self, x): return evaluate(x, self.model)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--popsize", type=int, default=6); ap.add_argument("--maxiter", type=int, default=12); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(); bounds = [(lo, hi) for _, lo, hi in C.BOUNDS]; nd = len(bounds); rng = np.random.default_rng(a.seed)
    init = np.array([[lo + (hi - lo)*u for (lo, hi), u in zip(bounds, row)] for row in rng.random((a.popsize*nd, nd))])
    seeds = list(C.SEEDS.values())
    for t in ("control_a", "treatment_a"):                                       # + the two round-1 (a) bests
        tr = [json.loads(l) for l in open(C.OUTD/f"trials_{t}.jsonl")]; b = min([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])["params"]
        seeds.append({k: b[k] for k, _, _ in C.BOUNDS})
    for i, s in enumerate(seeds): init[i] = np.clip([s[k] for k, _, _ in C.BOUNDS], [b[0] for b in bounds], [b[1] for b in bounds])
    hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUTD/f"history_{a.model}.json").write_text(json.dumps(hist))
        print(f"[{a.model} v2] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
        return len(hist) >= MIN_GEN and (hist[-4] - hist[-1]) < 0.005*abs(hist[-4])
    res = differential_evolution(_Obj(a.model), bounds=bounds, workers=a.workers, init=init, maxiter=a.maxiter, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUTD/f"result_{a.model}.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev), names=[b[0] for b in C.BOUNDS])))
    print(f"[{a.model} v2] DONE {res.fun:.4f}", flush=True)


if __name__ == "__main__":
    main()
