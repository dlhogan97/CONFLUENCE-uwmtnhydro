"""
Matched final calibration of the East baseline and deep-store models (2026-10-28): same optimizer, same budget, same objective,
from scratch (Latin-hypercube start, no seeding from earlier bests), fixed number of generations (no early stop).
  baseline   nonlinear bucket per HRU draining to the stream, Q = rate*(S/Sf)^exp; free: log10 rate, Sf, exp, routing scale (4)
  treatment  threshold shallow store (Q = max(S - S_th, 0)/tau_s to the stream, S/tau_l leak) + linear 2-yr basin deep store;
             free: log10 tau_s, S_th, routing scale (3); tau_l from the Dec-Feb volume balance
Objective (both) = mean of the routing-signature parts (WY2009-2020), the late-summer recession term (WY1984-2024) and |DJF/obs - 1|/0.10;
no low-flow level or memory term; KGE >= 0.80 guard. Population ~40 (popsize 10 x 4 / 13 x 3), 15 generations.
Every trial is written to matched_cal/trials_<model>.jsonl; memory is computed afterwards on the near-equal fits (east_matched_report.py).
REC_W (env, default 1) weights the recession term as in east_ctl_treat_cal_v2.py; REC_W != 1 writes to matched_cal_recw<W>.
WREC=1 (env) adds a winter-recession term for every model, computed like the Aug-Sep term on the Nov 15 - Mar 15 lower envelope
  (obs median tau 318 d, IQR 250-450, WY1984-2024): median_y |ln(tau_sim/tau_obs)| / ln 1.25. It is the part of the hydrograph the deep store
  controls (2026-10-29: deep-store fits gave 609-2186 d, baselines 151-182 d). It also enables model treatment_td: the treatment with the deep
  store's drainage time tau_d free (100-1500 d; fixed at 2 yr otherwise), 4 free parameters like the baseline. Writes to ..._wrec.
usage: [REC_W=3] [WREC=1] python east_matched_cal.py {baseline|treatment|treatment_td} [--workers 10]
"""
import os, json, time, argparse, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import theilslopes
import pandas as pd
import east_ctl_treat_cal as C, east_threshold_cal as T, east_independent_compare as I

KGE_MIN, FAIL = C.KGE_MIN, C.FAIL
SPEC = {"baseline": dict(bounds=C.BOUNDS, popsize=10), "treatment": dict(bounds=T.BOUNDS["treatment"], popsize=13),
        "treatment_td": dict(bounds=T.BOUNDS["treatment"] + [("log_tau_d", 2.0, np.log10(1500.0))], popsize=10)}
REC_W = float(os.environ.get("REC_W", "1"))
WREC = os.environ.get("WREC", "0") == "1"
OUTD = C.X.OUT/(("matched_cal" if REC_W == 1 else f"matched_cal_recw{REC_W:g}") + ("_wrec" if WREC else "")); OUTD.mkdir(exist_ok=True)


def winter_taus(q, y0=1984, y1=2024):
    low = q.rolling(7, center=True, min_periods=5).min().rolling(7, center=True, min_periods=5).mean(); out = {}
    for y in range(y0, y1 + 1):
        w = low[f"{y-1}-11-15":f"{y}-03-15"].dropna()
        if len(w) < 90: continue
        s = theilslopes(np.log(np.clip(w.values, 1e-4, None)), np.arange(len(w)))[0]; out[y] = -1.0/s if s < 0 else np.nan
    return pd.Series(out)


WTAU_OBS = winter_taus(C.obs)


def winter_term(q):
    j = pd.concat([winter_taus(q).rename("m"), WTAU_OBS.rename("o")], axis=1).dropna(subset=["o"]).fillna(5000.0).clip(upper=5000.0)
    return float(np.median(np.abs(np.log(j.m/j.o)))/np.log(1.25))


def simulate(model, p):
    if model == "baseline":
        Qs, Qd, _ = C.simulate(p, 0.0); return C.flow(Qs, Qd, p), None
    T.TAU_D = 10**p["log_tau_d"]*86400.0 if "log_tau_d" in p else 2.0*T.YR          # module global read by T.simulate
    q, tl = T.run_model(p, "treatment")[:2]; return q, tl


def evaluate(x, model):
    names = [b[0] for b in SPEC[model]["bounds"]]; p = dict(zip(names, map(float, x))); t0 = time.time()
    try:
        q, tl = simulate(model, p); obj, kge, tau, parts = I.objective(q)
        n = len(parts)
        if WREC: parts["winter_recession"] = winter_term(q); obj = (obj*n + parts["winter_recession"])/(n + 1); n += 1
        obj = (obj*n + (REC_W - 1.0)*parts["recession"])/(n + REC_W - 1.0)
        if kge < KGE_MIN: obj = FAIL + (KGE_MIN - kge)*10
        rec = dict(params=p, tau_l_d=tl, objective=obj, parts=parts, KGE=kge, tau_recess_median=tau, eval_s=time.time() - t0)
    except Exception as e:
        obj = FAIL; rec = dict(params=p, objective=obj, reason=f"{type(e).__name__}:{e}")
    with open(OUTD/f"trials_{model}.jsonl", "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
    return obj


class _Obj:
    def __init__(self, model): self.model = model
    def __call__(self, x): return evaluate(x, self.model)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=15); ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args(); s = SPEC[a.model]; bounds = [(lo, hi) for _, lo, hi in s["bounds"]]; hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUTD/f"history_{a.model}.json").write_text(json.dumps(hist))
        print(f"[matched {a.model}] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
    res = differential_evolution(_Obj(a.model), bounds=bounds, popsize=s["popsize"], init="latinhypercube", maxiter=a.maxiter, workers=a.workers,
                                 updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUTD/f"result_{a.model}.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev),
                                                               names=[b[0] for b in s["bounds"]])))
    print(f"[matched {a.model}] DONE {res.fun:.4f}  nfev {res.nfev}", flush=True)


if __name__ == "__main__":
    main()
