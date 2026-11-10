"""
Deep store ADDED to the paper baseline: only the store's own parameters are fitted, under the paper's calibration rules (2026-10-29).
  baseline   SUMMA East run of the paper (runs/East_River_20260720_094546), every parameter frozen (east_paper_booster.py: emulator r = 1.00000).
  store      singleBasin structure: each HRU aquifer leaks lam*rate*(S/Sf)^m_leak to one basin store that drains rate*(S_d/Sf_d)^exp_d to the stream.
             Free (4): log10 lam [-4, 0], m_leak [1, 3], log10 Sf_d [0, 2.5] m, exp_d [1, 5]. Seed = the overnight M1 store (0.0187, 1.35, 12.4, 2.41).
  objective  the paper's own: signature_objective (com_timing, monthly_volume, concentration, recession tau 27 d, lowflow_sepnov; equal weights)
             on WY1993-2002, same as the baseline calibration. No memory term. Memory and everything else are scored afterwards on WY1987-2024.
The question asked of it: once the store is there, sized by the same rules the baseline was, does the model carry low-flow memory?
--memory (manufactured store): the same four store parameters fitted to the OBSERVED lag-1 memory instead, |rho1(WY1987-2024) - 0.383|/0.05, with the
  paper objective as a guard (rejected if worse than 1.10 x the baseline's 0.3370). Circular on purpose: it builds a source of memory into the model and
  asks what it costs; the guard keeps the traditional fit. Sf_d up to 10^3 m. Trials -> stageB/paper_store_memory/.
Trials -> stageB/paper_store_fit/trials.jsonl.   usage: python east_paper_store_fit.py [--memory] [--workers 8]  |  [--memory] --report
"""
import os, sys, json, time, argparse, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"modeling"/"calibration"))
from signature_objective import signature_objective
import emu_core as E, east_paper_booster as PB, east_booster as EB

st, AT, P0, obs = PB.st, PB.AT, PB.P0, EB.obs
MEMORY = "--memory" in sys.argv
OUTD = PB.B.OUT/("paper_store_memory" if MEMORY else "paper_store_fit"); OUTD.mkdir(exist_ok=True)
BOUNDS = [("log_lam", -4.0, 0.0), ("m_leak", 1.0, 3.0), ("log_Sf_d", 0.0, 3.0 if MEMORY else 2.5), ("exp_d", 1.0, 5.0)]
BASE_OBJ, RHO_OBS, GUARD = 0.3370, 0.383, 1.10
SEED = [np.log10(0.0187), 1.35, np.log10(12.4), 2.41]
CAL0, CAL1, TAU_TARGET = "1992-10-01", "2002-09-30", 27.0


def run(p, passes=3):
    NT, NH = st["Rch"].shape; S = st["S0"].copy(); Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT); sdi = 0.0
    for _ in range(passes):
        S, sdi = E.run_buckets(st["Rch"], st["A"], P0["Sf"], P0["rate"], P0["exp"], 10**p["log_lam"], p["m_leak"], 10**p["log_Sf_d"], p["exp_d"], P0["rate"],
                               S, float(sdi), 3600.0, Qs, Qd, Sd)
    return PB.B.daily_Q(st, Qs, Qd, P0["scale"]), pd.Series(Qd*AT, index=st["tt"]).resample("1D").mean(), pd.Series(Sd, index=st["tt"]).resample("1D").mean()


def paper_objective(q):
    r = signature_objective(q[CAL0:CAL1], obs[CAL0:CAL1], TAU_TARGET, area_m2=AT); return r.objective, r.parts


def evaluate(x):
    p = dict(zip([b[0] for b in BOUNDS], map(float, x))); t0 = time.time()
    try:
        q, qd, _ = run(p); obj, parts = paper_objective(q); rec = dict(params=p, objective=obj, parts=parts,
                                                                      store_share=float(qd["1986-10-01":"2024-09-30"].mean()/q["1986-10-01":"2024-09-30"].mean()))
        if MEMORY:
            rho = float(E.lag_r(E.resid(*E.series(q, 1987, 2024)), 1)); rec.update(paper_obj=obj, rho1_87=rho)
            obj = abs(rho - RHO_OBS)/0.05 if obj <= GUARD*BASE_OBJ else 10.0 + (obj - GUARD*BASE_OBJ)*10; rec["objective"] = obj
    except Exception as e:
        obj = 10.0; rec = dict(params=p, objective=obj, reason=f"{type(e).__name__}:{e}")
    rec["eval_s"] = time.time() - t0
    with open(OUTD/"trials.jsonl", "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
    return obj


def report():
    tr = [json.loads(l) for l in open(OUTD/"trials.jsonl")]; ok = sorted([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])
    uniq, seen = [], set()
    for r in ok:
        k = tuple(round(v, 3) for v in r["params"].values())
        if k not in seen: seen.add(k); uniq.append(r)
    best = uniq[0]; near = [r for r in uniq if r["objective"] <= (best["objective"] + 1.0 if MEMORY else 1.05*best["objective"])]   # memory: within 0.05 of rho1
    base = PB.B.OUT/"east_final_models_daily.csv"; D = pd.read_csv(base, index_col="date", parse_dates=True); qb = D["paper_baseline"]
    rows = [dict(model="observed", **EB.score(obs)), dict(model="paper baseline", paper_obj=paper_objective(qb)[0], **EB.score(qb))]
    q, qd, sd = run(best["params"]); rows.append(dict(model="paper baseline + store (best)", paper_obj=best["objective"], store_share=best["store_share"],
                                                      **{k: round(v, 4) for k, v in best["params"].items()}, **EB.score(q, sd)))
    q1, qd1, sd1 = run(dict(zip([b[0] for b in BOUNDS], SEED))); rows.append(dict(model="paper baseline + M1 store (transplant)", paper_obj=paper_objective(q1)[0],
                                                                                    **EB.score(q1, sd1)))
    R = pd.DataFrame(rows).set_index("model"); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(R.round(3).T.to_string()); R.to_csv(OUTD/"report_best.csv")
    print(f"\nnear-equal fits: {len(near)}")
    N = []
    for r in near[:60]:
        qq, qdd, _ = run(r["params"]); s = EB.score(qq); N.append(dict(fit_obj=r["objective"], paper_obj=r.get("paper_obj", r["objective"]), store_share=r["store_share"], **r["params"],
                                                                         **{k: s[k] for k in ("KGE_eval", "NSE_eval", "recess_d", "winter_tau_d", "fallmin_bias", "rho1_87", "rho1_09")}))
    N = pd.DataFrame(N); print(N.describe().loc[["min", "50%", "max"]].round(3).T.to_string()); N.to_csv(OUTD/"report_near.csv", index=False)
    D["paper_store_memory" if MEMORY else "paper_store_best"] = q; D.to_csv(base, float_format="%.5g"); print("series paper_store_best written to", base)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.report: return report()
    q0 = PB.B.OUT/"east_final_models_daily.csv"; qb = pd.read_csv(q0, index_col="date", parse_dates=True)["paper_baseline"]
    o, parts = paper_objective(qb); print(f"paper baseline rescored on WY1993-2002: objective {o:.4f} (calibration run 0.3369)  parts {({k: round(v, 3) for k, v in parts.items()})}", flush=True)
    hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUTD/"history.json").write_text(json.dumps(hist))
        print(f"[paper store] gen {len(hist):2d}: best {hist[-1]:.4f}  x {np.round(intermediate_result.x, 3).tolist()}", flush=True)
    res = differential_evolution(evaluate, bounds=[(lo, hi) for _, lo, hi in BOUNDS], popsize=10, init="latinhypercube", x0=SEED, maxiter=15, workers=a.workers,
                                 updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=11)
    (OUTD/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), objective=float(res.fun), nfev=int(res.nfev))))
    print(f"[paper store] DONE {res.fun:.4f}", flush=True); report()


if __name__ == "__main__":
    main()
