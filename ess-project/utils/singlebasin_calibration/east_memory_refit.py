"""
Follow-up to east_memory_search.py (added AFTER seeing its result, 2026-10-30, so this is a second look, not part of the pre-set test). The frozen-baseline search could not keep a large store:
screen trials with delayed stores reach the observed rho1 but run fall minima +60-70% high, apparently because the frozen buckets still drain as if they carry all the slow water.
Here the HRU bucket scale and exponent refit WITH the added structure. Same paper objective (WY1993-2002), memory not in it; the bucket-only refit is the control, since it gets the same two extra parameters.
Same gates and same shared-permutation significance test as the search. -> stageB/memory_search/refit/report.csv
usage: python east_memory_refit.py [--workers 10]
"""
import json, time, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import differential_evolution
import emu_core as E, east_booster as EB, east_memory_search as M

OUTD = M.OUTD/"refit"; OUTD.mkdir(exist_ok=True)
BK = [("Sf_h", 0.3, 12.0, True), ("exp_h", 1.0, 8.0, False)]
M.STRUCT.update({"control_refit": (0, [("f", 0.0, 0.0, False)]+BK), "cascade_refit": (1, M.STRUCT["cascade"][1]+BK), "thresh_cascade_refit": (3, M.STRUCT["thresh_cascade"][1]+BK),
                 "sr_capture_refit": (6, M.STRUCT["sr_capture"][1]+BK), "linear_refit": (0, M.STRUCT["linear"][1]+BK)})
NAMES = ["control_refit", "linear_refit", "cascade_refit", "thresh_cascade_refit", "sr_capture_refit"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10); a = ap.parse_args(); rows, res = [], {}
    with Pool(a.workers) as pool:
        for name in NAMES:
            k = len(M.STRUCT[name][1]); fits = []
            for seed in (1, 2, 3):
                f = OUTD/f"fit_{name}_s{seed}.json"
                if not f.exists():
                    fh = OUTD/f"fit_{name}_s{seed}.jsonl"; fh.unlink(missing_ok=True)
                    r = differential_evolution(M.Objective(name, str(fh)), bounds=[(0.0, 1.0)]*k, popsize=12, maxiter=50, tol=1e-4, seed=seed, workers=pool.map, updating="deferred", polish=False)
                    f.write_text(json.dumps(dict(params=M.params_of(name, r.x), objective=float(r.fun)))); print(name, seed, round(r.fun, 4), flush=True)
                fits.append(json.loads(f.read_text()))
            b = min(fits, key=lambda x: x["objective"]); q, sd = M.simulate(name, b["params"]); res[name] = M.resid_series(q)
            tr = [json.loads(l) for s in (1, 2, 3) for l in open(OUTD/f"fit_{name}_s{s}.jsonl")]; ok = [t for t in tr if t["objective"] <= 1.02*b["objective"]]
            rows.append(dict(model=name, paper_obj=b["objective"], store_mm=float(sd["1986-10-01":"2024-09-30"].mean()*1000), next_fall_pct=M.pulse_response(name, b["params"], q), near_rho_lo=min(t["rho1_87"] for t in ok), near_rho_hi=max(t["rho1_87"] for t in ok), params=json.dumps({k: round(v, 3) for k, v in b["params"].items()}), **EB.score(q)))
    R = pd.DataFrame(rows).set_index("model"); X = pd.concat(res, axis=1).dropna(); rng = np.random.default_rng(7); act = {k: E.lag_r(X[k], 1) for k in X}; null = []
    for _ in range(5000):
        i = rng.permutation(len(X)); null.append(max(E.lag_r(pd.Series(X[k].values[i], index=X.index), 1) for k in X))
    R["p_family"] = pd.Series({k: (np.sum(np.array(null) >= v)+1)/5001 for k, v in act.items()}); R.to_csv(OUTD/"report.csv"); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80)
    print("REPORT"); print(R[["paper_obj", "KGE_eval", "NSE_eval", "recess_d", "winter_tau_d", "fallmin_bias", "rho1_87", "p_family", "rho1_09", "near_rho_lo", "near_rho_hi", "next_fall_pct", "store_mm", "params"]].round(3).to_string())


if __name__ == "__main__":
    main()
