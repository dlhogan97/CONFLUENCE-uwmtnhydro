"""
East structural test (2026-11-02). Hypothesis (user): without a deep store the model cannot match recession behaviour, low-flow level AND low-flow memory together; with one it can.
Both arms are calibrated to the SAME objective with the same budget, so a pass for the store and a fail for the control is the evidence.
  arms        control_refit (HRU bucket scale + exponent free, no store) vs added structures with the buckets also free (east_memory_refit.STRUCT).
  objective   J = (5 x paper objective [WY1993-2002, 5 parts] + summer + winter + fall + memory)/9, each added term clipped at 3:
                summer   |ln(median Aug-Sep recession tau / obs)| / ln 1.25                    WY1987-2024, obs 68.9 d
                winter   |ln(median Nov 15-Mar 15 recession tau / obs)| / ln 1.25              WY1987-2024, obs 344 d
                fall     |Aug-Nov 7-d minimum mean bias| / 10%                                 WY1987-2024
                memory   |rho1 - obs| / 0.10   rho1 = Spearman lag-1 of ln fall-min residual on ln Apr-Jul peak, WY1987-2024, obs +0.383
  held out    hydrograph skill on the non-calibration years (KGE_eval, NSE_eval). Memory has no hold-out: observed rho1 is +0.14 in WY1987-2008 and +0.54
              in WY2009-2024, so either half is a different target. Both halves are reported.
-> stageB/memory_search/joint/report.csv
usage: python east_memory_joint.py [--workers 20]
"""
import json, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import differential_evolution
import emu_core as E, east_booster as EB, east_ctl_treat_cal_v2 as V, east_memory_search as M, east_memory_refit as RF

OUTD = M.OUTD/"joint"; OUTD.mkdir(exist_ok=True)
NAMES = ["control_refit", "linear_refit", "cascade_refit", "thresh_cascade_refit", "sr_capture_refit"]
OB = dict(summer=float(V.recession_taus(EB.obs, 1987, 2024).median()), winter=float(EB.winter_taus(EB.obs, 1987, 2024).median()), rho1=float(M.rho1(EB.obs)))
LF0 = float(np.exp(E.series(EB.obs, 1987, 2024)[0]).mean())


def terms(q):
    t = dict(summer=abs(np.log(V.recession_taus(q, 1987, 2024).median()/OB["summer"]))/np.log(1.25), winter=abs(np.log(EB.winter_taus(q, 1987, 2024).median()/OB["winter"]))/np.log(1.25),
             fall=abs(100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/LF0-1))/10.0, memory=abs(M.rho1(q)-OB["rho1"])/0.10)
    return {k: float(np.clip(v, 0, 3)) if np.isfinite(v) else 3.0 for k, v in t.items()}


def trial(args):
    name, u, fh = args; p = M.params_of(name, u)
    try:
        q, _ = M.simulate(name, p); po = M.paper_objective(q); t = terms(q); J = (5*po+sum(t.values()))/9.0
    except Exception as e:
        J, po, t = 10.0, np.nan, dict(reason=str(e))
    with open(fh, "a") as g: g.write(json.dumps(dict(params=p, objective=J, paper_obj=po, **t))+"\n")
    return J


class Obj:
    def __init__(self, name, fh): self.name, self.fh = name, fh
    def __call__(self, u): return trial((self.name, u, self.fh))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=20); a = ap.parse_args(); rows = []
    print("observed targets", {k: round(v, 3) for k, v in OB.items()}, flush=True)
    with Pool(a.workers) as pool:
        for name in NAMES:
            k = len(M.STRUCT[name][1]); fits = []
            for seed in (1, 2, 3):
                f = OUTD/f"fit_{name}_s{seed}.json"
                if not f.exists():
                    fh = OUTD/f"fit_{name}_s{seed}.jsonl"; fh.unlink(missing_ok=True)
                    r = differential_evolution(Obj(name, str(fh)), bounds=[(0.0, 1.0)]*k, popsize=12, maxiter=60, tol=1e-4, seed=seed, workers=pool.map, updating="deferred", polish=False)
                    f.write_text(json.dumps(dict(params=M.params_of(name, r.x), objective=float(r.fun)))); print(name, seed, round(r.fun, 4), flush=True)
                fits.append(json.loads(f.read_text()))
            b = min(fits, key=lambda x: x["objective"]); q, sd = M.simulate(name, b["params"]); s = EB.score(q)
            tr = pd.DataFrame([json.loads(l) for sd_ in (1, 2, 3) for l in open(OUTD/f"fit_{name}_s{sd_}.jsonl")]); ok = tr[tr.objective <= 1.05*b["objective"]]
            rows.append(dict(model=name, J=b["objective"], seed_spread=max(x["objective"] for x in fits)-b["objective"], paper_obj=M.paper_objective(q), **{f"term_{k}": v for k, v in terms(q).items()},
                             rho1_87_08=float(E.lag_r(E.resid(*E.series(q, 1987, 2008)), 1)), store_mm=float(sd["1986-10-01":"2024-09-30"].mean()*1000), next_fall_pct=M.pulse_response(name, b["params"], q),
                             near_n=len(ok), near_rho_lo=ok.memory.min(), near_rho_hi=ok.memory.max(), params=json.dumps({k: round(v, 4) for k, v in b["params"].items()}), **s))
    R = pd.DataFrame(rows).set_index("model"); R.to_csv(OUTD/"report.csv"); pd.set_option("display.width", 280); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90)
    print("REPORT"); print(R[["J", "seed_spread", "paper_obj", "term_summer", "term_winter", "term_fall", "term_memory", "KGE_eval", "NSE_eval", "recess_d", "winter_tau_d", "fallmin_bias", "rho1_87", "p87", "rho1_87_08", "rho1_09", "next_fall_pct", "store_mm"]].round(3).to_string())
    print(R[["params"]].to_string())


if __name__ == "__main__":
    main()
