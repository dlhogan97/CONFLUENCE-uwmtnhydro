"""
East joint calibration v2 (2026-11-02): v1 (east_memory_joint.py) let the stores carry too much memory. Diagnosis of the v1 best fits:
  raw (not peak-adjusted) fall-min ACF lag 1: obs +0.17, sr_capture +0.50, cascade +0.54, thresh_cascade +0.50, control +0.10
  sd of ln fall minimum:                      obs 0.295, sr_capture 0.213 (too smooth), cascade 0.234, thresh_cascade 0.307, control 0.304
Observed lag 3-5 negatives follow the snow forcing (peak-flow ACF lag 4 -0.26), so they are NOT targeted: the test is whether a model that is not over-damped sees them.
Objective = v1 terms (paper x5, summer and winter recession, fall level, peak-adjusted lag-1 memory) PLUS one, divided by 10:
  variability |sd(ln fall min) sim/obs - 1| / 0.10      (keeps the year-to-year range)
  The raw (unadjusted) lag-1 is reported, not scored: it mixes memory with same-year snow, so the user judged it a superficial target. A first launch that scored
  it (stopped 2026-11-02, joint2/) was driving the cascade to drop memory entirely.
Same arms, bounds, budget and seeds as v1. -> stageB/memory_search/joint3/report.csv
usage: python east_memory_joint2.py [--workers 6]
"""
import json, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import differential_evolution
import emu_core as E, east_booster as EB, east_memory_search as M, east_memory_refit as RF, east_memory_joint as J1

OUTD = M.OUTD/"joint3"; OUTD.mkdir(exist_ok=True)
NAMES = ["control_refit", "cascade_refit", "thresh_cascade_refit", "sr_capture_refit"]
_lf, _ = E.series(EB.obs["1986-10-01":"2024-12-31"], 1987, 2024); OB = dict(raw1=float(E.lag_r(_lf, 1)), sd=float(_lf.std()))
acf = lambda s: [float(E.lag_r(s, l)) for l in range(1, 6)]


def terms(q):
    t = J1.terms(q); lf, pk = E.series(q, 1987, 2024)
    t["variability"] = float(np.clip(abs(lf.std()/OB["sd"]-1)/0.10, 0, 3))
    return t


def trial(args):
    name, u, fh = args; p = M.params_of(name, u)
    try:
        q, _ = M.simulate(name, p); po = M.paper_objective(q); t = terms(q); Jv = (5*po+sum(t.values()))/10.0
    except Exception as e:
        Jv, po, t = 10.0, np.nan, dict(reason=str(e))
    with open(fh, "a") as g: g.write(json.dumps(dict(params=p, objective=Jv, paper_obj=po, **t))+"\n")
    return Jv


class Obj:
    def __init__(self, name, fh): self.name, self.fh = name, fh
    def __call__(self, u): return trial((self.name, u, self.fh))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args(); rows = []
    print("targets", {**{k: round(v, 3) for k, v in J1.OB.items()}, **{k: round(v, 3) for k, v in OB.items()}}, flush=True)
    obs = EB.obs["1986-10-01":"2024-12-31"]; lf, pk = E.series(obs, 1987, 2024)
    rows.append(dict(model="observed", resid_acf=np.round(acf(E.resid(lf, pk)), 2).tolist(), raw_acf=np.round(acf(lf), 2).tolist(), sd_lnmin=lf.std()))
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
            b = min(fits, key=lambda x: x["objective"]); q, sd = M.simulate(name, b["params"]); q = q["1986-10-01":"2024-12-31"]; s = EB.score(q); lf, pk = E.series(q, 1987, 2024)
            rows.append(dict(model=name, J=b["objective"], seed_spread=max(x["objective"] for x in fits)-b["objective"], paper_obj=M.paper_objective(q), **{f"term_{k}": v for k, v in terms(q).items()},
                             resid_acf=np.round(acf(E.resid(lf, pk)), 2).tolist(), raw_acf=np.round(acf(lf), 2).tolist(), sd_lnmin=lf.std(),
                             rho1_87_08=float(E.lag_r(E.resid(*E.series(q, 1987, 2008)), 1)), store_mm=float(sd["1986-10-01":"2024-09-30"].mean()*1000),
                             params=json.dumps({k: round(v, 4) for k, v in b["params"].items()}), **s))
    R = pd.DataFrame(rows).set_index("model"); R.to_csv(OUTD/"report.csv"); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 100)
    print("REPORT"); print(R[["J", "paper_obj", "KGE_eval", "NSE_eval", "recess_d", "winter_tau_d", "fallmin_bias", "rho1_87", "rho1_87_08", "rho1_09", "sd_lnmin", "store_mm"]].round(3).to_string())
    print(R[["resid_acf", "raw_acf", "params"]].to_string())


if __name__ == "__main__":
    main()
