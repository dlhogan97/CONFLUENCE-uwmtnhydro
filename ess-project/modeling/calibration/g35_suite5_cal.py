"""
Tuolumne G35 calibration, round 5 (2026-11-04), SUMMA. Round 4 (g35_suite4_cal.py) plus snow liquid-water drainage, set with the user 2026-11-04.
Why: round-4 best misses the big Dec-Mar storm flows. Mid-elevation snow holds the storm rain as liquid (snowLiqFlx.f90: no drainage below
Fcapil*(1-volFracIce), which is 100-160 mm in deep fresh snow) and releases it in Jun-Jul (g35_suite4_storms.py).
Faster drainage (lower Fcapil, higher k_snow, lower mw_exp) fixes the winter flow but lowers SWE at the already-short mid bands. So precip_scale (pinned at
0.95 in round 4 by the Jun-Jul excess) is free to rise with it. The user judged tempCritRain 267.65 an overcorrection: range 268.5-271.0 K (rain ABOVE it).
  objective  as round 4: Apr-Jul KGE + Apr-May/Jun-Jul volume + Apr-Jul centroid (C4.score)
  limits     as round 4 (ASO nMAE <= 0.556, band melt order <= 0.10, ET + sublimation 350-450 mm) PLUS Aug-Sep bias >= -60 % (round-4 best -55 %; the user wants
             the late season a bit better, and every drainage gain so far cost Aug-Sep)
Free (13): the 10 round-4 parameters (tempCritRain narrowed to [268.5, 271.0], precip_scale [0.95, 1.25]) + Fcapil [0.02, 0.06] (all HRUs)
           log_k_snow [-2.3, -1.3] (0.005-0.05 m/s)  mw_exp [1.5, 3.5]
Seed: round-4 best with tempCritRain 269.15, Fcapil 0.03 (the sweep run that stayed feasible: obj 4.78, Aug-Sep -68 %, which this guard rejects).
Every trial -> OUT/suite5_cal/trials.jsonl
usage: python g35_suite5_cal.py [--workers 22] [--popsize 2] [--maxiter 14]      |      python g35_suite5_cal.py --score <cal_day.nc>
"""
import json, uuid, time, shutil, argparse, numpy as np, netCDF4 as nc
from pathlib import Path
from scipy.optimize import differential_evolution
import g35_suite4_cal as C4
import deepstore_build as DB
S, F, C3 = C4.S, C4.F, C4.C3

OUT = F.OUT/"suite5_cal"; OUT.mkdir(exist_ok=True)
B4 = dict((n, (lo, hi)) for n, lo, hi in C4.BOUNDS); B4["tempCritRain"] = (268.5, 271.0); B4["precip_scale"] = (0.95, 1.25)
BOUNDS = [(n, *B4[n]) for n, _, _ in C4.BOUNDS] + [("Fcapil", 0.02, 0.06), ("log_k_snow", -2.3, -1.3), ("mw_exp", 1.5, 3.5)]
AUGSEP_MIN = -60.0
SEED = json.loads((C4.OUT/"best"/"best_obj"/"trial.json").read_text())["x"][:10] + [0.03, float(np.log10(0.015)), 3.0]
SEED[[b[0] for b in BOUNDS].index("tempCritRain")] = 269.15
FAIL = C4.FAIL


def physical(x):
    p = C3.physical(list(x[:10])); p.update(Fcapil=float(x[10]), k_snow=float(10**x[11]), mw_exp=float(x[12])); return p


class Runner(C4.Runner):
    def _subset(self, work):
        super()._subset(work)
        with nc.Dataset(work/"trialParams.nc") as t: n = len(np.array(t["hruId"][:]).ravel())
        for k in ("Fcapil", "k_snow", "mw_exp"): DB.set_trial_param(work, k, np.full(n, self.soil[k]))


def score(out, obs):
    obj, parts, diag = C4.score(out, obs)
    v = max(0.0, AUGSEP_MIN - diag["AugSep_bias"])/5
    if v > 0: obj = (obj if obj >= FAIL else FAIL) + v
    return obj, parts, diag


def evaluate(x):
    soil = physical(x); t0 = time.time(); r = Runner(work_root=str(OUT/f"w_{uuid.uuid4().hex[:8]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.soil = soil
    w = Path(r.work_root)/"keep"
    try:
        _, proc = r.run_summa(np.zeros(len(r.specs)), w); out = w/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): obj, rec = FAIL + 50, dict(reason="summa_failed", stderr=proc.stderr[-300:])
        else: obj, parts, diag = score(out, C4.obs_mm()); diag.pop("band_order", None); rec = dict(parts=parts, diagnostics=diag)
    except Exception as e:
        obj, rec = FAIL + 50, dict(reason=f"{type(e).__name__}:{e}")
    finally:
        shutil.rmtree(r.work_root, ignore_errors=True)
    with open(OUT/"trials.jsonl", "a") as fh: fh.write(json.dumps(dict(x=list(map(float, x)), params=soil, objective=obj, eval_s=time.time() - t0, **rec), default=float) + "\n")
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=22); ap.add_argument("--popsize", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=14); ap.add_argument("--seed", type=int, default=15); ap.add_argument("--score"); a = ap.parse_args()
    if a.score:
        obj, parts, diag = score(a.score, C4.obs_mm()); print(obj, parts); print(json.dumps(diag, indent=1, default=float)); return
    C4.obs_mm(); hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUT/"history.json").write_text(json.dumps(hist))
        print(f"[g35 suite5] gen {len(hist):2d}: best {hist[-1]:.4f}", flush=True)
    res = differential_evolution(evaluate, bounds=[(lo, hi) for _, lo, hi in BOUNDS], popsize=a.popsize, init="latinhypercube", x0=SEED, maxiter=a.maxiter,
                                 workers=a.workers, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUT/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), params=physical(res.x), objective=float(res.fun), nfev=int(res.nfev))))
    print(f"[g35 suite5] DONE {res.fun:.4f} nfev {res.nfev}", flush=True)


if __name__ == "__main__":
    main()
