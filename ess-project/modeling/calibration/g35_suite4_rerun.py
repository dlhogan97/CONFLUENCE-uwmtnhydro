"""
Rerun round-4 SUMMA trials (g35_suite4_cal.py) to keep their daily output for figures (2026-11-04): the best-objective trial and the feasible trial with the
highest Apr-Jul KGE. -> OUT/suite4_cal/best/{best_obj,best_kge}/cal_day.nc + trial.json
usage: python g35_suite4_rerun.py
"""
import json, shutil, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import g35_suite4_cal as C4

BEST = C4.OUT/"best"


def pick():
    T = [json.loads(l) for l in open(C4.OUT/"trials.jsonl")]; F = [t for t in T if t["objective"] < C4.FAIL]
    return {"best_obj": min(T, key=lambda t: t["objective"]), "best_kge": max(F, key=lambda t: t["diagnostics"]["KGE_AprJul"])}


def rerun(item):
    name, t = item
    r = C4.Runner(work_root=str(C4.OUT/f"rerun_{name}"), **C4.S.CFG); r.variant = C4.F.VARIANTS["G"]; r.soil = C4.physical(t["x"]); w = Path(r.work_root)/"keep"
    _, proc = r.run_summa(np.zeros(len(r.specs)), w); keep = BEST/name
    shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    (keep/"trial.json").write_text(json.dumps(t, default=float, indent=1))
    return name, proc.returncode, t["objective"]


def main():
    with ProcessPoolExecutor(2) as ex:
        for name, rc, obj in ex.map(rerun, pick().items()): print(f"[suite4 rerun] {name}: rc {rc}, objective {obj:.3f}", flush=True)


if __name__ == "__main__":
    main()
