"""
Rerun the round-5 best trial (g35_suite5_cal.py) with daily water-balance output (g35_suite4_patch_fate.WBV) for figures and the winter/storm checks
(2026-11-05). -> OUT/suite5_cal/best/best_obj/{cal_day.nc, trial.json, score.json}
usage: python g35_suite5_rerun.py
"""
import json, shutil, numpy as np
from pathlib import Path
import g35_suite5_cal as C5
import g35_suite4_patch_fate as PF


def main():
    T = [json.loads(l) for l in open(C5.OUT/"trials.jsonl")]; t = min(T, key=lambda t: t["objective"])
    r = PF.with_wb(C5.Runner)(work_root=str(C5.OUT/"rerun_best"), **C5.S.CFG); r.variant = C5.F.VARIANTS["G"]; r.soil = C5.physical(t["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print("rc", proc.returncode, proc.stderr[-300:], flush=True)
    keep = C5.OUT/"best"/"best_obj"; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    (keep/"trial.json").write_text(json.dumps(t, default=float, indent=1))
    obj, parts, diag = C5.score(keep/"cal_day.nc", C5.C4.obs_mm()); diag.pop("band_order", None)
    (keep/"score.json").write_text(json.dumps(dict(objective=obj, parts=parts, diagnostics=diag), indent=1, default=float))
    print("rerun objective", obj, "(trial", t["objective"], ")", flush=True)


if __name__ == "__main__":
    main()
