"""
Rerun the round-2 calibration candidates and KEEP their output, since g35_suite2_cal.py deletes each trial's run (2026-10-30). No trial passed all the rejection rules, so:
  best_obj     result.json, the search's least-rejected trial (KGE just under 0.75, melt-order error above its limit)
  best_kge     highest KGE of all trials (fails melt order)
  best_order   highest KGE among trials inside the ASO SWE and melt-order limits (fails KGE)
Output -> OUT/suite2_cal/best/<name>/out/cal_day.nc, selected trials -> OUT/suite2_cal/best/selected.json.
usage: python g35_suite2_rerun.py
"""
import json, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import g35_suite2_cal as C

BEST = C.OUT/"best"; BEST.mkdir(exist_ok=True)


def run(item):
    name, x = item; soil = C.physical(x); r = C.Runner(work_root=str(BEST/name), **C.S.CFG); r.variant = C.F.VARIANTS["G"]; r.soil = soil
    _, proc = r.run_summa(np.zeros(len(r.specs)), Path(r.work_root)/"keep")
    return name, proc.returncode


def main():
    tr = [json.loads(l) for l in open(C.OUT/"trials.jsonl")]; tr = [t for t in tr if "diagnostics" in t]; kge = lambda t: t["diagnostics"]["KGE"]
    sel = dict(best_obj=min(tr, key=lambda t: t["objective"]), best_kge=max(tr, key=kge),
               best_order=max([t for t in tr if t["diagnostics"]["aso_swe_nmae"] <= C.ASO_MAX and t["diagnostics"]["aso_melt_order"] <= C.MO_MAX], key=kge))
    (BEST/"selected.json").write_text(json.dumps(sel, default=float, indent=1))
    with ProcessPoolExecutor(3) as ex:
        for name, rc in ex.map(run, [(k, v["x"]) for k, v in sel.items()]): print(name, "returncode", rc, flush=True)


if __name__ == "__main__":
    main()
