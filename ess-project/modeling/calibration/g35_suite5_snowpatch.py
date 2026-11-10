"""
Round-5 best (g35_suite5_cal.py) + the anchor_m080 snowpatch (2026-11-05). Same graft as g35_suite4_snowpatch.py (10 depth bins of HRU 2, placed right after
HRU 2 so their lateral outflow reaches HRU 7; drift on snowfall; bin NIR 0.70, tol 0.05; bins seeded from the 44-yr anchor run), on top of the round-5
runner (snow liquid-water parameters, which the bin clones inherit from HRU 2's trialParams row). Daily water-balance output (g35_suite4_patch_fate.WBV).
The user stopped chasing the winter AR storm flows (likely forcing error) on 2026-11-05.
-> OUT/suite5_cal/best/best_obj_patch/{cal_day.nc (37 HRUs), cal_day_28.nc (bins folded into HRU 2), score.json}
usage: python g35_suite5_snowpatch.py
"""
import json, shutil, numpy as np
from pathlib import Path
import g35_suite5_cal as C5
import g35_suite4_snowpatch as SP
import g35_suite4_patch_fate as PF

KEEP = C5.OUT/"best"/"best_obj_patch"


class Runner(SP.Runner, C5.Runner):                       # MRO: snowpatch graft -> round-5 snow params -> round-4 runner
    pass


def main():
    t = json.loads((C5.OUT/"best"/"best_obj"/"trial.json").read_text())
    r = PF.with_wb(Runner)(work_root=str(C5.OUT/"rerun_patch"), **C5.S.CFG); r.variant = C5.F.VARIANTS["G"]; r.soil = C5.physical(t["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print("rc", proc.returncode, proc.stderr[-400:], flush=True)
    shutil.rmtree(KEEP, ignore_errors=True); shutil.copytree(w/"out", KEEP); shutil.rmtree(r.work_root, ignore_errors=True)
    SP.fold(KEEP/"cal_day.nc", KEEP/"cal_day_28.nc")
    obj, parts, diag = C5.score(KEEP/"cal_day_28.nc", C5.C4.obs_mm()); diag.pop("band_order", None)
    (KEEP/"score.json").write_text(json.dumps(dict(objective=obj, parts=parts, diagnostics=diag, audit=r.audit, trial=t["x"]), indent=1, default=float))
    print(json.dumps(dict(objective=obj, audit=r.audit, **{k: diag[k] for k in ("KGE_AprJul", "AprMay_bias", "JunJul_bias", "AugSep_bias", "OctNov_bias",
          "vapour_mm", "aso_swe_nmae", "band_order_mae")}), indent=1, default=float))


if __name__ == "__main__":
    main()
