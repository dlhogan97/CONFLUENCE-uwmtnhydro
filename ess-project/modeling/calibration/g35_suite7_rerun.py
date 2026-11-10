"""
Rerun the round-7 best (suite7_cal/result.json) with the same physics plus daily soil water / rain+melt / ET / runoff output, so it compares in
g35_lever_table.py (2026-11-09).  -> OUT/suite7_cal/best/{cal_day.nc, score.json}
usage: python g35_suite7_rerun.py
"""
import json, shutil, numpy as np
from pathlib import Path
import g35_suite7_cal as C7
EXTRA = ["scalarTotalSoilWat", "scalarRainPlusMelt", "scalarTotalET", "scalarTotalRunoff", "scalarSurfaceRunoff", "scalarSoilBaseflow", "scalarSWE"]


class R(C7.Runner):
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing); oc = work/"outputControl.txt"; have = {l.split("|")[0].strip() for l in oc.read_text().splitlines() if "|" in l}
        oc.write_text(oc.read_text() + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in EXTRA if v not in have))


def main():
    res = json.loads((C7.OUT/"result.json").read_text()); keep = C7.OUT/"best"
    r = R(work_root=str(C7.OUT/"run_best"), **C7.S.CFG); r.variant = C7.F.VARIANTS["G"]; r.soil = C7.physical(res["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print("rc", proc.returncode, proc.stderr[-300:], flush=True)
    if proc.returncode != 0: print(proc.stdout[-1500:]); raise SystemExit(1)
    shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    obj, parts, diag = C7.score(keep/"cal_day.nc", C7.C4.obs_mm()); diag.pop("band_order", None)
    (keep/"score.json").write_text(json.dumps(dict(x=res["x"], params=res["params"], objective=obj, obj_round5=None, parts=parts, diagnostics=diag), indent=1, default=float))
    print("objective", obj, "(calibration", res["objective"], ")", flush=True)


if __name__ == "__main__":
    main()
