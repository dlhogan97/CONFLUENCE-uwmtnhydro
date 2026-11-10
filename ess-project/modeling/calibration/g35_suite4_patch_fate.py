"""
Where does the snowpatch meltwater go? (2026-11-04) Reruns the round-4 best with and without the anchor_m080 snowpatch (g35_suite4_snowpatch.py) with
per-HRU water-balance output, so the patch-minus-control difference can be followed down HRU 2's drainage chain (2 -> 7 -> 12 -> 17 -> 22 -> 26 -> stream).
-> OUT/suite4_cal/fate/{nopatch,patch}/cal_day.nc
usage: python g35_suite4_patch_fate.py {nopatch|patch}
"""
import sys, json, shutil, numpy as np
from pathlib import Path
import g35_suite4_cal as C4
import g35_suite4_snowpatch as SP

FATE = C4.OUT/"fate"
WBV = ["scalarRainPlusMelt", "scalarTotalRunoff", "scalarSurfaceRunoff", "scalarSoilBaseflow", "scalarExfiltration", "scalarTotalET",
       "scalarTotalSoilWat", "scalarSWE", "scalarSnowSublimation", "scalarCanopySublimation", "scalarSnowfall", "scalarRainfall", "scalarCanopyWat"]


def with_wb(base):
    class R(base):
        def _write_filemanager(self, work, prefix, forcing=None):
            super()._write_filemanager(work, prefix, forcing); oc = work/"outputControl.txt"; have = oc.read_text()
            oc.write_text(have + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in WBV if f"{v} |" not in have))
    return R


def main():
    which = sys.argv[1]; t = json.loads((C4.OUT/"best"/"best_obj"/"trial.json").read_text())
    r = with_wb(SP.Runner if which == "patch" else C4.Runner)(work_root=str(FATE/f"run_{which}"), **C4.S.CFG)
    r.variant = C4.F.VARIANTS["G"]; r.soil = C4.physical(t["x"]); w = Path(r.work_root)/"keep"
    _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(which, "rc", proc.returncode, proc.stderr[-300:], flush=True)
    keep = FATE/which; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
