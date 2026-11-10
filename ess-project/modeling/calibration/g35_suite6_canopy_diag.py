"""
One-year hourly diagnostic for the canopy SW intercomparison (2026-11-06). Same parameter set and physics as g35_suite6_canopy.py, WY2012 only
(sim 2011-10-01 .. 2012-09-30), hourly instantaneous: SWRadAtm, scalarCosZenith, scalarExposedLAI, scalarExposedSAI, scalarBelowCanopySolar,
scalarCanopyAbsorbedSolar, scalarGroundAbsorbedSolar, scalarSnowDepth, scalarCanopyTemp.
Why: realized daily transmission did not match either scheme's formula (Beer's too high; UEB below Beer's), so check what VAI and cosZ the model actually used.
-> OUT/suite6_cal/canopy/diag_<scheme>/
usage: python g35_suite6_canopy_diag.py BeersLaw|UEB_2stream
"""
import sys, shutil, numpy as np
from pathlib import Path
import g35_suite6_canopy as G
C6 = G.C6
HV = ["SWRadAtm", "scalarCosZenith", "scalarExposedLAI", "scalarExposedSAI", "scalarBelowCanopySolar", "scalarCanopyAbsorbedSolar",
      "scalarGroundAbsorbedSolar", "scalarSnowDepth", "scalarCanopyTemp"]


def main():
    scheme = sys.argv[1]; t = G.best_trial(); keep = G.ROOT/f"diag_{scheme}"
    class R(G.runner(scheme)):
        def _write_filemanager(self, work, prefix, forcing=None):
            super()._write_filemanager(work, prefix, forcing); oc = work/"outputControl.txt"; L = oc.read_text().splitlines()
            L = [l for l in L if l.split("|")[0].strip() not in HV]                      # drop daily versions, re-add hourly instantaneous
            oc.write_text("\n".join(L) + "\n" + "".join(f"{v} | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n" for v in HV))
    cfg = dict(C6.S.CFG, sim_end="2012-09-30", analysis_start="2011-10-01")
    r = R(work_root=str(C6.OUT/f"run_cdiag_{scheme}"), **cfg); r.variant = C6.F.VARIANTS["G"]; r.soil = C6.physical(t["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(scheme, "rc", proc.returncode, proc.stderr[-300:], flush=True)
    if proc.returncode != 0: print(proc.stdout[-1500:]); sys.exit(1)
    shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True); print(keep)


if __name__ == "__main__":
    main()
