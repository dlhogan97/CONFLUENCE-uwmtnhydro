"""
Snow-surface energy balance for the round-5 best (2026-11-06). The Tuolumne snowpack is far too cold: Jan snow-base -8 to -13 C above 2900 m, whole pack -11
to -17 C at 6 am, ~7 C diurnal swing (g35_suite5_ground.py). Ground heat alone does not ripen the base, so the cold enters from the top. The user suspects longwave.
Same run as g35_suite5_ground.py "zero" (round-5 control, zeroFlux, funcSoilWet), with hourly instantaneous output of:
  forcing        airtemp, LWRadAtm, SWRadAtm, windspd, spechum
  ground surface scalarSurfaceTemp, scalarLWRadUbound2Ground (sky LW absorbed), scalarLWRadCanopy2Ground (canopy LW absorbed), scalarLWRadGround (emitted),
                 scalarLWNetGround, scalarGroundAbsorbedSolar, scalarSenHeatGround, scalarLatHeatGround, scalarGroundAdvectiveHeatFlux, scalarGroundNetNrgFlux,
                 scalarGroundSnowFraction
  canopy         scalarCanopyTemp, scalarCanopyEmissivity
  snow           scalarSWE, scalarSnowDepth, nSnow, mLayerTemp
-> OUT/suite5_cal/energy/{cal_day.nc, cal_timestep.nc}
usage: python g35_suite5_energy.py
"""
import sys, shutil, numpy as np
from pathlib import Path
import g35_suite5_cal as C5
import g35_suite5_ground as GR
import json

EB = ["airtemp", "LWRadAtm", "SWRadAtm", "windspd", "spechum",
      "scalarSurfaceTemp", "scalarLWRadUbound2Ground", "scalarLWRadCanopy2Ground", "scalarLWRadGround", "scalarLWNetGround", "scalarGroundAbsorbedSolar",
      "scalarSenHeatGround", "scalarLatHeatGround", "scalarGroundAdvectiveHeatFlux", "scalarGroundNetNrgFlux", "scalarGroundSnowFraction",
      "scalarCanopyTemp", "scalarCanopyEmissivity", "scalarSWE", "scalarSnowDepth", "nSnow", "mLayerTemp"]
KEEP = C5.OUT/"energy"


class R(GR.runner(None)):
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing); oc = work/"outputControl.txt"     # columns: freq | sum | instant | mean ...
        have = {l.split("|")[0].strip() for l in oc.read_text().splitlines() if "|" in l}     # a var listed twice breaks SUMMA: scalarSWE stays daily,
        oc.write_text(oc.read_text() + "".join(f"{v} | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n" for v in EB if v not in have))   # nSnow/mLayerTemp already hourly


def main():
    t = json.loads((C5.OUT/"best"/"best_obj"/"trial.json").read_text())
    r = R(work_root=str(C5.OUT/"run_energy"), **C5.S.CFG); r.variant = C5.F.VARIANTS["G"]; r.soil = C5.physical(t["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print("rc", proc.returncode, proc.stderr[-300:], flush=True)
    if proc.returncode != 0: print(proc.stdout[-1500:]); sys.exit(1)
    shutil.rmtree(KEEP, ignore_errors=True); shutil.copytree(w/"out", KEEP); shutil.rmtree(r.work_root, ignore_errors=True); print(KEEP)


if __name__ == "__main__":
    main()
