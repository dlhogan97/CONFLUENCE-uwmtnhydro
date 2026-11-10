"""
Does initial soil moisture survive the one-year spin-up? (2026-11-05) Round-5 best (g35_suite5_cal.py) rerun with every soil layer's initial matric head set
to PSI (m). Note: SUMMA ignores coldState mLayerVolFracLiq for soil and recomputes it from mLayerMatricHead (check_icond.f90 updateSoil), so the runner's
_wet_soil (sets VolFracLiq = fieldCapacity) does nothing. The real default is psi = -1 m -> theta ~0.39 (theta_sat 0.45, vGn alpha -1, n 1.3).
Run starts 2011-10-01, analysis from 2012-10-01.  psi 0 = saturated (theta 0.45); psi -100 = dry (theta ~0.15).
-> OUT/suite5_cal/init/psi_<PSI>/cal_day.nc
usage: python g35_suite5_init.py <PSI>
"""
import sys, shutil, json, numpy as np, netCDF4 as nc
from pathlib import Path
import g35_suite5_cal as C5
import g35_suite4_patch_fate as PF


def runner(psi):
    class R(PF.with_wb(C5.Runner)):
        def _wet_soil(self, work):
            super()._wet_soil(work)
            with nc.Dataset(work/"coldState.nc", "a") as d:
                ns = np.array(d["nSnow"][:]).ravel().astype(int); assert (ns == 0).all(), "cold state has snow layers"
                d["mLayerMatricHead"][:] = psi
    return R


def main():
    psi = float(sys.argv[1]); t = json.loads((C5.OUT/"best"/"best_obj"/"trial.json").read_text())
    r = runner(psi)(work_root=str(C5.OUT/f"run_init_{psi:g}"), **C5.S.CFG); r.variant = C5.F.VARIANTS["G"]; r.soil = C5.physical(t["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(psi, "rc", proc.returncode, proc.stderr[-300:], flush=True)
    keep = C5.OUT/"init"/f"psi_{psi:g}"; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
