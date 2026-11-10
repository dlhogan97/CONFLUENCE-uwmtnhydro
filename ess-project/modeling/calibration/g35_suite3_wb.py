"""
Rerun the round-3 feasible trial (g35_suite3_cal.py, objective 2.471) with daily water-balance and forcing output, as training data for the Tuolumne
snow / below-snow emulators (2026-11-03). Forcing is written as SUMMA sees it, after the runner's precip, temperature and longwave adjustments.
-> OUT/suite3_cal/wb/cal_day.nc. With "dark", the same soil with the G35 albedo floors lowered by
DARK (earlier, faster melt), a hold-out for the below-snow emulator -> OUT/suite3_cal/wb_dark/cal_day.nc. (Variant names like "ref" do NOT work:
full_basin_G35.Runner overwrites the albedo after the variant is applied.)
usage: python g35_suite3_wb.py [dark]
"""
import sys, json, shutil, numpy as np
from pathlib import Path
import g35_suite3_cal as C
import g35_melt_vs_runoff as WB
import full_basin_G35 as G35
import deepstore_build as DB

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "G"
KEEP = C.OUT/("wb" if VARIANT == "G" else f"wb_{VARIANT}")
DARK = 0.08
FORCV = ["airtemp", "pptrate", "SWRadAtm", "LWRadAtm", "spechum", "windspd", "airpres"]
EXTRA = ["scalarSnowSublimation", "scalarCanopySublimation", "scalarSnowAlbedo", "scalarGroundEvaporation", "scalarCanopyTranspiration", "scalarCanopyEvaporation", "scalarSnowDepth"]


class Runner(C.Runner):
    def _apply_params(self, work, params):
        super()._apply_params(work, params)
        if VARIANT == "dark":
            z, V = self.elev, G35.V
            DB.set_trial_param(work, "albedoMinVisible", G35.A.ramp(z, 0.62, 0.62, V["vtop"]) - DARK)
            DB.set_trial_param(work, "albedoMinNearIR", G35.A.ramp(z, 0.48, 0.48, V["ntop"]) - DARK)

    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\naverageRoutedRunoff | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n"
                                              + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in WB.SUMV + WB.STATEV + FORCV + EXTRA))


def main():
    t = min((json.loads(l) for l in open(C.OUT/"trials.jsonl")), key=lambda t: t["objective"]); print("objective", t["objective"], flush=True)
    r = Runner(work_root=str(C.OUT/f"wb_run_{VARIANT}"), **C.S.CFG); r.variant = C.F.VARIANTS["G"]; r.soil = C.physical(t["x"]); w = Path(r.work_root)/"keep"
    _, proc = r.run_summa(np.zeros(len(r.specs)), w); print("rc", proc.returncode, proc.stderr[-300:], flush=True)
    shutil.rmtree(KEEP, ignore_errors=True); shutil.copytree(w/"out", KEEP); shutil.rmtree(r.work_root, ignore_errors=True)
    (KEEP/"trial.json").write_text(json.dumps(t, default=float, indent=1))


if __name__ == "__main__":
    main()
