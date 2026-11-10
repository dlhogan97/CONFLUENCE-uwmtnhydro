"""
Full-basin (28 HRU) run of the alpine-extended albedo setup G35 (2026-10-27): variant G below 3040 m (forest kept), and above 3040 m the ramps of
tuo_alpine_albedo.py continued to decay 7e6, floor vis 0.80 / NIR 0.64, fresh max 0.92 / 0.69 at 3686 m. Scored with full_basin_albedo.score and
compared against the earlier ref / G / GO rows (full_basin_albedo/scores.csv).
usage: python full_basin_G35.py
"""
import uuid, shutil, json, numpy as np, pandas as pd
from pathlib import Path
import deepstore_build as DB
import run_calibration_snow as S
import full_basin_albedo as F
import tuo_alpine_albedo as A

V = A.VAR["G35"] if "G35" in A.VAR else dict(dtop=7.0e6, vtop=0.80, ntop=0.64, mvtop=0.92, mntop=0.69)


class Runner(F.Runner):
    def _apply_params(self, work, params):
        super()._apply_params(work, params); z = self.elev
        DB.set_trial_param(work, "albedoDecayRate", A.ramp(z, 6e5, 1.29e6, V["dtop"], log=True))
        DB.set_trial_param(work, "albedoMinVisible", A.ramp(z, 0.62, 0.62, V["vtop"])); DB.set_trial_param(work, "albedoMinNearIR", A.ramp(z, 0.48, 0.48, V["ntop"]))
        DB.set_trial_param(work, "albedoMaxVisible", A.ramp(z, 0.88, 0.88, V["mvtop"])); DB.set_trial_param(work, "albedoMaxNearIR", A.ramp(z, 0.62, 0.62, V["mntop"]))


if __name__ == "__main__":
    r = Runner(work_root=str(F.OUT/f"G35_{uuid.uuid4().hex[:6]}"), **S.CFG); r.variant = F.VARIANTS["G"]; w = Path(r.work_root)/"keep"
    params, proc = r.run_summa(np.zeros(len(r.specs)), w); print("rc", proc.returncode, proc.stderr[-200:])
    keep = F.OUT/"G35"; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); (keep/"variant.json").write_text(json.dumps(V)); shutil.rmtree(r.work_root, ignore_errors=True)
    sc = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG); row, dg = F.score("G35", sc)
    old = pd.read_csv(F.OUT/"scores.csv"); T = pd.concat([old, pd.DataFrame([row])], ignore_index=True); T.to_csv(F.OUT/"scores_with_G35.csv", index=False)
    pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
    mo = pd.read_csv(F.OUT/"melt_order_by_hru.csv", index_col=0); mo["G35"] = pd.Series(dg["frac_left_mdl"]); print(mo.sort_values("elev", ascending=False).round(2).to_string())
    mo.to_csv(F.OUT/"melt_order_by_hru_with_G35.csv")
