"""
Round-5 best + cloud-based incoming LW correction (g35_cloud_lw.py), 2026-11-06. The user asked for this instead of a flat LW shift ("would nuke the snowpack").
Same hourly energy-balance + layer output as g35_suite5_energy.py, so it compares directly with the round-5 control in OUT/suite5_cal/energy.
Preview (raw forcing, WY2013-2020, Dec-Feb): dry days unchanged; precipitating days (>2 mm) eps 0.82-0.87 -> 0.98-1.0 (+42-49 W/m2); Dec-Feb mean +11-13 W/m2.
-> OUT/suite5_cal/cloudlw/s<S>/{cal_day.nc, cal_timestep.nc, score.json}
usage: python g35_suite5_cloudlw.py 1.0      (S = strength, 1 = full Crawford & Duchon)
"""
import sys, json, shutil, numpy as np
from pathlib import Path
import g35_suite5_cal as C5
import g35_suite5_energy as EN
import g35_cloud_lw as CL


def runner(s):
    class R(EN.R):
        def _apply_forcing(self, work, params):
            fdir = super()._apply_forcing(work, params); self.n_lw = CL.apply(fdir, self.forcing, s); return fdir
    return R


def main():
    s = float(sys.argv[1]); t = json.loads((C5.OUT/"best"/"best_obj"/"trial.json").read_text()); keep = C5.OUT/"cloudlw"/f"s{s:g}"
    r = runner(s)(work_root=str(C5.OUT/f"run_cloudlw_{s:g}"), **C5.S.CFG); r.variant = C5.F.VARIANTS["G"]; r.soil = C5.physical(t["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(s, "rc", proc.returncode, "files edited", r.n_lw, proc.stderr[-300:], flush=True)
    if proc.returncode != 0: print(proc.stdout[-1500:]); sys.exit(1)
    shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    obj, parts, diag = C5.score(keep/"cal_day.nc", C5.C4.obs_mm()); diag.pop("band_order", None)
    (keep/"score.json").write_text(json.dumps(dict(objective=obj, parts=parts, diagnostics=diag, strength=s), indent=1, default=float))
    print(s, json.dumps({k: round(float(v), 3) for k, v in dict(objective=obj, **diag).items() if k in ("objective", "KGE_AprJul", "AprMay_bias", "JunJul_bias",
          "AugSep_bias", "OctNov_bias", "aso_swe_nmae", "band_order_mae", "vapour_mm", "annual_mm")}), flush=True)


if __name__ == "__main__":
    main()
