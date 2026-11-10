"""
Ground heat test for Tuolumne winter baseflow (2026-11-05). In round 4/5 the soil column (3 layers, ~0.25 m, under the snowpack) is frozen all winter:
upper soil about -3 C at 2300-2900 m and -6 to -7 C above 2900 m (hourly layer run, g35_suite4_storms.py layers).
- Winter rain and melt that reaches the soil refreezes as soil ice (0.03 -> 0.06-0.09 volumetric by March) and comes out at spring thaw.
- Observed Dec-Mar baseflow RISES (15-day-min 0.27 -> 0.83 mm/d); the model's stays ~0.1.
- modelDecisions has bcLowrTdyn = zeroFlux, so there is no ground heat from below. Real ground heat (~1-3 W/m2 ≈ 0.3-0.8 mm/d basal melt) keeps soils
  under deep snow near 0 C.
Test: bcLowrTdyn = presTemp with lowerBoundTemp = T (K, all HRUs), on the round-5 best (g35_suite5_cal.py). Daily water-balance output.
-> OUT/suite5_cal/ground/lbt_<T>/{cal_day.nc, score.json}
Hourly instantaneous layer state (mLayerTemp, mLayerVolFracIce, nSnow) -> cal_timestep.nc, to check soil freezing.
T = "zero" keeps zeroFlux (round-5 control with the same output).
Optional 2nd arg LAM (2026-11-05, user: soil thermal conductivity is likely too high; these soils should be hard to freeze): switch thCondSoil funcSoilWet
(~1.7 W/m/K moist, from frac_sand/frac_clay) -> mixConstit with thCond_soil = LAM for the solid fraction (moist bulk ~0.55*LAM + 0.18). -> lbt_<T>_lam<LAM>
usage: python g35_suite5_ground.py 273.65 | zero  [LAM]
"""
import sys, json, shutil, numpy as np, netCDF4 as nc
from pathlib import Path
import g35_suite5_cal as C5
import g35_suite4_patch_fate as PF
import deepstore_build as DB


LAYERS = ["mLayerTemp", "mLayerVolFracIce", "nSnow"]                  # hourly: SUMMA writes index vars (nSnow) only at the model timestep


def runner(T, lam=None):
    class R(PF.with_wb(C5.Runner)):
        def _subset(self, work):
            super()._subset(work)
            with nc.Dataset(work/"trialParams.nc") as t: n = len(np.array(t["hruId"][:]).ravel())
            if lam is not None: DB.set_trial_param(work, "thCond_soil", np.full(n, lam))
            if T is not None: DB.set_trial_param(work, "lowerBoundTemp", np.full(n, T))

        def _write_filemanager(self, work, prefix, forcing=None):
            super()._write_filemanager(work, prefix, forcing); oc = work/"outputControl.txt"     # columns: freq | sum | instant | mean ...
            keep = [l for l in oc.read_text().splitlines(True) if l.split("|")[0].strip() not in LAYERS]
            oc.write_text("".join(keep) + "".join(f"{v} | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n" for v in LAYERS))
            md = work/"modelDecisions.txt"; txt = md.read_text()
            if T is not None:
                assert "bcLowrTdyn     zeroFlux" in txt, "unexpected bcLowrTdyn line"; txt = txt.replace("bcLowrTdyn     zeroFlux", "bcLowrTdyn     presTemp")
            if lam is not None:
                assert "thCondSoil     funcSoilWet" in txt, "unexpected thCondSoil line"; txt = txt.replace("thCondSoil     funcSoilWet", "thCondSoil     mixConstit ")
            md.write_text(txt)
    return R


def main():
    T = None if sys.argv[1] == "zero" else float(sys.argv[1]); lam = float(sys.argv[2]) if len(sys.argv) > 2 else None
    tag = ("zero" if T is None else f"{T:.2f}") + ("" if lam is None else f"_lam{lam:g}"); t = json.loads((C5.OUT/"best"/"best_obj"/"trial.json").read_text())
    r = runner(T, lam)(work_root=str(C5.OUT/f"run_ground_{tag}"), **C5.S.CFG); r.variant = C5.F.VARIANTS["G"]; r.soil = C5.physical(t["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(T, "rc", proc.returncode, proc.stderr[-300:], flush=True)
    if proc.returncode != 0: print(proc.stdout[-1500:]); sys.exit(1)
    keep = C5.OUT/"ground"/f"lbt_{tag}"; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    obj, parts, diag = C5.score(keep/"cal_day.nc", C5.C4.obs_mm()); diag.pop("band_order", None)
    (keep/"score.json").write_text(json.dumps(dict(objective=obj, parts=parts, diagnostics=diag, lowerBoundTemp=T, thCond_soil=lam), indent=1, default=float))
    print(T, json.dumps({k: round(float(v), 3) for k, v in dict(objective=obj, **diag).items() if k in ("objective", "KGE_AprJul", "AprMay_bias", "JunJul_bias",
          "AugSep_bias", "OctNov_bias", "aso_swe_nmae", "band_order_mae", "vapour_mm", "annual_mm")}), flush=True)


if __name__ == "__main__":
    main()
