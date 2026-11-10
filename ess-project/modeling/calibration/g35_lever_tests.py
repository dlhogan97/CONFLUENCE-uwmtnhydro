"""
Timing-lever forward runs on the round-5 best (2026-11-06, agreed with the user). Round 5 is ~2 weeks late: Oct-Apr -74 mm, May-Jul +50 mm vs observed,
and the water budget shows the lag is SOIL storage at <2900 m (soil 327 -> 658 mm Oct -> May below 2300 m, released Jul-Sep; Apr input 188 mm vs runoff
86 mm at 2300-2900 m) plus lateral flow that feeds downslope HRU soils instead of the stream. The user wants water to move early and Aug-Sep to dry out.
Common physics for all four (canopy stays Beer's law + canopyGapFrac):
  - LAI fix (g35_lai_fix.py): LAI_method monTable, evergreen needleleaf LAIM restored to the near-flat 5.1-6.4
  - soil heat: thCondSoil mixConstit with thCond_soil 0.5; bcLowrTdyn presTemp with lowerBoundTemp 274.15 K
  - NO cloud-LW correction (round 6 showed it plus higher precip overshoots; decide separately)
Variants:
  base       round-5 best + common physics
  fastsoil   + k_macropore below split_m x10 (1.33e-4 -> 1.33e-3 m/s; above-split value 3.9e-3 unchanged)
  nolateral  + downHRUindex = 0 for every HRU (lateral outflow goes straight to basin runoff, not into the downslope HRU's soil)
  fcapil     + Fcapil 0.023 -> 0.01 (snow holds less liquid water, so winter rain drains through the pack)
-> OUT/levers/<variant>/{cal_day.nc, score.json}
usage: python g35_lever_tests.py <variant>
"""
import sys, json, shutil, numpy as np, netCDF4 as nc
from pathlib import Path
import g35_suite5_cal as C5
import g35_suite6_cal as C6
import g35_lai_fix as LAI
import deepstore_build as DB

OUT = C5.F.OUT/"levers"; OUT.mkdir(exist_ok=True)
X5 = json.loads((C5.OUT/"best"/"best_obj"/"trial.json").read_text())["x"]; N5 = [b[0] for b in C5.BOUNDS]
THCOND, LBT = 0.5, 274.15
VARIANTS = {"base": {}, "fastsoil": {"log_kmac_lo": X5[N5.index("log_kmac_lo")] + 1.0}, "nolateral": {}, "fcapil": {"Fcapil": 0.01}}


def runner(variant):
    class R(C5.Runner):
        def _subset(self, work):
            super()._subset(work)
            with nc.Dataset(work/"trialParams.nc") as t: n = len(np.array(t["hruId"][:]).ravel())
            DB.set_trial_param(work, "thCond_soil", np.full(n, THCOND)); DB.set_trial_param(work, "lowerBoundTemp", np.full(n, LBT))
            if variant == "nolateral":
                with nc.Dataset(work/"attributes.nc", "a") as a: a["downHRUindex"][:] = 0

        def _write_filemanager(self, work, prefix, forcing=None):
            super()._write_filemanager(work, prefix, forcing); md = work/"modelDecisions.txt"; txt = md.read_text()
            assert "bcLowrTdyn     zeroFlux" in txt and "thCondSoil     funcSoilWet" in txt and "canopySrad     BeersLaw" in txt
            md.write_text(txt.replace("bcLowrTdyn     zeroFlux", "bcLowrTdyn     presTemp").replace("thCondSoil     funcSoilWet", "thCondSoil     mixConstit "))
            LAI.apply(work)
    return R


def main():
    v = sys.argv[1]; x = list(X5)
    for k, val in VARIANTS[v].items(): x[N5.index(k)] = val
    keep = OUT/v; r = runner(v)(work_root=str(OUT/f"run_{v}"), **C5.S.CFG); r.variant = C5.F.VARIANTS["G"]; r.soil = C5.physical(x)
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(v, "rc", proc.returncode, proc.stderr[-300:], flush=True)
    if proc.returncode != 0: print(proc.stdout[-1500:]); sys.exit(1)
    for f in ("modelDecisions.txt",): shutil.copy2(w/f, w/"out"/f) if (w/f).exists() else None
    shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    obj5, _, diag = C5.score(keep/"cal_day.nc", C5.C4.obs_mm()); obj6, parts6, d6 = C6.score(keep/"cal_day.nc", C5.C4.obs_mm())
    diag.update({k: d6[k] for k in ("Apr_bias", "May_bias", "Jun_bias", "Jul_bias")}); diag.pop("band_order", None)
    (keep/"score.json").write_text(json.dumps(dict(variant=v, x=x, params=r.soil, obj_round5=obj5, obj_round6=obj6, diagnostics=diag), indent=1, default=float))
    print(v, json.dumps({k: round(float(val), 3) for k, val in dict(obj5=obj5, obj6=obj6, **diag).items() if not isinstance(val, (list, dict))}), flush=True)


if __name__ == "__main__":
    main()
