"""
Why does round-4 best (g35_suite4_cal.py, no snowpatch) miss the big Dec-Mar storm flows? (2026-11-04) In the Jan/Feb 2017, Mar 2018 and Feb 2019 storms
25-67 mm of rain fell in a week but only 0-4 mm left the base of the snowpack (scalarRainPlusMelt = iLayerLiqFluxSnow(nSnow), computFlux.f90:599).
Snow liquid water (snowLiqFlx.f90): a layer drains only above its retention Fcapil*(1 - volFracIce) (Fcapil 0.06), then k_snow*relSat**mw_exp; liquid in a
layer below 0 C refreezes (snowfrz_scale 50). So rain is held by retention in deep fresh snow, or refrozen by cold content.
  layers      hourly instantaneous snow-layer state (mLayerTemp/VolFracLiq/VolFracIce/Depth, nSnow) + scalar fluxes -> OUT/suite4_cal/storms/layers
  snow <K> <Fcapil> <k_snow> <mw_exp>   as tcr, plus snow liquid-water parameters (written to trialParams, all HRUs) -> OUT/suite4_cal/storms/snow_<...>
  tcr <K>     round-4 best with tempCritRain set to K (rain ABOVE it), daily water-balance output, scored -> OUT/suite4_cal/storms/tcr_<K>
usage: python g35_suite4_storms.py layers | tcr 269.15 | snow 267.65 0.03 0.015 3
"""
import sys, json, shutil, numpy as np
from pathlib import Path
import g35_suite4_cal as C4
import g35_suite4_patch_fate as PF
import deepstore_build as DB
import netCDF4 as nc

ST = C4.OUT/"storms"
HOURLY = ["mLayerTemp", "mLayerVolFracLiq", "mLayerVolFracIce", "mLayerDepth", "nSnow", "scalarSWE", "scalarSnowDepth", "scalarRainPlusMelt",
          "scalarThroughfallRain", "scalarRainfall", "scalarSnowfall", "airtemp"]


class Layers(C4.Runner):
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing); oc = work/"outputControl.txt"      # drop existing lines (duplicates merge flags); columns: freq | sum | instant | mean ...
        keep = [l for l in oc.read_text().splitlines(True) if l.split("|")[0].strip() not in HOURLY]
        oc.write_text("".join(keep) + "".join(f"{v} | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n" for v in HOURLY))


def with_snow(base, fcapil, k_snow, mw_exp):
    class R(base):
        def _subset(self, work):
            super()._subset(work)
            with nc.Dataset(work/"trialParams.nc") as t: n = len(np.array(t["hruId"][:]).ravel())
            for k, v in (("Fcapil", fcapil), ("k_snow", k_snow), ("mw_exp", mw_exp)): DB.set_trial_param(work, k, np.full(n, v))
    return R


def main():
    mode = sys.argv[1]; t = json.loads((C4.OUT/"best"/"best_obj"/"trial.json").read_text())
    sp = [float(a) for a in sys.argv[3:6]] if mode == "snow" else None
    name = "layers" if mode == "layers" else f"tcr_{float(sys.argv[2]):.2f}" if mode == "tcr" else f"snow_{float(sys.argv[2]):.2f}_F{sp[0]:g}_k{sp[1]:g}_m{sp[2]:g}"
    base = Layers if mode == "layers" else PF.with_wb(C4.Runner)
    r = (with_snow(base, *sp) if mode == "snow" else base)(work_root=str(ST/f"run_{name}"), **C4.S.CFG)
    r.variant = C4.F.VARIANTS["G"]; r.soil = C4.physical(t["x"])
    if mode in ("tcr", "snow"): r.soil["tempCritRain"] = float(sys.argv[2])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(name, "rc", proc.returncode, proc.stderr[-300:], (proc.stdout or "")[-1500:], flush=True)
    keep = ST/name; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    if mode in ("tcr", "snow"):
        obj, parts, diag = C4.score(keep/"cal_day.nc", C4.obs_mm()); diag.pop("band_order")
        (keep/"score.json").write_text(json.dumps(dict(objective=obj, parts=parts, diagnostics=diag, tempCritRain=r.soil["tempCritRain"], snow=sp), indent=1, default=float))
        print(json.dumps(dict(objective=obj, **diag), default=float, indent=1))


if __name__ == "__main__":
    main()
