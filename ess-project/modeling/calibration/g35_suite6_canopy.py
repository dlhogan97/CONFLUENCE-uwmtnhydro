"""
Canopy shortwave intercomparison, Tuolumne G35 (2026-11-06, asked by the user): canopySrad BeersLaw (current) vs UEB_2stream.
Why: forest HRUs melt 40-90 d late (10-20 diagnosis). Beer's law in SUMMA (vegSWavRad.f90:482) applies tau = exp(-0.5/cosZ * VAI) to BOTH direct and diffuse
and has no scattering, so at the forest VAI (~4.2-6.3) December noon transmission is 0.2-1.6 %. UEB_2stream (vegSWavRad.f90:673) has the same hard-coded
G = 0.5 plus back-scatter b = 0.5, a separate diffuse transmission and canopy-ground multiple reflection: ~2-7x more SW reaches the snow. No new parameters.
Everything else is the round-6 physics (cloud LW s=1, mixConstit thCond_soil 0.5, presTemp) with ONE parameter set: the round-6 best trial at launch time
(read from suite6_cal/trials.jsonl and saved to params.json), so the two runs differ only in canopySrad.
Extra daily-mean output: below-canopy SW, canopy- and ground-absorbed SW, canopy LW to ground, canopy temperature, snow depth.
-> OUT/suite6_cal/canopy/<scheme>/{cal_day.nc, score.json, params.json}
usage: python g35_suite6_canopy.py BeersLaw      |      python g35_suite6_canopy.py UEB_2stream
"""
import sys, json, shutil, numpy as np
from pathlib import Path
import g35_suite6_cal as C6

EXTRA = ["scalarBelowCanopySolar", "scalarCanopyAbsorbedSolar", "scalarGroundAbsorbedSolar", "scalarLWRadCanopy2Ground", "scalarCanopyTemp", "scalarSnowDepth"]
ROOT = C6.OUT/"canopy"


def runner(scheme):
    class R(C6.Runner):
        def _write_filemanager(self, work, prefix, forcing=None):
            super()._write_filemanager(work, prefix, forcing)
            md = work/"modelDecisions.txt"; txt = md.read_text(); assert "canopySrad     BeersLaw" in txt, "unexpected canopySrad line"
            md.write_text(txt.replace("canopySrad     BeersLaw   ", f"canopySrad     {scheme:<11s}"))
            oc = work/"outputControl.txt"; have = {l.split("|")[0].strip() for l in oc.read_text().splitlines() if "|" in l}
            oc.write_text(oc.read_text() + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in EXTRA if v not in have))
    return R


def best_trial():
    pf = ROOT/"params.json"
    if pf.exists(): return json.loads(pf.read_text())
    T = [json.loads(l) for l in open(C6.OUT/"trials.jsonl")]; b = min(T, key=lambda t: t["objective"])
    ROOT.mkdir(exist_ok=True); pf.write_text(json.dumps(dict(x=b["x"], objective=b["objective"], params=b["params"]), indent=1)); return json.loads(pf.read_text())


def main():
    scheme = sys.argv[1]; assert scheme in ("BeersLaw", "UEB_2stream"); t = best_trial(); keep = ROOT/scheme
    r = runner(scheme)(work_root=str(C6.OUT/f"run_canopy_{scheme}"), **C6.S.CFG); r.variant = C6.F.VARIANTS["G"]; r.soil = C6.physical(t["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(scheme, "rc", proc.returncode, proc.stderr[-300:], flush=True)
    if proc.returncode != 0: print(proc.stdout[-1500:]); sys.exit(1)
    shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    obj, parts, diag = C6.score(keep/"cal_day.nc", C6.C4.obs_mm()); diag.pop("band_order", None)
    (keep/"score.json").write_text(json.dumps(dict(scheme=scheme, objective=obj, parts=parts, diagnostics=diag), indent=1, default=float))
    print(scheme, json.dumps({k: round(float(v), 3) for k, v in dict(objective=obj, **diag).items() if k in ("objective", "KGE_AprJul", "Apr_bias", "May_bias",
          "Jun_bias", "Jul_bias", "AugSep_bias", "aso_swe_nmae", "band_order_mae", "vapour_mm")}), flush=True)


if __name__ == "__main__":
    main()
