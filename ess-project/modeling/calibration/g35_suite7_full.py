"""
Round-7 best over the FULL forcing record (2026-11-09, asked by the user): sim 1981-10-01 .. 2025-09-30, WY1982 is spin-up, scoring WY1983-2025.
Rounds 5-7 calibrated on WY2013-2020 only, so WY1983-2012 and WY2021-2025 are a true out-of-sample check.
Two runs, same physics and parameters (g35_suite7_cal: Beer's law + gaps, monTable LAI with the evergreen row restored, mixConstit 0.5, presTemp 274.15):
  control  round-7 best
  patch    round-7 best + the anchor_m080 snowpatch graft (g35_suite4_snowpatch: 10 depth bins of HRU 2 placed right after HRU 2, drift on snowfall, bin NIR 0.70,
           tol 0.05), with bins seeded from the 44-yr anchor run's SWE at 1981-10-01 (not 2011-10-01)
Both with the daily water-balance output of g35_suite4_patch_fate.with_wb.
-> OUT/suite7_cal/full/<run>/{cal_day.nc, (patch: cal_day_28.nc with the bins folded into HRU 2), meta.json}
usage: python g35_suite7_full.py control|patch|control_wired|patch_wired   (_wired: HRU 2 + bins drain straight to the stream)
"""
import sys, json, shutil, numpy as np, netCDF4 as nc
from pathlib import Path
import g35_suite7_cal as C7
import g35_suite4_snowpatch as SP
import g35_suite4_patch_fate as PF

START, END = "1981-10-01", "2025-09-30"
CFG = dict(C7.S.CFG, sim_start=START, analysis_start="1982-10-01", sim_end=END)
ROOT = C7.OUT/"full"; ROOT.mkdir(exist_ok=True)
_init = SP.init_swe; SP.init_swe = lambda ts=START: _init(ts)          # SP.Runner._subset calls init_swe() with the 2011 default


class Patch(SP.Runner, C7.Runner):                                     # MRO: snowpatch graft -> round-7 physics -> round-5/4 runner
    pass


def wired(base):
    """HRU 2 (and, in the patch run, all its bins) send lateral outflow straight to the basin outflow (downHRUindex = 0) instead of into HRU 7's soil.
    Added 2026-11-09 (user): in the patch run HRU 7 took up the extra Aug-Sep melt (HRU 7 ET +30 mm/yr), so only 0.07 mm/yr reached the outlet."""
    class W(base):
        def _subset(self, work):
            super()._subset(work)
            with nc.Dataset(work/"attributes.nc", "a") as a:
                hid = np.array(a["hruId"][:]).ravel(); d = np.array(a["downHRUindex"][:]); d[np.isin(hid, SP.BINS)] = 0; a["downHRUindex"][:] = d
    return W


def main():
    name = sys.argv[1]; assert name in ("control", "patch", "control_wired", "patch_wired"); res = json.loads((C7.OUT/"result.json").read_text()); keep = ROOT/name
    base = Patch if name.startswith("patch") else C7.Runner
    cls = PF.with_wb(wired(base) if name.endswith("_wired") else base)
    r = cls(work_root=str(ROOT/f"run_{name}"), **CFG); r.variant = C7.F.VARIANTS["G"]; r.soil = C7.physical(res["x"])
    w = Path(r.work_root)/"keep"; _, proc = r.run_summa(np.zeros(len(r.specs)), w); print(name, "rc", proc.returncode, proc.stderr[-400:], flush=True)
    if proc.returncode != 0: print(proc.stdout[-2000:]); sys.exit(1)
    shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); shutil.rmtree(r.work_root, ignore_errors=True)
    meta = dict(run=name, sim=[START, END], x=res["x"], params=res["params"], calibration_objective=res["objective"])
    if name.startswith("patch"):
        SP.fold(keep/"cal_day.nc", keep/"cal_day_28.nc"); meta.update(audit=r.audit, init_swe_m=SP.init_swe())
    (keep/"meta.json").write_text(json.dumps(meta, indent=1, default=float)); print(json.dumps(meta.get("audit", {}), default=float), flush=True)


if __name__ == "__main__":
    main()
