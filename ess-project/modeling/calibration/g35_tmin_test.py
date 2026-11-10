"""
Shrubland TMIN test (2026-11-03). In the round-3 feasible run, transpiration is ~0 until May: open shrubland (MODIFIED_IGBP veg 7, 49% of the basin)
has MPTABLE TMIN = 273 K (Ball-Berry photosynthesis off below it), evergreen needleleaf 265 K. User: basin ET ~300 mm/yr is right; OpenET is a pattern
guide only (more ET in Apr-Jun, June peak). Same soil/snow as the round-3 trial; only the shrubland TMIN in the run's TBL_MPTABLE.TBL changes.
-> OUT/suite3_cal/tmin_test/<variant>/cal_day.nc + <variant>.json (g35_suite3_cal.score diagnostics), tmin_test.csv, monthly ET table printed
usage: python g35_tmin_test.py
"""
import re, json, shutil, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import g35_suite3_cal as C, g35_suite3_wb as WB, g35_suite_cal as G

OUT = C.OUT/"tmin_test"; OUT.mkdir(exist_ok=True)
SHRUB = 7                                                                   # MODIFIED_IGBP open shrubland (1-based column in the MODIS block)
VARIANTS = {"base": None, "shrub_268": 268, "shrub_265": 265}


def set_tmin(tbl, veg, value):
    t = tbl.read_text(); i = t.index("&noah_mp_modis_parameters"); m = re.compile(r"^( TMIN\s*=)(.*)$", re.M).search(t, i)
    vals = [v.strip() for v in m.group(2).split(",")]; vals[veg - 1] = str(value)
    tbl.write_text(t[:m.start()] + m.group(1) + " " + ", ".join(vals) + t[m.end():])


class Runner(WB.Runner):
    tmin = None
    def _apply_params(self, work, params):
        super()._apply_params(work, params)
        if self.tmin is not None: set_tmin(work/"TBL_MPTABLE.TBL", SHRUB, self.tmin)
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing); oc = work/"outputControl.txt"; have = oc.read_text()
        oc.write_text(have + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in G.HRUV + G.BASV if f"{v} |" not in have))


def run(name):
    soil = json.loads((C.OUT/"wb"/"trial.json").read_text())["params"]
    r = Runner(work_root=str(OUT/f"w_{name}"), **C.S.CFG); r.variant = C.F.VARIANTS["G"]; r.soil = soil; r.tmin = VARIANTS[name]; w = Path(r.work_root)/"keep"
    try:
        _, proc = r.run_summa(np.zeros(len(r.specs)), w); out = w/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): rec = dict(reason="summa_failed", stderr=proc.stderr[-300:])
        else:
            obj, parts, diag = C.score(out, C.S.RunnerSnow(work_root=str(OUT/f"score_{name}"), **C.S.CFG), soil); rec = dict(objective=obj, parts=parts, diagnostics=diag)
            (OUT/name).mkdir(exist_ok=True); shutil.copy(out, OUT/name/"cal_day.nc")
    finally:
        shutil.rmtree(r.work_root, ignore_errors=True); shutil.rmtree(OUT/f"score_{name}", ignore_errors=True)
    (OUT/f"{name}.json").write_text(json.dumps(dict(variant=name, tmin=VARIANTS[name], **rec), default=float, indent=1)); return name


def report():
    import tuo_snow_emulator as SE
    H = SE.hrus(); w = H.w.values; s = slice("2012-10-01", "2020-09-30")
    K = ["KGE_AprNov", "NSE_AprNov", "ET_mm", "aso_swe_nmae", "aso_melt_order", "com_bias_d", "AprMay_bias", "JunJul_bias", "AugSep_bias", "OctNov_bias", "min7_ratio"]
    rows, mon = [], {}
    for n in VARIANTS:
        r = json.loads((OUT/f"{n}.json").read_text()); rows.append(dict(variant=n, objective=r.get("objective"), **{k: r.get("diagnostics", {}).get(k) for k in K}))
        if (OUT/n/"cal_day.nc").exists():
            X = SE.load(OUT/n/"cal_day.nc"); e = (X["et"][s]*w).sum(1); mon[n] = e.groupby(e.index.month).mean()
    T = pd.DataFrame(rows); T.to_csv(OUT/"tmin_test.csv", index=False); pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
    M = pd.DataFrame(mon); M.to_csv(OUT/"tmin_test_monthly_et.csv"); print("basin ET mm/d by month\n", M.round(2).T.to_string())


if __name__ == "__main__":
    with ProcessPoolExecutor(len(VARIANTS)) as ex:
        for n in ex.map(run, VARIANTS): print("done", n, flush=True)
    report()
