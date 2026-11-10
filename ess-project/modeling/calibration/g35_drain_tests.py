"""
Can SUMMA drain spring melt the way the emulator says the observations need? (2026-11-03). One-at-a-time soil changes from the round-3 feasible trial
(g35_suite3_cal.py, objective 2.471: k_macropore 3.1e-4 below 2580 m, 7.3e-4 above, near the 1e-3 upper bound; fieldCapacity 0.122).
Emulator result (tuo_emulator_fit.py soil): with ASO-consistent snow, observed timing needs little water held above field capacity, ~1 d drainage below the
split, ~15 d above, and ET drawn from wet soil; ET at its 300 mm floor is acceptable (user, 2026-11-03).
Scored with g35_suite3_cal.score (Apr-Nov KGE, seasonal volumes, centre of mass, ET, ASO SWE and melt-order diagnostics).
-> OUT/suite3_cal/drain_tests/<variant>.json, drain_tests.csv
usage: python g35_drain_tests.py
"""
import json, shutil, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import g35_suite3_cal as C

OUT = C.OUT/"drain_tests"; OUT.mkdir(exist_ok=True)
BASE = json.loads((C.OUT/"wb"/"trial.json").read_text())["params"]
VARIANTS = {"base": {}, "hi_2e-3": dict(k_macropore_hi=2e-3), "hi_5e-3": dict(k_macropore_hi=5e-3), "lo_3e-3": dict(k_macropore=3e-3),
            "both_3e-3": dict(k_macropore=3e-3, k_macropore_hi=3e-3), "fc_0.08": dict(fieldCapacity=0.08), "aniso_x3": dict(kAnisotropic=3*BASE["kAnisotropic"]),
            "zscale_2": dict(zScale_TOPMODEL=2.0), "fast_fc": dict(k_macropore=3e-3, k_macropore_hi=3e-3, fieldCapacity=0.08),
            "fast_fc_aniso": dict(k_macropore=3e-3, k_macropore_hi=3e-3, fieldCapacity=0.08, kAnisotropic=3*BASE["kAnisotropic"])}


def run(name):
    soil = {**BASE, **VARIANTS[name]}; r = C.Runner(work_root=str(OUT/f"w_{name}"), **C.S.CFG); r.variant = C.F.VARIANTS["G"]; r.soil = soil
    w = Path(r.work_root)/"keep"
    try:
        _, proc = r.run_summa(np.zeros(len(r.specs)), w); out = w/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): rec = dict(reason="summa_failed", stderr=proc.stderr[-300:])
        else:
            obj, parts, diag = C.score(out, C.S.RunnerSnow(work_root=str(OUT/f"score_{name}"), **C.S.CFG), soil); rec = dict(objective=obj, parts=parts, diagnostics=diag)
            shutil.copy(out, OUT/f"{name}_cal_day.nc")
    finally:
        shutil.rmtree(r.work_root, ignore_errors=True); shutil.rmtree(OUT/f"score_{name}", ignore_errors=True)
    (OUT/f"{name}.json").write_text(json.dumps(dict(variant=name, soil=soil, **rec), default=float, indent=1)); return name


def main():
    with ProcessPoolExecutor(len(VARIANTS)) as ex:
        for n in ex.map(run, VARIANTS): print("done", n, flush=True)
    K = ["KGE_AprNov", "NSE_AprNov", "ET_mm", "aso_swe_nmae", "aso_melt_order", "com_bias_d", "AprMay_bias", "JunJul_bias", "AugSep_bias", "OctNov_bias", "min7_ratio"]
    rows = []
    for n in VARIANTS:
        r = json.loads((OUT/f"{n}.json").read_text()); rows.append(dict(variant=n, objective=r.get("objective"), **{k: r.get("diagnostics", {}).get(k) for k in K}))
    T = pd.DataFrame(rows); T.to_csv(OUT/"drain_tests.csv", index=False); pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
