"""
Tuolumne: how can ET rise ~100 mm/yr without losing spring runoff? (2026-10-30). One-at-a-time changes from the round-2 best-objective trial, each a full SUMMA run WY2012-2020.
ET is about 240 mm in that trial; OpenET is higher and uncertain, so the working target is about +100 mm. Winter is a forcing problem and leaves the score, recession is reported only:
  objective_AprNov  mean of peak_timing, peak_volume, falling_limb, com_timing, monthly_bias (Apr-Oct); KGE, NSE and volume bias below are Apr-Nov days only.
Reported: ET total and parts (transpiration, canopy and ground evaporation, sublimation) by elevation band and month, ASO SWE and melt-order error, recession, 7-day minimum.
-> OUT/et_tests/<variant>/cal_day.nc, figures/draft/tuo_et_tests.csv, tuo_et_tests_bands.csv, tuo_et_tests_monthly.csv
usage: python g35_et_tests.py [--score-only]
"""
import json, sys, shutil, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import deepstore_build as DB
import g35_suite2_cal as C, signature_objective_plus as P
from east_ctl_treat_cal_v2 import recession_taus
S, F = C.S, C.F
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
OUT = F.OUT/"et_tests"; OUT.mkdir(exist_ok=True)
BASE = json.load(open(C.OUT/"result.json"))["params"]
ETV = ["scalarCanopyTranspiration", "scalarCanopyEvaporation", "scalarGroundEvaporation", "scalarSnowSublimation", "scalarCanopySublimation"]
VARIANTS = {"base": {}, "precip_105": dict(precip_scale=1.05), "precip_115": dict(precip_scale=1.15),
            "fc_16": dict(fieldCapacity=0.16), "fc_22": dict(fieldCapacity=0.22), "lat_slow": dict(kAnisotropic=1.5), "macro_low": dict(k_macropore=1e-4),
            "no_thin": dict(depth=None), "roots_15": dict(depth=None, root_x=1.5), "lai_15": dict(lai_x=1.5), "vcmax_60": dict(vcmax25_canopyTop=60.0),
            "water": dict(precip_scale=1.10, fieldCapacity=0.16), "water_roots": dict(precip_scale=1.10, fieldCapacity=0.16, depth=None, root_x=1.5),
            "water_roots_lai": dict(precip_scale=1.10, fieldCapacity=0.16, depth=None, root_x=1.5, lai_x=1.5),
            # round 2: more precipitation fixes melt order but floods runoff; slower soil raises ET but cuts it, so pair them. split = fast below 2600 m (spring water), slow above (ET)
            "p100_macro": dict(precip_scale=1.00, k_macropore=1e-4), "p105_macro": dict(precip_scale=1.05, k_macropore=1e-4), "p105_lat": dict(precip_scale=1.05, kAnisotropic=1.5),
            "p105_macro_lat": dict(precip_scale=1.05, k_macropore=1e-4, kAnisotropic=1.5), "p100_macro_lat": dict(precip_scale=1.00, k_macropore=1e-4, kAnisotropic=1.5),
            "p105_macro5e4": dict(precip_scale=1.05, k_macropore=5e-4), "p105_split": dict(precip_scale=1.05, split=2600.0, k_macropore_hi=1e-4), "p100_split": dict(precip_scale=1.00, split=2600.0, k_macropore_hi=1e-4)}


class Runner(C.Runner):
    def _apply_params(self, work, params):
        super()._apply_params(work, params); s = self.soil
        with nc.Dataset(work/"coldState.nc") as c: col = np.array(c["iLayerHeight"][:])[-1]
        with nc.Dataset(work/"trialParams.nc") as tp: root, lai = np.array(tp["rootingDepth"][:]).ravel(), np.array(tp["summerLAI"][:]).ravel()
        if "root_x" in s: DB.set_trial_param(work, "rootingDepth", np.minimum(root*s["root_x"], col - 0.05))
        if "lai_x" in s: DB.set_trial_param(work, "summerLAI", lai*s["lai_x"])
        if "split" in s: DB.set_trial_param(work, "k_macropore", np.where(self.elev < s["split"], s["k_macropore"], s["k_macropore_hi"]))
        if "vcmax25_canopyTop" in s: DB.set_trial_param(work, "vcmax25_canopyTop", [s["vcmax25_canopyTop"]]*len(root))
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing)
        with open(work/"outputControl.txt", "a") as fh: fh.write("".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in ETV))


def run(name):
    r = Runner(work_root=str(OUT/f"w_{name}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.soil = {**BASE, **VARIANTS[name]}; w = Path(r.work_root)/"keep"
    _, proc = r.run_summa(np.zeros(len(r.specs)), w); keep = OUT/name; shutil.rmtree(keep, ignore_errors=True)
    if (w/"out"/"cal_day.nc").exists(): shutil.copytree(w/"out", keep)
    shutil.rmtree(r.work_root, ignore_errors=True); return name, proc.returncode, proc.stderr[-200:]


def score_all():
    r = S.RunnerSnow(work_root=str(F.OUT/"score"), **S.CFG); o = r.obs[r.analysis_start:r.sim_end]; w = r.hru_wt; rows, bands, mons = [], {}, {}; yrs = len(o)/365.25
    band = pd.cut(r.elev, [0, 2200, 2600, 2900, 3200, 4000], labels=["<2200", "2200-2600", "2600-2900", "2900-3200", ">3200"])
    for name in VARIANTS:
        if not (OUT/name/"cal_day.nc").exists(): print("no output", name); continue
        d = nc.Dataset(OUT/name/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
        q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).astype(float).ravel()[:len(t)], index=t)*r.area
        H = {v: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float), index=t, columns=r.hruid.astype(int)).where(lambda z: z.abs() < 1e8) for v in ETV+["scalarTotalET", "scalarSWE", "scalarRainfall", "scalarSnowfall"]}; d.close()
        j = pd.concat([q.rename("s"), o.rename("o")], axis=1)[r.analysis_start:r.sim_end].dropna(); rt = P.signature_objective_routing(j.s, j.o); parts = {k: rt.parts[k] for k in ("peak_timing", "peak_volume", "falling_limb", "com_timing", "monthly_bias")}
        a = j[j.index.month.isin(range(4, 12))]; c = a.s.corr(a.o); kge = 1-np.sqrt((c-1)**2+(a.s.std()/a.o.std()-1)**2+(a.s.mean()/a.o.mean()-1)**2)
        pct = lambda m: 100*(j.s[j.index.month.isin(m)].sum()/j.o[j.index.month.isin(m)].sum()-1); mm = lambda v: (H[v][r.analysis_start:r.sim_end]*w).sum(1)*86400/yrs
        low7 = lambda s: pd.Series({y: s[f"{y}-08-01":f"{y}-11-30"].rolling(7, center=True, min_periods=7).mean().min() for y in range(2013, 2020)})
        _, ad = S.aso_objective(H["scalarSWE"], r.aso, r.w_hru); ap, _ = S.aso_objective(H["scalarSWE"], r.aso, r.w_hru)
        y0, y1 = int(r.analysis_start[:4])+1, int(r.sim_end[:4])
        rows.append(dict(variant=name, objective_AprNov=float(np.mean(list(parts.values()))), KGE_AprNov=kge, NSE_AprNov=1-((a.s-a.o)**2).sum()/((a.o-a.o.mean())**2).sum(), ET_mm=-float(mm("scalarTotalET").sum()),
                         transp=-float(mm(ETV[0]).sum()), can_evap=-float(mm(ETV[1]).sum()), gnd_evap=-float(mm(ETV[2]).sum()), subl=-float(mm(ETV[3]).sum()+mm(ETV[4]).sum()),
                         AprNov_bias=pct(range(4, 12)), AprMay_bias=pct([4, 5]), JunJul_bias=pct([6, 7]), AugSep_bias=pct([8, 9]), OctNov_bias=pct([10, 11]), com_bias_d=rt.diagnostics["com_bias_d"],
                         recess_d=float(recession_taus(j.s, y0, y1).median()), recess_obs_d=float(recession_taus(j.o, y0, y1).median()), min7_ratio=float((low7(q)/low7(o)).median()),
                         aso_swe_nmae=ad["swe_nmae"], aso_melt_order=ap["melt_order"], **parts))
        e = -(H["scalarTotalET"][r.analysis_start:r.sim_end]*86400); bands[name] = pd.Series(e.mean().values*365.25, index=r.hruid.astype(int)).groupby(np.asarray(band)).apply(lambda z: np.average(z, weights=w[np.asarray(band) == z.name]))
        mons[name] = (e*w).sum(1).groupby(e.index.month).mean()*30.4
    T = pd.DataFrame(rows).set_index("variant"); T.to_csv(FIG+"tuo_et_tests.csv"); pd.DataFrame(bands).T.to_csv(FIG+"tuo_et_tests_bands.csv"); pd.DataFrame(mons).T.to_csv(FIG+"tuo_et_tests_monthly.csv")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); print(T.round(2).to_string()); print(pd.DataFrame(bands).T.round(0).to_string()); print(pd.DataFrame(mons).T.round(0).to_string())


def main():
    if "--score-only" not in sys.argv:
        todo = [v for v in VARIANTS if not (OUT/v/"cal_day.nc").exists()]
        with ProcessPoolExecutor(len(todo)) as ex:
            for x in ex.map(run, todo): print(x, flush=True)
    score_all()


if __name__ == "__main__":
    main()
