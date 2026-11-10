"""
Full-basin (28 HRU) test of the pillow-derived albedo setup (2026-10-27), WY2012-2020 (ASO scoring from WY2013).
Forcing for all runs: uniform -2.0 C with its Dilley-O'Brien longwave increment, swScale / lwOffset from the Dana-only melt best; precipitation and runoff frozen.
  ref  albedo of the Dana-only melt best (floor vis 0.463 / NIR 0.345, decay 1.29e6, SUMMA default maxima 0.95 / 0.70)
  G    compressed albedo range (floor 0.62 / 0.48, fresh maxima 0.88 / 0.62) + albedoDecayRate ramped log-linearly 6e5 at 2409 m -> 1.29e6 at 3040 m (clipped)
  GO   G with every evergreen-forest HRU (MODIFIED_IGBP vegType 1) switched to Grasslands (vegType 10)
Scores: ASO SWE amount and melt order (run_calibration_snow.aso_objective), peak SWE, streamflow KGE / peak timing / centre of mass / monthly ratios.
usage: python full_basin_albedo.py
"""
import uuid, shutil, json, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import deepstore_build as DB
import run_calibration_snow as S
import run_calibration_melt as M
import run_calibration_melt2 as M2

OUT = Path("/scratch/dlhogan/ess-project-data/calibration/diagnostics/full_basin_albedo"); OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(M2.SEED)
VARIANTS = {"ref": dict(albMinVis=PARAMS["albMinVis"], albMinNIR=PARAMS["albMinNIR"], maxVis=0.95, maxNIR=0.70, ramp=None, open=False),
            "G": dict(albMinVis=0.62, albMinNIR=0.48, maxVis=0.88, maxNIR=0.62, ramp=(2409.0, 6e5, 3040.0, 1.29e6), open=False),
            "GO": dict(albMinVis=0.62, albMinNIR=0.48, maxVis=0.88, maxNIR=0.62, ramp=(2409.0, 6e5, 3040.0, 1.29e6), open=True)}
OUTV = ["averageRoutedRunoff", "scalarSWE", "scalarSnowfall", "scalarSnowSublimation", "scalarCanopySublimation"]


class Runner(M.RunnerMelt):
    variant = None
    def __init__(self, **cfg):
        super().__init__(**cfg); self.idx = list(range(len(self.hruid)))          # all 28 HRUs
    def _vector_to_params(self, x):
        v = self.variant; p = {"swScale": ("forcing", PARAMS["swScale"]), "lwOffset": ("forcing", PARAMS["lwOffset"]), "albMinVis": ("snow", v["albMinVis"]),
                               "albMinNIR": ("snow", v["albMinNIR"]), "albedoDecayRate": ("snow", PARAMS["albedoDecayRate"])}
        for k, val in M.PRECIP.items(): p[k] = ("elev", val)
        return p
    def _apply_params(self, work, params):
        super()._apply_params(work, params); v = self.variant; n = len(self.hruid)
        DB.set_trial_param(work, "albedoMaxVisible", [v["maxVis"]]*n); DB.set_trial_param(work, "albedoMaxNearIR", [v["maxNIR"]]*n)
        if v["ramp"]:
            z0, d0, z1, d1 = v["ramp"]; w = np.clip((self.elev - z0)/(z1 - z0), 0.0, 1.0)
            DB.set_trial_param(work, "albedoDecayRate", 10**(np.log10(d0) + w*(np.log10(d1) - np.log10(d0))))
    def _subset(self, work):                                   # full basin: no subsetting (keeps downHRUindex / lateral flow); only the canopy swap
        if self.variant["open"]:
            with nc.Dataset(work/"attributes.nc", "a") as a:
                vt = a["vegTypeIndex"][:]; a["vegTypeIndex"][:] = np.where(vt == 1, 10, vt)
    def _write_filemanager(self, work, prefix, forcing=None):
        super()._write_filemanager(work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n" + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in OUTV))


def run(name):
    r = Runner(work_root=str(OUT/f"{name}_{uuid.uuid4().hex[:6]}"), **S.CFG); r.variant = VARIANTS[name]; w = Path(r.work_root)/"keep"
    params, proc = r.run_summa(np.zeros(len(r.specs)), w)
    keep = OUT/name; shutil.rmtree(keep, ignore_errors=True); shutil.copytree(w/"out", keep); (keep/"variant.json").write_text(json.dumps(VARIANTS[name]))
    shutil.rmtree(r.work_root, ignore_errors=True); return name, proc.returncode, proc.stderr[-300:]


def score(name, r):
    d = nc.Dataset(OUT/name/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    V = lambda v: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float), index=t, columns=r.hruid.astype(int)).where(lambda z: np.abs(z) < 1e8)
    swe = V("scalarSWE"); q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:, 0]), index=t)*r.area
    sel = (t >= r.analysis_start) & (t <= r.sim_end); wy = (t.year + (t.month >= 10))[sel]
    sf, sg, sc = [(V(v)[sel]*86400).groupby(wy).sum().mean() for v in ("scalarSnowfall", "scalarSnowSublimation", "scalarCanopySublimation")]; d.close()
    parts, diag = S.aso_objective(swe, r.aso, r.w_hru); w = r.hru_wt
    o = r.obs[r.analysis_start:r.sim_end]; s = q[r.analysis_start:r.sim_end]; j = pd.concat([s.rename("s"), o.rename("o")], axis=1).dropna()
    kge = 1 - np.sqrt((np.corrcoef(j.s, j.o)[0, 1] - 1)**2 + (j.s.std()/j.o.std() - 1)**2 + (j.s.mean()/j.o.mean() - 1)**2)
    wyj = j.index.year + (j.index.month >= 10); oct1 = pd.to_datetime((wyj - 1).astype(str) + "-10-01"); dd = np.asarray((j.index - oct1).days)
    com = lambda c: pd.Series(dd*j[c].values).groupby(wyj).sum()/j[c].groupby(wyj).sum()
    pk = lambda c: j[c].rolling(7, center=True).mean().groupby(wyj).idxmax().apply(lambda x: (x - pd.Timestamp(x.year - (x.month < 10), 10, 1)).days)
    mo = j.groupby(j.index.month).mean(); ratio = (mo.s/mo.o).reindex([3, 4, 5, 6, 7, 8]).round(2).to_dict()
    peak_swe = float((swe[sel]*w[None, :]).sum(1).groupby(wy).max().mean())
    return dict(variant=name, aso_swe_nmae=diag["swe_nmae"], aso_melt_order_mae=diag["melt_order_mae"], aso_obj=(parts["swe_amount"] + parts["melt_order"])/2, peak_swe_mm=peak_swe,
                subl_pct=float(100*np.sum(w*(-sg - sc))/np.sum(w*sf)), KGE=kge, peak_shift_d=float((pk("s") - pk("o")).mean()), com_shift_d=float((com("s") - com("o")).mean()),
                **{f"Q_ratio_m{k}": v for k, v in ratio.items()}), diag


def main():
    with ProcessPoolExecutor(3) as ex: res = list(ex.map(run, VARIANTS))
    print(res)
    r = Runner(work_root=str(OUT/"score"), **S.CFG); rows, diags = [], {}
    for n in VARIANTS:
        row, dg = score(n, r); rows.append(row); diags[n] = dg
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))
    fo = pd.DataFrame({"ASO": diags["ref"]["frac_left_aso"], **{n: diags[n]["frac_left_mdl"] for n in VARIANTS}})
    at = __import__("xarray").open_dataset(r.settings/"attributes.nc"); el = pd.Series(at.elevation.values.ravel(), index=at.hruId.values.astype(int)); vt = pd.Series(at.vegTypeIndex.values.ravel(), index=at.hruId.values.astype(int))
    fo.insert(0, "vegType", vt.reindex(fo.index)); fo.insert(0, "elev", el.reindex(fo.index).round(0))
    print("\nmelt order: fraction of early-spring SWE left on late ASO flights, by HRU"); print(fo.sort_values("elev", ascending=False).round(2).to_string())
    pd.DataFrame(rows).to_csv(OUT/"scores.csv", index=False); fo.to_csv(OUT/"melt_order_by_hru.csv")


if __name__ == "__main__":
    main()
