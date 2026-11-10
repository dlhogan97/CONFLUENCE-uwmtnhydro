"""
Follow-up tests to melt_energy_inversion.py (2026-10-27), same pillow years, same site HRUs (8 for Dana; 13/18 for Tuolumne Meadows), same forcing.
  albedo variants   the Dana inversion implied a near-constant melt-season albedo (~0.70-0.76) while the model swings 0.81 (fresh) -> 0.62 (old):
                    raise the floor (albedoMinVisible / albedoMinNearIR), lower the fresh-snow maximum (albedoMaxVisible / albedoMaxNearIR), faster early decay
  open canopy       HRUs 13 and 18 (evergreen forest, 56-60% trees) switched to Grasslands (MODIFIED_IGBP vegType 10), as the pillow sits in an open meadow
Reported per variant and site: median and mean missing melt energy R (W/m2), R by model-albedo tercile, melt timing errors vs the pillow (t50, t10, d8020; days).
usage: python melt_inversion_tests.py
"""
import uuid, shutil, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import deepstore_build as DB
import run_calibration_melt2 as M2
import melt_energy_inversion as I
from run_calibration import seed_vector

BASE = dict(albMinVis=I.PARAMS["albMinVis"], albMinNIR=I.PARAMS["albMinNIR"], albedoDecayRate=I.PARAMS["albedoDecayRate"], maxVis=0.95, maxNIR=0.70, open=False)
VARIANTS = {"A base": {}, "B floor 0.62/0.48": dict(albMinVis=0.62, albMinNIR=0.48), "C floor 0.70/0.52": dict(albMinVis=0.70, albMinNIR=0.52),
            "D B + max 0.88/0.62": dict(albMinVis=0.62, albMinNIR=0.48, maxVis=0.88, maxNIR=0.62),
            "E C + max 0.88/0.62": dict(albMinVis=0.70, albMinNIR=0.52, maxVis=0.88, maxNIR=0.62),
            "F D + decay 6e5": dict(albMinVis=0.62, albMinNIR=0.48, maxVis=0.88, maxNIR=0.62, albedoDecayRate=6e5),
            "O open canopy (13,18 grassland)": dict(open=True), "OD open + D": dict(albMinVis=0.62, albMinNIR=0.48, maxVis=0.88, maxNIR=0.62, open=True)}
OUT = I.OUT/"tests"; OUT.mkdir(exist_ok=True)


class Runner(I.Runner):
    variant = BASE; phys = None
    def _vector_to_params(self, x):                                       # physical values passed directly (the calibration bounds would clip the higher floors)
        p = {sp.name: (sp.kind, float(self.phys[sp.name])) for sp in self.specs}
        for k, v in M2.PRECIP.items(): p[k] = ("elev", v)
        return p
    def _apply_params(self, work, params):
        super()._apply_params(work, params); n = len(self.hruid); v = self.variant
        DB.set_trial_param(work, "albedoMaxVisible", [v["maxVis"]]*n); DB.set_trial_param(work, "albedoMaxNearIR", [v["maxNIR"]]*n)
    def _subset(self, work):
        super()._subset(work)
        if self.variant["open"]:
            with nc.Dataset(work/"attributes.nc", "a") as a:
                ids = np.asarray(a["hruId"][:]).astype(int); vt = a["vegTypeIndex"][:]
                for h in (13, 18): vt[ids == h] = 10
                a["vegTypeIndex"][:] = vt


def run(args):
    name, y = args; v = {**BASE, **VARIANTS[name]}
    r = Runner(work_root=f"/tmp/claude-1008/melttest/{uuid.uuid4().hex[:8]}", **M2.CFG); r.variant = v
    p = dict(I.PARAMS, albMinVis=v["albMinVis"], albMinNIR=v["albMinNIR"], albedoDecayRate=v["albedoDecayRate"]); r.phys = p; x = np.zeros(len(r.specs)); w = Path(r.work_root)/"w"
    r.run_start, r.run_end = f"{y-1}-10-01", f"{y}-09-30"; params, proc = r.run_summa(x, w)
    d = nc.Dataset(w/"out"/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    out = {k: pd.DataFrame(np.array(d[f"{k}_mean"][:]).astype(float), index=t, columns=I.HRUS).where(lambda z: np.abs(z) < 1e8) for k in ("scalarSWE", "scalarSnowfall", "scalarSnowAlbedo")}
    d.close(); shutil.rmtree(r.work_root, ignore_errors=True); return name, y, out


def main():
    yrs = sorted(set(I.YEARS["Dana"]) | set(I.YEARS["TUM"]))
    with ProcessPoolExecutor(16) as ex: res = list(ex.map(run, [(n, y) for n in VARIANTS for y in yrs]))
    R = {(n, y): o for n, y, o in res}
    PIL = {k: pd.read_csv(v[1], index_col=0, parse_dates=True).value.asfreq("D") for k, v in M2.PIL.items()}
    rows, tim = [], []
    for n in VARIANTS:
        for site in ("Dana", "TUM"):
            for y in I.YEARS[site]:
                s = I.site_series(R[(n, y)], site); swe_m = s["scalarSWE"]; swe_o = PIL[site].reindex(swe_m.index).interpolate(limit=3)
                sm, so = swe_m.rolling(3, center=True, min_periods=2).mean(), swe_o.rolling(3, center=True, min_periods=2).mean(); dm, do = sm.diff(), so.diff()
                ok = (sm > 50) & (so > 50) & (dm <= 1) & (do <= 1) & (s["scalarSnowfall"]*86400 < 1) & swe_m.index.month.isin([3, 4, 5, 6, 7])
                rows.append(pd.DataFrame(dict(variant=n, site=site, wy=y, R=(I.LF*(dm - do)/86400.0), albedo=s["scalarSnowAlbedo"]))[ok])
                a, b = M2.melt_times(swe_m), M2.melt_times(swe_o)
                if a and b: tim.append(dict(variant=n, site=site, wy=y, **{k: a[k] - b[k] for k in ("t50", "t10", "d8020", "peak")}, ratio=a["peak_mm"]/b["peak_mm"]))
    D = pd.concat(rows); T = pd.DataFrame(tim); D.to_csv(OUT/"residual_daily.csv"); T.to_csv(OUT/"timing.csv", index=False)
    pd.set_option("display.width", 250)
    s1 = D.groupby(["site", "variant"]).R.agg(["median", "mean", lambda z: np.mean(np.abs(z))]).rename(columns={"<lambda_0>": "mean_abs"})
    dana = D[(D.site == "Dana")].copy(); dana["abin"] = dana.groupby("variant").albedo.transform(lambda z: pd.qcut(z, 3, labels=["low", "mid", "high"]))
    s1 = s1.join(dana.groupby(["site", "variant", "abin"], observed=True).R.mean().unstack(), how="left")
    s1 = s1.join(D[D.site == "Dana"].groupby(["site", "variant"]).albedo.agg(alb_p10=lambda z: z.quantile(.1), alb_p90=lambda z: z.quantile(.9)), how="left")
    print("missing melt energy R (W/m2) and, for Dana, R by model-albedo tercile and the model albedo range"); print(s1.round(1).to_string())
    g = T.groupby(["site", "variant"]).agg(t50_mae=("t50", lambda z: np.mean(np.abs(z))), t10_mae=("t10", lambda z: np.mean(np.abs(z))), d8020_mae=("d8020", lambda z: np.mean(np.abs(z))),
                                           t50_bias=("t50", "mean"), t10_bias=("t10", "mean"), peak_ratio=("ratio", "mean"), n=("t50", "size"))
    g["score"] = g[["t50_mae", "t10_mae", "d8020_mae"]].mean(1)/7
    print("\nmelt timing vs pillow (days)"); print(g.round(2).to_string())


if __name__ == "__main__":
    main()
