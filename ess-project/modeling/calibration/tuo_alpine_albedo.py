"""
Extend the pillow-derived albedo setup into the alpine (2026-10-27). The ASO energy inversion (alpine_aso_inversion.py) found the alpine HRUs
(> 3300 m) carry ~24 W/m2 of excess melt energy, almost all through absorbed shortwave (implied model albedo 0.62-0.69 vs ~0.74-0.77 needed).
SUMMA has no wind redistribution, so the test is energy-only: continue the elevation ramps of variant G above 3040 m.
  G     floor vis 0.62 / NIR 0.48, fresh max 0.88 / 0.62, decay 6e5 @2409 -> 1.29e6 @3040 m, constant above (the full-basin best so far)
  G1    decay ramp continued to 3.0e6 at 3686 m
  G2    G1 + floor ramped 3040 -> 3686 m: vis 0.62 -> 0.70, NIR 0.48 -> 0.55
  G3    decay -> 5.0e6 and floor vis -> 0.75, NIR -> 0.60 at 3686 m
HRUs 1-12 (alpine and 3040 m bands) run alone (snow only), WY2012-2020; scored against ASO (SWE nMAE, melt order) on these HRUs and by the
early->late interval energy residual R (W/m2; negative = model has excess energy).
usage: [ALP_VARIANTS=G3,G4,G5] python tuo_alpine_albedo.py
  G4/G5 push the alpine albedo past the physically plausible range (user request), raising the fresh-snow maxima too.
"""
import uuid, shutil, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import deepstore_build as DB
import run_calibration_snow as S
import run_calibration_melt as M
import full_basin_albedo as F

HRUS = list(range(1, 13)); OUT = F.OUT/("alpine_albedo" + os.environ.get("ALP_TAG", "")) if False else F.OUT/"alpine_albedo"; OUT.mkdir(exist_ok=True); LF = 3.34e5
Z0, Z1, ZT = 2409.0, 3040.0, 3686.0
VAR = {"G": dict(dtop=1.29e6, vtop=0.62, ntop=0.48), "G1": dict(dtop=3.0e6, vtop=0.62, ntop=0.48),
       "G2": dict(dtop=3.0e6, vtop=0.70, ntop=0.55), "G3": dict(dtop=5.0e6, vtop=0.75, ntop=0.60),
       # beyond the plausible range (user, 2026-10-27): fresh-snow maxima also raised up high so floor < max
       "G4": dict(dtop=1.0e7, vtop=0.85, ntop=0.68, mvtop=0.95, mntop=0.75), "G5": dict(dtop=3.0e7, vtop=0.92, ntop=0.75, mvtop=0.97, mntop=0.80),
       "G35": dict(dtop=7.0e6, vtop=0.80, ntop=0.64, mvtop=0.92, mntop=0.69), "G38": dict(dtop=8.5e6, vtop=0.83, ntop=0.66, mvtop=0.94, mntop=0.72)}
import os
if os.environ.get("ALP_VARIANTS"): VAR = {k: VAR[k] for k in os.environ["ALP_VARIANTS"].split(",")}


def ramp(z, v0, v1, v2, log=False):
    """piecewise in elevation: v0 at Z0 -> v1 at Z1 -> v2 at ZT (log-linear if log)."""
    f = (lambda a: np.log10(a)) if log else (lambda a: a); g = (lambda a: 10**a) if log else (lambda a: a)
    lo = f(v0) + np.clip((z - Z0)/(Z1 - Z0), 0, 1)*(f(v1) - f(v0)); hi = f(v1) + np.clip((z - Z1)/(ZT - Z1), 0, 1)*(f(v2) - f(v1))
    return g(np.where(z <= Z1, lo, hi))


class Runner(F.Runner):
    vv = None
    def __init__(self, **cfg):
        super().__init__(**cfg); ids = self.hruid.astype(int); self.idx = [int(np.where(ids == h)[0][0]) for h in HRUS]
    def _apply_params(self, work, params):
        super()._apply_params(work, params); v = self.vv; z = self.elev
        DB.set_trial_param(work, "albedoDecayRate", ramp(z, 6e5, 1.29e6, v["dtop"], log=True))
        DB.set_trial_param(work, "albedoMinVisible", ramp(z, 0.62, 0.62, v["vtop"])); DB.set_trial_param(work, "albedoMinNearIR", ramp(z, 0.48, 0.48, v["ntop"]))
        DB.set_trial_param(work, "albedoMaxVisible", ramp(z, 0.88, 0.88, v.get("mvtop", 0.88))); DB.set_trial_param(work, "albedoMaxNearIR", ramp(z, 0.62, 0.62, v.get("mntop", 0.62)))
    def _subset(self, work): M.RunnerMelt._subset(self, work)                  # snow only: HRUs 1-12 alone
    def _write_filemanager(self, work, prefix, forcing=None):
        M.RunnerMelt._write_filemanager(self, work, prefix, forcing)
        (work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n" + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in
                                              ("scalarSWE", "scalarSnowAlbedo", "scalarGroundAbsorbedSolar", "scalarSnowSublimation", "scalarSnowfall")))


def run(name):
    r = Runner(work_root=str(OUT/f"{name}_{uuid.uuid4().hex[:6]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.vv = VAR[name]; w = Path(r.work_root)/"keep"
    params, proc = r.run_summa(np.zeros(len(r.specs)), w); d = nc.Dataset(w/"out"/"cal_day.nc")
    t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    out = {v: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float), index=t, columns=HRUS).where(lambda z: np.abs(z) < 1e8) for v in ("scalarSWE", "scalarSnowAlbedo", "scalarGroundAbsorbedSolar", "scalarSnowSublimation", "scalarSnowfall")}
    d.close(); shutil.rmtree(r.work_root, ignore_errors=True); return name, out


def main():
    with ProcessPoolExecutor(len(VAR)) as ex: res = dict(ex.map(run, VAR))
    r = S.RunnerSnow(work_root=str(OUT/"score"), **S.CFG); aso = r.aso[r.aso.hru.isin(HRUS)]; w = r.w_hru.loc[HRUS]
    el = pd.Series(r.elev, index=r.hruid.astype(int)).loc[HRUS]
    a = aso.copy(); a["wy"] = a.date.dt.year; early = a[a.date.dt.month.isin([3, 4])]; late = a[((a.date.dt.month == 5) & (a.date.dt.day >= 20)) | a.date.dt.month.isin([6, 7])]
    rows, fo, rr = [], {}, {}
    for n, o in res.items():
        swe = o["scalarSWE"]; parts, diag = S.aso_objective(swe, aso, w); fo[n] = diag["frac_left_mdl"]; fo["ASO"] = diag["frac_left_aso"]
        R = []
        for (wy, h), e in early.groupby(["wy", "hru"]):
            l = late[(late.wy == wy) & (late.hru == h)]
            if l.empty: continue
            e = e.sort_values("date").iloc[-1]; l = l.sort_values("date").iloc[0]
            if e.aso_mm < 100: continue
            days = (l.date - e.date).days; s = swe[h]; snow = s[e.date:l.date] > 1.0; m_loss = float(s.get(e.date, np.nan) - s.get(l.date, np.nan))
            sel = snow[snow].index; ab = o["scalarSnowAlbedo"][h].reindex(sel); ab = ab[(ab > 0) & (ab < 1)]
            R.append(dict(hru=h, R=LF*((e.aso_mm - l.aso_mm)/days - m_loss/max(int(snow.sum()), 1))/86400.0, albedo=float(ab.mean()) if len(ab) else np.nan))
        R = pd.DataFrame(R).join(el.rename("elev"), on="hru"); rr[n] = R.groupby("hru").R.mean()
        alp = R[R.elev > 3300]; mid = R[R.elev <= 3300]
        late_sw = (swe.loc[[d for d in sorted(late.date.unique()) if d in swe.index]]*w.values[None, :]).sum(1).mean()
        rows.append(dict(variant=n, aso_swe_nmae=diag["swe_nmae"], aso_melt_order_mae=diag["melt_order_mae"], R_alpine=alp.R.mean(), R_3040=mid.R.mean(),
                         melt_season_albedo_alpine=alp.albedo.mean(), melt_season_albedo_3040=mid.albedo.mean(), model_late_flight_SWE_mm=late_sw))
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))
    F_ = pd.DataFrame(fo); F_.insert(0, "elev", el.reindex(F_.index).round(0)); print("\nshare of early-spring SWE left on late flights"); print(F_.sort_values("elev", ascending=False).round(2).to_string())
    RR = pd.DataFrame(rr); RR.insert(0, "elev", el.reindex(RR.index).round(0)); print("\nenergy residual R by HRU (W/m2; negative = model excess)"); print(RR.sort_values("elev", ascending=False).round(1).to_string())
    pd.DataFrame(rows).to_csv(OUT/"scores.csv", index=False); F_.to_csv(OUT/"melt_order.csv"); RR.to_csv(OUT/"residual_by_hru.csv")


if __name__ == "__main__":
    main()
