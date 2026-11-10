"""
Tuolumne G35 calibration, round 2 (2026-10-29): the East matched-calibration objective, with precipitation free within realistic ranges.
What g35_suite_cal.py (round 1) taught: downslope soil drainage closes most of the Mar-May deficit (163 -> 258 mm); the low-elevation precipitation
boost was rejected because it fell as snow where ASO has none (boost vs ASO SWE error r = +0.93, low-band rain fraction stuck at 11%), so the rain/snow
threshold is now free alongside it; best KGE was only 0.61-0.66.
Fixed: G35 albedo ramps, temperature / SW / LW corrections, frozenPrecipMultip ramp, every ET parameter, thinned soil (depth 0.5 limited by roots),
downHRUindex chain intact.
Free (9): the six of round 1 (lateral drainage, fieldCapacity, zScale_TOPMODEL, k_macropore, lowP_boost, routing scale) plus
  tempCritRain    [270.65, 273.15] K   rain/snow threshold. SUMMA makes rain where air temperature is ABOVE it (derivforce.f90:297-304), so LOWERING it
                                       gives more rain: 271.15 undoes the uniform -2 C forcing offset for phase only. (A first launch on 2026-10-29 had
                                       the sign backwards, 273.15-275.65, and was stopped after 48 trials: suite2_cal/aborted_wrong_phase_sign/.)
  precip_scale    [0.90, 1.15]         basin-wide precipitation multiplier (PRISM-scale uncertainty); applied on top of the low-elevation boost
Objective = the East matched objective (east_independent_compare.objective), WY2013-2020, equal weights, no low-flow level or memory term:
  peak_timing, peak_volume, falling_limb, com_timing, monthly_bias (Apr-Oct)   signature_objective_routing defaults
  recession       median_y |ln(tau_sim / tau_obs)| / ln 1.25, Aug-Sep lower-envelope tau (east_ctl_treat_cal_v2.recession_taus); obs median ~31 d
  winter_volume   |DJF volume sim/obs - 1| / 10%
Rejection: KGE < 0.75 (objective = 10 + 10*(0.75 - KGE), so the search climbs KGE until it is feasible); ASO SWE nMAE > 1.15 x G35 (0.556) or
melt-order error > 1.15 x G35 (0.52) keeps the snowpack elevation structure as calibrated. Reported: everything g35_suite_cal.score reports.
Every trial -> OUT/suite2_cal/trials.jsonl.
usage: python g35_suite2_cal.py [--workers 27] [--popsize 3] [--maxiter 16]
"""
import sys, json, uuid, time, shutil, argparse, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
from scipy.optimize import differential_evolution
import run_calibration_snow as S
import signature_objective_plus as P
import full_basin_albedo as F
import g35_soil_tests as ST
import g35_suite_cal as G
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"utils"/"singlebasin_calibration"))
from east_ctl_treat_cal_v2 import recession_taus

OUT = F.OUT/"suite2_cal"; OUT.mkdir(exist_ok=True)
BOUNDS = G.BOUNDS + [("tempCritRain", 270.65, 273.15), ("precip_scale", 0.90, 1.15)]
SEED = [0.955, 0.137, 1.509, -2.556, 0.003, 4.14, 273.15, 1.0]                # round-1 best, threshold and scale at their old values
KGE_MIN, ASO_MAX, MO_MAX, FAIL = 0.75, 1.15*0.483, 1.15*0.452, 10.0


def physical(x):
    p = G.physical(x[:len(G.BOUNDS)]); p.update(tempCritRain=float(x[6]), precip_scale=float(x[7])); return p


class Runner(G.Runner):
    def _apply_forcing(self, work, params):
        fdir = ST.Runner._apply_forcing(self, work, params)
        m = self.soil["precip_scale"]*(1.0 + self.soil["lowP_boost"]*np.clip((G.Z_HI - self.elev)/(G.Z_HI - G.Z_LO), 0.0, 1.0))
        for f in sorted(fdir.glob("*.nc")):
            with nc.Dataset(f, "a") as d: d["pptrate"][:] = np.array(d["pptrate"][:])*m[None, :]
        return fdir


def score(out, r, soil):
    _, p1, diag = G.score(out, r, soil)                                           # round-1 terms kept as diagnostics
    d = nc.Dataset(out); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).astype(float).ravel()[:len(t)], index=t)*r.area; d.close()
    j = pd.concat([q.rename("s"), r.obs.rename("o")], axis=1)[r.analysis_start:r.sim_end].dropna()
    rt = P.signature_objective_routing(j.s, j.o); parts = dict(rt.parts)
    y0, y1 = int(r.analysis_start[:4]) + 1, int(r.sim_end[:4])
    tm, to = recession_taus(j.s, y0, y1), recession_taus(j.o, y0, y1)
    k = pd.concat([tm.rename("m"), to.rename("o")], axis=1).fillna(5000.0).clip(upper=5000.0)
    parts["recession"] = float(np.median(np.abs(np.log(k.m/k.o)))/np.log(1.25))
    dj = j[j.index.month.isin([12, 1, 2])].sum(); parts["winter_volume"] = float(abs(dj.s/dj.o - 1.0)/0.10)
    obj = float(np.mean(list(parts.values()))); kge = rt.diagnostics["KGE"]
    diag.update(recess_sim_d=float(tm.median()), recess_obs_d=float(to.median()), DJF_ratio=float(dj.s/dj.o), aso_melt_order=p1["melt_order"],
                **{f"r1_{a}": b for a, b in p1.items()})
    if kge < KGE_MIN: obj = FAIL + (KGE_MIN - kge)*10
    elif diag["aso_swe_nmae"] > ASO_MAX: obj = FAIL + (diag["aso_swe_nmae"] - ASO_MAX)*10
    elif p1["melt_order"] > MO_MAX: obj = FAIL + (p1["melt_order"] - MO_MAX)*10
    return obj, parts, diag


def evaluate(x):
    soil = physical(x); t0 = time.time(); r = Runner(work_root=str(OUT/f"w_{uuid.uuid4().hex[:8]}"), **S.CFG); r.variant = F.VARIANTS["G"]; r.soil = soil
    w = Path(r.work_root)/"keep"
    try:
        _, proc = r.run_summa(np.zeros(len(r.specs)), w); out = w/"out"/"cal_day.nc"
        if proc.returncode != 0 or not out.exists(): obj, rec = FAIL + 5, dict(reason="summa_failed", stderr=proc.stderr[-300:])
        else: obj, parts, diag = score(out, S.RunnerSnow(work_root=str(OUT/"score"), **S.CFG), soil); rec = dict(parts=parts, diagnostics=diag)
    except Exception as e:
        obj, rec = FAIL + 5, dict(reason=f"{type(e).__name__}:{e}")
    finally:
        shutil.rmtree(r.work_root, ignore_errors=True)
    with open(OUT/"trials.jsonl", "a") as fh: fh.write(json.dumps(dict(x=list(map(float, x)), params=soil, objective=obj, eval_s=time.time() - t0, **rec), default=float) + "\n")
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=27); ap.add_argument("--popsize", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=16); ap.add_argument("--seed", type=int, default=8); a = ap.parse_args(); hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun)); (OUT/"history.json").write_text(json.dumps(hist))
        print(f"[g35 suite2] gen {len(hist):2d}: best {hist[-1]:.4f}  x {np.round(intermediate_result.x, 3).tolist()}", flush=True)
    res = differential_evolution(evaluate, bounds=[(lo, hi) for _, lo, hi in BOUNDS], popsize=a.popsize, init="latinhypercube", x0=SEED, maxiter=a.maxiter,
                                 workers=a.workers, updating="deferred", polish=False, tol=0, callback=cb, disp=False, seed=a.seed)
    (OUT/"result.json").write_text(json.dumps(dict(x=list(map(float, res.x)), params=physical(res.x), objective=float(res.fun), nfev=int(res.nfev))))
    print(f"[g35 suite2] DONE {res.fun:.4f} nfev {res.nfev}", flush=True)


if __name__ == "__main__":
    main()
