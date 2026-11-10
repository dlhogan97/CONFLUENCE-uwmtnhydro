"""
Tuolumne snow + below-snow emulator fits (2026-11-03). Training data: g35_suite3_wb.py (round-3 feasible soil; wb = albedo G, wb_dark = albedo floors -0.08).
Stages:
  below    fit tuo_below_emulator to SUMMA wb (SUMMA's SWI in, SUMMA's routed flow and ET out); hold-out = wb_dark with ITS SWI (shifted melt, same soil)
  mimic    fit tuo_snow_emulator to SUMMA wb SWE (can the structure reproduce SUMMA?)
  aso      snow fitted to ASO only: HRU SWE amount + elevation-band melt order (no Landsat; user 2026-11-03)
  joint    snow fitted to ASO + observed Apr-Nov flow (KGE, Apr-May / Jun-Jul / Aug-Sep volumes, centre of mass) through the frozen below emulator
  soil     snow FROZEN at the aso fit; below-snow parameters refitted to observed flow (same flow terms as joint), ET held in 300-380 mm/yr:
           what soil/routing behaviour does ASO-consistent snow need? (joint showed timing can only be fixed by snow that ASO rules out)
Scores are WY2013-2020; spin-up from Oct 2011. Results -> OUT/suite3_cal/emulator/<stage>.json
usage: python tuo_emulator_fit.py below|mimic|aso|joint|soil [--seeds 3]
"""
import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
import tuo_snow_emulator as SE, tuo_below_emulator as BE

ROOT = Path("/scratch/dlhogan/ess-project-data/calibration/diagnostics/full_basin_albedo/suite3_cal")
OUT = ROOT/"emulator"; OUT.mkdir(exist_ok=True)
A0, A1 = "2012-10-01", "2020-09-30"
H = SE.hrus(); TOPO = BE.topology()
X = SE.load(ROOT/"wb"/"cal_day.nc"); SPLIT = json.loads((ROOT/"wb"/"trial.json").read_text())["params"]["split_m"]
SRATE = SE.subl_rate(X); ASO = SE.aso_table(H, A0, A1)


def obs_flow():
    f = OUT/"obs_mm_day.csv"
    if not f.exists():
        import g35_suite3_cal as C
        r = C.Runner(work_root=str(OUT/"_obs"), **C.S.CFG); (r.obs/r.area*86400*1000).rename("obs").to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)["obs"]


def nse(s, o): return float(1 - ((s - o)**2).sum()/((o - o.mean())**2).sum())
def kge(s, o):
    c = np.corrcoef(s, o)[0, 1]; return float(1 - np.sqrt((c - 1)**2 + (s.std()/o.std() - 1)**2 + (s.mean()/o.mean() - 1)**2))


def flow_diag(s, o):
    j = pd.concat([s.rename("s"), o.rename("o")], axis=1)[A0:A1].dropna(); m = j.index.month
    pct = lambda ms: float(100*(j.s[np.isin(m, ms)].sum()/j.o[np.isin(m, ms)].sum() - 1))
    wy = j.index.year + (j.index.month >= 10); d = np.asarray((j.index - pd.to_datetime((wy - 1).astype(str) + "-10-01")).days)
    com = lambda c: (pd.Series(d*j[c].values).groupby(wy).sum()/j[c].groupby(wy).sum())
    a = j[np.isin(m, range(4, 12))]
    return dict(KGE_AprNov=kge(a.s, a.o), NSE_AprNov=nse(a.s, a.o), AprMay=pct([4, 5]), JunJul=pct([6, 7]), AugSep=pct([8, 9]), OctNov=pct([10, 11]),
                com_bias_d=float((com("s") - com("o")).mean()))


def below_run(p, X, swi=None):
    swi = X["swi_mb"] if swi is None else swi
    return BE.simulate(p, swi.clip(lower=0), X["swe"] > 1.0, X["SW"], X["T"], H, SPLIT, TOPO)


def below_score(p, X):
    q, et = below_run(p, X); s, o = q[A0:A1], X["q_routed"][A0:A1]
    eta = (et[A0:A1]*H.w.values).sum(1).resample("YS-OCT").sum().mean(); ets = (X["et"][A0:A1]*H.w.values).sum(1).resample("YS-OCT").sum().mean()
    return dict(nse=nse(s, o), lnse=nse(np.log(s + 0.05), np.log(o + 0.05)), et_emu=float(eta), et_summa=float(ets), **{f"summa_{k}": v for k, v in flow_diag(s, o).items()})


def fit(fun, k, seeds, maxiter=200, popsize=15):
    best = None
    for sd in range(1, seeds + 1):
        r = differential_evolution(fun, [(0, 1)]*k, seed=sd, maxiter=maxiter, popsize=popsize, tol=1e-7, polish=False, workers=1)
        print(f"  seed {sd}: {r.fun:.4f}", flush=True)
        if best is None or r.fun < best.fun: best = r
    return best


def stage_below(seeds):
    def f(u):
        s = below_score(BE.params_of(u), X); return (1 - s["nse"]) + (1 - s["lnse"]) + abs(s["et_emu"]/s["et_summa"] - 1)
    r = fit(f, len(BE.PARAMS), seeds); p = BE.params_of(r.x)
    Xr = SE.load(ROOT/"wb_dark"/"cal_day.nc")
    res = dict(params=p, train=below_score(p, X), holdout_dark=below_score(p, Xr),
               summa_dark_vs_G=dict(summa_dAprMay_pct=float(100*(Xr["q_routed"][A0:A1].pipe(lambda z: z[z.index.month.isin([4, 5])].sum())
                                                         /X["q_routed"][A0:A1].pipe(lambda z: z[z.index.month.isin([4, 5])].sum()) - 1))))
    (OUT/"below.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


SCORER = SE.AsoScorer(X["T"].index, ASO, H)


def snow_diag(p):
    swe, swi = SE.simulate(p, X, H, SRATE); nmae, mo, fm = SCORER(swe.values)
    return swe, swi, dict(aso_nmae=nmae, band_order_mae=mo, band_order={int(e): [round(float(a), 2), round(float(m), 2)] for e, a, m in zip(SCORER.elev, SCORER.fo, fm)})


def stage_mimic(seeds):
    sw = X["swe"][A0:A1]
    def f(u):
        swe, _ = SE.simulate(SE.params_of(u), X, H, SRATE); return float(((swe[A0:A1] - sw).abs()*H.w.values).sum(1).sum()/(sw*H.w.values).sum(1).sum())
    r = fit(f, len(SE.PARAMS), seeds); p = SE.params_of(r.x); _, swi, d = snow_diag(p)
    res = dict(params=p, swe_nmae_vs_summa=float(r.fun), **d, summa=snow_summa())
    (OUT/"mimic.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def snow_summa():
    mo = SE.band_melt_order(X["swe"], ASO, H)
    return dict(aso_nmae=SE.aso_amount(X["swe"], ASO, H), band_order_mae=float((mo.mdl - mo.aso).abs().mean()),
                band_order={int(e): [round(float(a), 2), round(float(m), 2)] for e, a, m in zip(mo.elev, mo.aso, mo.mdl)})


def stage_flow(seeds, joint):
    pb = json.loads((OUT/"below.json").read_text())["params"]; obs = obs_flow()
    def run(p):
        swe, swi, d = snow_diag(p); q, et = BE.simulate(pb, swi, swe > 1.0, X["SW"], X["T"], H, SPLIT, TOPO)
        d.update(flow_diag(q, obs)); d["ET"] = float((et[A0:A1]*H.w.values).sum(1).resample("YS-OCT").sum().mean()); return d, q
    def f(u):
        d, _ = run(SE.params_of(u)); j = d["aso_nmae"]/0.25 + d["band_order_mae"]/0.10
        if joint: j += (1 - d["KGE_AprNov"])/0.1 + (abs(d["AprMay"]) + abs(d["JunJul"]) + abs(d["AugSep"]))/3/15 + abs(d["com_bias_d"])/10
        return j
    r = fit(f, len(SE.PARAMS), seeds, maxiter=120); p = SE.params_of(r.x); d, q = run(p)
    name = "joint" if joint else "aso"
    q.rename("q_mm_d").to_csv(OUT/f"{name}_q.csv")
    res = dict(params=p, objective=float(r.fun), **d)
    # reference: SUMMA's own SWI through the same below emulator, scored against obs
    q0, _ = below_run(pb, X); res["reference_summa_swi"] = flow_diag(q0, obs); res["reference_summa_actual"] = flow_diag(X["q_routed"], obs)
    (OUT/f"{name}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def stage_soil(seeds):
    ps = json.loads((OUT/"aso.json").read_text())["params"]; obs = obs_flow(); swe, swi, dsn = snow_diag(ps); snowy = swe > 1.0
    pb0 = json.loads((OUT/"below.json").read_text())["params"]
    def run(pb):
        q, et = BE.simulate(pb, swi, snowy, X["SW"], X["T"], H, SPLIT, TOPO); d = flow_diag(q, obs)
        d["ET"] = float((et[A0:A1]*H.w.values).sum(1).resample("YS-OCT").sum().mean()); return d, q
    def f(u):
        d, _ = run(BE.params_of(u))
        return ((1 - d["KGE_AprNov"])/0.1 + (abs(d["AprMay"]) + abs(d["JunJul"]) + abs(d["AugSep"]))/3/15 + abs(d["com_bias_d"])/10
                + max(300 - d["ET"], 0, d["ET"] - 380)/10)
    r = fit(f, len(BE.PARAMS), seeds, maxiter=120); pb = BE.params_of(r.x); d, q = run(pb); q.rename("q_mm_d").to_csv(OUT/"soil_q.csv")
    res = dict(params=pb, params_summa_fit=pb0, objective=float(r.fun), **d, snow=dsn, frozen_soil=run(pb0)[0])
    (OUT/"soil.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    {"below": lambda: stage_below(a.seeds), "mimic": lambda: stage_mimic(a.seeds), "aso": lambda: stage_flow(a.seeds, False),
     "joint": lambda: stage_flow(a.seeds, True), "soil": lambda: stage_soil(a.seeds)}[a.stage]()
