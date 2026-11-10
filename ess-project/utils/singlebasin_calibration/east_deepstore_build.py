"""
East River deep-store TREATMENT, minimally calibrated (2026-10-23).

Control = the routing/timing control (run_calibration_east_routing best): its fast-bucket parameters are used UNCHANGED.
Treatment adds, on top of that control:
  * a LEAK from every fast bucket to one basin store, proportional to the bucket's own storage-driven outflow:
        leak_i = f * baseflow_i       (leak exponent = baseflow exponent, rate = baseflow rate, so  lam = f)
  * a LINEAR deep store that releases to the stream:   Q_deep = S_deep / tau_d
No parameter is fitted to streamflow skill or to memory:
  tau_d  from the OBSERVED winter recession (Dec 1-Mar 15 e-folding of the 7-day flow, median ~363 d, WY1987-2025) -> primary tau_d = 1 yr
  f      from a volume balance: the f that makes the model's mean Dec-Feb flow equal the observed one over WY1987-2024
         (Dec-Feb, not Dec-Mar: March carries the control's early-melt bias). Solved separately for every tau_d.
  S_deep(0) = equilibrium (f * mean bucket baseflow * tau_d), then 4 passes over the record.
Everything else is DIAGNOSTIC: fall and winter lag-1 (Spearman, residual of ln low flow on ln Apr-Jul peak), prior-year storage vs low flow,
routing objective terms, seasonal volumes. A sensitivity grid over tau_d and f shows how little the answer depends on hitting one point.

The emulator is the one validated against SUMMA (emu_core); the first check printed here is that f=0 reproduces the SUMMA control output.
usage: STAGEB_T0=1981-10-01 STAGEB_END=2025-09-30 TREAT_STATE=routfull python east_deepstore_build.py
"""
import os, sys, json, glob, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd, netCDF4 as nc
from scipy.optimize import brentq
from scipy.stats import spearmanr
sys.path.insert(0, "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration")
import stageB_emulator as B, emu_core as E, signature_objective_plus as P
from run_calibration_v2 import best_trial

STATE = os.environ.get("TREAT_STATE", "routfull")
OUT = B.OUT; YR = 365.25*86400.0
RUNS = "/scratch/dlhogan/ess-project-data/calibration/runs/"
if STATE == "routfull":
    bt = best_trial(RUNS+"East_River_routing_20261022_153020")[0]["params"]
    RATE, SF, EX, SCALE = bt["aquiferBaseflowRate"], bt["aquiferScaleFactor"], bt["aquiferBaseflowExp"], bt["routingGammaScale"]
else:                                           # test state: the previous control used the stage-A frozen aquifer
    RATE, SF, EX, SCALE = 9.43986e-06, 2.65777, 2.056814, 51629.17
st = B.build_state(STATE); W = st["A"]/st["A"].sum(); obs = E.obs
NT, NH = st["Rch"].shape; ONE = lambda m, d: m(d)


def simulate(lam, tau_yr, b0=None, passes=4):
    """lam = f. Returns Qs (m/s per HRU), Qd (m/s basin), Sd (m)."""
    Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT); S = st["S0"].copy(); tau_s = tau_yr*YR
    sdi = 0.0 if (lam == 0 or b0 is None) else lam*b0*tau_s
    for _ in range(passes):
        S, sdi = E.run_buckets(st["Rch"], st["A"], SF, RATE, EX, float(lam), EX, 1.0, 1.0, 1.0/tau_s, S, float(sdi), 3600.0, Qs, Qd, Sd)
    return Qs, Qd, Sd


def flow(Qs, Qd): return B.daily_Q(st, Qs, Qd, SCALE)


# ---- 1. validation: f = 0 must reproduce the SUMMA control
Q0s, Q0d, S0d = simulate(0.0, 1.0); q0 = flow(Q0s, Q0d); b0 = float((Q0s*W).sum(1).mean())
d = nc.Dataset(f"{B.SRC[STATE][0]}/out/best_timestep.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
Atot = float(np.array(nc.Dataset(f"{B.SRC[STATE][0]}/attributes.nc")["HRUarea"][:]).sum())
summa = (pd.Series(np.array(d["averageRoutedRunoff"][:, 0]), index=t)*Atot).resample("D").mean()
c = pd.concat([summa.rename("s"), q0.rename("e")], axis=1)["1986-10-01":"2024-09-30"].dropna()
print(f"state {STATE}: bucket rate {RATE:.3g} m/s, Sf {SF:.3f} m, exp {EX:.3f}, routing scale {SCALE:.0f} s")
print(f"VALIDATION f=0 vs SUMMA control (WY1987-2024): daily r {np.corrcoef(c.s, c.e)[0, 1]:.4f}, mean ratio {c.e.mean()/c.s.mean():.4f}, max |diff| {np.abs(c.s-c.e).max():.3f} m3/s")
print(f"mean basin bucket baseflow {b0*YR*1000:.0f} mm/yr")

# ---- 2. diagnostics
A_, B_ = "1986-10-01", "2024-09-30"
mo = lambda q, m: q[A_:B_][q[A_:B_].index.month.isin(m)].mean()
OBS_DJF = mo(obs, [12, 1, 2])


def winter_series(q, y0=1987, y1=2024):
    out = {}
    for y in range(y0, y1+1):
        w = q[f"{y-1}-12-01":f"{y}-03-31"].rolling(30, center=True, min_periods=20).mean().dropna()
        if len(w) > 60: out[y] = np.log(max(w.min(), 0.01))
    return pd.Series(out)


def lag1_pair(q, y0, y1):
    fall = E.memory(q, y0, y1)["res1"]; lf, pk = E.series(q, y0, y1); ws = winter_series(q, y0, y1); r = E.resid(ws, pk.reindex(ws.index))
    return fall, E.lag_r(r, 1)


def partial(x, y, z):
    d_ = pd.concat([x.rename("x"), y.rename("y"), z.rename("z")], axis=1).dropna()
    rx = d_.x - np.polyval(np.polyfit(d_.z, d_.x, 1), d_.z); ry = d_.y - np.polyval(np.polyfit(d_.z, d_.y, 1), d_.z); return spearmanr(rx, ry)[0]


def stor_link(Sd, q, y0, y1):
    S = pd.Series(Sd, index=st["tt"]).resample("D").mean(); lf, pk = E.series(q, y0, y1)
    sp = pd.Series({y: S.get(pd.Timestamp(f"{y-1}-09-30"), np.nan) for y in lf.index}); idx = [y for y in lf.index if y0 <= y <= y1]
    return partial(lf[idx], sp[idx], pk[idx])


OBS_LAG = {w: lag1_pair(obs, *w) for w in ((1987, 2024), (2009, 2024))}
OBS_LINK = {w: None for w in ((1987, 2024), (2009, 2024))}


def evaluate(lam, tau_yr, tag=""):
    Qs, Qd, Sd = simulate(lam, tau_yr, b0); q = flow(Qs, Qd)
    unr = ((st["SR"]+Qs)*W).sum(1)+Qd; hr = pd.DataFrame(dict(Sd=Sd, Qd=Qd, unr=unr), index=st["tt"])[A_:B_]
    ann = hr.Sd.groupby(hr.index.year + (hr.index.month >= 10)).mean()
    r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], obs["2008-10-01":"2020-09-30"])
    f87, w87 = lag1_pair(q, 1987, 2024); f09, w09 = lag1_pair(q, 2009, 2024)
    lk = {w: stor_link(Sd, q, *w) for w in ((1987, 2024), (2009, 2024))}
    lko = {w: stor_link(Sd, obs, *w) for w in ((1987, 2024), (2009, 2024))}
    sea = lambda m: q[A_:B_][q[A_:B_].index.month.isin(m)].sum()*86400/Atot*1000/38.0
    return dict(tau_yr=tau_yr, f=lam, leak_share_pct=100*lam/(1+lam) if lam else 0.0, DJF_ratio=mo(q, [12, 1, 2])/OBS_DJF, vol_OctMar_mm=sea([10, 11, 12, 1, 2, 3]), vol_AprJul_mm=sea([4, 5, 6, 7]),
                vol_AugSep_mm=sea([8, 9]), obj=r.objective, **{f"t_{k}": v for k, v in r.parts.items()}, KGE=r.diagnostics["KGE"],
                fall_lag1_87=f87, fall_lag1_09=f09, win_lag1_87=w87, win_lag1_09=w09, link_model_87=lk[(1987, 2024)], link_model_09=lk[(2009, 2024)],
                link_obs_87=lko[(1987, 2024)], link_obs_09=lko[(2009, 2024)], deep_mean_m=float(hr.Sd.mean()), deep_range_m=float(ann.max()-ann.min()),
                deep_out_mm_yr=float(hr.Qd.mean()*YR*1000), deep_share_pct=float(100*hr.Qd.mean()/hr.unr.mean())), q, Sd


# ---- 3. f from the winter volume balance, per tau_d
def f_balance(tau_yr):
    g = lambda lam: mo(flow(*simulate(lam, tau_yr, b0)[:2]), [12, 1, 2]) - OBS_DJF
    lo, hi = 0.0, 0.25
    if g(lo) >= 0: return 0.0
    while g(hi) < 0 and hi < 16: lo, hi = hi, hi*2
    return float(brentq(g, lo, hi, xtol=1e-3)) if g(hi) >= 0 else np.nan


print(f"\nobserved mean Dec-Feb flow WY1987-2024: {OBS_DJF:.2f} m3/s ({OBS_DJF*86400*90/Atot*1000:.1f} mm per season); control: {mo(q0, [12, 1, 2]):.2f} m3/s")
TAUS = [0.5, 1.0, 2.0, 5.0, 10.0]; rows = []; store = {}
ev0, _, _ = evaluate(0.0, 1.0); ev0["tau_yr"] = np.nan; ev0["label"] = "control (no deep store)"; rows.append(ev0)
fb = {}
for tau in TAUS:
    fb[tau] = f_balance(tau); r_, q_, S_ = evaluate(fb[tau], tau); r_["label"] = f"balanced f, tau_d={tau:g} yr"; rows.append(r_); store[tau] = S_
    print(f"  tau_d {tau:>4g} yr -> f = {fb[tau]:.3f}  (leak = {100*fb[tau]/(1+fb[tau]):.0f}% of bucket drainage)", flush=True)
FREF = fb[1.0]
for tau in TAUS:
    for mult in (0.25, 0.5, 2.0, 4.0):
        r_, _, _ = evaluate(FREF*mult, tau); r_["label"] = f"grid: f = {mult:g} x f_ref(1 yr), tau_d={tau:g} yr"; rows.append(r_)
T = pd.DataFrame(rows); T.to_csv(OUT/f"deepstore_build_{STATE}.csv", index=False)
json.dump(dict(state=STATE, bucket=dict(rate=RATE, Sf=SF, exp=EX, routing_scale_s=SCALE), f_balance={str(k): v for k, v in fb.items()}, tau_primary_yr=1.0, f_primary=FREF,
               obs_lag1={f"{k[0]}-{k[1]}": dict(fall=v[0], winter=v[1]) for k, v in OBS_LAG.items()}), open(OUT/f"deepstore_build_{STATE}.json", "w"), indent=1)
print("\nobserved lag-1 (Spearman): " + "; ".join(f"WY{k[0]}-{k[1]}: fall {v[0]:+.2f}, winter {v[1]:+.2f}" for k, v in OBS_LAG.items()))
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
cols = ["label", "f", "DJF_ratio", "obj", "KGE", "fall_lag1_87", "fall_lag1_09", "win_lag1_87", "win_lag1_09", "link_model_87", "link_obs_87", "link_obs_09", "deep_mean_m", "deep_range_m", "deep_out_mm_yr", "deep_share_pct"]
print("\n"); print(T[cols].round(3).to_string(index=False))
print("\nPRIMARY (tau_d = 1 yr, f from winter balance) -> SUMMA build parameters:")
tau_s = YR; sEq = FREF*b0*tau_s
print(f"  aquiferLeakRate = f*aquiferBaseflowRate = {FREF*RATE:.4g} m/s; aquiferLeakExp = aquiferBaseflowExp = {EX:.4f}")
print(f"  basin__aquiferHydCond = 1/tau_d = {1/tau_s:.4g} m/s; basin__aquiferScaleFactor = 1.0; basin__aquiferBaseflowExp = 1.0; basin__aquiferInitStorage = {sEq:.4f} m (equilibrium)")
