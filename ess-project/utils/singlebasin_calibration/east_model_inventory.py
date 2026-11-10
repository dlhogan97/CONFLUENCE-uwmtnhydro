"""
Inventory of every East River model we have, scored identically (2026-10-28).
Same observed record (emu_core.obs), WY1987-2024 unless noted:
  rho1 / p         Spearman lag-1 of the residual of ln(Aug 1-Nov 30 7-d min) on ln(Apr-Jul 7-d peak), two-sided permutation p
  rho1 09-24       same, WY2009-2024
  recess           median late-summer recession tau (lower envelope Aug-Sep, Theil-Sen), WY1984-2024; obs 68 d
  fallmin bias     mean Aug-Nov 7-d minimum vs observed
  KGE / logKGE     daily, WY1987-2024
Sources: SUMMA runs (baseline, control, old slowdrain adjusted, overnight M0-M3), emulator fits on the full-record control
recharge (stage-B new M0/M1 and M1 with fixed deep scale), and the 2026-10-27 emulator series (rounds 1-3, threshold store).
usage: STAGEB_T0=1981-10-01 STAGEB_END=2025-09-30 python east_model_inventory.py
"""
import os, json, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd, netCDF4 as nc
from scipy.stats import spearmanr
import emu_core as E, stageB_emulator as B, east_ctl_treat_cal_v2 as V, east_ctl_treat_cal as C, east_threshold_cal as T

D = "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/"
SB = "/scratch/dlhogan/ess-project-data/calibration/bestruns/"
obs = E.obs


def summa(run, fn):
    d = nc.Dataset(f"{run}/out/{fn}") if os.path.exists(f"{run}/out/{fn}") else nc.Dataset(sorted(__import__("glob").glob(f"{run}/output*/*.nc"))[0])
    t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
    A = float(np.array(nc.Dataset(f"{run}/attributes.nc")["HRUarea"][:]).sum()); return (pd.Series(np.array(d["averageRoutedRunoff"][:]).ravel(), index=t)*A).resample("D").mean()


def best(path):
    tr = [json.loads(l) for l in open(path)]; return min([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])["params"]


def perm_p(r, est, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = r.values
    null = [E.lag_r(pd.Series(rng.permutation(v), index=r.index), 1) for _ in range(n)]
    return (np.sum(np.abs(null) >= abs(est)) + 1)/(n + 1)


def perm_pk(r, k, n=2000, seed=0):
    est = E.lag_r(r, k); rng = np.random.default_rng(seed + k); v = r.values
    null = [E.lag_r(pd.Series(rng.permutation(v), index=r.index), k) for _ in range(n)]
    return est, (np.sum(np.abs(null) >= abs(est)) + 1)/(n + 1)


def halves(q, k):
    """lag-k of the residual in WY1987-2005 and WY2006-2024 (stability check)"""
    return [E.lag_r(E.resid(*E.series(q, a, b)), k) for a, b in ((1987, 2005), (2006, 2024))]


def score(q):
    lf, pk = E.series(q, 1987, 2024); r = E.resid(lf, pk); r1 = E.lag_r(r, 1)
    r2, p2 = perm_pk(r, 2); r3, p3 = perm_pk(r, 3); h2 = halves(q, 2)
    lf9, pk9 = E.series(q, 2009, 2024); r9 = E.lag_r(E.resid(lf9, pk9), 1)
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1)["1986-10-01":"2024-09-30"].dropna()
    return dict(rho1=r1, p=perm_p(r, r1), rho2=r2, p2=p2, rho2_8705=h2[0], rho2_0624=h2[1], rho3=r3, p3=p3, rho1_0924=r9, recess_d=V.recession_taus(q).median(),
                fallmin_bias_pct=100*(np.exp(lf).mean()/np.exp(E.series(obs, 1987, 2024)[0]).mean() - 1),
                KGE=E.kge(j.o, j.s), logKGE=E.kge(np.log(j.o), np.log(j.s.clip(lower=.05))))


S = {}
# --- SUMMA runs
S["SUMMA baseline (bestruns)"] = summa(SB+"East_River_baseline_1981_2025/run", "baseline_timestep.nc")
S["SUMMA control, fast aquifer"] = summa(B.SRC["ctlfull"][0], "best_timestep.nc")
S["SUMMA old adjusted (slowdrain sf19.6)"] = summa(D.replace("deepstore_dev/", "slowdrain_sf19p6"), "x")
for m in ("M0", "M1", "M2", "M3"):
    S[f"SUMMA overnight {m}"] = pd.read_csv(D+f"overnight/summa_daily_{m}_full.csv", index_col=0, parse_dates=True)["q_cms"]
# --- emulator, full-record control recharge (stage-B fits)
st = B.build_state("ctlfull")
for lab, f in [("emu new M0 (ctl)", "fit_ctl_M0_to2025.json"), ("emu new M1 (ctl)", "fit_ctl_M1_to2025.json")] + \
              [(f"emu new M1, Sf_d={s}", f"fit_ctl_M1_sfd{s}_to2025.json") for s in ("0.5", "1", "2", "4", "8")]:
    p = json.load(open(D+"stageB/"+f))["params"]; Qs, Qd, Sd = B.emulate(st, p); S[lab] = B.daily_Q(st, Qs, Qd, p["scale"])
# --- emulator, routing-control recharge (2026-10-27 rounds)
for lab, d, f in [("r1 control (a)", "ctl_treat_cal", "trials_control_a.jsonl"), ("r1 treatment (a)", "ctl_treat_cal", "trials_treatment_a.jsonl"),
                  ("r1 treatment (b)", "ctl_treat_cal", "trials_treatment_b.jsonl"),
                  ("r2 control (current control)", "ctl_treat_cal_v2", "trials_control.jsonl"), ("r2 treatment", "ctl_treat_cal_v2", "trials_treatment.jsonl"),
                  ("r3 control (rec x3)", "ctl_treat_cal_v3_recw3", "trials_control.jsonl"), ("r3 treatment (rec x3)", "ctl_treat_cal_v3_recw3", "trials_treatment.jsonl")]:
    try:
        p = best(C.X.OUT/d/f); Qs, Qd, _ = C.simulate(p, p["f"]); S[lab] = C.flow(Qs, Qd, p)
    except FileNotFoundError:
        pass
S["threshold + deep, sweep (current deep store)"] = T.run_model(dict(log_tau_s=np.log10(35.0), S_th=0.1, lscale=np.log10(48921.0)), "treatment")[0]
S["threshold + deep, DE-calibrated"] = T.run_model(best(T.OUTD/"trials_treatment.jsonl"), "treatment")[0]
S["threshold, leak to stream (DE)"] = T.run_model(best(T.OUTD/"trials_control.jsonl"), "control")[0]

rows = [dict(model="observed", **score(obs))]
for k, q in S.items():
    try: rows.append(dict(model=k, **score(q)))
    except Exception as e: print("skip", k, type(e).__name__, e)
R = pd.DataFrame(rows).set_index("model"); pd.set_option("display.width", 250)
print(R.round(3).to_string()); R.to_csv(C.X.OUT/"east_model_inventory.csv")

# inputs shared by every emulator model: does the SUMMA recharge (or the climate, via the observed peak) carry a lag-2 signal?
I = pd.read_csv(C.X.OUT/"east_recharge_memory.csv", index_col=0).loc[1987:2024]
res = lambda y, x: pd.Series(y.values - np.polyval(np.polyfit(x, y, 1), x), index=y.index)
ins = {"SUMMA recharge Apr-Jul (resid on model peak)": res(np.log(I.R_AprJul), I.ln_peak_model),
       "SUMMA recharge Aug-Nov (resid on model peak)": res(np.log(I.R_AugNov.clip(lower=1e-3)), I.ln_peak_model),
       "SUMMA recharge, water year (resid on model peak)": res(np.log(I.R_year), I.ln_peak_model),
       "observed ln Apr-Jul peak (climate)": I.ln_peak_obs}
print("\nshared inputs, WY1987-2024:")
for k, r in ins.items():
    r.index = r.index.astype(int); out = [perm_pk(r, j) for j in (1, 2, 3)]
    print(f"  {k:50s} " + "  ".join(f"lag{j} {a:+.2f} (p {b:.3f})" for j, (a, b) in zip((1, 2, 3), out)))
