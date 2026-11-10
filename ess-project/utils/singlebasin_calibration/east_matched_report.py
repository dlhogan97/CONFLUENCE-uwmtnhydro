"""
Report for east_matched_cal.py (2026-10-28): baseline vs deep store, each at its best fit and across its near-equal fits.
  near-equal = every unique trial within 5% of that model's best objective (re-simulated)
  per fit: objective, KGE/NSE in WY2009-2020 (cal) and WY1987-2008 + WY2021-2023 (eval), late-summer recession, fall-min bias,
           rho1 of the peak residual and of the Apr-Jul-volume residual (WY1987-2024; WY2009-2024), with two-sided permutation p for the bests
  knockout from the treatment best: same tau_s and routing, no deep store (linear bucket to the stream, S_th = 0, no leak)
Best series are written to east_final_models_daily.csv as matched_baseline / matched_treatment / matched_knockout.
usage: [REC_W=3] python east_matched_report.py   (reads M.OUTD; series columns get a _recw<W> suffix when REC_W != 1)
"""
import json, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
import east_matched_cal as M, east_threshold_cal as T, east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V, emu_core as E

obs = C.obs


def drivers(q, y0, y1):
    """per WY: ln fall minimum (Aug 1-Nov 30, 7-d mean), ln Apr-Jul 7-d peak, ln Apr-Jul mean flow (as east_residual_basis.py)"""
    lf, pk, aj = {}, {}, {}
    for y in range(y0, y1 + 1):
        a = q[f"{y}-08-01":f"{y}-11-30"]
        if a.isna().mean() <= 0.1 and len(a) > 100:
            m = a.rolling(7, center=True, min_periods=7).mean().min()
            if np.isfinite(m) and m > 0.01: lf[y] = m
        b = q[f"{y}-04-01":f"{y}-07-31"]
        if b.isna().mean() <= 0.05: pk[y] = b.interpolate(limit=5).rolling(7, center=True, min_periods=7).mean().max(); aj[y] = b.mean()
    return np.log(pd.DataFrame(dict(lf=lf, peak=pk, AprJul=aj))).dropna()


def lag_p(r, k, n=2000, seed=0):
    est = E.lag_r(r, k); rng = np.random.default_rng(seed + k)
    null = [E.lag_r(pd.Series(rng.permutation(r.values), index=r.index), k) for _ in range(n)]
    return est, (np.sum(np.abs(null) >= abs(est)) + 1)/(n + 1)

CAL = list(range(2009, 2021)); EVAL = [y for y in range(1987, 2024) if y not in CAL]
LF0 = np.exp(E.series(obs, 1987, 2024)[0]).mean()


def skill(q, years):
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1).dropna(); j = j[np.isin(j.index.year + (j.index.month >= 10), years)]
    return E.kge(j.o, j.s), 1 - ((j.s - j.o)**2).sum()/((j.o - j.o.mean())**2).sum()


def rho(q, y0, basis, with_p=False):
    d = drivers(q, y0, 2024); r = pd.Series(d.lf - np.polyval(np.polyfit(d[basis], d.lf, 1), d[basis]), index=d.index)
    return lag_p(r, 1, n=1000) if with_p else (E.lag_r(r, 1), np.nan)


def describe(q, with_p=False):
    (kc, nc), (ke, ne) = skill(q, CAL), skill(q, EVAL); out = dict(winter_tau_d=M.winter_taus(q).median(), KGE_cal=kc, KGE_eval=ke, NSE_cal=nc, NSE_eval=ne, recess_d=V.recession_taus(q).median(),
                                                                    fallmin_bias=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/LF0 - 1))
    for y0 in (1987, 2009):
        for b in ("peak", "AprJul"):
            r, p = rho(q, y0, b, with_p); out[f"rho1_{b}_{y0}"] = r
            if with_p: out[f"p_{b}_{y0}"] = p
    return out


rows, near_rows, series = [], [], {}
for model in ("baseline", "treatment_td" if M.WREC else "treatment"):
    tr = [json.loads(l) for l in open(M.OUTD/f"trials_{model}.jsonl")]; ok = sorted([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])
    uniq, seen = [], set()
    for r in ok:
        k = tuple(round(v, 4) for v in r["params"].values())
        if k not in seen: seen.add(k); uniq.append(r)
    best = uniq[0]; near = [r for r in uniq if r["objective"] <= 1.05*best["objective"]]
    q, tl = M.simulate(model, best["params"]); series[model] = q
    rows.append(dict(model=model, n_trials=len(tr), n_near=len(near), objective=best["objective"], **{k: round(v, 4) for k, v in best["params"].items()},
                     tau_l_d=tl, **describe(q, with_p=True)))
    for r in near:
        qq, _ = M.simulate(model, r["params"]); near_rows.append(dict(model=model, objective=r["objective"], **describe(qq)))
    if model.startswith("treatment"):
        p = best["params"]; T.TAU_D = 10**p["log_tau_d"]*86400.0 if "log_tau_d" in p else 2.0*T.YR      # the near-fit loop left TAU_D at its last fit
        Qs, Qd, _ = T.simulate(0.0, 10**p["log_tau_s"], 1e30, True); series["knockout"] = T.flow(Qs, Qd*0.0, p["lscale"])
        rows.append(dict(model="knockout (treatment best, no deep store)", **describe(series["knockout"], with_p=True)))

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
R = pd.DataFrame(rows).set_index("model").T; print("best fits"); print(R.to_string(float_format=lambda v: f"{v:.3f}"))
N = pd.DataFrame(near_rows); cols = ["objective", "winter_tau_d", "KGE_eval", "NSE_eval", "recess_d", "fallmin_bias", "rho1_peak_1987", "rho1_AprJul_1987", "rho1_peak_2009", "rho1_AprJul_2009"]
print("\nnear-equal fits (within 5% of each model's best): min / median / max")
print(N.groupby("model")[cols].agg(["min", "median", "max"]).T.round(3).to_string())
R.to_csv(M.OUTD/"report_best.csv"); N.to_csv(M.OUTD/"report_near.csv", index=False)
F = C.X.OUT/"east_final_models_daily.csv"; D = pd.read_csv(F, index_col="date", parse_dates=True)
sfx = ("" if M.REC_W == 1 else f"_recw{M.REC_W:g}") + ("_wrec" if M.WREC else "")
for k, v in series.items(): D[f"matched_{k.replace('treatment_td', 'treatment')}{sfx}"] = v
D.to_csv(F, float_format="%.5g")
print("series written to", F)
