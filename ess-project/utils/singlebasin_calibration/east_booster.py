"""
Deep store as an observation-parameterised BOOSTER on the calibrated baseline (2026-10-29). No recalibration.
  baseline   the matched-calibration baseline (east_matched_cal.py, standard objective: routing signatures + Aug-Sep recession + DJF volume;
             no low-flow or memory term), every parameter frozen at its best fit.
  booster    the SAME buckets + a linear leak f*rate*(S/Sf) from every HRU bucket to one basin deep store, Q_d = S_d/tau_d.
             Both deep-store numbers come from the observed hydrograph, not from fitting the model:
               tau_d      winter baseflow recession, Nov 15 - Mar 15 lower envelope (Theil-Sen, as the Aug-Sep term): median 318 d, IQR 250-450
               release    late-winter baseflow, Feb 15 - Mar 15 lower envelope: median 1.55 m3/s (18% of mean flow) -> storage Q*tau ~ 57 mm.
                          This includes some bucket drainage, so it bounds the deep release from above; share = 1.0 and 0.5 are both run.
             f is then fixed by mass balance: mean deep release (WY1987-2024) = share * 1.55 m3/s (one root-find on the flux, no skill metric).
Grid: tau_d in {250, 318, 450} d x share in {0.5, 1.0}; central = (318 d, 1.0).
Scored: KGE / NSE (WY2009-2020 cal years, WY1987-2008 + 2021-2023 eval), Aug-Sep and winter recession, fall-min bias, deep storage vs 57 mm,
rho1 of the peak residual (WY1987-2024, WY2009-2024) with two-sided permutation p.
usage: python east_booster.py      (writes stageB/east_booster.csv and booster_* columns to east_final_models_daily.csv)
"""
import json, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.stats import theilslopes
import emu_core as E, east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V

obs = C.obs
CAL = list(range(2009, 2021)); EVAL = [y for y in range(1987, 2024) if y not in CAL]
TAUS, SHARES, CENTRAL = (250.0, 318.0, 450.0), (0.5, 1.0), (318.0, 1.0)


def winter_taus(q, y0=1984, y1=2024):
    low = q.rolling(7, center=True, min_periods=5).min().rolling(7, center=True, min_periods=5).mean(); out = {}
    for y in range(y0, y1 + 1):
        w = low[f"{y-1}-11-15":f"{y}-03-15"].dropna()
        if len(w) < 90: continue
        s = theilslopes(np.log(np.clip(w.values, 1e-4, None)), np.arange(len(w)))[0]; out[y] = -1.0/s if s < 0 else np.nan
    return pd.Series(out)


def late_winter(q): low = q.rolling(7, center=True, min_periods=5).min(); return pd.Series({y: low[f"{y}-02-15":f"{y}-03-15"].mean() for y in range(1984, 2025)}).dropna()


Q_DEEP_OBS = float(late_winter(obs).median())


def run(p, lam, tau_d):
    C.TAU_D_YR = tau_d/365.25; Qs, Qd, Sd = C.simulate(p, lam)
    q = C.flow(Qs, Qd, p); z = np.zeros_like(Qs)
    qd = C.flow(z, Qd, p) - C.flow(z, Qd*0.0, p)                  # daily_Q adds SUMMA surface runoff; remove it to isolate the deep release
    return q, qd, pd.Series(Sd, index=C.st["tt"]).resample("D").mean()


def lam_for(p, tau_d, target):
    g = lambda lf: run(p, 10**lf, tau_d)[1]["1986-10-01":"2024-09-30"].mean() - target
    return float(10**brentq(g, -6.0, -1.0, xtol=0.01))


def score(q, sd=None):
    def sk(ys):
        j = pd.concat([q.rename("s"), obs.rename("o")], axis=1).dropna(); j = j[np.isin(j.index.year + (j.index.month >= 10), ys)]
        return E.kge(j.o, j.s), 1 - ((j.s - j.o)**2).sum()/((j.o - j.o.mean())**2).sum()
    def r1(y0, n=2000):
        r = E.resid(*E.series(q, y0, 2024)); est = E.lag_r(r, 1); rng = np.random.default_rng(1)
        null = [E.lag_r(pd.Series(rng.permutation(r.values), index=r.index), 1) for _ in range(n)]; return est, (np.sum(np.abs(null) >= abs(est)) + 1)/(n + 1)
    (kc, nc), (ke, ne) = sk(CAL), sk(EVAL); a, pa = r1(1987); b, pb = r1(2009)
    lf0 = np.exp(E.series(obs, 1987, 2024)[0]).mean()
    return dict(KGE_cal=kc, KGE_eval=ke, NSE_cal=nc, NSE_eval=ne, recess_d=V.recession_taus(q).median(), winter_tau_d=winter_taus(q).median(),
                late_winter_q=late_winter(q).median(), fallmin_bias=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/lf0 - 1),
                deep_storage_mm=np.nan if sd is None else float(sd["1986-10-01":"2024-09-30"].mean()*1000),       # S_d is a basin depth in m
                rho1_87=a, p87=pa, rho1_09=b, p09=pb)


def main():
    tr = [json.loads(l) for l in open(C.X.OUT/"matched_cal"/"trials_baseline.jsonl")]
    p = min([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])["params"]
    print("baseline (frozen):", {k: round(v, 4) for k, v in p.items()}, f"| observed deep release {Q_DEEP_OBS:.2f} m3/s, winter tau {winter_taus(obs).median():.0f} d")
    rows = [dict(model="observed", **score(obs))]
    q0, _, _ = run(p, 0.0, 318.0); rows.append(dict(model="baseline (calibrated, frozen)", **score(q0))); out = {"booster_baseline": q0}
    for tau in TAUS:
        for sh in SHARES:
            lam = lam_for(p, tau, sh*Q_DEEP_OBS); q, qd, sd = run(p, lam, tau)
            rows.append(dict(model=f"+ deep store tau {tau:.0f} d, release {sh:.0%}", f=lam, deep_release=qd["1986-10-01":"2024-09-30"].mean(), **score(q, sd)))
            if (tau, sh) == CENTRAL: out["booster_central"] = q
            print("  done", tau, sh, round(lam, 4), flush=True)
    R = pd.DataFrame(rows).set_index("model"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(R.round(3).to_string()); R.to_csv(C.X.OUT/"east_booster.csv")
    F = C.X.OUT/"east_final_models_daily.csv"; D = pd.read_csv(F, index_col="date", parse_dates=True)
    for k, s in out.items(): D[k] = s
    D.to_csv(F, float_format="%.5g"); print("series written to", F)


if __name__ == "__main__":
    main()
