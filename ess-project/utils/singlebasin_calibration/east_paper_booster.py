"""
Deep store added to the PAPER baseline, no recalibration (2026-10-29).
  baseline   SUMMA East run of the paper (calibration runs/East_River_20260720_094546: DE, WY1993-2002, spin-up WY1990-1992), frozen as
             bestruns/East_River_baseline_1981_2025. HRU aquifers: rate 9.4e-6 m/s, scale factor 2.658 m, exponent 2.057; routing 51629 s.
             Emulated from that run's recharge (stageB_emulator state "old"); leak off reproduces the SUMMA streamflow.
  + store    every HRU aquifer leaks lam*rate*(S/Sf) (linear) to ONE basin store that releases linearly, Q_d = S_d/tau_d (the singleBasin build).
             Nothing is fitted to skill. The two store numbers come from the observed hydrograph (east_booster.py):
               tau_d    winter baseflow recession, Nov 15 - Mar 15: median 318 d (IQR 250-450)
               release  late-winter baseflow, Feb 15 - Mar 15: 1.55 m3/s; it includes some aquifer drainage, so share 1.0 is an upper bound, 0.5 also run
             lam follows from mass balance: mean store release (WY1987-2024) = share * 1.55 m3/s. The store starts at release*tau_d (no spin-up drift).
Grid tau_d {250, 318, 450} d x share {0.5, 1.0}; scored as east_booster.score. Writes stageB/east_paper_booster.csv and paper_* columns.
Result (linear store): fit and winter recession improve, memory does not (rho1 +0.05 -> +0.06-0.08), as the overnight linear M2 found.
--nonlinear: store release Q_d = rate*(S_d/Sf_d)^n with n from the observed winter recession (Brutsaert-Nieber -dQ/dt = a Q^b on Nov 15 - Mar 15
  7-d flow, n = 1/(2-b): lower envelope b 1.21 -> n 1.27, all points b 1.42 -> n 1.72). Sf_d puts the recession time S/(n Q) at tau_d = 318 d at the
  mean release. Release targets: the baseline's late-winter deficit (1.55 - 0.92 = 0.63 m3/s) and the full 1.55 upper bound. Grid n {1, 1.27, 1.72}.
usage: STAGEB_T0=1981-10-01 STAGEB_END=2025-09-30 python east_paper_booster.py [--nonlinear]
"""
import os, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd, netCDF4 as nc
from scipy.optimize import brentq
import emu_core as E, stageB_emulator as B, east_booster as EB

st = B.build_state("old"); AT = float(st["A"].sum())
P0 = dict(rate=9.4e-6, Sf=2.6577658, exp=2.056814, scale=51629.17)
TAUS, SHARES, CENTRAL = (250.0, 318.0, 450.0), (0.5, 1.0), (318.0, 1.0)


def run(lam, tau_d, q_init=0.0, passes=3, n=1.0):
    NT, NH = st["Rch"].shape; S = st["S0"].copy(); Qs, Qd, Sd = np.zeros((NT, NH)), np.zeros(NT), np.zeros(NT)
    qbar = max(q_init, 1e-3)/AT; sbar = n*tau_d*86400.0*qbar                # mean storage (m) whose recession time S/(nQ) is tau_d
    sfd = sbar/(qbar/P0["rate"])**(1.0/n)                                    # Q_d = rate*(S/sfd)^n passes qbar at sbar (n = 1: sfd = tau_d*rate)
    sdi = sbar if q_init > 0 else 0.0
    for _ in range(passes):
        S, sdi = E.run_buckets(st["Rch"], st["A"], P0["Sf"], P0["rate"], P0["exp"], float(lam), 1.0, sfd, float(n), P0["rate"], S, float(sdi), 3600.0, Qs, Qd, Sd)
    q = B.daily_Q(st, Qs, Qd, P0["scale"]); qd = pd.Series(Qd*AT, index=st["tt"]).resample("1D").mean()
    return q, qd, pd.Series(Sd, index=st["tt"]).resample("1D").mean()


def lam_for(tau_d, target, n=1.0):
    g = lambda ll: run(10**ll, tau_d, target, passes=2, n=n)[1]["1986-10-01":"2024-09-30"].mean() - target
    return float(10**brentq(g, -5.0, 1.0, xtol=0.01))


def main():
    d = nc.Dataset(B.SRC["old"][0] + "/out/" + B.SRC["old"][1]); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
    summa = (pd.Series(np.array(d["averageRoutedRunoff"][:]).ravel(), index=t)*AT).resample("1D").mean(); d.close()
    q0, _, _ = run(0.0, 318.0); j = pd.concat([q0, summa], axis=1).dropna()
    print(f"emulator vs SUMMA paper baseline (leak off): r = {j.corr().iloc[0, 1]:.5f}, mean {j.iloc[:, 0].mean():.3f} vs {j.iloc[:, 1].mean():.3f} m3/s", flush=True)
    rows = [dict(model="observed", **EB.score(EB.obs)), dict(model="paper baseline (SUMMA)", **EB.score(summa)), dict(model="paper baseline (emulator)", **EB.score(q0))]
    out = {"paper_baseline": summa}; nonlin = "--nonlinear" in __import__("sys").argv
    deficit = EB.Q_DEEP_OBS - float(EB.late_winter(summa).median())
    grid = [(318.0, tg, n) for n in (1.0, 1.27, 1.72) for tg in (deficit, EB.Q_DEEP_OBS)] if nonlin else [(tau, sh*EB.Q_DEEP_OBS, 1.0) for tau in TAUS for sh in SHARES]
    for tau, target, n in grid:
            sh = target/EB.Q_DEEP_OBS; lam = lam_for(tau, target, n); q, qd, sd = run(lam, tau, target, n=n)
            leak_share = float(qd["1986-10-01":"2024-09-30"].mean()/q["1986-10-01":"2024-09-30"].mean())
            rows.append(dict(model=f"+ store tau {tau:.0f} d, n {n:g}, release {target:.2f} m3/s", lam=lam, n=n, store_release=qd["1986-10-01":"2024-09-30"].mean(), store_share_of_flow=leak_share, **EB.score(q, sd)))
            if not nonlin and (tau, sh) == CENTRAL: out["paper_deepstore"] = q
            if nonlin: out[f"paper_deepstore_n{n:g}_q{target:.2f}"] = q
            print("  done", tau, n, round(target, 2), f"lam {lam:.4g}", flush=True)
    R = pd.DataFrame(rows).set_index("model"); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(R.round(3).to_string()); R.to_csv(B.OUT/("east_paper_booster_nonlinear.csv" if nonlin else "east_paper_booster.csv"))
    F = B.OUT/"east_final_models_daily.csv"; D = pd.read_csv(F, index_col="date", parse_dates=True)
    for k, s in out.items(): D[k] = s
    D.to_csv(F, float_format="%.5g"); print("series written to", F)


if __name__ == "__main__":
    main()
