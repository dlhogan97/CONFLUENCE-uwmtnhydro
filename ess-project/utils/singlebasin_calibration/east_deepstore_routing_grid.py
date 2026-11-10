"""
Can the deep store's hydrograph skill improve without giving up its memory? (2026-10-28)
The sweep case (tau_s 35 d, S_th 0.1 m) had the routing gamma scale FIXED at the routing-control value (48921 s, ~0.6 d), while the
calibrated control uses ~150000 s (~1.7 d): sharper melt pulses -> Q7max +28%, NSE 0.56. Here tau_s stays fixed at 35 d (set by the
late-summer recession, not by memory); routing scale and S_th are gridded; tau_l from the Dec-Feb balance as before.
Reported: the calibration objective (east_ctl_treat_cal_v2 formula), NSE/KGE/logNSE in WY2009-2020 and outside it, Q7max bias, recession,
fall-min bias, rho1 (WY1987-2024, WY2009-2024).
usage: python east_deepstore_routing_grid.py
"""
import logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from multiprocessing import Pool
import east_threshold_cal as T, east_ctl_treat_cal_v2 as V, east_independent_compare as I, emu_core as E

obs = T.obs
CAL = range(2009, 2021); OUTY = [y for y in range(1987, 2024) if y not in CAL]


def per(q, years):
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1).dropna(); j = j[np.isin(j.index.year + (j.index.month >= 10), years)]
    nse = lambda m, o: 1 - ((m - o)**2).sum()/((o - o.mean())**2).sum(); eps = 0.01*j.o.mean()
    return nse(j.s, j.o), E.kge(j.o, j.s), nse(np.log(j.s + eps), np.log(j.o + eps))


def point(a):
    sth, scale_s = a; p = dict(log_tau_s=np.log10(35.0), S_th=sth, lscale=np.log10(scale_s))
    try: q, tl = T.run_model(p, "treatment")[:2]
    except ValueError: return dict(S_th=sth, scale_d=scale_s/86400, note="DJF unreachable")
    obj, kge, tau, parts = I.objective(q); (nc, kc, lc), (no, ko, lo) = per(q, CAL), per(q, OUTY)
    w = q.index.year + (q.index.month >= 10); wo = obs.index.year + (obs.index.month >= 10); q7 = lambda x: x.rolling(7, center=True, min_periods=7).mean()
    yy = list(range(1987, 2024)); q7max = 100*(q7(q)[np.isin(w, yy)].groupby(w[np.isin(w, yy)]).max().mean()/q7(obs)[np.isin(wo, yy)].groupby(wo[np.isin(wo, yy)]).max().mean() - 1)
    m87, m09 = E.memory(q, 1987, 2024), E.memory(q, 2009, 2024)
    lf0 = np.exp(E.series(obs, 1987, 2024)[0]).mean()
    return dict(S_th=sth, scale_d=scale_s/86400, tau_l=tl, objective=obj, NSE_cal=nc, NSE_out=no, KGE_cal=kc, KGE_out=ko, logNSE_cal=lc, logNSE_out=lo,
                Q7max_bias=q7max, recess=tau, fallmin_bias=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/lf0 - 1), rho1_8724=m87["res1"], rho1_0924=m09["res1"])


if __name__ == "__main__":
    grid = [(s, sc) for s in (0.05, 0.1, 0.15) for sc in (48921.0, 80000.0, 110000.0, 140000.0, 170000.0, 200000.0)]
    with Pool(18) as pool: R = pd.DataFrame(pool.map(point, grid))
    pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
    R.to_csv(T.C.X.OUT/"east_deepstore_routing_grid.csv", index=False)
