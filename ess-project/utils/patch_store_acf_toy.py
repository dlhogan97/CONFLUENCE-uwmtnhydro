"""Toy mass balance: what autocorrelation of precip-adjusted low flow does a depleting patch store produce?
S_{t+1} = max(S_t + R_t - M_t, 0);  M_t = min(S_t + R_t, E_t * S_t^beta);  mean recharge R = P_t (WY precip, lognormal CV 0.35), E_t = E0 * lognormal(CV 0.25) (melt energy / warmth, independent of P).
The equilibrium store is S_eq = (1/E0)^(1/beta) x mean recharge (set by E0): beta = 1 is a linear reservoir (supply-limited), beta = 0.5 has melt area shrinking with the store.
Low flow Q_t = (1-share) * base_t + share * M_t (rescaled), base_t = P_t^0.5 * noise (non-patch, same-year-precip driven). Residual = ln Q regressed on ln P (same year), rank ACF lags 1-5, n = 43 years, 2000 simulations.
'Warming': E_t grows 15%/decade after year 20 (store declines). -> figures/draft/patch_store_acf_toy.csv"""
import numpy as np, pandas as pd
from scipy.stats import rankdata
rng = np.random.default_rng(11); N = 43; NS = 2000; SPIN = 60
def sp(a, b): return np.corrcoef(rankdata(a), rankdata(b))[0, 1]
def sim(E0, beta, share, warming=0.0, detr=False, cvE=0.25, cvP=0.35):
    out = []; Seq = (1.0/E0)**(1.0/beta)
    for _ in range(NS):
        T = N+SPIN; P = np.exp(rng.normal(0, cvP, T)-cvP**2/2); E = E0*np.exp(rng.normal(0, cvE, T)-cvE**2/2); tt = np.arange(T)-SPIN; E = E*(1+warming*np.clip(tt, 0, None)/10.0); S = Seq; M = np.zeros(T)
        for t in range(T):
            avail = S+P[t]; M[t] = min(avail, E[t]*max(S, 1e-9)**beta); S = max(avail-M[t], 0.0)
        M, P_ = M[SPIN:], P[SPIN:]; base = P_**0.5*np.exp(rng.normal(0, 0.15, N)); Q = (1-share)*base+share*(M/np.mean(M))*np.mean(base)
        y = np.log(Q); X = np.column_stack([np.ones(N), np.log(P_)]+([np.arange(N)/10.] if detr else [])); r = y-X@np.linalg.lstsq(X, y, rcond=None)[0]; out.append([sp(r[:-k], r[k:]) for k in range(1, 6)])
    return np.array(out)
rows = []; band = 1.96/np.sqrt(N)
def add(lab, A, extra=None): m = A.mean(0); rows.append(dict(case=lab, **{f"lag{k+1}": m[k] for k in range(5)}, p_sig_pos_lag1=np.mean(A[:, 0] > band), p_sig_neg_lag1=np.mean(A[:, 0] < -band), **(extra or {})))
for beta, lab_b in ((1.0, "linear store"), (0.5, "area-limited store (beta 0.5)")):
    for E0, nm in ((0.8, "small store (~1-2x annual)"), (0.4, "medium store"), (0.15, "large store")):
        S_eq = (1/E0)**(1/beta); add(f"{lab_b}, {nm}, patch = 50% of low flow", sim(E0, beta, 0.5), dict(store_x_annual=S_eq))
for sh in (0.25, 0.8): add(f"area-limited, medium store, patch = {int(sh*100)}% of low flow", sim(0.5, 0.5, sh))
add("area-limited, medium store, 50%, WARMING (+15%/decade), not detrended", sim(0.5, 0.5, 0.5, warming=0.15)); add("area-limited, medium store, 50%, WARMING, detrended", sim(0.5, 0.5, 0.5, warming=0.15, detr=True))
add("linear, medium store, 50%, WARMING, not detrended", sim(0.4, 1.0, 0.5, warming=0.15)); add("NULL: no patch contribution", sim(0.5, 0.5, 0.0))
T = pd.DataFrame(rows); pd.set_option("display.width", 250); print(f"mean rank ACF of the precip-adjusted low-flow residual (n={N}, noise band ±{band:.2f}); p_sig = fraction of simulated records with a significant lag-1 value"); print(T[["case", "store_x_annual", "lag1", "lag2", "lag3", "lag4", "lag5", "p_sig_pos_lag1", "p_sig_neg_lag1"]].round(2).to_string(index=False)); T.to_csv("../figures/draft/patch_store_acf_toy.csv", index=False)
