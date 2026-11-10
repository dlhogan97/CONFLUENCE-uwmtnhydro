"""
Baselines for the slow-routing deep store (tau_s 35 d, S_th 0.05 m, routing 140000 s, tau_l from the Dec-Feb balance), same parameters,
deep store switched off (2026-10-28):
  no leak          the baseline: each HRU aquifer is a linear bucket that releases everything to the stream (S_th = 0, same tau_s and
                   routing, no leak, no deep store). Keeping S_th without a leak would trap the water below it forever.
  leak to stream   same tau_l, but the leak goes straight to the stream (no 2-yr store)
Scored like east_independent_compare (objective, KGE, recession, fall-min bias, rho1 87-24 / 09-24 with permutation p) and written as
extra columns of east_final_models_daily.csv for the notebook.
usage: python east_slowroute_ablation.py
"""
import logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
import east_threshold_cal as T, east_independent_compare as I

C, st = T.C, T.st
p = dict(log_tau_s=np.log10(35.0), S_th=0.05, lscale=np.log10(140000.0)); ts = 35.0
q_trt, tl = T.run_model(p, "treatment")[:2]
Qs, Qd, _ = T.simulate(0.0, ts, 1e30, True); q_noleak = T.flow(Qs, Qd*0.0, p["lscale"])        # S_th = 0, no leak
Qs, Qd, _ = T.simulate(p["S_th"], ts, tl, False); q_stream = T.flow(Qs, Qd, p["lscale"])
S = {"deep store (slow routing)": q_trt, "baseline: linear bucket to stream": q_noleak, "leak straight to stream": q_stream}
rows = [I.diag(k, q) for k, q in S.items()]
R = pd.DataFrame(rows).set_index("model").T; pd.set_option("display.width", 220)
print(f"tau_l from the winter balance: {tl:.0f} d"); print(R.to_string(float_format=lambda v: f"{v:.3f}"))
F = C.X.OUT/"east_final_models_daily.csv"; D = pd.read_csv(F, index_col="date", parse_dates=True)
D["slowroute_noleak"] = q_noleak; D["slowroute_leak_to_stream"] = q_stream; D.to_csv(F, float_format="%.5g")
