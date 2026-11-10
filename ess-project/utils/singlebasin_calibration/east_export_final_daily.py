"""
Export the two East models of east_independent_compare.py as daily series for the results notebook (2026-10-28).
  control     round-2 single nonlinear bucket (east_ctl_treat_cal_v2 best), no leak, no threshold
  treatment   threshold shallow store (tau_s 35 d, S_th 0.1 m, tau_l by the Dec-Feb balance) + linear deep store (tau_d 2 yr)
  treatment_slowroute  same tau_s (35 d), S_th 0.05 m, routing scale 140000 s (~1.6 d); from east_deepstore_routing_grid.py
               (objective 1.213, within 3% of the grid best; out-of-period NSE 0.55 vs 0.46)
Columns: obs (the record both were scored against), control, treatment, treatment_no_deep (deep outflow removed), deep_storage_m (daily mean),
         treatment_slowroute, deep_storage_slowroute_m.
usage: python east_export_final_daily.py   -> stageB/east_final_models_daily.csv
"""
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
import east_independent_compare as I

C, T, st = I.C, I.T, I.st
pc = I.best(I.V.OUTD.parent/"ctl_treat_cal_v2"/"trials_control.jsonl"); Qs, Qd, _ = C.simulate(pc, 0.0); ctl = C.flow(Qs, Qd, pc)
p = dict(log_tau_s=np.log10(35.0), S_th=0.1, lscale=np.log10(48921.0)); trt, tl, Qs, Qd, Sd = T.run_model(p, "treatment")
nodeep = C.B.daily_Q(st, Qs, np.zeros_like(Qd), 10**p["lscale"])
p2 = dict(log_tau_s=np.log10(35.0), S_th=0.05, lscale=np.log10(140000.0)); trt2, tl2, _, _, Sd2 = T.run_model(p2, "treatment")
D = pd.DataFrame(dict(obs=I.obs, control=ctl, treatment=trt, treatment_no_deep=nodeep, deep_storage_m=pd.Series(Sd, index=st["tt"]).resample("D").mean(),
                      treatment_slowroute=trt2, deep_storage_slowroute_m=pd.Series(Sd2, index=st["tt"]).resample("D").mean()))
D.index.name = "date"; out = C.X.OUT/"east_final_models_daily.csv"; D.to_csv(out, float_format="%.5g")
print(out, D.index.min().date(), D.index.max().date(), "tau_l", round(tl), "| slowroute tau_l", round(tl2), "| control params", {k: round(v, 3) for k, v in pc.items()})
