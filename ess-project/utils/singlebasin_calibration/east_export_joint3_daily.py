"""Daily series of the joint3 East models (variability-only joint fit, east_memory_joint2.py, 2026-11-02) for model_memory_result.ipynb.
Emulator on SUMMA recharge, best seed per structure from memory_search/joint3/report.csv.
-> stageB/east_joint3_models_daily.csv: obs, <model> flow (m3/s) and <model>_store_m (added store, m) for each structure."""
import json, pandas as pd
import east_booster as EB, east_memory_refit as RF, east_memory_search as M

R = pd.read_csv(M.OUTD/"joint3"/"report.csv", index_col=0)
out = {"obs": EB.obs}
for name in ("control_refit", "cascade_refit", "thresh_cascade_refit", "sr_capture_refit"):
    q, s = M.simulate(name, json.loads(R.loc[name, "params"]))
    out[name], out[name+"_store_m"] = q, s
f = M.OUTD.parent/"east_joint3_models_daily.csv"
pd.DataFrame(out).rename_axis("date").to_csv(f); print("wrote", f)
