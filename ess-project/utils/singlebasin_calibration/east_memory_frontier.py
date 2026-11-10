"""What does memory cost in the screen trials of east_memory_search.py? For each structure, the best paper objective among trials reaching rho1 >= 0.2, 0.3, 0.38 (WY1987-2024), then full scores of those trials.
Descriptive only: picking by rho1 here is the fitted-memory question, so nothing from this table is a pass under the search's gates. -> stageB/memory_search/frontier.csv"""
import json, numpy as np, pandas as pd
import east_memory_search as M, east_booster as EB
rows = []
for name in M.STRUCT:
    t = pd.read_json(M.OUTD/f"screen_{name}.jsonl", lines=True)
    for rho in (0.2, 0.3, 0.38):
        g = t[t.rho1_87 >= rho]
        if not len(g): rows.append(dict(model=name, rho_min=rho, n=0)); continue
        b = g.loc[g.objective.idxmin()]; p = b.params; q, sd = M.simulate(name, p); s = EB.score(q); wy = q["1986-10-01":"2024-09-30"]
        rows.append(dict(model=name, rho_min=rho, n=len(g), paper_obj=b.objective, rho1_87=s["rho1_87"], p87=s["p87"], rho1_09=s["rho1_09"], KGE_eval=s["KGE_eval"], NSE_eval=s["NSE_eval"], recess_d=s["recess_d"], winter_tau_d=s["winter_tau_d"], fallmin_bias=s["fallmin_bias"], store_mm=float(sd["1986-10-01":"2024-09-30"].mean()*1000), params=json.dumps({k: round(v, 3) for k, v in p.items()})))
R = pd.DataFrame(rows); R.to_csv(M.OUTD/"frontier.csv", index=False); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70); print(R.round(2).to_string(index=False))
