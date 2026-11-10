"""
Is the East control (single nonlinear bucket, east_ctl_treat_cal_v2 best) overfit? (2026-10-28)
  (1) parameters vs bounds; calibration history
  (2) skill in the signature-calibration window (WY2009-2020) vs outside it (WY1987-2008, WY2021-2024), control and deep store
  (3) memory across the near-equivalent control fits: every trial within 5% of the best objective, re-simulated
usage: python east_control_overfit_check.py
"""
import json, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
import east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V, emu_core as E

obs = C.obs
tr = [json.loads(l) for l in open(V.OUTD/"trials_control.jsonl")]; ok = sorted([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])
b = ok[0]; print(f"(1) trials {len(tr)} ({len(ok)} valid); best objective {b['objective']:.4f}; history {json.load(open(V.OUTD/'history_control.json'))}")
for k, lo, hi in C.BOUNDS: v = b["params"][k]; print(f"    {k:7s} {v:8.3f}   bounds [{lo}, {hi}]   position {(v - lo)/(hi - lo):.2f}")
print("    best parts:", {k: round(v, 2) for k, v in b["parts"].items()})


def skill(q, a, z):
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1)[a:z].dropna(); lo = j[j.index.month.isin([8, 9, 10, 11])]
    return dict(KGE=E.kge(j.o, j.s), logKGE=E.kge(np.log(j.o), np.log(j.s.clip(lower=.05))), bias=100*(j.s.sum()/j.o.sum() - 1), AugNov_bias=100*(lo.s.sum()/lo.o.sum() - 1))


ef = pd.read_csv(C.X.OUT/"east_final_models_daily.csv", index_col=0, parse_dates=True)
rows = []
for name, q in (("control", ef.control), ("deep store", ef.treatment)):
    for lab, (a, z) in {"cal WY2009-2020": ("2008-10-01", "2020-09-30"), "out WY1987-2008": ("1986-10-01", "2008-09-30"), "out WY2021-2024": ("2020-10-01", "2024-09-30")}.items():
        rows.append(dict(model=name, period=lab, **skill(q, a, z)))
print("\n(2) skill inside vs outside the signature window"); print(pd.DataFrame(rows).round(3).to_string(index=False))

near = [r for r in ok if r["objective"] <= 1.05*b["objective"]]
print(f"\n(3) {len(near)} control trials within 5% of the best objective")
out = []
for r in near:
    p = r["params"]; Qs, Qd, _ = C.simulate(p, 0.0); q = C.flow(Qs, Qd, p)
    m87, m09 = E.memory(q, 1987, 2024), E.memory(q, 2009, 2024)
    out.append(dict(objective=r["objective"], lrate=p["lrate"], Sf=p["Sf"], exp=p["exp"], lscale=p["lscale"], rho1_8724=m87["res1"], rho1_0924=m09["res1"],
                    recess=V.recession_taus(q).median(), fallmin_bias=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/np.exp(E.series(obs, 1987, 2024)[0]).mean() - 1)))
O = pd.DataFrame(out); print(O.round(3).to_string(index=False))
print("\n    spread:", O[["rho1_8724", "rho1_0924", "recess", "fallmin_bias"]].describe().loc[["min", "50%", "max"]].round(2).to_string())
