"""
East River: independent control vs threshold-store treatment, scored on the identical objective (2026-10-27).
  control      round-2 single nonlinear bucket per HRU, no leak, no threshold (east_ctl_treat_cal_v2 best, REC_W = 1)
  treatment    threshold shallow store + 2-yr deep store at the sweep case selected by the routing objective (tau_s 35 d, S_th 0.1 m, tau_l by DJF balance)
  reference    round-2 bucket + linear leak treatment, and the DE-calibrated threshold treatment (tau_s drifted to 48 d)
The threshold CONTROL of east_threshold_cal.py is not used: its slow leak straight to the stream made it a per-HRU deep store (not independent).
Objective = east_ctl_treat_cal_v2 formula (routing parts + recession + DJF volume); memory / low-flow quantities are diagnostics only.
usage: python east_independent_compare.py
"""
import json, logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V, east_shallow_tau_sweep as TS, east_threshold_cal as T
E, st, P, obs = C.E, C.st, C.P, C.obs
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"


def best(path):
    tr = [json.loads(l) for l in open(path)]; return min([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])["params"]


def objective(q):
    r = P.signature_objective_routing(q["2008-10-01":"2020-09-30"], obs["2008-10-01":"2020-09-30"]); tm = V.recession_taus(q)
    j = pd.concat([tm.rename("m"), V.TAU_OBS.rename("o")], axis=1).fillna(5000.0).clip(upper=5000.0); parts = dict(r.parts)
    parts["recession"] = float(np.median(np.abs(np.log(j.m/j.o)))/np.log(1.25)); parts["winter_volume"] = float(abs(C.djf(q) - 1.0)/0.10)
    return float(np.mean(list(parts.values()))), r.diagnostics["KGE"], tm.median(), parts


def diag(name, q, Sd=None):
    rng = np.random.default_rng(7); obj, kge, tau, parts = objective(q); a87, a09 = TS.r1(q, 1987, 2024, rng), TS.r1(q, 2009, 2024, rng)
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1)["2008-10-01":"2024-09-30"].dropna(); lk = np.nan
    if Sd is not None:
        S = pd.Series(Sd, index=st["tt"]).resample("D").mean(); lk = spearmanr(a87[2], pd.Series({y: S.get(pd.Timestamp(f"{y-1}-09-30"), np.nan) for y in a87[2].index}))[0]
    return dict(model=name, objective=obj, KGE=kge, logKGE=E.kge(np.log(j.o.clip(lower=.05)), np.log(j.s.clip(lower=.05))), recess_tau_d=tau, DJF=C.djf(q),
                fallmin_bias_pct=100*(np.exp(E.series(q, 1987, 2024)[0]).mean()/TS.LF_OBS - 1), rho1_87=a87[0], p87=a87[1], rho1_09=a09[0], p09=a09[1], deep_link=lk,
                **{f"part_{k}": v for k, v in parts.items()})


rows, Q = [], {}
pc = best(V.OUTD.parent/"ctl_treat_cal_v2"/"trials_control.jsonl"); Qs, Qd, Sd = C.simulate(pc, 0.0); Q["control: single bucket"] = C.flow(Qs, Qd, pc); rows.append(diag("control: single bucket", Q["control: single bucket"]))
pt = best(V.OUTD.parent/"ctl_treat_cal_v2"/"trials_treatment.jsonl"); Qs, Qd, Sd = C.simulate(pt, pt["f"]); q = C.flow(Qs, Qd, pt); rows.append(diag("ref: bucket + linear leak (round 2)", q, Sd))
p = dict(log_tau_s=np.log10(35.0), S_th=0.1, lscale=np.log10(48921.0)); q, tl, Qs, Qd, Sd = T.run_model(p, "treatment"); Q["treatment: threshold + deep (35 d, 0.1 m)"] = q
rows.append(diag("treatment: threshold + deep (35 d, 0.1 m)", q, Sd)); print("sweep-case tau_l", round(tl))
pb = best(T.OUTD/"trials_treatment.jsonl"); q, tl, Qs, Qd, Sd = T.run_model(pb, "treatment"); rows.append(diag("ref: threshold + deep, DE-calibrated", q, Sd))
R = pd.DataFrame(rows).set_index("model").T; pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 30)
print(R.to_string(float_format=lambda v: f"{v:.3f}")); print(f"observed: recession {V.TAU_OBS.median():.0f} d; rho1 0.38 (p 0.02) / 0.54 (p 0.035)")
R.to_csv(C.X.OUT/"east_independent_compare.csv")
fig, ax = plt.subplots(3, 1, figsize=(15, 12))
for a, (t0, t1, ttl) in zip(ax, [("2008-10-01", "2024-09-30", "WY2009-2024"), ("2011-06-01", "2013-06-30", "WY2012-2013 low-flow seasons"), ("2019-06-01", "2019-12-31", "2019 summer-fall recession")]):
    a.plot(obs[t0:t1].index, obs[t0:t1], color="black", lw=2.2, label="observed", zorder=5)
    for (k, s), c in zip(Q.items(), ("#eb6834", "#2a78d6")): a.plot(s[t0:t1].index, s[t0:t1], color=c, lw=1.3, label=k)
    a.set_yscale("log"); a.set_ylim(0.05, 150); a.set_ylabel("discharge (m3/s)"); a.set_title(ttl, loc="left"); a.grid(alpha=.3, which="both")
ax[0].legend(fontsize=9); fig.tight_layout(); fig.savefig(FIG + "east_independent_control_vs_threshold.png", dpi=120); print("figure written")
