"""Leave-one-year-out analysis of the low-flow vs perennial-snow-area rank correlation (raw and partial|peak), shadow-inclusive snow area. -> figures/draft/tuo_lowflow_snowpatch_loo.png, tuo_lowflow_snowpatch_loo.csv, tuo_lowflow_snowpatch_loo_summary.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
src = open("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/utils/tuo_lowflow_vs_snowpatch_area.py").read(); ns = {}; exec(compile(src[:src.index("rows = []; DATA")], "base", "exec"), ns)
G, names, MET, PEAK, eos, partial, D = (ns[k] for k in ("G", "names", "MET", "PEAK", "eos", "partial", "D"))
rows = []; ms = list(MET)
for g in names:
    for mn in ms:
        m = MET[mn](G[g]); d = pd.concat([m.rename("q"), eos["A_shadow"].rename("a"), PEAK.rename("pk")], axis=1).dropna()
        if len(d) < 8: continue
        r0 = spearmanr(d.a, d.q)[0]; p0 = partial(d.a.values, d.q.values, d.pk.values)[0]
        for y in d.index:
            e = d.drop(index=y); rows.append(dict(gauge=g, metric=mn, dropped=y, n=len(e), rho=spearmanr(e.a, e.q)[0], rho_partial=partial(e.a.values, e.q.values, e.pk.values)[0], rho_full=r0, rho_partial_full=p0, snow_area_dropped=d.a[y]))
L = pd.DataFrame(rows); L.to_csv(D+"tuo_lowflow_snowpatch_loo.csv", index=False); sm = []
for (g, mn), x in L.groupby(["gauge", "metric"]):
    i = (x.rho-x.rho_full).abs().idxmax(); j = (x.rho_partial-x.rho_partial_full).abs().idxmax()
    sm.append(dict(gauge=g, metric=mn, n_full=x.n.iloc[0]+1, rho_full=x.rho_full.iloc[0], rho_min=x.rho.min(), rho_max=x.rho.max(), most_influential_year=int(x.dropped[i]), rho_without_it=x.rho[i],
                   part_full=x.rho_partial_full.iloc[0], part_min=x.rho_partial.min(), part_max=x.rho_partial.max(), part_most_influential=int(x.dropped[j]), part_without_it=x.rho_partial[j]))
S = pd.DataFrame(sm); S.to_csv(D+"tuo_lowflow_snowpatch_loo_summary.csv", index=False); pd.set_option("display.width", 250); print(S.round(2).to_string(index=False))
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False}); gs = ["LBM", "LABvTB"]; fig, ax = plt.subplots(2*len(gs), len(ms), figsize=(4.6*len(ms), 3.0*2*len(gs)))
for i, g in enumerate(gs):
    for j, mn in enumerate(ms):
        x = L[(L.gauge == g) & (L.metric == mn)].sort_values("dropped")
        for k, (col, full, lab) in enumerate([("rho", "rho_full", "raw"), ("rho_partial", "rho_partial_full", "partial | peak")]):
            a = ax[2*i+k, j]; cols = ["#d55e00" if (abs(v-x[full].iloc[0]) > 0.15) else "#2a78d6" for v in x[col]]
            a.bar(range(len(x)), x[col], color=cols, width=.7); a.axhline(x[full].iloc[0], color="#0b0b0b", lw=1.2, ls="--"); a.axhline(0, color="#999", lw=.7)
            a.set_xticks(range(len(x))); a.set_xticklabels([str(int(y))[2:] for y in x.dropped], fontsize=7); a.set_ylim(-1, 1)
            a.set_title(f"{g} — {mn} ({lab}); full-sample ρ={x[full].iloc[0]:+.2f}, LOO range {x[col].min():+.2f}…{x[col].max():+.2f}", fontsize=8, loc="left")
            if j == 0: a.set_ylabel("ρ with year dropped")
            if 2*i+k == 2*len(gs)-1: a.set_xlabel("year dropped")
fig.suptitle("Leave-one-year-out: low flow vs end-of-season snow area (dashed = full sample; orange = changes ρ by >0.15)", y=1.0); fig.tight_layout(); fig.savefig(D+"tuo_lowflow_snowpatch_loo.png", dpi=140, bbox_inches="tight")
