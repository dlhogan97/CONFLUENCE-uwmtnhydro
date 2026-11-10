"""Integrate: (i) diurnal discharge signal + water temperature (QC'd nested gauges, Aug-Oct), (ii) end-of-season / Aug-1 snow area and the patch-melt capacity index, (iii) autumn air temperature and shortwave (METSIM forcing, basin and high-elevation),
(iv) low-flow metrics (Sep volume, Aug-Nov min 7-d flow). Spearman matrices + PARTIAL correlations (controlling for autumn T_high and SW_high, i.e. the weather that drives melt).
-> figures/draft/tuo_patch_signal_heatmap.png, tuo_patch_signal_scatter.png, tuo_patch_signal_corr.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, t as tdist
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBS = U+"modeling/calibration/observations/"; D = U+"figures/draft/"; FS = "/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/"
src = open(U+"utils/tuo_augnov_lag1_acf_qc.py").read(); ns = {}; exec(compile(src[:src.index('L = pd.DataFrame({k: low7')], "base", "exec"), ns); G, low7 = ns["G"], ns["low7"]
NEST = ["T120", "LBM", "LABvTB", "DFkBC"]
met = pd.read_csv(OBS+"tuolumne_daily_met.csv", index_col=0, parse_dates=True)
def per_year(s, months, how="mean"):
    x = s[s.index.month.isin(months)]; g = x.groupby(x.index.year); return (g.mean() if how == "mean" else g.sum()).rename(None)
PD = {}
PD["Aug-Oct T basin"] = per_year(met.T_basin, [8, 9, 10]); PD["Aug-Oct T high"] = per_year(met.T_high, [8, 9, 10]); PD["Aug-Oct SW basin"] = per_year(met.SW_basin, [8, 9, 10]); PD["Aug-Oct SW high"] = per_year(met.SW_high, [8, 9, 10])
PD["Aug-Sep PDD high"] = per_year(met.T_high.clip(lower=0), [8, 9], "sum")
bud = pd.read_csv(D+"tuo_patch_melt_budget.csv").set_index("year"); eos = pd.read_csv(FS+"eos_final.csv").set_index("year")
PD["snow area (EOS min)"] = eos.A_shadow; PD["snow area Aug 1"] = bud.A_aug1; PD["patch melt capacity (DDF8)"] = bud.V_DDF8_Mm3
PRED = pd.DataFrame(PD)
ann = pd.read_csv(OBS+"tuo_diurnal_annual.csv"); ann = ann[ann.window == "Aug-Oct"]
def sepvol(q):
    q = q.asfreq("D"); r = {}
    for y in sorted(set(q.index.year)):
        z = q[(q.index.year == y) & (q.index.month == 9)].dropna()
        if len(z) >= 25: r[y] = z.mean()*30*86400/1e6
    return pd.Series(r)
TARG = {}
for g in NEST:
    a = ann[ann.gauge == g].set_index("year")
    TARG[(g, "diurnal Q (signed melt-like)")] = a.q_proj; TARG[(g, "diurnal Q (rel. amp.)")] = a.q_rel_amp; TARG[(g, "water temp (Aug-Oct)")] = a.wt_mean; TARG[(g, "water temp diurnal range")] = a.wt_range
    TARG[(g, "Sep volume")] = sepvol(G[g]); TARG[(g, "Aug-Nov min 7-d")] = low7(G[g])
def pcor(x, y, Z):
    X = np.column_stack([np.ones(len(x))]+[rankdata(z) for z in Z]); rx = rankdata(x); ry = rankdata(y)
    ex = rx-X@np.linalg.lstsq(X, rx, rcond=None)[0]; ey = ry-X@np.linalg.lstsq(X, ry, rcond=None)[0]; r = np.corrcoef(ex, ey)[0, 1]; n = len(x); df = n-2-len(Z)
    return r, (2*tdist.sf(abs(r*np.sqrt(df/(1-r**2))), df) if df > 0 else np.nan)
rows = []
for (g, t), s in TARG.items():
    for c in PRED.columns:
        d = pd.concat([s.rename("y"), PRED[c].rename("x"), PRED["Aug-Oct T high"].rename("Tw"), PRED["Aug-Oct SW high"].rename("Sw")], axis=1).dropna()
        if len(d) < 8: continue
        r, p = spearmanr(d.x, d.y); rec = dict(gauge=g, target=t, predictor=c, n=len(d), rho=r, p=p)
        if c in ("snow area (EOS min)", "snow area Aug 1", "patch melt capacity (DDF8)"): rp, pp = pcor(d.x.values, d.y.values, [d.Tw.values, d.Sw.values]); rec.update(rho_partial_TSW=rp, p_partial=pp)
        rows.append(rec)
R = pd.DataFrame(rows); R.to_csv(D+"tuo_patch_signal_corr.csv", index=False); pd.set_option("display.width", 250)
print(R[R.predictor.isin(["snow area (EOS min)", "patch melt capacity (DDF8)"])][["gauge", "target", "predictor", "n", "rho", "p", "rho_partial_TSW", "p_partial"]].round(2).to_string(index=False))
print("\nautumn T / SW vs targets (rho):"); print(R[R.predictor.isin(["Aug-Oct T high", "Aug-Oct SW high", "Aug-Sep PDD high"])].pivot_table(index=["gauge", "target"], columns="predictor", values="rho").round(2))
# ---- heatmap: targets rows x predictors
tl = ["diurnal Q (signed melt-like)", "diurnal Q (rel. amp.)", "water temp (Aug-Oct)", "water temp diurnal range", "Sep volume", "Aug-Nov min 7-d"]; fig, ax = plt.subplots(1, 4, figsize=(24, 5.6), sharey=True)
for k, g in enumerate(NEST):
    M = R[R.gauge == g].pivot(index="target", columns="predictor", values="rho").reindex(index=tl, columns=PRED.columns); P = R[R.gauge == g].pivot(index="target", columns="predictor", values="p").reindex(index=tl, columns=PRED.columns); N = R[R.gauge == g].pivot(index="target", columns="predictor", values="n").reindex(index=tl, columns=PRED.columns)
    im = ax[k].imshow(M.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto"); ax[k].set_xticks(range(M.shape[1])); ax[k].set_xticklabels(M.columns, rotation=50, ha="right", fontsize=8); ax[k].set_yticks(range(len(tl))); ax[k].set_yticklabels(tl, fontsize=8); ax[k].set_title(g, fontsize=10, loc="left")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v): ax[k].text(j, i, f"{v:+.2f}{'*' if P.values[i, j] < 0.05 else ''}\nn={int(N.values[i, j])}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else "#222")
fig.colorbar(im, ax=ax, fraction=0.01); fig.suptitle("Spearman ρ (* p<0.05): diurnal / stream-temperature / low-flow metrics vs snow area, patch-melt capacity, autumn temperature and shortwave (nested gauges, QC'd record)", y=1.02, fontsize=11); fig.savefig(D+"tuo_patch_signal_heatmap.png", dpi=140, bbox_inches="tight")
# ---- scatter: diurnal signal and water temperature vs snow area (+ weather colouring)
fig, ax = plt.subplots(2, 4, figsize=(17, 7.2))
for k, g in enumerate(NEST):
    for i, (t, lab) in enumerate([("diurnal Q (signed melt-like)", "signed diurnal Q amplitude (rel.)"), ("water temp (Aug-Oct)", "median water temperature (°C)")]):
        d = pd.concat([TARG[(g, t)].rename("y"), PRED["snow area (EOS min)"].rename("x"), PRED["Aug-Oct T high"].rename("Tw")], axis=1).dropna(); a = ax[i, k]
        sc_ = a.scatter(d.x, d.y, c=d["Tw"], cmap="coolwarm", s=45, edgecolor="white", lw=.6); [a.annotate(str(int(y))[2:], (r.x, r.y), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#52514e") for y, r in d.iterrows()]
        rr = R[(R.gauge == g) & (R.target == t) & (R.predictor == "snow area (EOS min)")]; a.set_xscale("symlog", linthresh=0.5); a.axhline(0, color="#ddd", lw=.7) if i == 0 else None
        if len(rr): r0 = rr.iloc[0]; a.set_title(f"{g}: ρ={r0.rho:+.2f} (p={r0.p:.2f}); partial|T,SW={r0.rho_partial_TSW:+.2f} (p={r0.p_partial:.2f}); n={int(r0.n)}", fontsize=8, loc="left")
        a.set_xlabel("end-of-season snow area (km², symlog)"); a.set_ylabel(lab) if k == 0 else None; a.grid(color="#e3e2de", lw=.5)
fig.colorbar(sc_, ax=ax, fraction=0.015, label="Aug–Oct air temp, high elev. (°C)"); fig.savefig(D+"tuo_patch_signal_scatter.png", dpi=140, bbox_inches="tight")
