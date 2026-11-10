"""DMC of Lyell (LBM) against two references: Dana Fork (DFkBC, independent) and T120 (189 km2, contains Lyell+Dana: NOT independent, LBM is 8% of its area), for baseline cutoffs (day of year) 180-240, excluding 2011 and 2017.
Same method as tuo_double_mass_csc.py (literal): cumulative specific discharge from May 1, s0 = OLS DMC slope for day<=cutoff, onset = first 7-d run after cutoff with local slope > s0, termination = first 7-d run below s0.
T120 values after 2016-10-01 divided by 35.3146. -> figures/draft/tuo_dmc_multi_table.csv, tuo_dmc_multi_summary.png, tuo_dmc_multi_curves_<ref>.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SC = "/scratch/dlhogan/ess-project-data/"; R = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = R+"figures/draft/"
A = pd.read_csv(R+"modeling/calibration/observations/tuolumne_nested_gauge_areas.csv").set_index("id").area_km2
n = pd.read_csv(SC+"streamflow-data/TuolumneRiver/TuolumneStreamflow_Hydroclimate_2002_2021.csv", parse_dates=["Date"]).set_index("Date").loc[:"2021-08-12"]
n["Estimated_Discharge_T120"] = n.Estimated_Discharge_T120.where(n.index < "2016-10-01", n.Estimated_Discharge_T120/35.3146)
sp = lambda col, a: n[col]/A[a]*86400/1e3
Q = {"LBM": sp("Estimated_Discharge_LBM", "LBM"), "DFkBC": sp("Estimated_Discharge_DFkBC", "DFkBC"), "T120": sp("Estimated_Discharge_T120", "T120")}
EXC = {2011, 2017}; CUTS = [180, 200, 215, 230]; W = 7
def run(ref, cut, drop=EXC):
    rows, cur = [], {}
    for y in range(2004, 2022):
        if y in drop: continue
        w = pd.DataFrame({"G": Q["LBM"], "N": Q[ref]}).loc[f"{y}-05-01":f"{y}-10-31"]
        if len(w) < 150: continue
        ok = w.notna().all(axis=1)
        if (~ok).sum() > 12 or not ok.iloc[:30].all() or ok.iloc[-45:-14].sum() < 30: continue
        w = w.interpolate(limit=10).dropna(); doy = w.index.dayofyear.values; cG, cN = w.G.cumsum(), w.N.cumsum(); base = doy <= cut
        X = np.c_[np.ones(base.sum()), cN[base].values]; b, res, *_ = np.linalg.lstsq(X, cG[base].values, rcond=None); s0 = b[1]
        se = np.sqrt((res[0]/(base.sum()-2))/((cN[base]-cN[base].mean())**2).sum())
        loc = (cG.diff(W)/cN.diff(W)).shift(-(W//2)); hi = (loc > s0).values; on = term = None
        for i in np.where(doy > cut)[0]:
            if hi[i:i+W].all(): on = i; break
        r = dict(ref=ref, cut=cut, year=y, s0=s0, s0_se=se)
        if on is not None:
            for i in range(on+W, len(w)):
                if (~hi[i:i+W]).all(): term = i; break
            term = len(w)-1 if term is None else term; sl = slice(on, term+1); vol = w.G.values[sl].sum(); csc = (w.G.values[sl]-s0*w.N.values[sl]).sum()
            r.update(on_doy=doy[on], off_doy=doy[term], csc_mm=csc, csc_err=se*w.N.values[sl].sum(), lyell_mm=vol, csc_frac=csc/vol)
        rows.append(r); cur[y] = (cN, cG, s0, on, term)
    return pd.DataFrame(rows), cur
out = []; curves = {}
for ref in ["DFkBC", "T120"]:
    for c in CUTS:
        t, cur = run(ref, c); out.append(t); curves[(ref, c)] = cur
T = pd.concat(out); T.to_csv(D+"tuo_dmc_multi_table.csv", index=False)
print(T.groupby(["ref", "cut"]).agg(n=("year", "size"), s0_med=("s0", "median"), on_doy_med=("on_doy", "median"), on_min=("on_doy", "min"), on_max=("on_doy", "max"), csc_frac_med=("csc_frac", "median"), csc_frac_min=("csc_frac", "min"), csc_frac_max=("csc_frac", "max")).round(2).to_string())
print(T[T.cut == 215].pivot(index="year", columns="ref", values="on_doy").to_string()); print(T[T.cut == 215].pivot(index="year", columns="ref", values="csc_frac").round(2).to_string())
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}); cols = ["#2a78d6", "#d55e00", "#009e73", "#cc79a7"]
fig, ax = plt.subplots(2, 3, figsize=(13, 6.4))
for ri, ref in enumerate(["DFkBC", "T120"]):
    for c, col in zip(CUTS, cols):
        t = T[(T.ref == ref) & (T.cut == c)]; ax[ri, 0].plot(t.year, t.on_doy, "o-", color=col, label=f"cutoff {c}"); ax[ri, 1].plot(t.year, t.csc_frac, "o-", color=col); ax[ri, 2].plot(t.year, t.s0, "o-", color=col)
    ax[ri, 0].set_ylabel(f"ref {ref}\nonset day of year"); ax[ri, 1].set_ylabel("CSC / Lyell flow in window"); ax[ri, 2].set_ylabel("baseline slope s0")
ax[0, 0].legend(fontsize=7); [a.grid(color="#e3e2de", lw=.5) for a in ax.ravel()]; fig.suptitle("Lyell DMC onset, cryospheric fraction and baseline slope by reference and cutoff (2011, 2017 excluded)"); fig.tight_layout(); fig.savefig(D+"tuo_dmc_multi_summary.png", dpi=130, bbox_inches="tight")
for ref in ["DFkBC", "T120"]:
    cur = curves[(ref, 215)]; ys = list(cur); fig, ax = plt.subplots(2, 5, figsize=(16, 6.2)); ax = ax.ravel()
    for a, y in zip(ax, ys):
        cN, cG, s0, on, term = cur[y]; a.plot(cN, cG, color="#2a78d6"); a.plot(cN, s0*cN, "--", color="#999", lw=.9)
        if on is not None: a.plot(cN.iloc[on], cG.iloc[on], "o", color="#d55e00"); a.plot(cN.iloc[term], cG.iloc[term], "s", color="#009e73")
        a.set_title(f"{y} s0={s0:.2f}" + (f" on doy {cN.index[on].dayofyear}" if on is not None else " no onset"), fontsize=8, loc="left"); a.grid(color="#e3e2de", lw=.5)
    for a in ax[len(ys):]: a.axis("off")
    fig.suptitle(f"Lyell vs {ref}, cutoff doy 215 (cumulative mm from May 1)"); fig.tight_layout(); fig.savefig(D+f"tuo_dmc_multi_curves_{ref}.png", dpi=125, bbox_inches="tight")
