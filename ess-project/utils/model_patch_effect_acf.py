"""(1) Post-spin-up (1998-2024) precip-adjusted low-flow trends: observed HH, control, patch runs, and the model's PATCH EFFECT (patch run / control) per year and its trend.
(2) Lag-1..5 autocorrelation of the precip-adjusted low-flow residuals (log metrics regressed on ln P) for obs and model runs, raw and after removing a linear trend (rank ACF, n~27-43; +-1.96/sqrt(n) band).
-> figures/draft/model_patch_effect_acf.csv/.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = U+"figures/draft/"
src = open(U+"utils/model_lowflow_trend_compare.py").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src[:src.index("rows = []\ndef one")], "b", "exec"), ns)
M, fit, P, MET = ns["M"], ns["fit"], ns["P"], ns["MET"]; RUNS = ["control (no patches)", "patch m090q089", "patch anchor m080"]; WIN = list(range(1998, 2025))
print("=== 1998-2024 precip-adjusted trend (%/decade [95% CI] p) ===")
rows = []
for lab in ["HH unimpaired"]+RUNS:
    for m in ["WY min 7-d", "WY Q10", "Aug-Nov min 7-d", "Sep mean", "Aug mean"]:
        r = fit(M[lab][m], [(P, True)], WIN); rows.append((lab, m, r))
print(pd.DataFrame([{ "series": l, "metric": m, "txt": f"{r['pct_per_decade']:+5.1f} [{r['lo']:+.0f},{r['hi']:+.0f}] p={r['p']:.2f}"} for l, m, r in rows]).pivot(index="series", columns="metric", values="txt").to_string())
print("\n=== the model's PATCH EFFECT = patch run / control, by year (all low-flow metrics); trend 1998-2024 ===")
for run in RUNS[1:]:
    eff = pd.DataFrame({m: M[run][m]/M["control (no patches)"][m] for m in ["Aug mean", "Sep mean", "Aug-Nov min 7-d"]}).loc[1998:2024]; 
    print(run, "| median effect: ", {m: f"{100*(eff[m].median()-1):+.1f}%" for m in eff}, "| mean effect first/last 9 yr (Sep mean): %+.1f%% -> %+.1f%%" % (100*(eff["Sep mean"].iloc[:9].mean()-1), 100*(eff["Sep mean"].iloc[-9:].mean()-1)))
    for m in eff:
        y = np.log(eff[m]); b = np.polyfit(eff.index, y, 1)[0]*10; print(f"   trend of the patch effect on {m:16s}: {100*(np.exp(b)-1):+.1f} %/decade  (obs needs about -10 to -19 %/decade)")
print("\n=== ACF of precip-adjusted low-flow residuals (rank, lags 1-5) ===")
def resid(ser, yrs, detr):
    d = pd.concat([np.log(ser.clip(lower=1e-2)).rename("y"), np.log(P).rename("p")], axis=1).dropna(); d = d[d.index.isin(yrs)]; cols = [np.ones(len(d)), d.p.values]+([(d.index.values-2000)/10.] if detr else []); X = np.column_stack(cols); r = d.y.values-X@np.linalg.lstsq(X, d.y.values, rcond=None)[0]; return pd.Series(r, index=d.index)
def acf(r, nl=5):
    out = []
    for k in range(1, nl+1):
        a = r.values[:-k] if False else None
        idx = [y for y in r.index if y-k in r.index]
        out.append(spearmanr(r[[y-k for y in idx]].values, r[idx].values)[0] if len(idx) > 6 else np.nan)
    return np.array(out)
tab = []
for lab, yrs in (("HH unimpaired", range(1982, 2025)), ("control (no patches)", range(1982, 2025)), ("patch m090q089", range(1998, 2025)), ("control (no patches), 1998-2024", range(1998, 2025)), ("USGS 11274790", range(2007, 2026))):
    key = lab.split(",")[0]
    for m in ["Aug-Nov min 7-d", "Sep mean"]:
        for detr in (False, True):
            r = resid(M[key][m], list(yrs), detr); a = acf(r); tab.append(dict(series=lab, metric=m, detrended=detr, n=len(r), band=1.96/np.sqrt(len(r)), **{f"lag{k+1}": a[k] for k in range(5)}))
T = pd.DataFrame(tab); T.to_csv(D+"model_patch_effect_acf.csv", index=False); pd.set_option("display.width", 220); print(T.round(2).to_string(index=False))
