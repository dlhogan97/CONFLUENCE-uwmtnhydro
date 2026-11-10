"""Does the model reproduce the observed precipitation-adjusted low-flow trend, with and without perennial snow patches?
Runs (daily discharge at the basin outlet, 1981-2025): control = SUMMA baseline_wbias10 (no patches); patch variants = snowpatch_m090q089 and snowpatch_anchorm080 (N-bin patch store). Observations: Hetch Hetchy unimpaired (HH, 1970-2024) and USGS 11274790 (2007-2025).
Per series/metric: ln Q_metric = a + b ln P_WY + c * decade (P = PRISM-based basin WY precipitation, same for every series); c in %/decade, 95% CI and p from OLS. 'obs - model' = the same fit applied to ln(Q_obs / Q_model) (difference of the metrics), i.e. does the model miss the observed trend?
-> figures/draft/model_lowflow_trend_compare.csv/.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = U+"figures/draft/"; SF = "/scratch/dlhogan/ess-project-data/streamflow-data/TuolumneRiver/"
src = open(U+"utils/tuo_baseflow_trend_precip.py").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src[:src.index("rows = []; keep = {}")], "b", "exec"), ns)
G, metrics, fit, P = ns["G"], ns["metrics"], ns["fit"], ns["P"]
def csv_series(name): d = pd.read_csv(SF+name, parse_dates=["time"]).set_index("time")["discharge"]; return d.asfreq("D")
RUN = {"control (no patches)": csv_series("TuolumneRiver_SUMMA_baseline_dailyDischarge_1981_2025.csv"), "patch m090q089": csv_series("TuolumneRiver_SUMMA_snowpatch_m090q089_dailyDischarge_1981_2025.csv"), "patch anchor m080": csv_series("TuolumneRiver_SUMMA_snowpatch_anchorm080_dailyDischarge_1981_2025.csv")}
OBS = {"HH unimpaired": G["HH unimpaired"], "USGS 11274790": G["USGS11274790"]}
MET = ["WY min 7-d", "WY Q10", "Aug-Nov min 7-d", "Sep mean", "Aug mean"]; M = {k: metrics(v) for k, v in {**OBS, **RUN}.items()}
for k, v in RUN.items(): print(k, "first/last date", v.first_valid_index().date(), v.last_valid_index().date(), "| mean flow %.1f m3/s" % v.mean())
rows = []
def one(label, ser, yrs, kind):
    for m in MET:
        r = fit(ser[m] if hasattr(ser, "columns") else ser, [(P, True)], yrs) if False else fit(ser[m], [(P, True)], yrs)
        if r: rows.append(dict(window=f"{min(yrs)}-{max(yrs)}", series=label, kind=kind, metric=m, n=r["n"], pct_per_decade=r["pct_per_decade"], lo=r["lo"], hi=r["hi"], p=r["p"]))
for win, yrs in (("1985-2024", range(1985, 2025)), ("2008-2025", range(2008, 2026))):
    for lab in OBS: one(lab, M[lab], list(yrs), "obs")
    for lab in RUN: one(lab, M[lab], list(yrs), "model")
    # obs - model: trend of the log-ratio of the metrics
    for ol in OBS:
        if (win == "2008-2025") != (ol == "USGS 11274790") and not (win == "1985-2024" and ol == "HH unimpaired"): continue
        for rl in RUN:
            diff = pd.DataFrame({m: M[ol][m]/M[rl][m] for m in MET}); one(f"{ol} / {rl}", diff, list(yrs), "obs/model")
R = pd.DataFrame(rows); R.to_csv(D+"model_lowflow_trend_compare.csv", index=False); pd.set_option("display.width", 220)
for win in R.window.unique():
    x = R[(R.window == win) & (R.kind != "obs/model")]; print(f"\n=== precip-adjusted trend, {win} (%/decade [95% CI] p) ==="); 
    t = x.assign(txt=lambda d: d.apply(lambda r: f"{r.pct_per_decade:+5.1f} [{r.lo:+.0f},{r.hi:+.0f}] p={r.p:.2f}", axis=1)).pivot(index="series", columns="metric", values="txt")[MET]; print(t.to_string())
    y = R[(R.window == win) & (R.kind == "obs/model")]
    if len(y): print(f"--- observed / modelled (trend of the ratio: positive = model declines too much, negative = model misses the decline) ---"); print(y.assign(txt=lambda d: d.apply(lambda r: f"{r.pct_per_decade:+5.1f} [{r.lo:+.0f},{r.hi:+.0f}] p={r.p:.2f}", axis=1)).pivot(index="series", columns="metric", values="txt")[MET].to_string())
