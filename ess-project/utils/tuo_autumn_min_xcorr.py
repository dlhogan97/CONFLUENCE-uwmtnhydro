"""Tuolumne observed autumn minimum flows (Sep 1-Nov 30, only days >=5 d after the last day-to-day rise) and their cross-correlation across ALL observed gauges.
Gauges: 4 nested headwater gauges (T120 / LBM / LABvTB / DFkBC, 2002-2021), USGS 11274790 (WY2007-2025), HH unimpaired reconstruction (1970-2024).
-> ess-project/figures/draft/tuo_autumn_min_xcorr.png , tuo_autumn_min_table.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
SC = "/scratch/dlhogan/ess-project-data/"; SF = SC+"streamflow-data/TuolumneRiver/"
RISE, QUIET, MIN_DAYS = 0.05, 5, 10          # a "rise" = day-to-day increase >5%; a day is eligible when none of the previous 5 days rose; need >=10 eligible days
def hh_unimpaired():
    d = pd.read_csv(SF+"TuolumneRiver_HH_unimpaired_19702024.csv", encoding="utf-8-sig").set_index("DATE"); d = d[d.index.astype(str).str.fullmatch(r"\d+/\d+")].apply(pd.to_numeric, errors="coerce"); out = []
    for c in d.columns:
        wy = int(c.split()[1])
        for md, v in d[c].items():
            m, dd = map(int, md.split("/")); y = wy-1 if m >= 10 else wy
            try: out.append((pd.Timestamp(y, m, dd), v*0.0283168))
            except ValueError: pass
    return pd.Series(dict(out)).sort_index()
nest = pd.read_csv(SF+"TuolumneStreamflow_Hydroclimate_2002_2021.csv", parse_dates=["Date"]).set_index("Date"); nest.columns = [c.replace("Estimated_Discharge_", "") for c in nest.columns]
G = {k: nest[k].astype(float) for k in nest}
G["T120"] = G["T120"].where(G["T120"].index < "2016-10-01", G["T120"]/35.3146)             # unapplied cfs->cms step after WY2016 (see tuolumne-lowflow-memory-is-splice-artifact)
usgs = pd.read_csv(SC+"domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/preprocessed/Tuolumne_River_USGS11274790_dailyQ_WY2007_2025.csv", parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
G["USGS11274790"] = usgs; G["HH unimpaired"] = hh_unimpaired()
def autumn_min(q):
    q = q.asfreq("D"); rise = (q.pct_change() > RISE).astype(float); quiet = rise.rolling(QUIET, min_periods=QUIET).sum().shift(0).eq(0)   # no rise on this day or the previous 4 -> >=5 d after last rise
    ok = quiet & q.index.month.isin([9, 10, 11]) & q.notna() & (q > 0); r = {}
    for y in sorted(set(q.index.year)):
        s = q[ok & (q.index.year == y)]
        if len(s) >= MIN_DAYS: r[y] = s.min()
    return pd.Series(r)
T = pd.DataFrame({k: autumn_min(v) for k, v in G.items()}); T.index.name = "year"
T.to_csv("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/tuo_autumn_min_table.csv")
names = list(T.columns); n = len(names); R = pd.DataFrame(np.nan, names, names); N = R.copy(); P = R.copy()
for a in names:
    for b in names:
        if a == b: N.loc[a, b] = T[a].notna().sum(); R.loc[a, b] = 1.0; P.loc[a, b] = 0.0; continue
        j = T[[a, b]].dropna(); N.loc[a, b] = len(j)
        if len(j) >= 5: r, p = spearmanr(j[a], j[b]); R.loc[a, b] = r; P.loc[a, b] = p
print("autumn-min years per gauge:", T.notna().sum().to_dict()); print(R.round(2)); print(N.astype(int))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(15, 6.6)); ax = fig.add_axes([0.04, 0.12, 0.30, 0.78]); im = ax.imshow(R.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_yticklabels(names)
for i in range(n):
    for k in range(n):
        if not np.isnan(R.iloc[i, k]): ax.text(k, i, f"{R.iloc[i,k]:+.2f}{'*' if P.iloc[i,k] < 0.05 and i != k else ''}\nn={int(N.iloc[i,k])}", ha="center", va="center", fontsize=7.5, color="white" if abs(R.iloc[i, k]) > 0.6 else "#222")
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Spearman correlation of annual autumn minimum flow\n(* p<0.05; pairwise over overlapping years)", fontsize=10, loc="left"); fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
axs = fig.add_axes([0.42, 0.12, 0.56, 0.78]); cols = plt.cm.tab10(np.arange(n)); yrs = T.index
for k, c in enumerate(names):
    z = T[c].dropna()
    if len(z): axs.plot(z.index, z.values, "-o", ms=3.2, lw=1.1, color=cols[k], label=f"{c} (n={len(z)})")
axs.set_yscale("log"); axs.set_xlim(1969, 2026); axs.set_ylabel("autumn minimum flow (m³/s, log)"); axs.grid(axis="y", color="#e3e2de", lw=.6); axs.legend(ncol=2, frameon=False, fontsize=8, loc="lower right")
axs.set_title(f"Autumn (Sep-Nov) minimum flow by year, days ≥{QUIET} d after any >{int(RISE*100)}% daily rise; T120 divided by 35.3 after WY2016", fontsize=10, loc="left")
fig.savefig("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/tuo_autumn_min_xcorr.png", dpi=150, bbox_inches="tight")
# pairwise scatter matrix (log-log) for the same gauges
fig2, A = plt.subplots(n, n, figsize=(13, 12))
for i, a in enumerate(names):
    for k, b in enumerate(names):
        ax = A[i, k]; ax.set_xticks([]); ax.set_yticks([])
        if i == k: ax.text(.5, .5, a, ha="center", va="center", fontsize=10, transform=ax.transAxes); ax.set_axis_off(); continue
        j = T[[b, a]].dropna()
        if i < k: ax.set_axis_off(); ax.text(.5, .5, f"ρ={R.loc[a,b]:+.2f}{'*' if P.loc[a,b]<0.05 else ''}\nn={int(N.loc[a,b])}" if len(j) >= 5 else f"n={len(j)}", ha="center", va="center", fontsize=12, transform=ax.transAxes); continue
        ax.loglog(j[b], j[a], "o", ms=4, color="#2a78d6")
        if len(j) >= 2: lo, hi = min(j.min()), max(j.max()); ax.plot([lo, hi], [lo, hi], color="#bbb", lw=.7)
        if k == 0: ax.set_ylabel(a, fontsize=8)
        if i == n-1: ax.set_xlabel(b, fontsize=8)
fig2.suptitle("Annual autumn-minimum flow, gauge vs gauge (m³/s, log-log; grey = 1:1)", y=0.995); fig2.tight_layout()
fig2.savefig("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/tuo_autumn_min_scatter_matrix.png", dpi=130, bbox_inches="tight")
