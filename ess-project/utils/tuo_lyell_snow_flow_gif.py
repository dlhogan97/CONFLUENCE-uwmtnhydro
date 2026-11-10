"""Lyell (LBM, 15 km2; LABvTB, 107 km2) snow-covered area (Landsat, sensor rule) vs streamflow between images. Needs fsca_catch_masks.py output.
Per usable scene (>=35% valid px in the catchment, Jul 1-Oct 10): snow area km2 (snow px / valid px x catchment area) and mean specific discharge (mm/d) since the previous usable scene of the same year (>=80% of days gauged).
Outputs (figures/draft): tuo_lyell_snow_flow_table.csv, tuo_lyell_snow_flow_static.png, tuo_lyell_snow_flow_stats.csv, tuo_lyell_snow_flow_<year>.gif (years with gauge data), tuo_lyell_snow_flow_all.gif"""
import warnings; warnings.filterwarnings("ignore")
import glob, os, sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.colors as mc
from matplotlib.animation import PillowWriter
from scipy.stats import spearmanr
SC = "/scratch/dlhogan/ess-project-data/"; R = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = R+"figures/draft/"; MD = SC+"geospatial-data/snowpatch_fsca/catchment_masks/"
st = np.load(MD+"static.npz"); dem = st["dem"]; MK = {k: st[f"mask_{k}"] for k in ["LBM", "LABvTB", "DFkBC"]}; PX = 0.0009; AREA = {k: v.sum()*PX for k, v in MK.items()}
A = pd.read_csv(R+"modeling/calibration/observations/tuolumne_nested_gauge_areas.csv").set_index("id").area_km2
n = pd.read_csv(SC+"streamflow-data/TuolumneRiver/TuolumneStreamflow_Hydroclimate_2002_2021.csv", parse_dates=["Date"]).set_index("Date").loc[:"2021-08-12"]
Q = {"LBM": n.Estimated_Discharge_LBM/A["LBM"]*86.4, "LABvTB": n.Estimated_Discharge_LABvTB/A["LABvTB"]*86.4}
gy, gx = np.gradient(dem, 30.); hs = np.clip(.5+.5*(gx*np.cos(np.deg2rad(315))+gy*np.sin(np.deg2rad(315)))/np.maximum(np.hypot(gx, gy), 1e-6)*np.minimum(np.hypot(gx, gy), 1)*1.6, 0, 1)
scenes = {}
for f in sorted(glob.glob(MD+"2*.npz")):
    d = pd.Timestamp(os.path.basename(f)[:8]); scenes[d] = np.load(f)["cls"]
rows = []
for d, c in scenes.items():
    r = dict(date=d, year=d.year, doy=d.dayofyear)
    for k, m in MK.items():
        v = ((c == 1) | (c == 2))[m].mean(); s = (c == 2)[m].sum(); nv = ((c == 1) | (c == 2))[m].sum(); r[f"valid_{k}"] = v; r[f"snow_{k}"] = s/nv*AREA[k] if nv else np.nan
    rows.append(r)
S = pd.DataFrame(rows).set_index("date")
def between(q, d0, d1):
    w = q.loc[d0+pd.Timedelta(days=1):d1]; return w.mean() if len(w) and w.notna().mean() >= .8 else np.nan
recs = []
for y, g in S.groupby("year"):
    g = g[(g.valid_LBM >= .35) & (g.index.month >= 7)]; ds = list(g.index)
    for i, d in enumerate(ds):
        r = dict(date=d, year=y, doy=d.dayofyear, snow_LBM=g.loc[d, "snow_LBM"], snow_LABvTB=g.loc[d, "snow_LABvTB"], valid_LABvTB=g.loc[d, "valid_LABvTB"], prev=ds[i-1] if i else pd.NaT)
        for k in Q: r[f"q_{k}"] = between(Q[k], ds[i-1], d) if i else np.nan; r[f"q_{k}_day"] = Q[k].get(d, np.nan)
        recs.append(r)
T = pd.DataFrame(recs); T["month"] = T.date.dt.month; T["gap_d"] = (T.date-T.prev).dt.days; T.to_csv(D+"tuo_lyell_snow_flow_table.csv", index=False)
print(len(S), "scenes;", len(T), "usable (LBM); with LBM flow between:", T.q_LBM.notna().sum(), " LABvTB:", T.q_LABvTB.notna().sum())
EPS = 0.02
def rho(a, b): d = pd.concat([a, b], axis=1).dropna(); return (spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0], len(d)) if len(d) >= 5 else (np.nan, len(d))
res = []
for k in ["LBM", "LABvTB"]:
    t = T[(T.get(f"valid_{k}", 1) >= .35) if k == "LABvTB" else slice(None)] if False else T
    res.append(dict(gauge=k, subset="all", **dict(zip(["rho", "n"], rho(t[f"snow_{k}"], t[f"q_{k}"])))))
    for lab, sel in [("Jul", t.month == 7), ("Aug", t.month == 8), ("Sep", t.month == 9), ("Oct", t.month == 10), ("Jul-Aug", t.month <= 8), ("Sep-Oct", t.month >= 9)]:
        res.append(dict(gauge=k, subset=lab, **dict(zip(["rho", "n"], rho(t[sel][f"snow_{k}"], t[sel][f"q_{k}"])))))
    # within-year (demeaned logs) pooled slope, early vs late
    d = t[[f"snow_{k}", f"q_{k}", "year", "month"]].dropna(); d = d[d[f"q_{k}"] > 0]; d["ls"] = np.log10(d[f"snow_{k}"]+EPS); d["lq"] = np.log10(d[f"q_{k}"]); d = d[d.groupby("year").year.transform("size") >= 3]
    for lab, sel in [("all", d.month > 0), ("Jul-Aug", d.month <= 8), ("Sep-Oct", d.month >= 9)]:
        e = d[sel]; e = e[e.groupby("year").year.transform("size") >= 2]
        if len(e) > 4: x = e.ls-e.groupby("year").ls.transform("mean"); yv = e.lq-e.groupby("year").lq.transform("mean"); b = (x*yv).sum()/(x*x).sum(); res.append(dict(gauge=k, subset=f"within-year slope dlog q/dlog SCA {lab}", rho=b, n=len(e)))
    for y, e in t.groupby("year"):
        r_, n_ = rho(e[f"snow_{k}"], e[f"q_{k}"])
        if n_ >= 4: res.append(dict(gauge=k, subset=f"year {y}", rho=r_, n=n_))
R_ = pd.DataFrame(res); R_.to_csv(D+"tuo_lyell_snow_flow_stats.csv", index=False); print(R_.round(2).to_string(index=False))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}); mcol = {7: "#e69f00", 8: "#2a78d6", 9: "#009e73", 10: "#cc79a7"}
fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
for j, k in enumerate(["LBM", "LABvTB"]):
    a = ax[0, j]; d = T[[f"snow_{k}", f"q_{k}", "month", "year"]].dropna(); a.set_xscale("log"); a.set_yscale("log")
    for mo, c in mcol.items(): e = d[d.month == mo]; a.plot(e[f"snow_{k}"]+EPS, e[f"q_{k}"], "o", color=c, ms=4, alpha=.8, label=["", "", "", "", "", "", "Jul", "Aug", "Sep", "Oct"][mo-1+0] if False else {7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}[mo])
    for y, e in d.groupby("year"):
        if len(e) > 1: a.plot(e[f"snow_{k}"]+EPS, e[f"q_{k}"], "-", color="#bbb", lw=.6, zorder=0)
    a.set_xlabel(f"{k} catchment snow area + {EPS} (km²)"); a.set_ylabel("mean specific discharge since previous image (mm/d)"); a.set_title(f"{k}: all images", loc="left"); a.legend(fontsize=7); a.grid(color="#e3e2de", lw=.5, which="both")
    a = ax[1, j]; d2 = T[[f"snow_{k}", f"q_{k}", "doy", "year"]].dropna(); d2 = d2[d2[f"snow_{k}"] > 0.05]; sc = a.scatter(d2.doy, d2[f"q_{k}"]/d2[f"snow_{k}"]*AREA[k]/AREA[k], c=d2.year, cmap="viridis", s=18); a.set_yscale("log"); a.set_xlabel("day of year"); a.set_ylabel("flow (mm/d) per km² snow"); a.set_title(f"{k}: flow per unit snow area through the season", loc="left"); a.grid(color="#e3e2de", lw=.5, which="both"); plt.colorbar(sc, ax=a, label="year")
a = ax[0, 2]
for y, e in T.groupby("year"):
    e = e.dropna(subset=["snow_LBM"]); 
    if len(e) > 2 and y in (2006, 2009, 2011, 2012, 2013, 2014, 2015, 2017, 2018): a.plot(e.doy, e.snow_LBM+EPS, "o-", ms=3, lw=1, label=str(y))
a.set_yscale("log"); a.set_xlabel("day of year"); a.set_ylabel("LBM snow area + eps (km²)"); a.legend(fontsize=6, ncol=3); a.set_title("LBM snow area through each summer", loc="left"); a.grid(color="#e3e2de", lw=.5, which="both")
ax[1, 2].axis("off"); fig.tight_layout(); fig.savefig(D+"tuo_lyell_snow_flow_static.png", dpi=120, bbox_inches="tight")
# ---- GIFs
cm = mc.ListedColormap([(0, 0, 0, 0), (0, 0, 0, 0), "#00e5ff", (.55, .55, .55, .65)]); ext = (0, dem.shape[1]*30/1000, 0, dem.shape[0]*30/1000)
def frame_fig(y, i, e):
    d = e.iloc[i]; fig = plt.figure(figsize=(13, 6.2)); gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=.38, wspace=.18)
    am = fig.add_subplot(gs[:, 0]); am.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext, origin="upper"); c = scenes[d.date]; am.imshow(np.where(c == 0, 0, c), cmap=cm, vmin=0, vmax=3, extent=ext, origin="upper", interpolation="nearest")
    for k, col in [("LABvTB", "#009e73"), ("LBM", "#d55e00")]: am.contour(MK[k].astype(float), levels=[.5], colors=col, linewidths=1.4, extent=ext, origin="upper")
    am.set_title(f"{d.date:%Y-%m-%d}   LBM snow {d.snow_LBM:.2f} km²   LABvTB snow {d.snow_LABvTB:.2f} km²", fontsize=9, loc="left"); am.set_xticks([]); am.set_yticks([]); am.text(.01, .01, "cyan = snow, grey = cloud/shadow;\norange = LBM, green = LABvTB catchment", transform=am.transAxes, fontsize=7, color="w", va="bottom")
    at = fig.add_subplot(gs[0, 1]); w0, w1 = pd.Timestamp(y, 6, 20), pd.Timestamp(y, 10, 12)
    for k, col in [("LBM", "#d55e00"), ("LABvTB", "#009e73")]: at.plot(Q[k].loc[w0:w1], color=col, lw=1.2, label=k)
    if pd.notna(d.prev): at.axvspan(d.prev, d.date, color="#2a78d6", alpha=.2)
    at.axvline(d.date, color="#2a78d6", lw=1); [at.axvline(x, color="#999", lw=.4, ls=":") for x in e.date]; at.set_yscale("log"); at.set_ylim(.05, 40); at.set_ylabel("specific discharge (mm/d)"); at.legend(fontsize=7, loc="upper right"); at.set_title(f"{y}: flow between images (shaded)", fontsize=9, loc="left"); at.grid(color="#e3e2de", lw=.5, which="both")
    as_ = fig.add_subplot(gs[1, 1]); a0 = T.dropna(subset=["snow_LBM", "q_LBM"]); as_.loglog(a0.snow_LBM+EPS, a0.q_LBM, "o", color="#ccc", ms=3); ey = e.dropna(subset=["snow_LBM", "q_LBM"]); ey = ey[ey.date <= d.date]
    as_.plot(ey.snow_LBM+EPS, ey.q_LBM, "o-", color="#2a78d6", ms=4, lw=1); 
    if pd.notna(d.q_LBM): as_.plot(d.snow_LBM+EPS, d.q_LBM, "o", color="#d55e00", ms=9, mfc="none", mew=2)
    as_.set_xlim(.015, 6); as_.set_ylim(.4, 30); as_.set_xlabel("LBM snow area + eps (km²)"); as_.set_ylabel("flow since prev. image (mm/d)"); as_.set_title("grey: all years; blue: this year so far", fontsize=9, loc="left"); as_.grid(color="#e3e2de", lw=.5, which="both")
    return fig
import io
from PIL import Image
def make_gif(years, path, ms=900):
    frames = []
    for y in years:
        e = T[T.year == y].reset_index(drop=True)
        for i in range(len(e)):
            f = frame_fig(y, i, e); buf = io.BytesIO(); f.savefig(buf, format="png", dpi=85); plt.close(f); buf.seek(0); frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0); return len(frames)
have = [y for y in sorted(T.year.unique()) if T[T.year == y].q_LBM.notna().sum() >= 2]; print("years with flow between images:", have)
for y in have: print(y, make_gif([y], D+f"tuo_lyell_snow_flow_{y}.gif"), "frames")
print("all:", make_gif(have, D+"tuo_lyell_snow_flow_all.gif", 700), "frames")
