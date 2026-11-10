"""Does the MODEL's perennial patch store recede like the observed snow area?  Patch bins = hruId 2, 29-36 (+37 bare) of the m090_q089 run (bin_spec.csv).
Per water year: end-of-season patch area = minimum over Aug 1-Oct 10 of the area of bins with SWE > 10 mm; patch store = SWE volume (mm x km2 -> 10^3 m3) of the bins on Sep 30. Compared with the Landsat end-of-season basin snow area (eos_final.csv).
Trend given PRISM WY precipitation (log-log, %/decade) for model patch area / store and Landsat area; rank correlation model vs Landsat. -> figures/draft/model_patch_store_trend.png/.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr, io, contextlib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = U+"figures/draft/"; SIM = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_sweep_20260812/m090_q089/"
src = open(U+"utils/tuo_baseflow_trend_precip.py").read(); ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src[:src.index("rows = []; keep = {}")], "b", "exec"), ns)
fit, P = ns["fit"], ns["P"]; spec = pd.read_csv(SIM+"bin_spec.csv"); d = xr.open_dataset(SIM+"output/m090_q089_timestep.nc"); t = pd.DatetimeIndex(d.time.values); ids = d.hruId.values.astype(int); area = xr.open_dataset(SIM+"attributes.nc").HRUarea.values/1e6
patch = [i for i, h in enumerate(ids) if h in set(spec.hruId[spec.regime != "bare"])]; swe = d["scalarSWE"].values[:, patch].astype(float); a = area[patch]; print("patch HRUs:", ids[patch], "area km2:", a.round(3), "total %.2f" % a.sum())
S = pd.DataFrame(swe, index=t).resample("D").mean(); rows = []
for y in range(1982, 2026):
    w = S[(S.index >= pd.Timestamp(y, 8, 1)) & (S.index <= pd.Timestamp(y, 10, 10))]
    if len(w) < 60: continue
    ar = ((w.values > 10)*a).sum(1); sep = S.loc[pd.Timestamp(y, 9, 30)].values if pd.Timestamp(y, 9, 30) in S.index else np.full(len(a), np.nan)
    rows.append(dict(year=y, model_area_km2=ar.min(), model_area_sep30=ar[-1] if False else float(((sep > 10)*a).sum()), model_store_1e3m3=float((sep/1000*a*1e6/1e3).sum()), model_store_mean_mm=float((sep*a).sum()/a.sum())))
R = pd.DataFrame(rows).set_index("year"); eos = pd.read_csv("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/eos_final.csv").set_index("year")
R["landsat_area_km2"] = eos.A_shadow; R.to_csv(D+"model_patch_store_trend.csv"); pd.set_option("display.width", 200); print(R.round(2).T.to_string())
for lab, ser in (("model patch area (EOS min)", R.model_area_km2.clip(lower=0.02)), ("model patch store at Sep 30 (mean SWE, mm)", R.model_store_mean_mm.clip(lower=1)), ("Landsat EOS snow area (basin)", R.landsat_area_km2.clip(lower=0.02))):
    r = fit(ser, [(P, True)], list(range(1985, 2025))); print(f"precip-adjusted trend, 1985-2024: {lab:45s} {r['pct_per_decade']:+6.1f} %/decade [{r['lo']:+.0f}, {r['hi']:+.0f}] p={r['p']:.3f} n={r['n']}")
z = R[["model_area_km2", "landsat_area_km2"]].dropna(); print("rank correlation model patch area vs Landsat EOS area: rho=%.2f (n=%d)" % (spearmanr(z.model_area_km2, z.landsat_area_km2)[0], len(z)))
zz = R.loc[2013:2025, ["model_area_km2", "landsat_area_km2"]].dropna(); print("  2013-2025 only: rho=%.2f (n=%d)" % (spearmanr(zz.model_area_km2, zz.landsat_area_km2)[0], len(zz)))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); ax[0].semilogy(R.index, R.model_area_km2.clip(lower=0.01), "o-", color="#2a78d6", label="model patch bins (area with SWE>10 mm, EOS min)"); ax[0].semilogy(R.index, R.landsat_area_km2.clip(lower=0.01), "s-", color="#d55e00", label="Landsat basin snow area (EOS min)"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_ylabel("km² (log)"); ax[0].set_title("End-of-season perennial snow area: model patch bins vs Landsat", fontsize=10, loc="left"); ax[0].grid(color="#e3e2de", lw=.5)
ax[1].scatter(z.landsat_area_km2.clip(lower=0.01), z.model_area_km2.clip(lower=0.01), c=z.index, cmap="viridis"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); [ax[1].annotate(str(y)[2:], (r.landsat_area_km2, max(r.model_area_km2, 0.01)), fontsize=7, xytext=(3, 3), textcoords="offset points") for y, r in z.iterrows()]; ax[1].set_xlabel("Landsat EOS snow area (km²)"); ax[1].set_ylabel("model patch area (km²)"); ax[1].set_title("ρ=%.2f (n=%d)" % (spearmanr(z.model_area_km2, z.landsat_area_km2)[0], len(z)), fontsize=10, loc="left"); fig.tight_layout(); fig.savefig(D+"model_patch_store_trend.png", dpi=140)
