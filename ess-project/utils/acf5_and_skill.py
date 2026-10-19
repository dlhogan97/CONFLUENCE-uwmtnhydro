"""Lag-1..5 low-flow ACF (obs / baseline / adjusted) + daily NSE-KGE for every sweep member.

ACF is the notebook's estimator VERBATIM (Spearman, year-aligned reindex, permutation p).
Skill scores are on DAILY routed streamflow over the obs-model overlap, plus a low-flow-
season-only variant so we can see whether the store trades whole-hydrograph skill for
late-season behaviour.
"""
from pathlib import Path
import glob, re, json
import numpy as np, pandas as pd, netCDF4 as nc4
from scipy.stats import rankdata

SC = Path("/scratch/dlhogan/ess-project-data")
EAST = SC/"domain_East_River_distributed_elevAspect"
TUO = SC/"domain_Tuolumne_River_distributed_elevAspect"
EAST_SW = EAST/"simulations/store_memory_sandbox/East/sweep_sf_20260812"
TUO_SW = TUO/"simulations/snowpatch_sweep_20260812"
YR0, YR1 = 1970, 2024
NLAG = 5


# ---------------------------------------------------------------- notebook cell 4
def compute_acf(series, n_lags=NLAG, n_permutations=5000, seed=0):
    def _spearman(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        ra = rankdata(a) - (len(a)+1)/2.0; rb = rankdata(b) - (len(b)+1)/2.0
        den = np.sqrt((ra*ra).sum()*(rb*rb).sum())
        return (ra*rb).sum()/den if den > 0 else 0.0
    rng = np.random.default_rng(seed)
    s = series.dropna().astype(float)
    years = s.index.values.astype(int); v = s.values; n = v.size
    posn = {int(y): i for i, y in enumerate(years)}
    rho = np.full(n_lags, np.nan); pval = np.ones(n_lags)
    for k in range(1, n_lags+1):
        ip = np.array([posn[int(y)-k] for y in years if int(y)-k in posn], int)
        ic = np.array([i for i, y in enumerate(years) if int(y)-k in posn], int)
        if ip.size <= 2: continue
        rho[k-1] = _spearman(v[ip], v[ic])
        perm = np.array([abs(_spearman(rr[ip], rr[ic])) for rr in
                         (rng.permutation(v) for _ in range(n_permutations))])
        pval[k-1] = np.mean(perm >= abs(rho[k-1]))
    return dict(rho=rho, p=pval, band=1.96/np.sqrt(n), n=n)


def low(d, months, rs):
    s = (d.where(d.index.month.isin(months))
          .rolling(7, center=True, min_periods=7).mean().resample(rs).min())
    s[s < 0.01] = np.nan
    s = s.copy(); s.index = s.index.year
    return s[(s.index >= YR0) & (s.index <= YR1)]


# ---------------------------------------------------------------- skill scores
def nse(o, m):
    return 1 - np.sum((m-o)**2)/np.sum((o-o.mean())**2)


def kge(o, m):
    r = np.corrcoef(o, m)[0, 1]
    a = m.std(ddof=0)/o.std(ddof=0)
    b = m.mean()/o.mean()
    return 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2), r, a, b


SPINUP_END = "1983-09-30"   # meta.json analysis_start for both basins; also drops the
                            # WY1981 aquifer-init spike (717 m3/s in East sf=7)


def scores(obs, mod, months=None):
    j = pd.DataFrame({"o": obs, "m": mod}).dropna()
    j = j[j.index > SPINUP_END]
    if months is not None:
        j = j[j.index.month.isin(months)]
    j = j[j.o > 0]
    o, m = j.o.values, j.m.values
    k, r, a, b = kge(o, m)
    lo, lm = np.log(np.maximum(o, 1e-4)), np.log(np.maximum(m, 1e-4))
    return dict(n=len(j), NSE=nse(o, m), KGE=k, r=r, alpha=a, beta=b,
                logNSE=nse(lo, lm), PBIAS=100*(m.sum()-o.sum())/o.sum())


# ---------------------------------------------------------------- model I/O
def qdaily(run):
    run = Path(run)
    at = nc4.Dataset(run/"attributes.nc"); a = float(np.array(at["HRUarea"][:]).sum()); at.close()
    f = sorted(glob.glob(str(run/"output"/"*.nc")) + glob.glob(str(run/"out"/"*.nc")))[0]
    ds = nc4.Dataset(f); t = ds["time"]
    tt = nc4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
    q = np.array(ds["averageRoutedRunoff"][:]).ravel()*a; ds.close()
    return pd.Series(q, index=pd.DatetimeIndex(tt)).resample("1D").mean()


def report(basin, obs_daily, runs, months, rs):
    print("\n" + "="*96)
    print(f"{basin}   low-flow window = months {months}, water-year bins '{rs}'")
    print("="*96)

    ob = low(obs_daily, months, rs)
    a_ob = compute_acf(ob)
    rows = [dict(model="OBSERVED", n=a_ob["n"], band=a_ob["band"],
                 **{f"rho{k+1}": a_ob["rho"][k] for k in range(NLAG)},
                 **{f"p{k+1}": a_ob["p"][k] for k in range(NLAG)})]
    srows = []
    for label, path in runs:
        q = qdaily(path)
        a = compute_acf(low(q, months, rs))
        rows.append(dict(model=label, n=a["n"], band=a["band"],
                         **{f"rho{k+1}": a["rho"][k] for k in range(NLAG)},
                         **{f"p{k+1}": a["p"][k] for k in range(NLAG)},
                         mae=float(np.nanmean(np.abs(a["rho"]-a_ob["rho"])))))
        s_all = scores(obs_daily, q)
        s_low = scores(obs_daily, q, months)
        srows.append(dict(model=label, n_days=s_all["n"],
                          NSE=s_all["NSE"], KGE=s_all["KGE"], logNSE=s_all["logNSE"],
                          r=s_all["r"], alpha=s_all["alpha"], beta=s_all["beta"],
                          PBIAS=s_all["PBIAS"],
                          KGE_low=s_low["KGE"], logNSE_low=s_low["logNSE"],
                          PBIAS_low=s_low["PBIAS"]))

    acf_t = pd.DataFrame(rows)
    print("\n--- Spearman low-flow ACF, lags 1-5  (bold = |rho| > noise band) ---")
    hdr = f"{'model':<14s}{'n':>4s}{'band':>7s}" + "".join(f"{'rho'+str(k):>8s}" for k in range(1, 6)) + f"{'MAE':>8s}"
    print(hdr); print("-"*len(hdr))
    for _, r in acf_t.iterrows():
        cells = ""
        for k in range(1, 6):
            v, p = r[f"rho{k}"], r[f"p{k}"]
            cells += f"{v:>+7.2f}{'*' if p < 0.05 else ' '}"
        mae = "" if pd.isna(r.get("mae", np.nan)) else f"{r['mae']:>8.3f}"
        print(f"{r['model']:<14s}{int(r['n']):>4d}{r['band']:>7.3f}{cells}{mae}")
    print("  * = permutation p < 0.05")

    sk = pd.DataFrame(srows)
    print("\n--- daily streamflow skill (full record / low-flow months only) ---")
    print(sk.round(3).to_string(index=False))
    return acf_t, sk


def main():
    # ---------------- East ----------------
    o = pd.read_csv(EAST/"observations/streamflow/raw_data/"
                    "EastRiver_dailyDischarge_WY1970_2025.csv",
                    parse_dates=["time"]).set_index("time")["discharge"]
    east_obs = o.groupby(o.index.normalize()).mean()
    east_runs = [("baseline", SC/"calibration/bestruns/East_River_baseline_1981_2025/run")]
    for d in sorted(glob.glob(str(EAST_SW/"sf*"))):
        if glob.glob(d+"/output/*.nc"):
            sf = float(re.search(r"sf(\d+)p(\d+)", Path(d).name).expand(r"\1.\2"))
            east_runs.append((f"sf={sf:g}", d))
    ea, es = report("EAST RIVER", east_obs, east_runs, [9, 10, 11], "YE-NOV")
    ea.to_csv(EAST_SW/"acf_lag1to5.csv", index=False)
    es.to_csv(EAST_SW/"skill_scores.csv", index=False)

    # ---------------- Tuolumne ----------------
    # NOTE: use the gauge the calibration used (meta.json cfg.obs_csv). The other file
    # in this directory, Tuolumne_River_lumped_streamflow_processed.csv, is a DIFFERENT
    # (downstream, ~2x) gauge -- ratio p05-p95 = 0.30-0.75, so not a unit conversion.
    t = pd.read_csv(TUO/"observations/streamflow/preprocessed/"
                    "Tuolumne_River_merged_dailyQ_WY1971_2025.csv",
                    parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
    tuo_obs = t.groupby(t.index).mean()
    tuo_runs = [("baseline", SC/"calibration/bestruns/Tuolumne_River_baseline_wbias10_1981_2025/run")]
    for d in sorted(glob.glob(str(TUO_SW/"m*"))):
        if glob.glob(d+"/output/*.nc"):
            tuo_runs.append((Path(d).name, d))
    ta, ts = report("TUOLUMNE RIVER", tuo_obs, tuo_runs, [8, 9, 10], "YS-OCT")
    ta.to_csv(TUO_SW/"acf_lag1to5.csv", index=False)
    ts.to_csv(TUO_SW/"skill_scores.csv", index=False)


if __name__ == "__main__":
    main()
