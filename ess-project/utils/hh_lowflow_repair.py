"""Can HH's Aug-Oct low flows be repaired by regression on a better short record?

Two candidate predictors/targets on the overlap:
  (a) the four nested headwater gauges  (WY2005-2023, tiny basins)
  (b) the real basin gauge that the merged record splices in from WY2007  <- much better

Tests, in order:
  1. Aug-Oct DAILY correlation, HH vs each candidate (log space).
  2. Is HH's low-flow *ranking* informative at all, or is it noise?
  3. Fit a power-law/quantile map on the overlap, apply to WY1970-2006, and see what the
     long-record 7-day-min ACF becomes.
  4. Cross-validate: does the map hold out of sample?
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, linregress

SC = Path("/scratch/dlhogan/ess-project-data")
TUO = SC/"domain_Tuolumne_River_distributed_elevAspect"
SD = SC/"streamflow-data/TuolumneRiver"
CFS = 35.3146667
NLAG = 5


def compute_acf(series, n_lags=NLAG, n_permutations=10000, seed=0):
    def _sp(a, b):
        ra = rankdata(a)-(len(a)+1)/2.; rb = rankdata(b)-(len(b)+1)/2.
        den = np.sqrt((ra*ra).sum()*(rb*rb).sum())
        return (ra*rb).sum()/den if den > 0 else 0.0
    rng = np.random.default_rng(seed)
    s = series.dropna().astype(float)
    yrs = s.index.values.astype(int); v = s.values; n = v.size
    pos = {int(y): i for i, y in enumerate(yrs)}
    rho = np.full(n_lags, np.nan); p = np.full(n_lags, np.nan)
    for k in range(1, n_lags+1):
        ip = np.array([pos[int(y)-k] for y in yrs if int(y)-k in pos], int)
        ic = np.array([i for i, y in enumerate(yrs) if int(y)-k in pos], int)
        if ip.size <= 3: continue
        rho[k-1] = _sp(v[ip], v[ic])
        pm = np.array([abs(_sp(r[ip], r[ic])) for r in
                       (rng.permutation(v) for _ in range(n_permutations))])
        p[k-1] = np.mean(pm >= abs(rho[k-1]))
    return dict(rho=rho, p=p, band=1.96/np.sqrt(n), n=n)


def fmt(a):
    return " ".join(f"{r:+.2f}{'*' if (p == p and p < 0.05) else ' '}"
                    for r, p in zip(a["rho"], a["p"]))


def min7(s, y0=1970, y1=2024):
    s = s[s > 0]
    x = (s.where(s.index.month.isin([8, 9, 10]))
          .rolling(7, center=True, min_periods=7).mean().resample("YS-OCT").min())
    x.index = x.index.year
    return x[(x.index >= y0) & (x.index <= y1)]


def load_hh_raw():
    d = pd.read_csv(SD/"TuolumneRiver_HH_unimpaired_19702024.csv")
    d = d[d.DATE.astype(str).str.match(r"^\d{1,2}/\d{1,2}$", na=False)]
    out = {}
    for c in [c for c in d.columns if c.startswith("WY ")]:
        wy = int(c.split()[1])
        v = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
        for ds, val in zip(d.DATE, v):
            m, day = (int(x) for x in ds.split("/"))
            try: out[pd.Timestamp(wy-1 if m >= 10 else wy, m, day)] = val
            except ValueError: pass
    return (pd.Series(out).sort_index().dropna())/CFS


hh = load_hh_raw()
merged = pd.read_csv(TUO/"observations/streamflow/preprocessed/"
                     "Tuolumne_River_merged_dailyQ_WY1971_2025.csv",
                     parse_dates=["datetime"]).set_index("datetime")
gauge = merged.loc[merged.source == "gauge", "discharge_cms"]
lump = pd.read_csv(TUO/"observations/streamflow/preprocessed/"
                   "Tuolumne_River_lumped_streamflow_processed.csv",
                   parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
nd = pd.read_csv(SD/"TuolumneStreamflow_Hydroclimate_2002_2021.csv",
                 parse_dates=["Date"]).set_index("Date")
nd.columns = [c.replace("Estimated_Discharge_", "") for c in nd.columns]

print("="*100)
print("0. WHERE THE ORIGINAL NUMBERS CAME FROM  (7-day min Aug-Oct, full record)")
print("="*100)
for tag, s, note in [
        ("regulated downstream", lump, "<- the ORIGINAL notebook series (lumped_processed)"),
        ("merged (HH<=2006 + gauge)", merged["discharge_cms"], "<- the inflated one (2.6x splice)"),
        ("HH unimpaired, whole record", hh, "")]:
    a = compute_acf(min7(s))
    print(f"  {tag:<28s} n={a['n']:2d}  {fmt(a)}   {note}")

print("\n" + "="*100)
print("1. AUG-OCT DAILY CORRELATION: HH vs each candidate  (overlap only)")
print("="*100)
cands = {g: nd[g].dropna() for g in nd.columns}
cands["basin gauge (WY2007+)"] = gauge
print(f"  {'candidate':<24s}{'n days':>8s}{'Spearman':>10s}{'Pearson(log)':>14s}{'log-log slope':>15s}")
print("  " + "-"*71)
for k, v in cands.items():
    j = pd.DataFrame({"hh": hh, "y": v}).dropna()
    j = j[(j.index.month.isin([8, 9, 10])) & (j.hh > 0) & (j.y > 0)]
    if len(j) < 50: continue
    sp = spearmanr(j.hh, j.y).statistic
    lr = linregress(np.log(j.hh), np.log(j.y))
    print(f"  {k:<24s}{len(j):>8d}{sp:>10.2f}{lr.rvalue:>14.2f}{lr.slope:>15.2f}")

print("\n" + "="*100)
print("2. IS HH's LOW-FLOW RANKING INFORMATIVE?  (annual 7-day min, overlap years)")
print("="*100)
m_hh = min7(hh)
targets = {"basin gauge": min7(gauge), **{f"nested {g}": min7(nd[g]) for g in nd.columns}}
for k, v in targets.items():
    j = pd.DataFrame({"hh": m_hh, "y": v}).dropna()
    if len(j) < 5: continue
    sp = spearmanr(j.hh, j.y)
    lr = linregress(np.log(j.hh), np.log(j.y))
    print(f"  HH vs {k:<22s} n={len(j):2d}  Spearman={sp.statistic:+.2f} (p={sp.pvalue:.3f})   "
          f"log-log slope={lr.slope:.2f}  r2={lr.rvalue**2:.2f}")

print("\n" + "="*100)
print("3. THE CORE PROBLEM: HH's low-flow dynamic range is compressed")
print("="*100)
ov = m_hh.index.intersection(min7(gauge).index)
print(f"  overlap WY{ov.min()}-{ov.max()}  (n={len(ov)})")
for k, v in [("HH unimpaired", m_hh), ("basin gauge", min7(gauge))]:
    s = v.reindex(ov).dropna()
    print(f"    {k:<16s} min={s.min():.3f} max={s.max():.3f}  max/min={s.max()/s.min():.1f}  "
          f"CV={s.std()/s.mean():.2f}  log-sd={np.log(s).std():.2f}")
j = pd.DataFrame({"hh": m_hh, "g": min7(gauge)}).dropna()
lr = linregress(np.log(j.hh), np.log(j.g))
print(f"\n  log-log fit  ln(gauge) = {lr.slope:.2f}*ln(HH) + {lr.intercept:.2f}   "
      f"r2={lr.rvalue**2:.2f}  p={lr.pvalue:.4f}")
print(f"  slope > 1 would mean HH compresses the range and a power law can re-expand it.")

print("\n" + "="*100)
print("4. APPLY THE MAP TO WY1970-2006 AND RE-DO THE LONG-RECORD ACF")
print("="*100)
hh_all = min7(hh)
mapped = np.exp(lr.intercept + lr.slope*np.log(hh_all))
gm = min7(gauge)
hyb = mapped.copy()
hyb.loc[hyb.index.intersection(gm.index)] = gm.reindex(hyb.index).dropna()

# non-parametric quantile map, for comparison
qm = pd.Series(np.quantile(j.g, rankdata(hh_all)/(len(hh_all)+1)), index=hh_all.index)

for tag, s in [("HH raw", hh_all),
               ("HH x constant K=0.5051", hh_all*0.5051),
               ("HH power-law mapped", mapped),
               ("HH quantile-mapped", qm),
               ("hybrid: mapped + real gauge", hyb)]:
    a = compute_acf(s.dropna())
    print(f"  {tag:<30s} n={a['n']:2d}  {fmt(a)}")
print("\n  ^ note rows 1-4: every MONOTONE map gives the IDENTICAL rank ACF.")
print("    A bias correction can fix the level; it cannot change which year outranks which.")
print(f"    Spearman(raw, power-law)={spearmanr(hh_all, mapped).statistic:.3f}  "
      f"Spearman(raw, quantile)={spearmanr(hh_all, qm).statistic:.3f}")
print(f"\n  hybrid splice check: mapped median WY<=2006 = "
      f"{mapped[mapped.index <= 2006].median():.3f}, real gauge median WY>=2007 = "
      f"{gm.median():.3f}  -> step {mapped[mapped.index <= 2006].median()/gm.median():.2f}x"
      f"   (constant-K version was 2.6x)")

print("\n" + "="*100)
print("5. LEAVE-ONE-OUT CHECK ON THE MAP")
print("="*100)
err = []
for y in j.index:
    tr = j.drop(y)
    f = linregress(np.log(tr.hh), np.log(tr.g))
    pred = np.exp(f.intercept + f.slope*np.log(j.loc[y, "hh"]))
    err.append((y, j.loc[y, "g"], pred))
e = pd.DataFrame(err, columns=["wy", "obs", "pred"]).set_index("wy")
print(f"  LOO n={len(e)}  Spearman(pred,obs)={spearmanr(e.pred, e.obs).statistic:+.2f}  "
      f"median |log ratio|={np.median(np.abs(np.log(e.pred/e.obs))):.2f}")
print(f"  constant-K baseline: median |log ratio|="
      f"{np.median(np.abs(np.log((j.hh*0.5051)/j.g))):.2f}")
print(e.round(3).to_string())
