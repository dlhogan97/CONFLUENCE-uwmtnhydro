"""Trend-aware low-flow memory summary, WY1986-2023 (the span the models cover).

Policy, applied identically to every series so no cherry-picking is possible:
  * test the monotonic trend (Spearman on rank, Mann-Kendall as a cross-check)
  * remove it ONLY if p < 0.05, and then as a Theil-Sen slope in LOG space
    (multiplicative, robust to outliers, cannot drive a positive series negative)
  * recompute the lag-1..5 Spearman ACF on whatever comes out

Both the as-is and the adjusted ACF are reported, so the reader sees what the detrending
did rather than being handed the adjusted number alone.

Series: Aug-Oct (Tuolumne) / Sep-Nov (East) 7-day-minimum low flow for observations, the
baseline run and the store-added run; plus end-of-summer snowpatch area, observed (Landsat)
and modelled (the anchor_m080 depth bins).
"""
import warnings; warnings.filterwarnings("ignore")
import json, io, contextlib, glob
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from scipy.stats import spearmanr, kendalltau, theilslopes
import matplotlib; matplotlib.use("Agg")

SCRATCH = Path("/scratch/dlhogan/ess-project-data")
REPO = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro")
NB = REPO/"ess-project/notebooks/exploration/result_viz_sandbox.ipynb"
WY0, WY1 = 1981, 2024                      # default: the full span the model runs cover.
                                           # Override with --wy0/--wy1; every output file is
                                           # tagged with the window so two windows cannot be
                                           # confused for each other.
P_TREND = 0.05                             # remove a trend only at this significance
A_HRU2 = 8.611236                          # km2, the parent snowpatch HRU
SWE_TH = 0.05                              # m; a bin counts as covered above this


# ---------------- series ----------------
# cells 9af69ffa and adj0001 define the low-flow series we want, but also run ten
# compute_acf calls at 5000 permutations each in pure Python -- ~60 s, and pure side effect
# for any caller that only wants the series. `fast` swaps in a stub for the duration.
_ACF_CELLS = ("9af69ffa", "adj0001")


def _acf_stub(series, n_lags=5, **kw):
    n = len(getattr(series, "dropna", lambda: series)())
    z = np.full(n_lags, np.nan)
    return dict(lags=np.arange(1, n_lags+1), rho=z, p=np.ones(n_lags),
                band=np.nan, band_perm=z, n=n)


def notebook_ns(fast=False):
    """Namespace after running the result_viz_sandbox cells that build the daily and
    annual low-flow series. `fast=True` stubs out compute_acf, which is ~60 s of
    permutation testing that only matters if you actually read the ACF results."""
    nb = json.load(open(NB))
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cid in ["3ee09baa", "e8e9e5e2", "142d63de", "3b37d977", "197d7372",
                    "bb2569fe", "23695d74", "9af69ffa", "adj0001"]:
            if fast and cid in _ACF_CELLS:
                real = ns.get("compute_acf")
                ns["compute_acf"] = _acf_stub
                exec(compile(cells[cid], cid, "exec"), ns)
                ns["compute_acf"] = real          # restore for anyone who needs the real one
            else:
                exec(compile(cells[cid], cid, "exec"), ns)
    return ns


def modelled_patch_area(run, bin_spec=None):
    """End-of-summer (30 Sep) snow-covered area of HRU 2, km2, per water year.

    With bins: area-weighted count of bins holding >SWE_TH. Without them (the baseline) the
    HRU is binary, so the series can only ever be 0 or the full 8.61 km2 -- which is exactly
    the limitation the binning exists to remove, so it is worth plotting."""
    cache = Path(run)/f"eos_patch_area_swe{SWE_TH:g}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0).iloc[:, 0]
    cand = sorted(glob.glob(str(Path(run)/"output/*.nc"))) + sorted(glob.glob(str(Path(run)/"out/*.nc")))
    nc = cand[0]                                   # bestruns use out/, sandbox runs use output/
    ds = xr.open_dataset(nc)
    hid = np.asarray(ds["hruId"].values).ravel()
    t = pd.DatetimeIndex(ds["time"].values)
    if bin_spec is not None:
        b = pd.read_csv(bin_spec)
        ids, w = b["hruId"].values, b["area_frac"].values
    else:
        ids, w = np.array([2]), np.array([1.0])
    swe = np.column_stack([np.asarray(ds["scalarSWE"].isel(
        hru=int(np.where(hid == i)[0][0])).values).ravel()/1000.0 for i in ids])
    ds.close()
    df = pd.DataFrame(swe, index=t)
    sep = df[(t.month == 9) & (t.day == 30)]
    sep = sep.groupby(sep.index.year).last()               # hourly output -> 24 rows per day
    out = pd.Series(((sep.values > SWE_TH) * w).sum(axis=1)*A_HRU2, index=sep.index,
                    name="patch_area_km2")
    out.index.name = "water_year"
    try:
        out.to_csv(cache)
    except OSError:
        pass                                       # read-only run dir: just skip the cache
    return out


# ---------------- statistics ----------------
def trend_stats(s):
    s = s.dropna(); x = s.index.values.astype(float); v = s.values.astype(float)
    rho, p = spearmanr(x, v)
    tau, p_mk = kendalltau(x, v)
    pos = v.min() > 0
    sl, ic, lo, hi = theilslopes(np.log(v) if pos else v, x)
    pct = 100*(np.exp(sl)-1) if pos else np.nan
    return dict(n=len(s), rho=rho, p=p, tau=tau, p_mk=p_mk, slope=sl, icept=ic,
                pct=pct, lo=100*(np.exp(lo)-1) if pos else np.nan,
                hi=100*(np.exp(hi)-1) if pos else np.nan, pos=pos)


def detrend(s, tr):
    """Divide out the fitted multiplicative trend, rescaled to preserve the series mean."""
    x = s.index.values.astype(float)
    fit = np.exp(tr["slope"]*x + tr["icept"])
    out = s / fit
    return out * (s.mean()/out.mean())


def main(wy0=WY0, wy1=WY1):
    ns = notebook_ns()
    yr_low, compute_acf = ns["yr_low"], ns["compute_acf"]
    clip = lambda s: s.dropna().loc[wy0:wy1]

    eos = pd.read_csv(SCRATCH/"geospatial-data/snowpatch_fsca/eos_final.csv").set_index("year")
    SNOW = SCRATCH/"domain_Tuolumne_River_distributed_elevAspect/simulations"
    SERIES = [
        ("East", "low flow", "observed",   clip(yr_low(ns["east_obs_augoct_low_flow"]))),
        ("East", "low flow", "baseline",   clip(yr_low(ns["east_mod_augoct_low_flow"]))),
        ("East", "low flow", "deep store", clip(yr_low(ns["east_adj_augoct_low_flow"]))),
        ("Tuolumne", "low flow", "observed",  clip(yr_low(ns["tuolumne_obs_augoct_low_flow"]))),
        ("Tuolumne", "low flow", "baseline",  clip(yr_low(ns["tuolumne_mod_augoct_low_flow"]))),
        ("Tuolumne", "low flow", "snowpatch", clip(yr_low(ns["tuolumne_adj_augoct_low_flow"]))),
        ("Tuolumne", "patch area", "observed", clip(eos["A_shadow"])),
        ("Tuolumne", "patch area", "baseline",
         clip(modelled_patch_area(SCRATCH/"calibration/bestruns/"
                                  "Tuolumne_River_baseline_wbias10_1981_2025/run"))),
        ("Tuolumne", "patch area", "snowpatch",
         clip(modelled_patch_area(SNOW/"snowpatch_anchor_20260812/anchor_m080",
                                  SNOW/"snowpatch_anchor_20260812/anchor_m080/bin_spec.csv"))),
    ]

    rows, keep = [], {}
    print(f"=== TRENDS, WY{wy0}-{wy1} "
          f"(detrended only where Spearman p < {P_TREND}) ===\n")
    print(f"{'basin':9s} {'variable':11s} {'series':10s} {'n':>3s} {'rho':>7s} {'p':>7s} "
          f"{'MK tau':>7s} {'%/yr':>8s} {'95% CI':>18s}  detrended?")
    for basin, var, lab, s in SERIES:
        if len(s) < 8:
            print(f"{basin:9s} {var:11s} {lab:10s} n={len(s)} -- too short"); continue
        tr = trend_stats(s)
        sig = (tr["p"] < P_TREND) and tr["pos"]
        keep[(basin, var, lab)] = (s, detrend(s, tr) if sig else s, tr, sig)
        ci = (f"[{tr['lo']:+.2f},{tr['hi']:+.2f}]" if tr["pos"] else "--")
        print(f"{basin:9s} {var:11s} {lab:10s} {tr['n']:3d} {tr['rho']:+7.2f} {tr['p']:7.3f} "
              f"{tr['tau']:+7.2f} {tr['pct']:+8.2f} {ci:>18s}  "
              f"{'YES' if sig else 'no'}")
        rows.append(dict(basin=basin, variable=var, series=lab, **{k: tr[k] for k in
                    ("n", "rho", "p", "tau", "p_mk", "pct", "lo", "hi")}, detrended=sig))

    print(f"\n=== LOW-FLOW LAG AUTOCORRELATION, WY{wy0}-{wy1} "
          "(Spearman, year-aligned; * = p<0.10, ** = p<0.05) ===\n")
    print(f"{'basin':9s} {'series':10s} {'form':11s} " + "".join(f"{'lag'+str(k):>9s}" for k in range(1, 6)))
    acf_rows = []
    for (basin, var, lab), (raw, adj, tr, sig) in keep.items():
        if var != "low flow":
            continue
        for form, s in (("as-is", raw), ("trend-adj." if sig else "(unchanged)", adj)):
            a = compute_acf(s, n_lags=5, n_permutations=5000)
            stars = ["**" if p < 0.05 else "* " if p < 0.10 else "  " for p in a["p"]]
            print(f"{basin:9s} {lab:10s} {form:11s} " +
                  "".join(f"{r:+7.2f}{st}" for r, st in zip(a["rho"], stars)))
            acf_rows.append(dict(basin=basin, series=lab, form=form, n=a["n"],
                                 **{f"rho{k+1}": a["rho"][k] for k in range(5)},
                                 **{f"p{k+1}": a["p"][k] for k in range(5)}))
        if sig:
            print(f"{'':9s} {'':10s} {'':11s}   ^ trend removed ({tr['pct']:+.2f}%/yr, p={tr['p']:.3f})")

    out = REPO/"ess-project/figures/draft"
    tag = f"wy{wy0}_{wy1}"
    pd.DataFrame(rows).to_csv(out/f"table_trends_{tag}.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(out/f"table_lowflow_acf_trendadj_{tag}.csv", index=False)
    print(f"\nwrote {out/f'table_trends_{tag}.csv'}")
    print(f"wrote {out/f'table_lowflow_acf_trendadj_{tag}.csv'}")
    return keep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wy0", type=int, default=WY0)
    ap.add_argument("--wy1", type=int, default=WY1)
    a = ap.parse_args()
    main(a.wy0, a.wy1)
