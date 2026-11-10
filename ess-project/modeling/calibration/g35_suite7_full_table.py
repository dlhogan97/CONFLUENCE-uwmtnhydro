"""
Evaluate the full-record runs (g35_suite7_full.py, control and patch) against observed flow (2026-11-09).
Observed: the runner's merged daily record (S.CFG obs_csv, WY1971-2025). CAUTION: this merged record has splices/unit steps (memory
tuolumne-lowflow-memory-is-splice-artifact); high-flow scores are fine, low-flow / log scores are suspect across the joins.
Periods: WY1983-2012 (before calibration, out of sample), WY2013-2020 (calibration), WY2021-2025 (after, out of sample), WY1983-2025.
Per period and run: daily KGE (r, alpha, beta), Apr-Jul KGE, log-flow KGE, monthly volume bias (%), Aug-Sep and Aug-Oct bias, annual runoff.
Patch run is read from cal_day_28.nc (bins folded into HRU 2), so routed flow is identical in meaning.
-> figures/draft/tuo_suite7_full.csv
usage: python g35_suite7_full_table.py
"""
import numpy as np, pandas as pd, netCDF4 as nc
import g35_suite7_full as G
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
PER = {"WY1983-2012": ("1982-10-01", "2012-09-30"), "WY2013-2020 (cal)": ("2012-10-01", "2020-09-30"), "WY2021-2025": ("2020-10-01", "2025-09-30"),
       "WY1983-2025": ("1982-10-01", "2025-09-30")}
MON = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]; ORDER = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def sim(name):
    f = G.ROOT/name/("cal_day_28.nc" if name.startswith("patch") else "cal_day.nc")
    d = nc.Dataset(f); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).ravel()*86400e3, index=t); d.close(); return q


def kge(s, o):
    r = np.corrcoef(s, o)[0, 1]; a = s.std()/o.std(); b = s.mean()/o.mean(); return 1 - np.sqrt((r - 1)**2 + (a - 1)**2 + (b - 1)**2), r, a, b


def main():
    runs = {n: sim(n) for n in ("control", "patch", "control_wired", "patch_wired") if (G.ROOT/n/"meta.json").exists()}
    obs = obs_mm_full(); rows, mons = [], []
    for pn, (a0, a1) in PER.items():
        for n, q in runs.items():
            j = pd.concat([q.rename("s"), obs.rename("o")], axis=1)[a0:a1].dropna(); ny = len(j)/365.25; m = j.index.month
            k, r, al, be = kge(j.s.values, j.o.values); aj = j[m.isin([4, 5, 6, 7])]
            row = dict(period=pn, run=n, years=round(ny, 1), KGE=k, r=r, alpha=al, beta=be, KGE_AprJul=kge(aj.s.values, aj.o.values)[0],
                       KGE_log=kge(np.log(j.s.values + 0.01), np.log(j.o.values + 0.01))[0],
                       AugSep_bias=100*(j.s[m.isin([8, 9])].sum()/j.o[m.isin([8, 9])].sum() - 1), AugOct_bias=100*(j.s[m.isin([8, 9, 10])].sum()/j.o[m.isin([8, 9, 10])].sum() - 1),
                       annual_sim=j.s.sum()/ny, annual_obs=j.o.sum()/ny)
            rows.append(row)
            g = j.groupby(m).sum(); mons.append(dict(period=pn, run=n, **{MON[i]: 100*(g.s[mm]/g.o[mm] - 1) for i, mm in enumerate(ORDER)}))
    X = pd.DataFrame(rows).set_index(["period", "run"]); Mb = pd.DataFrame(mons).set_index(["period", "run"]); pd.set_option("display.width", 250)
    print(X.round(3).to_string()); print("\nmonthly volume bias, %"); print(Mb.round(0).to_string())
    pd.concat([X.assign(table="scores"), Mb.assign(table="monthly_bias_pct")]).to_csv(FIG + "tuo_suite7_full.csv"); print("\n", FIG + "tuo_suite7_full.csv")


def obs_mm_full():
    """Observed daily runoff (mm/d) over WY1983-2025, with exactly the calibration's conversion (g35_suite4_cal.obs_mm: runner.obs/area*86400*1000)."""
    import shutil, g35_suite7_cal as C7
    f = G.ROOT/"obs_mm_day_full.csv"
    if not f.exists():
        r = C7.Runner(work_root=str(G.ROOT/"_obs"), **G.CFG); (r.obs/r.area*86400*1000).rename("obs").to_csv(f); shutil.rmtree(G.ROOT/"_obs", ignore_errors=True)
    return pd.read_csv(f, index_col=0, parse_dates=True)["obs"]


if __name__ == "__main__":
    main()
