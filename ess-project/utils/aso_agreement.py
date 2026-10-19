"""ASO vs Landsat band-8 fSCA agreement over all 38 matched spring/summer flights.

The published figure used aso_landsat_pairs_all.csv (n=3, late-season only).
This uses aso_landsat_pairs_spring.csv (n=38, Mar-Aug, |lag| <= 6 d).
"""
import numpy as np, pandas as pd
from pathlib import Path
D = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
p = pd.read_csv(D/"aso_landsat_pairs_spring.csv", parse_dates=["aso_date","ls_date"])

def stats(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    k = (x*y).sum()/(x*x).sum()                       # slope through origin
    sl, ic = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return dict(n=len(x), k0=k, slope=sl, icept=ic, r=r,
                rmse=float(np.sqrt(((y-x)**2).mean())),
                bias=float((y-x).mean()), mad=float(np.abs(y-x).mean()),
                xbar=float(x.mean()), ybar=float(y.mean()))

print("=== band 8: Landsat (y) vs ASO (x), n=38 ===")
rows = {}
for c, lab in [("aso_b8_t001","SWE>0.01"), ("aso_b8_t025","SWE>0.25"), ("aso_b8_t05","SWE>0.50")]:
    rows[lab] = stats(p[c], p.ls_b8)
t = pd.DataFrame(rows).T
print(t.round(3).to_string())

print("\n=== band 7 (SWE>0.25) ===")
print(pd.Series(stats(p.aso_b7_t025, p.ls_b7)).round(3).to_string())

print("\n=== by season half (band 8, SWE>0.25) ===")
for lab, sub in [("Mar-May", p[p.month <= 5]), ("Jun-Aug", p[p.month >= 6])]:
    s = stats(sub.aso_b8_t025, sub.ls_b8)
    print(f"  {lab}: n={s['n']:2d}  slope0={s['k0']:.2f}  r={s['r']:+.3f}  "
          f"bias={s['bias']:+.3f}  ASO mean {s['xbar']:.3f}  LS mean {s['ybar']:.3f}")

print("\n=== which threshold minimises |bias| and maximises r? ===")
best_r = t.r.idxmax(); best_b = t.bias.abs().idxmin()
print(f"  highest r      : {best_r}  (r={t.r.max():+.3f})")
print(f"  smallest |bias|: {best_b}  (bias={t.loc[best_b,'bias']:+.3f})")
print(f"\nmonths {sorted(p.month.unique())}, lag {p.lag.min():+d} to {p.lag.max():+d} d, "
      f"cloud {p.cloud.min():.0f}-{p.cloud.max():.0f}%")
print(f"ASO flight dates span {p.aso_date.min().date()} to {p.aso_date.max().date()}")
