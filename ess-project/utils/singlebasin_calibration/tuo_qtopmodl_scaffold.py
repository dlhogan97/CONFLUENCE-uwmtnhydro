"""Back-of-envelope scaffolding for Tuolumne's qTopmodl subsurface, straight from groundwatr.f90:
   tran0 = kAnisotropic*K*D/zScale ;  Q_out = tran0*(z/D)^zScale * tan_slope * contourLength ;  z = drainable saturated thickness
=> per unit area  q = c_h*(z/D)^b with b = zScale,  c_h = kAniso*K*(D/zScale)*tan_slope*(L/A)   [m/s]
A single store q = c*S^b recedes as -dQ/dt = a*Q^p with p = 2 - 1/b  (Brutsaert-Nieber exponent).  Compare with the OBSERVED recession exponent and test what the observed Aug-Oct decline requires."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc
B = "/scratch/dlhogan/ess-project-data/calibration/bestruns/Tuolumne_River_baseline_wbias10_1981_2025/run/"
OBS = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/preprocessed/Tuolumne_River_merged_dailyQ_WY1971_2025.csv"
tp = nc.Dataset(B+"trialParams.nc"); g = lambda k: float(np.unique(np.array(tp[k][:]))[0])
K, kA, zS, th, fc = g("k_soil"), g("kAnisotropic"), g("zScale_TOPMODEL"), g("theta_sat"), g("fieldCapacity")
a = xr.open_dataset(B+"attributes.nc"); hid = a.hruId.values.astype(int); A = a.HRUarea.values.ravel(); L = a.contourLength.values.ravel(); tb = a.tan_slope.values.ravel(); dn = a.downHRUindex.values.astype(int)
D = np.array(nc.Dataset(B+"coldState.nc")["mLayerDepth"][:]).sum(0)
print(f"parameters: K={K:.2e} m/s, kAnisotropic={kA:.2f}, zScale(exponent)={zS:.2f}, theta_sat={th}, fieldCap={fc}  -> active porosity {th-fc:.2f}")
c_h = kA*K*(D/zS)*tb*(L/A); qsat_mmd = c_h*86400*1000
tab = pd.DataFrame({"hru": hid, "D_m": D.round(2), "tanb": tb.round(2), "L/A_1e-4": (1e4*L/A).round(1), "q_sat_mm_d": qsat_mmd.round(1), "area_km2": (A/1e6).round(1), "terminal": dn == 0})
tab["Q_sat_m3s"] = (c_h*A).round(2); tab["tau_sat_d"] = ((th-fc)*D/(zS*c_h)/86400).round(1)    # S/(b q) at saturation, S = active porosity*D
print(tab.to_string(index=False))
up = {h: A[hid == h][0] for h in hid}; tot = {}
for h in hid:
    x = h; 
    while dn[hid == x][0] > 0: x = dn[hid == x][0]
    tot[x] = tot.get(x, 0)+up[h]
print("\nterminal HRUs: own area vs area they must drain (km2) | capacity at full saturation (m3/s) | demand at mean runoff 574 mm/yr (m3/s)")
for x, s in tot.items():
    i = np.where(hid == x)[0][0]; print(f"  HRU {x:2d}: own {A[i]/1e6:5.1f}  drains {s/1e6:6.1f}  (x{s/A[i]:.0f})  capacity {c_h[i]*A[i]:5.2f}  demand {0.574/3.156e7*s:5.2f}  -> mean saturation (z/D)={(0.574/3.156e7*s/(c_h[i]*A[i]))**(1/zS):.2f}")
# ---- observed vs modelled recession exponent (Brutsaert-Nieber, Aug-Oct, gauge era) ----
def bn(q, months=(8, 9, 10), y0=2008):
    q = q[q.index.year >= y0]; mmd = q*86400/(774.5e6)*1000; dq = -(mmd.shift(-1)-mmd.shift(1))/2; pos = dq > 0
    run = (pos & pos.shift(1, fill_value=False) & pos.shift(-1, fill_value=False)) | (pos & pos.shift(1, fill_value=False) & pos.shift(2, fill_value=False)) | (pos & pos.shift(-1, fill_value=False) & pos.shift(-2, fill_value=False))
    m = run & q.index.month.isin(months) & (mmd > 0.02) & (dq > 0); x, y = np.log(mmd[m]), np.log(dq[m]); p, lna = np.polyfit(x, y, 1)
    return p, np.exp(lna), int(m.sum()), mmd[m].median()
o = pd.read_csv(OBS, parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
d = xr.open_dataset(B+"out/baseline_timestep.nc"); t = pd.DatetimeIndex(d.time.values); sim = (pd.Series(d["averageRoutedRunoff"].isel(gru=0).values, index=t)*A.sum()).resample("D").mean()
for nm, q in (("observed (gauge, WY2008-25)", o), ("model control", sim)):
    for mo, lab in (((8, 9, 10), "Aug-Oct"), ((6, 7, 8, 9, 10), "Jun-Oct")):
        p, aa, n, qm = bn(q, mo); print(f"{nm:28s} {lab:8s}: recession exponent p={p:.2f}  (n={n}, median Q={qm:.2f} mm/d)  => equivalent store exponent b=1/(2-p)={1/(2-p) if p<2 else np.inf:.2f}")
print(f"structural expectation for zScale={zS:.2f}: p = 2-1/b = {2-1/zS:.2f}")
# ---- what does the observed Aug1 -> Oct1 decline require? ----
def clim(q, md):
    x = [q.get(pd.Timestamp(y, *md)) for y in range(2008, 2025)]; return np.nanmedian([v for v in x if v is not None])
for nm, q in (("observed", o), ("model", sim)):
    qa = np.nanmedian([q[f"{y}-07-29":f"{y}-08-02"].mean() for y in range(2008, 2025)]); qo = np.nanmedian([q[f"{y}-09-28":f"{y}-10-02"].mean() for y in range(2008, 2025)])
    print(f"{nm:9s}: median flow around Aug 1 = {qa:5.2f}, around Oct 1 = {qo:5.2f} m3/s  -> ratio {qo/qa:.2f}")
qa_m = np.nanmedian([sim[f"{y}-07-29":f"{y}-08-02"].mean() for y in range(2008, 2025)]); pm, am, _, _ = bn(sim, (8, 9, 10)); r0 = am*(qa_m*86400/774.5e6*1000)**pm      # model's mm/d^2 decline rate at its Aug-1 flow
print("\nlumped recession from the MODEL's Aug-1 flow, holding the Aug-1 decline RATE fixed, varying the store exponent b (p=2-1/b):")
for b in (1.0, 1.1, 1.25, 1.5, 1.78, 2.5):
    p = 2-1/b; Q = qa_m*86400/774.5e6*1000; a_ = r0/Q**p; 
    for _ in range(61): Q += -a_*Q**p
    print(f"  b={b:4.2f} (p={p:.2f}): Oct-1 / Aug-1 flow ratio = {Q/(qa_m*86400/774.5e6*1000):.2f}   -> Oct-1 flow {Q*774.5e6/86400/1000:.2f} m3/s")
