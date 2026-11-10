"""
Below-snow emulator for Tuolumne (2026-11-03): SWI (rain + melt reaching the soil) per HRU -> routed streamflow, fitted to SUMMA's own response
(g35_suite3_wb.py output) so emulated snow can be scored against observed flow in seconds.

Per HRU, daily, upslope HRUs first (SUMMA's downHRUindex chain, qTopmodl-style lateral flow):
  S   += SWI + lateral inflow from upslope (volume / own area)
  ET   = pet * min(1, S / (f_et smax)),  pet = c_pet * 0.0353 mm/d per W/m2 * SW * (T > 0), zero while the HRU has snow
  out  = (S - fc smax)/tau  above field capacity (tau_lo below the split elevation, tau_hi above), plus overflow above smax
  out goes to the downslope HRU, or to the stream at outlet HRUs; basin outflow is convolved with a gamma unit hydrograph.
Hold-out: the same soil with a different albedo (wb_ref), i.e. shifted melt, which is exactly what the snow search will do to it.
"""
import numpy as np, pandas as pd, xarray as xr
from numba import njit
from scipy.stats import gamma
import tuo_snow_emulator as SE

PARAMS = [("smax", 50.0, 600.0, True), ("fc", 0.05, 0.8, False), ("f_et", 0.1, 1.0, False), ("c_pet", 0.05, 1.0, False),
          ("tau_lo", 0.5, 60.0, True), ("tau_hi", 0.5, 300.0, True), ("uh_k", 0.5, 4.0, False), ("uh_theta", 0.1, 5.0, True)]


def topology():
    a = xr.open_dataset(SE.ATTR); down = a.downHRUindex.values.astype(int) - 1; a.close()        # 0-based index of the downslope HRU, -1 = outlet
    order = []
    def visit(i, seen):
        if i in seen: return
        for j in np.where(down == i)[0]: visit(j, seen)
        seen.add(i); order.append(i)
    seen = set()
    for i in range(len(down)): visit(i, seen)
    return down, np.array(order)


@njit(cache=True)
def _run(swi, snowy, SW, T, area, down, order, hi, smax, fc, f_et, c_pet, tau_lo, tau_hi, et_out, q_out):
    nt, nh = swi.shape
    S = np.full(nh, fc*smax)
    for i in range(nt):
        lat = np.zeros(nh)
        for o in range(nh):
            h = order[o]
            s = S[h] + swi[i, h] + lat[h]
            pet = 0.0 if snowy[i, h] or T[i, h] <= 0.0 else c_pet*0.0353*SW[i, h]
            et = min(pet*min(1.0, s/(f_et*smax)), s); s -= et
            tau = tau_hi if hi[h] else tau_lo
            out = max(s - fc*smax, 0.0)*min(1.0/tau, 1.0); s -= out
            if s > smax: out += s - smax; s = smax
            S[h] = s; et_out[i, h] = et
            if down[h] >= 0: lat[down[h]] += out*area[h]/area[down[h]]
            else: q_out[i] += out*area[h]


def uh(k, theta, n=60):
    x = np.arange(n + 1, dtype=float); w = np.diff(gamma.cdf(x, k, scale=theta)); return w/w.sum()


def simulate(p, swi, snowy, SW, T, H, split_m, topo=None):
    down, order = topo or topology(); area = H.area.values; nt, nh = swi.shape
    et, q = np.zeros((nt, nh)), np.zeros(nt)
    _run(swi.values, snowy.values, SW.values, T.values, area, down, order, H.elev.values >= split_m,
         *(float(p[k]) for k in ("smax", "fc", "f_et", "c_pet", "tau_lo", "tau_hi")), et, q)
    qb = np.convolve(q/area.sum(), uh(p["uh_k"], p["uh_theta"]))[:nt]
    return pd.Series(qb, index=swi.index), pd.DataFrame(et, index=swi.index, columns=swi.columns)


def params_of(u):
    return {k: float(10**(np.log10(lo) + v*(np.log10(hi) - np.log10(lo))) if lg else lo + v*(hi - lo)) for (k, lo, hi, lg), v in zip(PARAMS, u)}
