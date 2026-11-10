"""
Round-4 best (g35_suite4_cal.py) + the anchor_m080 snowpatch (2026-11-04). The user accepted the round-4 best objective as the Tuolumne base; this asks how
much of its Aug-Oct deficit the perennial snowpatch fills.
Snowpatch = utils/build_member_anchor.py design anchor_m080 (snowpatch-bins-drift-design memory), grafted onto the G35 runner:
  - HRU 2 (3679 m, 8.6 km2) split into 10 depth bins (hruIds 2, 29-37) with the design's area fractions; clones keep HRU 2's downHRUindex (7) and are
    placed right after HRU 2 in every file (see clone(): appended bins lose their lateral outflow)
  - each bin's SNOWFALL x drift_a (rain untouched). Snow fraction uses SUMMA's rule (derivforce.f90:295-304: wet-bulb +/- tempRangeTimestep/2 about
    tempCritRain) with a Stull (2011) wet bulb, applied AFTER the runner's precip_scale / lowP_boost / -2 C forcing
  - bin albedoMinNearIR 0.70 and relConvTol_energy 0.05, as in the design (the multipliers were balanced at NIR 0.70; 0.05 avoids the melt-out crash)
  - bins start with their 2011-10-01 SWE from the 44-yr anchor_m080 run (bins under 0.1 m start bare: thin layers crash SUMMA at melt-out)
Output cal_day.nc keeps all 37 HRUs; cal_day_28.nc folds the bins back into HRU 2 (area-weighted) so g35_suite4_cal.score applies unchanged.
-> OUT/suite4_cal/best/best_obj_patch/{cal_day.nc, cal_day_28.nc, score.json}
usage: python g35_suite4_snowpatch.py
"""
import sys, types, json, shutil, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
import g35_suite4_cal as C4
import deepstore_build as DB

DESIGN = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/designs_anchor_20260812/anchor_m080.csv")
ANCHOR_RUN = Path("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_anchor_20260812/anchor_m080/output/anchor_m080_timestep.nc")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"utils"))
_argv = sys.argv; sys.argv = [_argv[0], str(DESIGN), "unused"]                      # build_member_anchor reads its design from argv at import
sys.modules.setdefault("bin_spec", types.SimpleNamespace(A_HRU2=8.611236))           # its bin_spec module is gone; only build() used it
import build_member_anchor as BA
sys.argv = _argv

SRC, NIR_BIN, TOL_BIN, MIN_INIT_M, TRANGE = 2, 0.70, 0.05, 0.1, 2.0
SPEC = pd.read_csv(DESIGN); BINS = list(SPEC.hruId); ADDED = BINS[1:]
AREA = dict(zip(BINS, SPEC.area_frac)); DRIFT = dict(zip(BINS, SPEC.drift_a))
KEEP = C4.OUT/"best"/"best_obj_patch"


def init_swe(ts="2011-10-01"):
    d = nc.Dataset(ANCHOR_RUN); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
    i = t.get_indexer([pd.Timestamp(ts)], method="nearest")[0]; hid = np.array(d["hruId"][:]).ravel(); s = np.array(d["scalarSWE"][i]).ravel()/1000.0; d.close()
    return {int(h): float(m) for h, m in zip(hid, s) if int(h) in BINS and m >= MIN_INIT_M}


def wet_bulb(T, q, p):
    """Stull (2011) wet bulb (K) from air temperature (K), specific humidity (kg/kg), pressure (Pa)."""
    Tc = T - 273.15; e = q*p/(0.622 + 0.378*q); es = 611.2*np.exp(17.67*Tc/(Tc + 243.5)); rh = np.clip(100*e/es, 1.0, 100.0)
    return 273.15 + (Tc*np.arctan(0.151977*np.sqrt(rh + 8.313659)) + np.arctan(Tc + rh) - np.arctan(rh - 1.676331)
                     + 0.00391838*rh**1.5*np.arctan(0.023101*rh) - 4.686035)


def clone(src, dst, area_split=None):
    """BA.clone_hru_along, then move the clones to sit right after HRU 2. SUMMA runs HRUs in file order and zeroes lateral inflow every step
    (run_oneGRU.f90:163-238): a bin placed AFTER its downslope HRU 7 hands its subsurface outflow to a column that has already run, and the water is
    deleted. Appending the bins at the end (as build_member_anchor.py does) lost ~0.25 m3/s of Jun-Jul lateral flow here."""
    BA.clone_hru_along(src, dst, SRC, ADDED, area_split)
    with nc.Dataset(dst, "a") as d:
        hid = list(np.array(d["hruId"][:]).ravel().astype(int)); i = hid.index(SRC) + 1
        order = list(range(i)) + [hid.index(h) for h in ADDED] + [k for k in range(i, len(hid)) if hid[k] not in ADDED]
        for v in d.variables.values():
            if "hru" in v.dimensions: v[:] = np.take(np.array(v[:]), order, axis=v.dimensions.index("hru"))
    return dst


class Runner(C4.Runner):
    def _subset(self, work):
        super()._subset(work)
        for fn in ("attributes.nc", "trialParams.nc"):
            clone(work/fn, work/(fn + ".tmp"), AREA if fn == "attributes.nc" else None); (work/(fn + ".tmp")).replace(work/fn)
        clone(work/"coldState.nc", work/"cs37.nc"); BA.inject_multi(work/"cs37.nc", work/"coldState.nc", init_swe()); (work/"cs37.nc").unlink()
        with nc.Dataset(work/"trialParams.nc") as t: hid = np.array(t["hruId"][:]).ravel(); nir = np.array(t["albedoMinNearIR"][:]).ravel()
        b = np.isin(hid, BINS)
        DB.set_trial_param(work, "albedoMinNearIR", np.where(b, NIR_BIN, nir)); DB.set_trial_param(work, "relConvTol_energy", np.where(b, TOL_BIN, 0.01))

    def _apply_forcing(self, work, params):
        fdir = super()._apply_forcing(work, params); tc = self.soil["tempCritRain"]; add = base = 0.0
        for f in sorted(fdir.glob("*.nc")):
            clone(f, f.with_suffix(".tmp")); f.with_suffix(".tmp").replace(f)
            with nc.Dataset(f, "a") as d:
                hid = np.array(d["hruId"][:]).ravel(); dt = float(np.array(d["data_step"][:]).ravel()[0]); P = np.array(d["pptrate"][:])
                for h in BINS:
                    j = int(np.where(hid == h)[0][0])
                    tw = wet_bulb(np.array(d["airtemp"][:, j]), np.array(d["spechum"][:, j]), np.array(d["airpres"][:, j]))
                    snow = P[:, j]*np.clip((tc - tw + TRANGE/2)/TRANGE, 0.0, 1.0)
                    base += AREA[h]*snow.sum()*dt; add += AREA[h]*(DRIFT[h] - 1.0)*snow.sum()*dt; P[:, j] += (DRIFT[h] - 1.0)*snow
                d["pptrate"][:] = P
        self.audit = dict(hru2_snowfall_mm=base, added_mm=add, added_frac=add/base)
        return fdir


def fold(src, dst):
    """Copy cal_day.nc folding the bin HRUs back into HRU 2, area-weighted (fluxes and states in per-area units)."""
    with nc.Dataset(src) as a, nc.Dataset(dst, "w") as b:
        hid = np.array(a["hruId"][:]).ravel().astype(int); keep = hid <= 28; w = np.array([AREA.get(int(h), 0.0) for h in hid]); bi = np.isin(hid, BINS)
        for n, dm in a.dimensions.items(): b.createDimension(n, int(keep.sum()) if n == "hru" else (None if dm.isunlimited() else len(dm)))
        for n, v in a.variables.items():
            o = b.createVariable(n, v.dtype, v.dimensions); o.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"}); x = np.array(v[:])
            if "hru" in v.dimensions:
                ax = v.dimensions.index("hru"); folded = np.tensordot(np.compress(bi, x, axis=ax), w[bi], axes=([ax], [0])) if x.dtype.kind == "f" else None
                x = np.compress(keep, x, axis=ax)
                if folded is not None:
                    i2 = int(np.where(hid[keep] == SRC)[0][0]); sl = [slice(None)]*x.ndim; sl[ax] = i2; x[tuple(sl)] = folded
            o[:] = x


def main():
    t = json.loads((C4.OUT/"best"/"best_obj"/"trial.json").read_text())
    r = Runner(work_root=str(C4.OUT/"rerun_patch"), **C4.S.CFG); r.variant = C4.F.VARIANTS["G"]; r.soil = C4.physical(t["x"]); w = Path(r.work_root)/"keep"
    _, proc = r.run_summa(np.zeros(len(r.specs)), w); print("rc", proc.returncode, proc.stderr[-400:], flush=True)
    shutil.rmtree(KEEP, ignore_errors=True); shutil.copytree(w/"out", KEEP); shutil.rmtree(r.work_root, ignore_errors=True)
    fold(KEEP/"cal_day.nc", KEEP/"cal_day_28.nc")
    obj, parts, diag = C4.score(KEEP/"cal_day_28.nc", C4.obs_mm())
    out = dict(objective=obj, parts=parts, diagnostics=diag, audit=r.audit, init_swe_m=init_swe(), trial=t["x"])
    (KEEP/"score.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(dict(objective=obj, audit=r.audit, **{k: diag[k] for k in
        ("KGE_AprJul", "AprMay_bias", "JunJul_bias", "AugSep_bias", "OctNov_bias", "vapour_mm", "aso_swe_nmae", "band_order_mae")}), indent=1, default=float))


if __name__ == "__main__":
    main()
