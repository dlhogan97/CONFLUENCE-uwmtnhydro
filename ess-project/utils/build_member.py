"""Build + run the 5-year snowpatch-bin smoke test for Tuolumne HRU 2.

Splits HRU 2 into 9 depth bins whose AREAS discretise the fitted fSCA recession
curve, and gives each bin a PRECIPITATION multiplier (drift/avalanche) so its
supply a_j*S melts out at the peak SWE the curve says it should. Ablation is
uniform (albedoMinNearIR = 0.70) because that lever is already saturated.

Only the bin HRUs' precipitation is touched, and only the SNOWFALL part of it
(pptrate where airtemp < TSNOW) -- drift redistributes snow, not rain.
"""
from __future__ import annotations
from pathlib import Path
import re, shutil, subprocess, sys
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc4

sys.path.insert(0, str(Path(__file__).parent))
from bin_spec import A_HRU2
import pandas as _pd
DESIGN = Path(sys.argv[1])
MEMBER = sys.argv[2]
def spec():
    return _pd.read_csv(DESIGN), None

SCRATCH  = Path("/scratch/dlhogan/ess-project-data")
SETT     = SCRATCH/"calibration/bestruns/Tuolumne_River_baseline_wbias10_1981_2025/run"
LONGTERM = SETT/"out/baseline_timestep.nc"
FORCING  = SETT/"forcing"
EXP      = SCRATCH/"domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_sweep_20260812"
SRC_HRU  = 2
SIM_START, SIM_END = "1981-10-01 00:00", "2025-09-30 23:00"
MONTHS   = pd.date_range("1981-10-01", "2025-09-01", freq="MS")
TSNOW    = 274.15        # K; scale pptrate only below this (snowfall)
NIR_FLOOR, NIR_DEFAULT = 0.70, 0.30
SNOW_DENSITY, SNOW_TEMP_K, SNOW_N_LAYERS, SOIL_TEMP_K = 500.0, 272.15, 5, 273.15
RHO_ICE = 917.0
SETTINGS_LINK = ["modelDecisions.txt", "outputControl.txt", "localParamInfo.txt",
                 "basinParamInfo.txt", "TBL_VEGPARM.TBL", "TBL_SOILPARM.TBL",
                 "TBL_GENPARM.TBL", "TBL_MPTABLE.TBL"]


def clone_hru_along(nc_in, nc_out, src_id, new_ids, area_split=None, scale_contour=True):
    """Append clones of hru `src_id` along the hru dim. area_split={hruId: frac}
    rescales HRUarea AND contourLength (which scales with area, not per-HRU)."""
    din = nc4.Dataset(nc_in); hid = np.array(din["hruId"][:]).ravel(); n_old = hid.size
    si = int(np.where(hid == src_id)[0][0])
    with nc4.Dataset(nc_out, "w") as d:
        for nm, dim in din.dimensions.items():
            d.createDimension(nm, n_old + len(new_ids) if nm == "hru"
                              else (None if dim.isunlimited() else len(dim)))
        for nm, var in din.variables.items():
            fv = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
            v = d.createVariable(nm, var.datatype, var.dimensions, fill_value=fv)
            for a in var.ncattrs():
                if a != "_FillValue": v.setncattr(a, var.getncattr(a))
            arr = np.array(var[:])
            if "hru" in var.dimensions:
                ax = var.dimensions.index("hru")
                arr = np.concatenate([arr, np.take(arr, [si]*len(new_ids), axis=ax)], axis=ax)
            v[:] = arr
        hid2 = np.array(d["hruId"][:]).ravel()
        for k, nid in enumerate(new_ids): hid2[n_old + k] = nid
        d["hruId"][:] = hid2
        if area_split:
            for nm in (["HRUarea"] + (["contourLength"] if scale_contour else [])):
                if nm not in d.variables: continue
                a = np.array(d[nm][:]).ravel(); src = a[np.where(hid2 == src_id)[0][0]]
                for h, f in area_split.items(): a[np.where(hid2 == h)[0][0]] = f*src
                d[nm][:] = a
    din.close(); return nc_out


def inject_multi(base_ws, out_ws, swe_by_hru):
    """Prepend SNOW_N_LAYERS snow layers carrying swe_by_hru[hruId] metres w.e."""
    src = nc4.Dataset(base_ws); hid = np.array(src["hruId"][:]).ravel(); nhru = hid.size
    nsoil = int(np.array(src["nSoil"][:]).ravel()[0])
    swe_mm = np.zeros(nhru)
    for h, m in swe_by_hru.items(): swe_mm[hid == h] = m*1000.0
    nsnow = np.where(swe_mm > 0, SNOW_N_LAYERS, 0).astype("i4")
    nmid = int(nsnow.max()) + nsoil; nifc = nmid + 1
    def soil(nm):
        a = np.atleast_2d(np.array(src[nm][:])); return (a if a.shape[1] == nhru else a.T)[:nsoil].copy()
    sT, sIce, sLiq, sDep, matric = (soil("mLayerTemp"), soil("mLayerVolFracIce"),
                                    soil("mLayerVolFracLiq"), soil("mLayerDepth"),
                                    soil("mLayerMatricHead"))
    sT[:, swe_mm > 0] = SOIL_TEMP_K
    T = np.zeros((nmid, nhru)); Ice = np.zeros((nmid, nhru)); Liq = np.zeros((nmid, nhru))
    Dep = np.zeros((nmid, nhru)); iH = np.zeros((nifc, nhru))
    for h in range(nhru):
        ns = int(nsnow[h])
        if ns:
            dtot = swe_mm[h]/SNOW_DENSITY
            T[:ns, h] = SNOW_TEMP_K; Ice[:ns, h] = SNOW_DENSITY/RHO_ICE; Dep[:ns, h] = dtot/ns
        T[ns:ns+nsoil, h] = sT[:, h]; Ice[ns:ns+nsoil, h] = sIce[:, h]
        Liq[ns:ns+nsoil, h] = sLiq[:, h]; Dep[ns:ns+nsoil, h] = sDep[:, h]
        dep = Dep[:ns+nsoil, h]
        iH[:ns+nsoil+1, h] = -dep[:ns].sum() + np.r_[0.0, np.cumsum(dep)]
    with nc4.Dataset(out_ws, "w") as d:
        for n, s in [("hru",nhru),("scalarv",1),("midSoil",nsoil),("midToto",nmid),("ifcToto",nifc)]:
            d.createDimension(n, s)
        d.createVariable("hruId","i4",("hru",))[:] = hid
        d.createVariable("nSoil","i4",("scalarv","hru"))[:] = nsoil
        d.createVariable("nSnow","i4",("scalarv","hru"))[0,:] = nsnow
        d.createVariable("dt_init","f8",("scalarv","hru"))[:] = 3600.0
        for nm in ["scalarCanopyIce","scalarCanopyLiq","scalarSfcMeltPond",
                   "scalarAquiferStorage","scalarCanopyTemp","scalarCanairTemp"]:
            d.createVariable(nm,"f8",("scalarv","hru"))[0,:] = np.array(src[nm][:]).ravel()
        swe_v = np.array(src["scalarSWE"][:]).ravel()
        dep_v = np.array(src["scalarSnowDepth"][:]).ravel()
        alb_v = np.array(src["scalarSnowAlbedo"][:]).ravel()
        for h, m in swe_by_hru.items():
            swe_v[hid==h] = m*1000.0; dep_v[hid==h] = m*1000.0/SNOW_DENSITY; alb_v[hid==h] = 0.60
        d.createVariable("scalarSWE","f8",("scalarv","hru"))[0,:] = swe_v
        d.createVariable("scalarSnowDepth","f8",("scalarv","hru"))[0,:] = dep_v
        d.createVariable("scalarSnowAlbedo","f8",("scalarv","hru"))[0,:] = alb_v
        for nm, arr, dim in [("mLayerTemp",T,"midToto"),("mLayerVolFracIce",Ice,"midToto"),
                             ("mLayerVolFracLiq",Liq,"midToto"),("mLayerDepth",Dep,"midToto"),
                             ("mLayerMatricHead",matric,"midSoil"),("iLayerHeight",iH,"ifcToto")]:
            d.createVariable(nm,"f8",(dim,"hru"))[:] = arr
    src.close(); return out_ws


def extract_warmstate(longterm, cold, out, ts):
    """Slice `ts` from the long-term run into a coldState-shaped warmState."""
    SCAL = [("scalarCanopyIce","scalarCanopyIce_mean"),("scalarCanopyLiq","scalarCanopyLiq_mean"),
            ("scalarSnowDepth","scalarSnowDepth_mean"),("scalarSWE","scalarSWE"),
            ("scalarSfcMeltPond","scalarSfcMeltPond_mean"),
            ("scalarAquiferStorage","scalarAquiferStorage_mean"),
            ("scalarCanopyTemp","scalarCanopyTemp_mean"),("scalarCanairTemp","scalarCanairTemp_mean")]
    LAYER = [("mLayerTemp","mLayerTemp_mean"),("mLayerVolFracIce","mLayerVolFracIce_mean"),
             ("mLayerVolFracLiq","mLayerVolFracLiq_mean"),("mLayerMatricHead","mLayerMatricHead_mean")]
    lt, cd = xr.open_dataset(longterm), xr.open_dataset(cold)
    i = pd.DatetimeIndex(lt.time.values).get_indexer([ts], method="nearest")[0]
    t = lt.isel(time=i)
    nhru = int(cd.sizes["hru"]); nsoil = int(cd["nSoil"].values.flat[0])
    pick = lambda n: n if n in t else (n.removesuffix("_mean") if n.removesuffix("_mean") in t else None)
    with nc4.Dataset(out, "w") as d:
        for n, s in [("hru",nhru),("scalarv",1),("midSoil",nsoil),("midToto",nsoil),("ifcToto",nsoil+1)]:
            d.createDimension(n, s)
        d.createVariable("hruId","i4",("hru",))[:] = cd["hruId"].values
        d.createVariable("nSoil","i4",("scalarv","hru"))[:] = nsoil
        d.createVariable("nSnow","i4",("scalarv","hru"))[:] = 0
        d.createVariable("dt_init","f8",("scalarv","hru"))[:] = 3600.0
        for cs, ln in SCAL:
            v = d.createVariable(cs,"f8",("scalarv","hru")); a = pick(ln)
            if a: 
                arr = np.atleast_1d(t[a].values).ravel()
                v[0,:] = arr if arr.size == nhru else np.full(nhru, float(np.nanmean(arr)))
            else: v[0,:] = np.array(cd[cs].values).ravel() if cs in cd else 0.0
        d.createVariable("scalarSnowAlbedo","f8",("scalarv","hru"))[:] = (
            np.array(cd["scalarSnowAlbedo"].values).ravel() if "scalarSnowAlbedo" in cd else 0.85)
        for cs, ln in LAYER:
            dim = "midSoil" if cs == "mLayerMatricHead" else "midToto"
            v = d.createVariable(cs,"f8",(dim,"hru")); a = pick(ln)
            src = np.atleast_2d(t[a].values) if a else np.atleast_2d(cd[cs].values)
            if src.shape[1] != nhru: src = src.T
            v[:min(v.shape[0], src.shape[0]),:] = src[:v.shape[0],:]
        for cs, dim in [("mLayerDepth","midToto"),("iLayerHeight","ifcToto")]:
            src = np.atleast_2d(cd[cs].values)
            if src.shape[1] != nhru: src = src.T
            d.createVariable(cs,"f8",(dim,"hru"))[:] = src[:d.dimensions[dim].size,:]
    lt.close(); cd.close(); return out


def scale_precip(path, drift_by_hru):
    """Multiply pptrate on the bin HRUs, SNOWFALL ONLY (airtemp < TSNOW).
    Returns (added_mm, base_mm) area-agnostic column totals for the mass audit."""
    add = base = 0.0
    with nc4.Dataset(path, "a") as d:
        hid = np.array(d["hruId"][:]).ravel()
        ppt = np.array(d["pptrate"][:]); tair = np.array(d["airtemp"][:])
        dt = float(np.array(d["data_step"][:]).ravel()[0])
        for h, a in drift_by_hru.items():
            j = int(np.where(hid == h)[0][0])
            snow = tair[:, j] < TSNOW
            b = ppt[snow, j].sum()*dt
            base += b; add += b*(a - 1.0)
            ppt[snow, j] *= a
        d["pptrate"][:] = ppt
    return add, base


def build(name=None):
    name = name or MEMBER
    b, S = spec()
    bins = list(b.hruId); added = bins[1:]
    run = EXP/name; (run/"output").mkdir(parents=True, exist_ok=True)
    forc = EXP/(name + "_forcing"); forc.mkdir(parents=True, exist_ok=True)
    area_split = dict(zip(bins, b.area_frac))
    drift = dict(zip(bins, b.drift_a))

    clone_hru_along(SETT/"attributes.nc", run/"attributes.nc", SRC_HRU, added, area_split)
    base = run/"base28_ws.nc"   # per-member: a shared path corrupts parallel builds
    extract_warmstate(LONGTERM, SETT/"coldState.nc", base, pd.Timestamp(SIM_START))
    b30 = run/f"base{28+len(added)}_ws.nc"; clone_hru_along(base, b30, SRC_HRU, added)
    inject_multi(b30, run/"warmState.nc", dict(zip(bins, b.init_m)))

    clone_hru_along(SETT/"trialParams.nc", run/"trialParams.nc", SRC_HRU, added)
    with nc4.Dataset(run/"trialParams.nc", "a") as t:
        hid = np.array(t["hruId"][:]).ravel()
        v = (t.createVariable("albedoMinNearIR","f8",("hru",))
             if "albedoMinNearIR" not in t.variables else t["albedoMinNearIR"])
        v[:] = np.where(np.isin(hid, bins), NIR_FLOOR, NIR_DEFAULT)
        # WY1994 crash: solver could not converge on a 4 mm snow layer at melt-out.
        # Loosen the energy tolerance on the bin HRUs only (default 0.01, max 0.1).
        ce = (t.createVariable("relConvTol_energy","f8",("hru",))
              if "relConvTol_energy" not in t.variables else t["relConvTol_energy"])
        ce[:] = np.where(np.isin(hid, bins), 0.05, 0.01)

    pre = next(re.match(r"^(.+_)(\d{6})$", f.stem).group(1)
               for f in sorted(FORCING.glob("*.nc")) if re.match(r"^(.+_)(\d{6})$", f.stem))
    add = tot = 0.0
    for d0 in MONTHS:
        fn = f"{pre}{d0.year:04d}{d0.month:02d}.nc"
        clone_hru_along(FORCING/fn, forc/fn, SRC_HRU, added)
        a_, b_ = scale_precip(forc/fn, drift); add += a_; tot += b_

    for fn in SETTINGS_LINK:
        dst = run/fn
        if dst.is_symlink() or dst.exists(): dst.unlink()
        if (SETT/fn).exists(): dst.symlink_to((SETT/fn).resolve())
    tz = next((l.split("'")[1] for l in (SETT/"fileManager.txt").read_text().splitlines()
               if l.strip().startswith("tmZoneInfo")), "localTime")
    (run/"forcingFileList.txt").write_text(
        "".join(f"{pre}{d.year:04d}{d.month:02d}.nc\n" for d in MONTHS))
    (run/"fileManager.txt").write_text(
        "controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n"
        f"simStartTime         '{SIM_START}'\nsimEndTime           '{SIM_END}'\n"
        f"tmZoneInfo           '{tz}'\noutFilePrefix        '{name}'\n"
        f"settingsPath         '{run}/'\nforcingPath          '{forc}/'\n"
        f"outputPath           '{run}/output/'\n"
        "initConditionFile    'warmState.nc'\nattributeFile        'attributes.nc'\n"
        "trialParamFile       'trialParams.nc'\nforcingListFile      'forcingFileList.txt'\n"
        "decisionsFile        'modelDecisions.txt'\noutputControlFile    'outputControl.txt'\n"
        "globalHruParamFile   'localParamInfo.txt'\nglobalGruParamFile   'basinParamInfo.txt'\n"
        "vegTableFile         'TBL_VEGPARM.TBL'\nsoilTableFile        'TBL_SOILPARM.TBL'\n"
        "generalTableFile     'TBL_GENPARM.TBL'\nnoahmpTableFile      'TBL_MPTABLE.TBL'\n")

    b.to_csv(run/"bin_spec.csv", index=False)
    a = nc4.Dataset(run/"attributes.nc"); h = np.array(a["hruId"][:]).ravel()
    ar = np.array(a["HRUarea"][:]).ravel(); a.close()
    print(f"built {run}  ({h.size} HRUs)")
    print(f"  bin areas km2: {[round(float(ar[h==x][0])/1e6, 3) for x in bins]}"
          f"  sum {sum(float(ar[h==x][0]) for x in bins)/1e6:.3f} (was {A_HRU2:.3f})")
    print(f"  snowfall added to bins = {add/1000:.1f} m column-sum "
          f"({add/tot*100:.1f}% of the 9 bin columns' snowfall)")
    print(f"  basin-relative import  = {(b.area_frac*b.drift_a).sum()-1:.3f} x HRU2 snowfall "
          f"= {((b.area_frac*b.drift_a).sum()-1)*A_HRU2/1287.0*100:.2f}% of basin area-snowfall")
    return run


if __name__ == "__main__":
    run = build()
    print("BUILT_OK", run/"fileManager.txt")
    raise SystemExit(0)
    # stream SUMMA stdout to <run>/summa.log so progress is visible DURING the run.
    # capture_output=True buffers until exit -- the log stays empty for hours.
    with open(run/"summa.log", "w") as lg:
        r = subprocess.run(["summa", "-m", str(run/"fileManager.txt")],
                           stdout=lg, stderr=subprocess.STDOUT, text=True)
    print("RC", r.returncode, "| log:", run/"summa.log")
