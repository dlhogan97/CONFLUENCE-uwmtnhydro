"""Build, launch and score full 1981-2025 SUMMA runs (new build in /home/dlhogan/summa_deepstore) for fitted parameter sets."""
import shutil, os, json, subprocess, numpy as np, pandas as pd, netCDF4 as nc
from pathlib import Path
import emu_core as E
BASE = E.BASE_RUN
ROOT = Path("/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/overnight/summa")
EXE = "/tmp/claude-1008/summa_ds"        # wrapper: LD_LIBRARY_PATH -> new libsumma.so, exec /home/dlhogan/summa_deepstore/bin/summa.exe
def ln(n, a, b, c): return f"{n:<25} | {a:>12} | {b:>12} | {c:>12}"
def num(v):
    # the parameter files are read with a fixed-width real format: a value without a decimal point (e.g. "1") is read as 1e-4
    s = f"{v:.5g}"
    return s if ("." in s or "e" in s) else s+".0"
def build(name, fit, singlebasin):
    p = fit["params"]; T = ROOT/name; (T/"output").mkdir(parents=True, exist_ok=True)
    for f in ("attributes.nc", "coldState.nc", "trialParams.nc", "basinParamInfo.txt", "localParamInfo.txt", "modelDecisions.txt", "outputControl.txt", "forcingFileList.txt", "TBL_GENPARM.TBL", "TBL_MPTABLE.TBL", "TBL_SOILPARM.TBL", "TBL_VEGPARM.TBL"):
        shutil.copy2(BASE/f, T/f)
    if singlebasin:
        s = open(T/"modelDecisions.txt").read(); assert "spatial_gw     localColumn" in s
        open(T/"modelDecisions.txt", "w").write(s.replace("spatial_gw     localColumn", "spatial_gw     singleBasin"))
    # local parameters: declare the leak parameters (value is the default for every HRU)
    out = []
    for l in open(T/"localParamInfo.txt").read().split("\n"):
        out.append(l)
        if l.startswith("aquiferBaseflowExp"):
            out.append(ln("aquiferLeakRate", num(p["lam"]*E.RATE0), "0", "1.0e-3")); out.append(ln("aquiferLeakExp", num(p["m_leak"]), "1.0", "10.0"))
    open(T/"localParamInfo.txt", "w").write("\n".join(out))
    out = []
    for l in open(T/"basinParamInfo.txt").read().rstrip("\n").split("\n"):
        k = l.split("|")[0].strip()
        if k == "basin__aquiferHydCond": l = ln(k, num(E.RATE0), "1.0e-9", "10.0")
        elif k == "basin__aquiferScaleFactor": l = ln(k, num(p["Sf_d"]), "0.1", "100.0")
        elif k == "basin__aquiferBaseflowExp": l = ln(k, num(p["exp_d"]), "1.0", "10.0")
        out.append(l)
        if k == "basin__aquiferBaseflowExp": out.append(ln("basin__aquiferInitStorage", num(fit["S_deep"]) if singlebasin else "0.0", "0.0", "1000.0"))
    open(T/"basinParamInfo.txt", "w").write("\n".join(out)+"\n")
    with nc.Dataset(T/"trialParams.nc", "a") as t:
        t["aquiferScaleFactor"][:] = p["Sf_h"]; t["aquiferBaseflowExp"][:] = p["exp_h"]; t["routingGammaScale"][:] = p["scale"]
    with nc.Dataset(T/"coldState.nc", "a") as c:
        v = c["scalarAquiferStorage"]; v[:] = np.array(fit["S_small"]).reshape(np.array(v[:]).shape)
    if singlebasin:
        with open(T/"outputControl.txt", "a") as f:
            f.write("scalarAquiferLeak         | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\nbasin__AquiferStorage     | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\nbasin__AquiferLeak        | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\nbasin__DeepBaseflow       | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n")
    fm = open(BASE/"fileManager.txt").read().replace("settingsPath         '"+str(BASE)+"/", "settingsPath         '"+str(T)+"/").replace("outputPath           '"+str(BASE)+"/out/", "outputPath           '"+str(T)+"/output/").replace("outFilePrefix        'baseline'", f"outFilePrefix        '{name}'")
    open(T/"fileManager.txt", "w").write(fm); return T
def launch(T):
    return subprocess.Popen(f"{EXE} -m {T}/fileManager.txt > {T}/summa.log 2>&1", shell=True, start_new_session=True)
def finished(T):
    f = T/"summa.log"; return f.exists() and "finished simulation successfully" in f.read_text()[-4000:]
def score(T, name, singlebasin):
    d = nc.Dataset(next((T/"output").glob("*_timestep.nc"))); t = d["time"]; t2 = pd.DatetimeIndex(nc.num2date(t[:], t.units, only_use_cftime_datetimes=False))
    q = pd.Series(np.array(d["averageRoutedRunoff"][:]).ravel()*E.Atot, index=t2).resample("1D").mean(); Sd = None
    if singlebasin: Sd = pd.Series(np.array(d["basin__AquiferStorage_mean"][:]).ravel(), index=t2).resample("1D").mean()
    out = {}
    for wn, W in E.WINDOWS.items(): out[wn] = dict(skill=E.skill(q, W["skill"]), memory=E.memory(q, *W["mem"], Sdeep=Sd))
    return out, q
