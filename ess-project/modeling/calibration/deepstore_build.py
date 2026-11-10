"""deepstore_build.py -- staging helpers for the modified SUMMA build in /home/dlhogan/summa_deepstore (wrapper ~/bin/summa_ds).

The build adds parameters that every settings directory must declare, or SUMMA stops at startup:
  aquiferLeakRate, aquiferLeakExp   (local)   leak from the HRU aquifer to the basin store   (0 / 1 = off)
  canopyGapFrac                     (local)   between-canopy gap fraction in Beer's-law SW and canopy emissivity (0 = original)
  basin__aquiferInitStorage         (basin)   initial basin-store storage                     (0 = empty)
With all of them at their 'off' values the build reproduces the original executable (checked 2026-10-20).

Gap prior (`observations/<basin>_canopy_gap_prior.csv`): for forest-assigned HRUs (IGBP 1-5), 1 - tree fraction of the HRU from the 30 m
land-cover intersection; 0 elsewhere (shrub/barren canopies are buried by snow). It is a fixed prior, not a calibrated value.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, netCDF4 as nc

SUMMA_DS = "/home/dlhogan/bin/summa_ds"
OBS = Path(__file__).parent/"observations"


def _ln(n, a, b, c): return f"{n:<25} | {a:>12} | {b:>12} | {c:>12}"


def _num(v):
    # the parameter files are read with a fixed-width real format: a value without a decimal point (e.g. "1") is read as 1e-4
    s = f"{v:.5g}"
    return s if ("." in s or "e" in s) else s+".0"


def declare_params(work: Path, leak_rate=0.0, leak_exp=1.0, gap_default=0.0, basin_init=0.0):
    """Append the build's extra parameters to the staged localParamInfo.txt / basinParamInfo.txt (idempotent)."""
    lp = work/"localParamInfo.txt"; s = lp.read_text()
    add = [(k, v, lo, hi) for k, v, lo, hi in (("aquiferLeakRate", leak_rate, 0.0, 1.0e-3), ("aquiferLeakExp", leak_exp, 1.0, 10.0),
                                               ("canopyGapFrac", gap_default, 0.0, 1.0)) if f"\n{k} " not in "\n"+s]
    if add: lp.write_text(s.rstrip("\n")+"\n"+"\n".join(_ln(k, _num(v), _num(lo), _num(hi)) for k, v, lo, hi in add)+"\n")
    bp = work/"basinParamInfo.txt"; s = bp.read_text()
    if "basin__aquiferInitStorage" not in s:
        bp.write_text(s.rstrip("\n")+"\n"+_ln("basin__aquiferInitStorage", _num(basin_init), "0.0", "1000.0")+"\n")


def gap_prior(basin: str, hru_ids) -> np.ndarray:
    """Per-HRU canopy gap prior, ordered like `hru_ids`. basin: 'East_River' | 'Tuolumne_River'."""
    g = pd.read_csv(OBS/f"{basin.split('_')[0].lower()}_canopy_gap_prior.csv", index_col="hruId")["gap"]
    return g.reindex(np.asarray(hru_ids).astype(int)).fillna(0.0).values


def set_trial_param(work: Path, name: str, values):
    """Write a per-HRU parameter into the staged trialParams.nc, creating the variable if needed (trialParams overrides localParamInfo)."""
    with nc.Dataset(work/"trialParams.nc", "a") as ds:
        if name not in ds.variables: ds.createVariable(name, "f8", ("hru",))
        ds.variables[name][:] = np.asarray(values, dtype=float)
