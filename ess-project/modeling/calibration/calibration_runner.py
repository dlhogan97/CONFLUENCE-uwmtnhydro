#!/usr/bin/env python3
"""
calibration_runner.py — One SUMMA evaluation for the signature calibration.

Given a parameter vector, this: maps it to physical parameters, writes trialParams,
sets the wet-soil (field-capacity) cold state, runs SUMMA over spin-up + analysis,
extracts routed streamflow, and returns the aggregate-signature objective plus guard
checks.  Each call uses an isolated worker directory so it is safe under
differential_evolution(workers=N).

Parameter spec is per basin (see PARAM_SPECS below); mirrors CALIBRATION_SPEC.md.
Multiplier parameters scale the a-priori per-HRU field (preserving spatial structure);
absolute parameters are set uniformly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

from signature_objective import signature_objective, guard_runoff_ratio, guard_snow

SUMMA_EXE = "/home/dlhogan/bin/summa.exe"
FAIL_PENALTY = 10.0


@dataclass
class ParamSpec:
    name: str            # trialParams variable, or a synthetic name for kind='elev'
    kind: str            # 'mult' (scale a-priori field) | 'abs' (uniform) | 'elev' (ramp; see below)
    lo: float            # search lower bound (in the OPTIMIZED coordinate)
    hi: float            # search upper bound
    log: bool = False    # optimize in log10 space

    def to_physical(self, x: float) -> float:
        return 10.0 ** x if self.log else x

    @property
    def bounds(self) -> Tuple[float, float]:
        return (np.log10(self.lo), np.log10(self.hi)) if self.log else (self.lo, self.hi)


# --------------------------------------------------------------------------- #
# per-basin parameter specifications  (see CALIBRATION_SPEC.md)
# --------------------------------------------------------------------------- #
def _routing_scale_bounds(mean_lo_h, mean_hi_h, shape=2.5):
    return (mean_lo_h * 3600 / shape, mean_hi_h * 3600 / shape)

# theta_sat dropped (fixed at a-priori 0.45; not a dominant control).
# qSurfScale added both (saturation-excess surface-runoff = fast/slow partition).
# zScale_TOPMODEL added to East (requires hc_profile=pow_prof, now set) for depth-decaying K.
#
# frozenPrecipMultip is DISTRIBUTED as a linear ramp in elevation:
#     fpm_i = fpm_low + fpm_delta * znorm_i,     znorm = (elev-elev_min)/(elev_max-elev_min)
# Rationale: snowfall gauge undercatch grows with elevation (wind, colder, more snow-phase),
# and adding snow HIGH shifts the melt centroid later far more efficiently than a uniform
# multiplier -- which can only move volume and timing together. fpm_delta >= 0 by
# construction, so the ramp can only increase with elevation (encodes the physical prior).
PARAM_SPECS: Dict[str, List[ParamSpec]] = {
    "East_River": [
        ParamSpec("k_soil",                  "mult", 0.1, 10.0, log=True),
        ParamSpec("qSurfScale",              "abs",  1.0, 100.0, log=True),
        ParamSpec("zScale_TOPMODEL",         "abs",  1.0, 8.0),
        ParamSpec("aquiferBaseflowRate",     "abs",  1e-8, 1e-5, log=True),
        ParamSpec("aquiferScaleFactor",      "abs",  0.5, 5.0),
        ParamSpec("aquiferBaseflowExp",      "abs",  1.0, 5.0),
        ParamSpec("frozenPrecipMultip_low",  "elev", 0.85, 1.20),
        ParamSpec("frozenPrecipMultip_delta","elev", 0.0, 0.40),
        ParamSpec("routingGammaScale",       "abs",  *_routing_scale_bounds(16, 36)),
    ],
    "Tuolumne_River": [
        ParamSpec("k_soil",                  "mult", 0.1, 10.0, log=True),
        ParamSpec("qSurfScale",              "abs",  1.0, 100.0, log=True),
        ParamSpec("zScale_TOPMODEL",         "abs",  1.0, 8.0),
        ParamSpec("aquiferBaseflowExp",      "abs",  1.0, 5.0),
        ParamSpec("frozenPrecipMultip_low",  "elev", 0.85, 1.20),
        ParamSpec("frozenPrecipMultip_delta","elev", 0.0, 0.40),
        ParamSpec("routingGammaScale",       "abs",  *_routing_scale_bounds(12, 30)),
    ],
}

TAU_TARGET = {"East_River": 27.0, "Tuolumne_River": 15.0}


class CalibrationRunner:
    def __init__(self, domain: str, sim_start: str, sim_end: str,
                 analysis_start: str, obs_csv: str, work_root: str,
                 precip_mm_yr: float, peak_swe_apriori: float):
        self.domain = domain
        self.specs = PARAM_SPECS[domain]
        self.tau_target = TAU_TARGET[domain]
        self.sim_start, self.sim_end = sim_start, sim_end
        self.analysis_start = analysis_start          # drop spin-up before this
        self.settings = Path(f"/scratch/dlhogan/ess-project-data/"
                             f"domain_{domain}_distributed_elevAspect/settings/SUMMA")
        self.forcing = self.settings.parent.parent / "forcing" / "SUMMA_input"
        with xr.open_dataset(self.settings / "attributes.nc") as _a:
            _area = _a["HRUarea"].values.astype(float)
            _elev = _a["elevation"].values.astype(float)
        self.area = float(_area.sum())
        self.hru_wt = _area / _area.sum()
        # normalized elevation (0 at lowest HRU, 1 at highest) for the frozen-precip ramp
        self.elev_norm = ((_elev - _elev.min()) / (_elev.max() - _elev.min())
                          if _elev.max() > _elev.min() else np.zeros_like(_elev))
        with xr.open_dataset(self.settings / "trialParams.nc") as _tp:
            self.fieldcap = float(np.ravel(_tp["fieldCapacity"].values)[0])
            # a-priori per-HRU fields for the multiplier parameters (loaded once)
            self.base_vals = {s.name: _tp[s.name].values.copy()
                              for s in self.specs if s.kind == "mult" and s.name in _tp}
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.precip_mm_yr = precip_mm_yr
        self.peak_swe_apriori = peak_swe_apriori
        obs = pd.read_csv(obs_csv, parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
        self.obs = obs[analysis_start:sim_end]

    @property
    def bounds(self):
        return [s.bounds for s in self.specs]

    def _vector_to_params(self, x: np.ndarray) -> Dict[str, Tuple[str, float]]:
        out = {}
        for s, xi in zip(self.specs, x):
            out[s.name] = (s.kind, s.to_physical(xi))
        return out

    def _stage(self, work: Path):
        (work / "out").mkdir(parents=True, exist_ok=True)
        for f in self.settings.iterdir():
            if f.suffix in (".txt", ".TBL", ".nc"):
                shutil.copy2(f, work / f.name)

    def _apply_params(self, work: Path, params):
        with nc.Dataset(work / "trialParams.nc", "a") as ds:
            for name, (kind, val) in params.items():
                if kind == "elev" or name not in ds.variables:
                    continue                      # 'elev' pair handled below
                if kind == "mult":
                    ds.variables[name][:] = self.base_vals[name] * val
                else:
                    ds.variables[name][:] = val
            # elevation-ramped frozen-precip multiplier (undercatch grows with elevation)
            if "frozenPrecipMultip_low" in params:
                lo = params["frozenPrecipMultip_low"][1]
                dl = params["frozenPrecipMultip_delta"][1]
                ds.variables["frozenPrecipMultip"][:] = lo + dl * self.elev_norm

    def _wet_soil(self, work: Path):
        with nc.Dataset(work / "coldState.nc", "a") as ds:
            ds.variables["mLayerVolFracLiq"][:] = self.fieldcap
            # keep matric head consistent-ish: leave as-is (SUMMA re-derives from theta)

    def _write_filemanager(self, work: Path, prefix: str):
        (work / "fileManager.txt").write_text(
            f"controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n"
            f"simStartTime         '{self.sim_start} 00:00'\n"
            f"simEndTime           '{self.sim_end} 23:00'\n"
            f"tmZoneInfo           'localTime'\n"
            f"outFilePrefix        '{prefix}'\n"
            f"settingsPath         '{work}/'\n"
            f"forcingPath          '{self.forcing}/'\n"
            f"outputPath           '{work}/out/'\n"
            f"initConditionFile    'coldState.nc'\n"
            f"attributeFile        'attributes.nc'\n"
            f"trialParamFile       'trialParams.nc'\n"
            f"forcingListFile      'forcingFileList.txt'\n"
            f"decisionsFile        'modelDecisions.txt'\n"
            f"outputControlFile    'outputControl.txt'\n"
            f"globalHruParamFile   'localParamInfo.txt'\n"
            f"globalGruParamFile   'basinParamInfo.txt'\n"
            f"vegTableFile         'TBL_VEGPARM.TBL'\n"
            f"soilTableFile        'TBL_SOILPARM.TBL'\n"
            f"generalTableFile     'TBL_GENPARM.TBL'\n"
            f"noahmpTableFile      'TBL_MPTABLE.TBL'\n")
        (work / "outputControl.txt").write_text(
            "hruId                | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n"
            "averageRoutedRunoff  | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
            "scalarSWE            | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n"
            "scalarAquiferStorage | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0\n")

    def evaluate(self, x: np.ndarray) -> float:
        r = self.evaluate_full(x)
        self._log_trial(x, r)
        return r["objective"]

    def _log_trial(self, x, r):
        import json, time
        logdir = self.work_root / "logs"
        logdir.mkdir(exist_ok=True)
        rec = {"t": time.time(), "domain": self.domain,
               "x": [float(v) for v in np.asarray(x)],
               "objective": r["objective"], "reason": r.get("reason", "ok"),
               "parts": r.get("parts", {}), "diagnostics": r.get("diagnostics", {}),
               "params": r.get("params", {}), "runoff_mm": r.get("runoff_mm"),
               "peak_swe": r.get("peak_swe")}
        with open(logdir / f"trials_{os.getpid()}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def evaluate_full(self, x: np.ndarray) -> dict:
        work = self.work_root / f"w_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        prefix = "cal"
        try:
            self._stage(work)
            params = self._vector_to_params(np.asarray(x))
            self._apply_params(work, params)
            self._wet_soil(work)
            self._write_filemanager(work, prefix)
            # OMP_NUM_THREADS=1: domains are 1 GRU, so SUMMA's GRU-level OpenMP has nothing to
            # parallelize; leaving it unset makes each worker grab all cores and oversubscribe
            # (20 workers x 24 threads). Single-threaded workers run clean in parallel (~2-3x).
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            # 50-min cap: a legit 13-yr eval is well under; headroom for slow (high-zScale)
            # param sets, while still killing runaway (tiny-timestep) trials.
            proc = subprocess.run([SUMMA_EXE, "-m", str(work / "fileManager.txt")],
                                  capture_output=True, text=True, timeout=3000, env=env)
            out = work / "out" / f"{prefix}_timestep.nc"
            if proc.returncode != 0 or not out.exists():
                return {"objective": FAIL_PENALTY, "reason": "summa_failed",
                        "params": {k: v[1] for k, v in params.items()}}
            ds = xr.open_dataset(out)
            t = pd.DatetimeIndex(ds["time"].values)
            sim = (pd.Series(ds["averageRoutedRunoff"].isel(gru=0).values, index=t)
                   * self.area).resample("D").mean()[self.analysis_start:self.sim_end]
            swe = pd.Series((ds["scalarSWE"].values * self.hru_wt[None, :]).sum(1), index=t)
            swe_d = swe.resample("D").mean()
            peak_swe = swe_d.groupby(swe_d.index.year
                                     + (swe_d.index.month >= 10).astype(int)).max().mean()
            # basin runoff depth (mm/yr) for the runoff-ratio guard
            sim_depth = (pd.Series(ds["averageRoutedRunoff"].isel(gru=0).values, index=t)
                         * 86400 * 1000).resample("D").mean()[self.analysis_start:self.sim_end]
            sim_mm_yr = sim_depth.sum() / (len(sim_depth) / 365.25)
            ds.close()
            # guards
            if not guard_runoff_ratio(sim_mm_yr, self.precip_mm_yr):
                return {"objective": FAIL_PENALTY, "reason": "runoff_ratio",
                        "runoff_mm": sim_mm_yr}
            if not guard_snow(float(peak_swe), self.peak_swe_apriori):
                return {"objective": FAIL_PENALTY, "reason": "snow_guard",
                        "peak_swe": float(peak_swe)}
            res = signature_objective(sim, self.obs, self.tau_target)
            return {"objective": res.objective, "parts": res.parts,
                    "diagnostics": res.diagnostics,
                    "params": {k: v[1] for k, v in params.items()},
                    "runoff_mm": sim_mm_yr, "peak_swe": float(peak_swe)}
        except subprocess.TimeoutExpired:
            return {"objective": FAIL_PENALTY, "reason": "timeout"}
        except Exception as e:
            return {"objective": FAIL_PENALTY, "reason": f"exc:{type(e).__name__}:{e}"}
        finally:
            shutil.rmtree(work, ignore_errors=True)
