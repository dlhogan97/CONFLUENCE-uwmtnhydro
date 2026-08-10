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

from signature_objective import (signature_objective, guard_runoff_ratio, guard_snow,
                                 guard_lowflow, min7_median)

SUMMA_EXE = "/home/dlhogan/bin/summa.exe"
FAIL_PENALTY = 10.0


def _esat_kpa(T):
    """Saturation vapour pressure (kPa) over water, Magnus/Tetens. T in K."""
    Tc = T - 273.15
    return 0.6108 * np.exp(17.27 * Tc / (Tc + 237.3))


def _lw_dilley_obrien(Tair, p, q):
    """Dilley & O'Brien (1998) clear-sky downwelling LW (W m-2) from T, p, q.

    Used ONLY to form an INCREMENT when temperature is adjusted -- never to replace the stored
    LWRadAtm, which carries cloud/bias structure this clear-sky formula does not reproduce.
    """
    MV_CST = 0.622
    e_0 = (q * (p / 1000.0)) / (MV_CST + q * (1 - MV_CST))          # vapour pressure (kPa)
    return (59.38 + 113.7 * (Tair / 273.16) ** 6
            + 96.96 * np.sqrt((4650 * e_0) / (25.0 * Tair)))


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
# qSurfScale added both (saturation-excess surface-runoff = fast/slow partition). Capped at 50.
# zScale_TOPMODEL added to East (requires hc_profile=pow_prof, now set) for depth-decaying K.
#
# rootingDepth is NOT calibrated: it is fixed to each HRU's soil-column depth (see
# set_soil_ramp), so roots thin with elevation along with the soil and never reach past the
# column. Previously 2.495 m uniform against <=1.5 m of soil, which let East transpire straight
# out of the aquifer.
#
# aquiferBaseflowExp is East-only: it appears solely in bigAquifer.f90 (Qb = rate*(S/scale)^exp),
# so under Tuolumne's qTopmodl it is inert -- calibrating it there just wasted a dimension.
#
# kAnisotropic is calibrated in BOTH, but note it does DIFFERENT work in each:
#   East     bcLowrSoiH='drainage' -> freeDrainage: scalarDrainage = nodeHydCond*kAnisotropic
#            i.e. a vertical soil->aquifer RECHARGE multiplier  (soilLiqFlx.f90:1714)
#   Tuolumne groundwatr=qTopmodl:  tran0 = kAnisotropic*surfaceHydCond*soilDepth/zScale_TOPMODEL
#            i.e. the LATERAL transmissivity scale             (groundwatr.f90:405)
# It is not redundant with k_soil in either: k_soil scales conductivity everywhere, while
# kAnisotropic scales only the lateral/bottom term -- together they set the ratio. They are
# correlated in the drainage term, though, so expect a soft ridge between them.
#
# frozenPrecipMultip is DISTRIBUTED as a linear ramp in elevation:
#     fpm_i = fpm_low + fpm_delta * znorm_i,     znorm = (elev-elev_min)/(elev_max-elev_min)
# Rationale: snowfall gauge undercatch grows with elevation (wind, colder, more snow-phase),
# and adding snow HIGH shifts the melt centroid later far more efficiently than a uniform
# multiplier -- which can only move volume and timing together. fpm_delta >= 0 by
# construction, so the ramp can only increase with elevation (encodes the physical prior).
PARAM_SPECS: Dict[str, List[ParamSpec]] = {
    # k_soil is NOT calibrated: pegged to its calibrated value (1.342e-4 m/s, uniform) in
    # trialParams. It is uniform across HRUs (single soil class, and trialParams overrides the
    # ROSETTA class value -- verified: summa_setup reads trialParams AFTER pOverwrite), so
    # calibrating a uniform scalar bought nothing that a fixed value doesn't. Fixing it drops a
    # dimension and is cleaner to defend ("calibrated, then fixed").
    # STAGE 2 (snow-timing only). Stage 1 calibrated the hydrology over 16 generations and
    # plateaued at objective 0.3758 (KGE 0.841, tau 25.8 d, recession 0.045) with com_timing
    # stuck at ~1.55 -- 83% of the remaining objective. Root cause: albedoDecayRate was set to
    # 1.2e6 in localParamInfo but trialParams held a stale 8.0e5, and trialParams WINS
    # (summa_setup.f90: read_pinit L182 -> pOverwrite L227 -> read_param L249). So the melt
    # knob we thought we had set was silently pinned the whole time.
    #
    # Stage 1's hydrology is now PEGGED in trialParams (qSurfScale, zScale_TOPMODEL,
    # kAnisotropic, aquiferBaseflow{Rate,Exp}, aquiferScaleFactor, routingGammaScale) and
    # dropped from the spec, leaving DE only the knobs that move melt timing.
    "East_River": [
        # albedoDecayRate is NOT calibrated: PEGGED at 680193 (its stage-2 optimum) in
        # trialParams. Stage 2 gave DE free rein over [5e5, 5e7] -- two orders of magnitude,
        # 10x past the physical ceiling -- and it returned 6.8e5, essentially the value it
        # started from, while com_timing moved only 1.557 -> 1.534. Melt timing is simply not
        # reachable through albedo decay, so the dimension was spent for nothing.
        # NOTE: it had to be written into trialParams explicitly; the source file still held
        # the stale 8.0e5, so merely dropping it from the spec would have SILENTLY REVERTED it.
        #
        # frozenPrecipMultip floor raised 0.85 -> 1.0 (2026-07-19): gauge undercatch of snowfall
        # is one-signed, so the multiplier should never be allowed below unity. Combined with
        # delta this spans 1.0 at the base to at most 1.4 at the crest.
        ParamSpec("frozenPrecipMultip_low",  "elev", 1.00, 1.20),
        ParamSpec("frozenPrecipMultip_delta","elev", 0.0, 0.40),
        # STAGE 3 -- forcing bias correction as tunable parameters (2026-07-19).
        #
        # tempLapse: East's forcing lapse measured -2.607 C/km, less than HALF the -6.5 C/km
        # standard environmental lapse (Tuolumne is -3.289, same pathology -> systematic in the
        # PRISM/METSIM chain). The profile is rotated about the LOWEST HRU:
        #     dT(z) = (L/1000 - lapse_ref) * (z - z_min)
        # Anchoring low (not at the area-weighted mean) is deliberate: a mean pivot makes the
        # basin-mean temperature algebraically invariant to L, so it could only redistribute
        # heat, never correct a bulk bias. Anchored low, one knob does both -- e.g. L=-6.5
        # gives crest -4.2 C and basin-mean -1.9 C, reproducing almost exactly the ad-hoc
        # uniform -2 C that closed 62% of East's center-of-timing error in the A/B tests.
        # Range brackets inversion (-2) through super-adiabatic (-12); dry adiabatic is -9.8,
        # so treat an optimum beyond that as compensating for some OTHER error, not as a
        # defensible atmospheric lapse rate.
        # tempLapse DROPPED after stage 3: DE explored the full [-12,-2] range and cooling was
        # monotonically harmful (r(lapse, com_timing) = -0.858; com saturated at 3.0 = >=21 d
        # for lapse < -7). Cause was the z_min anchor -- rotating about the lowest HRU means the
        # LOW elevations are never cooled, but the March surplus (+66%) is a low/mid-elevation
        # early-melt error, so the parameter could only hammer the high country and shove water
        # past June. The spatial distribution wants to be left alone; the BULK bias may not.
        #
        # tempOffset: uniform shift, the thing the lapse structurally could not do. A hand A/B
        # of uniform -2 C cut East's centroid error 13.6 -> 5.1 d and fixed July exactly
        # (-35.6 -> -0.2 mm), but that was a 3-yr window with a -8.0% volume bias vs -3.7% on
        # the full record, so it needs testing over the whole period before it is believed.
        ParamSpec("tempOffset",              "forcing", -4.0, 1.0),
        # precipRampHigh DROPPED 2026-07-20: DE explored its full [0, 0.30] range and drove the
        # best trials to ~0.014. Its apparent value in the WY1995-97 A/B was a window artifact
        # (that window carried a -8.0% volume bias vs -3.7% on the full record, so there was
        # room to add water that the full record does not have).
        #
        # ---- RE-OPENED for a sensitivity check (2026-07-20) --------------------------------
        # These five were pegged from stage 1/2, but BOTH of those stages ran under conditions
        # that no longer hold: the old objective (dead `monthly_volume`, no Q_FLOOR, no
        # log-lowflow) and -- for stage 1 -- a silently stale albedoDecayRate of 8.0e5. So the
        # pegs may be carrying compensation for a melt regime that no longer exists rather than
        # representing their own physics. Warm-started AT the pegged values (mult seeds to 1.0,
        # absolutes to their trialParams value), so this asks "do they stay?" not "start over".
        # aquiferScaleFactor is the prime suspect: stage 1 moved it 2.0 -> 3.245 (+62%), exactly
        # the shape of extra storage buffering a melt pulse arriving too sharply.
        ParamSpec("k_soil",                  "mult", 0.1, 10.0, log=True),
        ParamSpec("albedoDecayRate",         "abs",  1e5, 5e6, log=True),
        ParamSpec("aquiferBaseflowRate",     "abs",  1e-9, 1e-4, log=True),
        ParamSpec("aquiferScaleFactor",      "abs",  0.5, 5.0),
        ParamSpec("routingGammaScale",       "abs",  *_routing_scale_bounds(16, 36)),
    ],
    # Tuolumne DOES calibrate k_soil: unlike East it has never been calibrated, so there is no
    # stable value to peg to yet. Fix it after this run, once it settles (as we did for East).
    "Tuolumne_River": [
        ParamSpec("k_soil",                  "mult", 0.1, 10.0, log=True),
        # qSurfScale RE-OPENED 2026-07-20 after being pegged the same morning. It was pegged on
        # a 0.15 correlation with an objective in which late-season low flow carried only 6% of
        # the weight and was scored SYMMETRICALLY -- that correlation was measuring almost
        # nothing relevant. qSurfScale sets the saturation-excess fast/slow partition, i.e. it
        # is the mechanistic lever for getting melt water out in June rather than dribbling it
        # through the fall (Jun -40.5 mm deficit reappears as a +21 mm Jul-Oct surplus: one
        # mechanism, not two). With lowflow_sepnov now weighted x3 AND asymmetric, it has a
        # real gradient to respond to.
        ParamSpec("qSurfScale",              "abs",  1.0, 50.0, log=True),
        # PEGGED in trialParams 2026-07-20, from the gen_03 best of the 8-param run:
        #   zScale_TOPMODEL   = 1.781   routingGammaScale = 24864
        # Evidence over 88 feasible trials (relative position in bounds, 0=low 1=high):
        #   zScale_TOPMODEL    top-quartile IQR 0.11, corr w/ objective  0.04, and pinned at
        #                      0.03-0.14 in EVERY generation -- flat and uncorrelated. It was
        #                      also the dominant lowflow_dry driver (AUC 0.86): no feasible
        #                      trial exceeded ~2.3 while the bound ran to 8, so ~80% of that
        #                      dimension was infeasible dead space burning whole SUMMA runs.
        #   qSurfScale         top-quartile IQR 0.13, corr 0.15, stable ~0.5 every generation.
        #   routingGammaScale  corr 0.04 and never converges (0.60->0.58->0.43->0.29->0.88):
        #                      wide scatter with no objective response = insensitive.
        # Freeing 8 -> 5 params cuts the DE population 64 -> 40 trials/generation AND removes
        # the main infeasibility source. CAUTION: the source trialParams.nc held STALE values
        # (3.33 / 10.0 / 28800) which silently override localParamInfo -- they were rewritten
        # to the pegged values above, not just deleted from this spec.
        ParamSpec("kAnisotropic",            "abs",  0.01, 10.0, log=True),
        # albedoDecayRate PEGGED at 126754 (2026-07-20). Opened over [5e4, 5e6] it was the only
        # Tuolumne parameter to converge -- top-quintile IQR [8.6e4, 1.35e5], 10% of the search
        # range -- and it came OFF the 5e4 lower bound it was pinned to in gen 0, so the range
        # was not binding. Direction was opposite to East (Tuolumne's melt was ~16 d LATE, so it
        # needed FASTER decay = smaller value; East needed slower).
        #
        # frozenPrecipMultip_low floor deliberately NOT raised to 1.0 as it was for East. East
        # needs MORE snow (gauge undercatch); Tuolumne demonstrably has too much low-elevation
        # snow (400-500 mm at 1607 m, melting out DOWY ~205), and its optimum sits at 0.950.
        # Forcing >=1.0 here would push the known error further the wrong way.
        # delta ceiling raised 0.40 -> 0.60: it reached 0.392 against the old bound in gen 3.
        ParamSpec("frozenPrecipMultip_low",  "elev", 0.85, 1.20),
        ParamSpec("frozenPrecipMultip_delta","elev", 0.0, 0.60),
        # tempOffset replaces the hardcoded +2 K forcing set (SUMMA_input_dt2), which was a
        # magic number picked by hand off a 3-year A/B. Forcing reverts to the ORIGINAL
        # SUMMA_input and the offset becomes a calibrated parameter with a reportable value.
        # Range is warm-shifted vs East's [-4, +1] because the basins need opposite corrections:
        # East melts ~10 d EARLY (wants cooling), Tuolumne melted ~16 d LATE (wants warming).
        # The seed carries +2.0 so the warm start reproduces the current state exactly.
        ParamSpec("tempOffset",              "forcing", -2.0, 4.0),
    ],
}

TAU_TARGET = {"East_River": 27.0, "Tuolumne_River": 15.0}

# Months dropped from the monthly_volume signature because the FORCING cannot support them.
# Tuolumne Dec-Mar (2026-07-20): PRISM does not resolve the magnitude of large winter rain
# events, so the model runs 5-6 mm/month against an observed 24-41 and no parameter closes it.
# Measured on the gen_03 best: Dec-Mar was 45% of monthly_volume, which is 63% of the
# objective -> 29% of the entire objective was an unreachable forcing artefact, versus 6% on
# lowflow_sepnov (the late-fall period that is the scientific target). East excludes nothing.
MONTHLY_EXCLUDE = {"East_River": (), "Tuolumne_River": (12, 1, 2, 3)}

# Per-domain signature weights (None = equal, the DEFAULT_WEIGHTS in signature_objective).
# Tuolumne up-weights lowflow_sepnov x3 (2026-07-20): the Sep-Nov recession is the scientific
# target of the memory experiments, but at equal weight it carried only 6% of the objective
# (8.9% after the Dec-Mar exclusion) -- less than a third of the weight sitting on melt-season
# volume. x3 lifts it to ~23%, making it the clear second term behind monthly_volume without
# letting it dominate. Safe against the lowflow_dry guard: the model currently runs too WET
# there (7-day min too high in 8 of 10 years), and the guard only trips below 10% of observed,
# so there is a wide feasible band to move down into before infeasibility becomes a risk.
SIG_WEIGHTS = {
    "East_River": None,
    "Tuolumne_River": {"com_timing": 1.0, "monthly_volume": 1.0, "concentration": 1.0,
                       "recession": 1.0, "lowflow_sepnov": 3.0},
}

# Asymmetry on the Sep-Nov low-flow signature: >1 penalises OVER-prediction more than under.
# Tuolumne 2.0 (2026-07-20): it ran too high in 9 of 11 years (mean x1.54, max x3.5), and an
# over-wet late season manufactures the storage memory the imposed-IC experiments test for.
# Running slightly dry is the conservative error. 2.0 puts the optimum near the 33rd percentile
# of observed low flow. East stays 1.0 (symmetric, unchanged).
LOWFLOW_OVER_WEIGHT = {"East_River": 1.0, "Tuolumne_River": 2.0}

# Measured 2026-07-19: area-weighted regression of Nov-Jun mean airtemp on HRU elevation.
# BOTH basins' forcing is anomalously FLAT with elevation -- the standard environmental lapse
# is -6.5 C/km and moist adiabatic is ~-5, so the PRISM/METSIM chain is under-lapsing by a
# factor of ~2. This is the reference the tunable `tempLapse` parameter rotates away from.
# Note -6.5 C/km alone yields a basin-mean dT of -1.88 C on East, i.e. almost exactly the
# ad-hoc uniform -2 C that empirically closed 62% of East's center-of-timing error.
CURRENT_LAPSE = {"East_River": -2.607e-3, "Tuolumne_River": -3.289e-3}   # C per m


class CalibrationRunner:
    def __init__(self, domain: str, sim_start: str, sim_end: str,
                 analysis_start: str, obs_csv: str, work_root: str,
                 precip_mm_yr: float, peak_swe_apriori: float):
        self.domain = domain
        self.specs = PARAM_SPECS[domain]
        self.tau_target = TAU_TARGET[domain]
        self.monthly_exclude = MONTHLY_EXCLUDE[domain]
        self.sig_weights = SIG_WEIGHTS[domain]
        self.lowflow_over_weight = LOWFLOW_OVER_WEIGHT[domain]
        self.sim_start, self.sim_end = sim_start, sim_end
        self.analysis_start = analysis_start          # drop spin-up before this
        self.settings = Path(f"/scratch/dlhogan/ess-project-data/"
                             f"domain_{domain}_distributed_elevAspect/settings/SUMMA")
        # Forcing subdir is overridable so a bias-corrected forcing set can be swapped in
        # WITHOUT touching the original (every existing product stays reproducible).
        # Tuolumne 2026-07-19: SUMMA_input_dt2 = uniform +2 K airtemp, RH-preserving spechum,
        # LWRadAtm recomputed (Dilley-O'Brien). Set FORCING_SUBDIR to change.
        _sub = os.environ.get("FORCING_SUBDIR", "SUMMA_input")
        self.forcing = self.settings.parent.parent / "forcing" / _sub
        with xr.open_dataset(self.settings / "attributes.nc") as _a:
            _area = _a["HRUarea"].values.astype(float)
            _elev = _a["elevation"].values.astype(float)
        self.area = float(_area.sum())
        self.hru_wt = _area / _area.sum()
        # normalized elevation (0 at lowest HRU, 1 at highest) for the frozen-precip ramp
        self.elev_norm = ((_elev - _elev.min()) / (_elev.max() - _elev.min())
                          if _elev.max() > _elev.min() else np.zeros_like(_elev))
        # absolute elevation + reference lapse, for the forcing-level temperature rotation.
        # Anchored at the LOWEST HRU (not the area-weighted mean): pivoting at the mean would
        # make basin-mean temperature algebraically invariant to the lapse (sum w*(z-zbar)=0),
        # so the parameter could only redistribute, never correct a bulk warm bias. Anchoring
        # low lets one parameter deliver both the gradient and the bulk shift.
        self.elev = _elev
        self.elev_ref = float(_elev.min())
        self.lapse_ref = CURRENT_LAPSE[domain]
        with xr.open_dataset(self.settings / "trialParams.nc") as _tp:
            self.fieldcap = float(np.ravel(_tp["fieldCapacity"].values)[0])
            # a-priori per-HRU fields for the multiplier parameters (loaded once)
            self.base_vals = {s.name: _tp[s.name].values.copy()
                              for s in self.specs if s.kind == "mult" and s.name in _tp}
            # a-priori scalar value of every spec'd param, so a seed written before a param was
            # added can be carried forward instead of silently falling back to a cold start.
            self.apriori = {s.name: float(np.ravel(_tp[s.name].values)[0])
                            for s in self.specs if s.name in _tp}
        # 'forcing'-kind params live in the forcing files, not trialParams, so they have no
        # a-priori there. Without an explicit entry migrate_seed cannot fill them and the whole
        # seed is rejected -> silent COLD START. Defaults are the identity transform: the
        # measured lapse (no rotation) and no precip ramp.
        for _s in self.specs:
            if _s.kind == "forcing" and _s.name not in self.apriori:
                self.apriori[_s.name] = {"tempLapse": self.lapse_ref * 1000.0,
                                         "tempOffset": 0.0,
                                         "precipRampHigh": 0.0}.get(_s.name, 0.0)
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
                if kind in ("elev", "forcing") or name not in ds.variables:
                    continue     # 'elev' pair handled below; 'forcing' in _apply_forcing
                if kind == "mult":
                    ds.variables[name][:] = self.base_vals[name] * val
                else:
                    ds.variables[name][:] = val
            # elevation-ramped frozen-precip multiplier (undercatch grows with elevation)
            if "frozenPrecipMultip_low" in params:
                lo = params["frozenPrecipMultip_low"][1]
                dl = params["frozenPrecipMultip_delta"][1]
                ds.variables["frozenPrecipMultip"][:] = lo + dl * self.elev_norm

    def _apply_forcing(self, work: Path, params) -> Path:
        """Build per-trial bias-corrected forcing if any 'forcing'-kind params are active.

        tempLapse (C/km)   rotate the temperature profile about the LOWEST HRU:
                             dT(z) = (L/1000 - lapse_ref) * (z - z_min)
                           A steeper lapse therefore cools the high country hard and the basin
                           mean modestly -- both the gradient and the bulk shift from one knob.
        precipRampHigh (-) linear precip multiplier, 1.0 at the base -> 1+R at the crest.

        Consistency: spechum is rescaled to PRESERVE RH, and LWRadAtm receives the
        Dilley-O'Brien INCREMENT only. It is never replaced with a raw DO field -- the stored
        LW sits +92 W/m2 above DO on East (+26 on Tuolumne), so replacement would delete far
        more energy than the adjustment adds. Returns the forcing dir to point SUMMA at.
        """
        lapse = params.get("tempLapse", (None, None))[1]
        offset = params.get("tempOffset", (None, None))[1]
        pramp = params.get("precipRampHigh", (None, None))[1]
        if lapse is None and offset is None and pramp is None:
            return self.forcing

        dT = None
        if lapse is not None or offset is not None:
            dT = np.zeros_like(self.elev, dtype=float)
            if lapse is not None:      # rotate about the lowest HRU
                dT += (lapse / 1000.0 - self.lapse_ref) * (self.elev - self.elev_ref)
            if offset is not None:     # uniform bulk shift
                dT += float(offset)
        span = self.elev.max() - self.elev.min()
        pm = (1.0 + pramp * (self.elev - self.elev.min()) / span) if pramp is not None else None

        fdir = work / "forcing"
        fdir.mkdir(parents=True, exist_ok=True)
        y0, m0 = int(self.sim_start[:4]), int(self.sim_start[5:7])
        y1, m1 = int(self.sim_end[:4]), int(self.sim_end[5:7])
        for f in sorted(self.forcing.glob("*.nc")):
            stem = f.name.split("_")[-1].replace(".nc", "")
            try:
                yy, mm = int(stem[:4]), int(stem[4:6])
            except ValueError:
                (fdir / f.name).symlink_to(f); continue
            if not ((y0, m0) <= (yy, mm) <= (y1, m1)):
                (fdir / f.name).symlink_to(f)          # outside sim window: cheap symlink
                continue
            with xr.open_dataset(f) as _d:
                ds = _d.load().copy()
            if dT is not None:
                T0 = ds["airtemp"].values
                q0 = ds["spechum"].values
                p = ds["airpres"].values
                T1 = T0 + dT[None, :]
                q1 = q0 * (_esat_kpa(T1) / _esat_kpa(T0))          # preserve RH
                ds["airtemp"].values[:] = T1
                ds["spechum"].values[:] = q1
                ds["LWRadAtm"].values[:] = (ds["LWRadAtm"].values
                                            + _lw_dilley_obrien(T1, p, q1)
                                            - _lw_dilley_obrien(T0, p, q0))
            if pm is not None:
                ds["pptrate"].values[:] = ds["pptrate"].values * pm[None, :]
            ds.to_netcdf(fdir / f.name)
            ds.close()
        return fdir

    def _wet_soil(self, work: Path):
        with nc.Dataset(work / "coldState.nc", "a") as ds:
            ds.variables["mLayerVolFracLiq"][:] = self.fieldcap
            # keep matric head consistent-ish: leave as-is (SUMMA re-derives from theta)

    def _write_filemanager(self, work: Path, prefix: str, forcing: Path = None):
        (work / "fileManager.txt").write_text(
            f"controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n"
            f"simStartTime         '{self.sim_start} 00:00'\n"
            f"simEndTime           '{self.sim_end} 23:00'\n"
            f"tmZoneInfo           'localTime'\n"
            f"outFilePrefix        '{prefix}'\n"
            f"settingsPath         '{work}/'\n"
            f"forcingPath          '{forcing or self.forcing}/'\n"
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
            fdir = self._apply_forcing(work, params)
            self._write_filemanager(work, prefix, forcing=fdir)
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
            # dry-stream guard: the log-space lowflow signature gives DE a gradient, but only
            # this makes a dried-out basin INFEASIBLE rather than cheaply penalised.
            sim_min7, obs_min7 = min7_median(sim), min7_median(self.obs)
            if not guard_lowflow(sim_min7, obs_min7):
                return {"objective": FAIL_PENALTY, "reason": "lowflow_dry",
                        "min7": sim_min7, "obs_min7": obs_min7}
            res = signature_objective(sim, self.obs, self.tau_target, area_m2=self.area,
                                      exclude_months=self.monthly_exclude,
                                      weights=self.sig_weights,
                                      lowflow_over_weight=self.lowflow_over_weight)
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
