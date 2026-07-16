#!/usr/bin/env python3
"""
signature_objective.py — Aggregate-signature objective for the memory-hypothesis calibration.

The model is an instrument for a memory hypothesis, not a water-resource predictor, so we
calibrate to AGGREGATE SIGNATURES, not daily KGE/NSE (those are reported to the SI only).
See CALIBRATION_SPEC.md.

Five equal-weighted signatures (each returned as a normalized error, 0 = perfect):
  1. center-of-mass timing   — directional t_Q (circular); mean |Δ| days, normalized by 7 d
  2. monthly volume          — Σ|sim_m − obs_m| / Σ obs_m over the 12-month mean hydrograph
  3. concentration           — directional R (peakiness); replaces AMJJ (redundant w/ monthly)
  4. baseflow recession      — |τ_sim − τ_target| / τ_target  (τ_target: East 27 d, Tuo 15 d)
  5. Aug–Sep low flow        — |Δ| / obs of Aug–Sep total volume (the memory signal)

Guards are hard constraints (return True/False), evaluated by the driver — a failed guard
rejects the trial with a fixed large penalty rather than trading against the objective.

All streamflow inputs are pandas Series with a daily DatetimeIndex, in consistent units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from objective_functions import (
    center_of_mass_error,
    monthly_dist_bias,
    kge as _kge_cost,
    nse as _nse_cost,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _wy(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.year + (idx.month >= 10).astype(int)


def _season_volume(q: pd.Series, months: tuple) -> pd.Series:
    """Per-water-year summed volume over the given calendar months."""
    q = q.dropna()
    sel = q[q.index.month.isin(months)]
    return sel.groupby(_wy(sel.index)).sum()


def flow_centroid_day_by_wy(q: pd.Series) -> pd.Series:
    """Flow-weighted center-of-mass timing per water year (Stewart et al. 'center of timing').

        t_Qhat = sum(t * Q(t)) / sum(Q(t))

    t = day of water year (1 = Oct 1). This is the FIRST MOMENT (centroid) of the
    hydrograph -- NOT the 50%-cumulative median. For right-skewed snowmelt hydrographs
    the centroid sits later than the median (the long recession tail pulls the mean),
    so the two are not interchangeable.
    """
    q = q.dropna()
    q = q[q > 0]
    if len(q) == 0:
        return pd.Series(dtype=float)
    wy = q.index.year + (q.index.month >= 10).astype(int)
    # Day of WATER year, Oct 1 -> t=1, computed from the actual WY start date.
    # (A dayofyear-offset shortcut is off by one in leap years: Oct 1 is dayofyear 275,
    #  not 274, once Feb 29 has passed.)
    wy_start = pd.to_datetime(pd.Series(wy - 1, index=q.index).astype(str) + "-10-01")
    t = (q.index - pd.DatetimeIndex(wy_start)).days + 1
    df = pd.DataFrame({"q": q.values, "wy": wy, "t": t})

    def _ct(g):
        s = g["q"].sum()
        return np.nan if s <= 0 else float((g["t"] * g["q"]).sum() / s)

    return df.groupby("wy", group_keys=False).apply(_ct, include_groups=False).dropna()


def directional_stats_by_wy(q: pd.Series) -> pd.DataFrame:
    """Circular (directional) statistics of streamflow timing, per water year.

        xbar = (1/sum Q) * sum( cos(2*pi*t) * Q(t) )
        ybar = (1/sum Q) * sum( sin(2*pi*t) * Q(t) )
        t_Q  = atan2(ybar, xbar) / (2*pi)      -> center-of-mass timing, fraction of year
        R    = sqrt(xbar^2 + ybar^2)           -> concentration (0 = uniform, 1 = one instant)

    t is the fraction of the WATER year elapsed. Unlike the linear centroid, this has no
    artificial discontinuity at the Oct 1 boundary (flow on Sep 30 vs Oct 1 sits adjacent on
    the circle, not at opposite ends), so it is less sensitive to where the year is cut.

    Returns a DataFrame indexed by water year with columns [tQ_days, R].
    """
    q = q.dropna()
    q = q[q > 0]
    if len(q) == 0:
        return pd.DataFrame(columns=["tQ_days", "R"])
    wy = np.asarray(q.index.year + (q.index.month >= 10).astype(int))
    wy_start = pd.to_datetime(pd.Series(wy - 1, index=q.index).astype(str) + "-10-01")
    doy = np.asarray((q.index - pd.DatetimeIndex(wy_start)).days, dtype=float)
    qv = np.asarray(q.values, dtype=float)
    rows = {}
    for y in np.unique(wy):
        m = wy == y
        n = float((pd.Timestamp(f"{y}-10-01") - pd.Timestamp(f"{y-1}-10-01")).days)
        t = doy[m] / n
        Q = qv[m]
        s = Q.sum()
        if s <= 0:
            continue
        xb = float((np.cos(2 * np.pi * t) * Q).sum() / s)
        yb = float((np.sin(2 * np.pi * t) * Q).sum() / s)
        tq = (np.arctan2(yb, xb) / (2 * np.pi)) % 1.0
        rows[int(y)] = (tq * n, float(np.sqrt(xb ** 2 + yb ** 2)))
    return pd.DataFrame(rows, index=["tQ_days", "R"]).T


def centroid_timing_error(sim: pd.Series, obs: pd.Series, signed: bool = False):
    """Mean (signed or absolute) centroid-timing difference, days."""
    s = flow_centroid_day_by_wy(sim)
    o = flow_centroid_day_by_wy(obs)
    common = s.index.intersection(o.index)
    if len(common) == 0:
        return float("nan")
    d = s.loc[common] - o.loc[common]
    return float(d.mean()) if signed else float(d.abs().mean())


def recession_constant(q: pd.Series,
                       months: tuple = (8, 9, 10, 11, 12, 1, 2, 3),
                       min_len: int = 5) -> float:
    """Master baseflow recession time constant tau (days).

    Isolated to the baseflow-dominated season (Aug–Mar) to exclude the melt limb.
    On each falling limb of >= `min_len` consecutive days, fit ln(Q) vs t; tau =
    -1/slope.  Returns the median tau over all qualifying limbs (robust to noise).
    """
    q = q.dropna()
    q = q[q.index.month.isin(months)]
    q = q[q > 0]
    if len(q) < min_len:
        return float("nan")
    v = q.values
    falling = np.r_[False, np.diff(v) < 0]
    taus = []
    i = 0
    n = len(v)
    while i < n:
        if falling[i]:
            j = i
            while j < n and falling[j]:
                j += 1
            seg = v[i - 1:j] if i > 0 else v[i:j]
            if len(seg) >= min_len and np.all(seg > 0):
                t = np.arange(len(seg))
                slope = np.polyfit(t, np.log(seg), 1)[0]
                if slope < 0:
                    taus.append(-1.0 / slope)
            i = j
        else:
            i += 1
    return float(np.median(taus)) if taus else float("nan")


# --------------------------------------------------------------------------- #
# the five signature errors
# --------------------------------------------------------------------------- #
def sig_com_timing(sim: pd.Series, obs: pd.Series, norm_days: float = 7.0) -> float:
    """Directional (circular) center-of-mass timing error, normalized by the 'good enough'
    target. norm_days = 7 -> a 7-day mean error scores 1.0, pushing the optimizer to <=7 d.

    Uses the circular t_Q (no water-year-boundary seam) rather than the linear centroid.
    """
    s = directional_stats_by_wy(sim)["tQ_days"]
    o = directional_stats_by_wy(obs)["tQ_days"]
    common = s.index.intersection(o.index)
    if len(common) == 0:
        return 3.0
    d = float((s.loc[common] - o.loc[common]).abs().mean())
    return float(np.clip(d / norm_days, 0.0, 3.0))


def sig_concentration(sim: pd.Series, obs: pd.Series) -> float:
    """Seasonal concentration (directional R) error, relative.

    R is the circular concentration of flow: 0 = spread evenly through the year, 1 = all
    flow at one instant.  Captures hydrograph PEAKINESS -- information no volume signature
    carries (a model can hit Apr-Jul volume exactly while being far too flat).  Replaces the
    AMJJ-volume signature, which was largely redundant with `monthly_volume`.
    """
    s = directional_stats_by_wy(sim)["R"]
    o = directional_stats_by_wy(obs)["R"]
    common = s.index.intersection(o.index)
    if len(common) == 0 or o.loc[common].mean() <= 0:
        return 3.0
    err = float((s.loc[common] - o.loc[common]).abs().mean() / o.loc[common].mean())
    return float(np.clip(err, 0.0, 3.0))


def sig_monthly_volume(sim: pd.Series, obs: pd.Series) -> float:
    return float(monthly_dist_bias(sim, obs))


def sig_amjj_runoff(sim: pd.Series, obs: pd.Series) -> float:
    s = _season_volume(sim, (4, 5, 6, 7)).mean()
    o = _season_volume(obs, (4, 5, 6, 7)).mean()
    if not np.isfinite(o) or o <= 0:
        return 3.0
    return float(np.clip(abs(s - o) / o, 0.0, 3.0))


def sig_augsep_lowflow(sim: pd.Series, obs: pd.Series) -> float:
    s = _season_volume(sim, (8, 9)).mean()
    o = _season_volume(obs, (8, 9)).mean()
    if not np.isfinite(o) or o <= 0:
        return 3.0
    return float(np.clip(abs(s - o) / o, 0.0, 3.0))


def sig_recession(sim: pd.Series, tau_target_days: float) -> float:
    tau = recession_constant(sim)
    if not np.isfinite(tau):
        return 3.0
    return float(np.clip(abs(tau - tau_target_days) / tau_target_days, 0.0, 3.0))


# --------------------------------------------------------------------------- #
# combined objective
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {
    "com_timing": 1.0,
    "monthly_volume": 1.0,
    "concentration": 1.0,      # directional R (replaced amjj_runoff)
    "recession": 1.0,
    "augsep_lowflow": 1.0,
}


@dataclass
class SignatureResult:
    objective: float
    parts: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)  # KGE/NSE etc. for SI


def signature_objective(sim: pd.Series,
                        obs: pd.Series,
                        tau_target_days: float,
                        weights: Optional[Dict[str, float]] = None) -> SignatureResult:
    """Equal-weighted (by default) normalized-error objective; lower is better."""
    w = weights or DEFAULT_WEIGHTS
    joint = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna()
    if len(joint) < 60:
        return SignatureResult(objective=1e6, parts={"error": np.nan})
    s, o = joint["s"], joint["o"]
    parts = {
        "com_timing":     sig_com_timing(s, o),       # directional t_Q
        "monthly_volume": sig_monthly_volume(s, o),
        "concentration":  sig_concentration(s, o),    # directional R (peakiness)
        "recession":      sig_recession(s, tau_target_days),
        "augsep_lowflow": sig_augsep_lowflow(s, o),
    }
    wsum = sum(w[k] for k in parts)
    obj = sum(w[k] * parts[k] for k in parts) / wsum
    diag = {
        "KGE": 1.0 - _kge_cost(s.values, o.values),
        "NSE": 1.0 - _nse_cost(s.values, o.values),
        "logNSE": 1.0 - _nse_cost(np.log(s.values + 0.01), np.log(o.values + 0.01)),
        "tau_sim_days": recession_constant(s),
        "R_sim": float(directional_stats_by_wy(s)["R"].mean()),
        "R_obs": float(directional_stats_by_wy(o)["R"].mean()),
    }
    return SignatureResult(objective=float(obj), parts=parts, diagnostics=diag)


# --------------------------------------------------------------------------- #
# guards (hard constraints; True = pass)
# --------------------------------------------------------------------------- #
def guard_runoff_ratio(sim_mm_yr: float, precip_mm_yr: float,
                       lo: float = 0.2, hi: float = 0.9) -> bool:
    if precip_mm_yr <= 0:
        return False
    return lo < (sim_mm_yr / precip_mm_yr) < hi


def guard_snow(peak_swe_sim: float, peak_swe_apriori: float,
               lo: float = 0.6, hi: float = 1.4) -> bool:
    if peak_swe_apriori <= 0:
        return False
    return lo <= (peak_swe_sim / peak_swe_apriori) <= hi


if __name__ == "__main__":
    # smoke test on synthetic data
    idx = pd.date_range("2014-10-01", "2019-09-30", freq="D")
    rng = np.random.default_rng(0)
    base = 2 + 3 * np.exp(-((idx.dayofyear - 160) ** 2) / (2 * 40 ** 2))
    obs = pd.Series(base * (1 + 0.1 * rng.standard_normal(len(idx))), index=idx).clip(0.1)
    sim = pd.Series(base * 0.9, index=idx).clip(0.1)
    r = signature_objective(sim, obs, tau_target_days=30.0)
    print("objective:", round(r.objective, 3))
    print("parts:", {k: round(v, 3) for k, v in r.parts.items()})
    print("diagnostics:", {k: round(v, 3) for k, v in r.diagnostics.items()})
