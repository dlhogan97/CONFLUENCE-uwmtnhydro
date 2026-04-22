#!/usr/bin/env python3
"""
objective_functions.py — Anchor + coherence metrics for staged SUMMA calibration.

All functions take numpy arrays and return a scalar float (lower = better, i.e.,
these are *cost* functions to be minimized).

Reference
---------
Gupta, H.V. et al. (2009). Decomposition of the MSE and NSE performance criteria.
J. Hydrol. 377, 80–91. doi:10.1016/j.jhydrol.2009.08.003
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import lfilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level metrics
# ---------------------------------------------------------------------------

def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency (Gupta et al. 2009).  Returns 1 - KGE as a cost."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[mask], obs[mask]
    if len(s) < 10 or np.std(o) == 0:
        return 1.0  # worst cost

    r = np.corrcoef(s, o)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o)
    kge_val = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(1.0 - kge_val)  # cost: 0 = perfect


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency. Returns 1 - NSE as a cost."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[mask], obs[mask]
    if len(s) < 10 or np.var(o) == 0:
        return 1.0
    nse_val = 1.0 - np.sum((s - o) ** 2) / np.sum((o - np.mean(o)) ** 2)
    return float(1.0 - nse_val)


def nrmse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Normalised RMSE: RMSE / mean(obs).  Returns scalar cost (≥0)."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[mask], obs[mask]
    if len(s) == 0 or np.mean(o) == 0:
        return 1.0
    return float(np.sqrt(np.mean((s - o) ** 2)) / np.mean(np.abs(o)))

def total_bias(sim, obs):
    denom = max(obs.sum(), 1e-12)
    return abs(sim.sum() - obs.sum()) / denom  # lower is better

def monthly_dist_bias(sim, obs):
    # sim/obs should be pandas Series with DatetimeIndex
    sim_m = sim.resample("MS").sum().groupby(sim.index.month).sum()
    obs_m = obs.resample("MS").sum().groupby(obs.index.month).sum()

    sim_f = (sim_m / max(sim_m.sum(), 1e-12)).reindex(range(1, 13), fill_value=0.0)
    obs_f = (obs_m / max(obs_m.sum(), 1e-12)).reindex(range(1, 13), fill_value=0.0)
    return (sim_f - obs_f).abs().mean()  # lower is better

def monthly_plus_total(sim, obs, w_monthly=0.8):
    return w_monthly * monthly_dist_bias(sim, obs) + (1 - w_monthly) * total_bias(sim, obs)

def _peak_and_meltout(swe: np.ndarray, meltout_threshold_mm: float = 10.0) -> Tuple[int, float, int]:
    """Return (peak_day_idx, peak_value, meltout_day_idx) for a single SWE timeseries."""
    if np.all(~np.isfinite(swe)) or np.nanmax(swe) < meltout_threshold_mm:
        return 0, 0.0, 0
    peak_idx = int(np.nanargmax(swe))
    peak_val = float(swe[peak_idx])
    # Melt-out: first day after peak where SWE drops below threshold
    post_peak = swe[peak_idx:]
    below = np.where(post_peak < meltout_threshold_mm)[0]
    meltout_idx = peak_idx + int(below[0]) if len(below) > 0 else len(swe) - 1
    return peak_idx, peak_val, meltout_idx


# ---------------------------------------------------------------------------
# Baseflow separation (Eckhardt digital filter)
# ---------------------------------------------------------------------------

def eckhardt_baseflow(q: np.ndarray, bfi_max: float = 0.8, a: float = 0.98) -> np.ndarray:
    """Two-parameter digital Eckhardt baseflow filter.

    Parameters
    ----------
    q:      streamflow array [m³/s or mm/d]
    bfi_max: maximum baseflow index (0.8 for perennial streams)
    a:      recession constant (0.98 is typical)
    """
    q = np.where(np.isfinite(q), np.maximum(q, 0.0), 0.0)
    b = np.zeros_like(q)
    b[0] = q[0] * bfi_max
    for t in range(1, len(q)):
        num = (1 - bfi_max) * a * b[t - 1] + (1 - a) * bfi_max * q[t]
        denom = 1 - a * bfi_max
        b[t] = min(num / denom, q[t])
    return b


# ---------------------------------------------------------------------------
# Stage 1: Snow anchor metric
# ---------------------------------------------------------------------------

def compute_anchor_snow(
    sim_swe: np.ndarray,
    obs_swe: np.ndarray,
    w_kge: float = 0.5,
    w_peak: float = 0.3,
    w_timing: float = 0.2,
) -> float:
    """Anchor cost for snow stage.

    Combines KGE on the full SWE timeseries, normalised peak magnitude error,
    peak timing error, and melt-out timing error.
    """
    # Align lengths
    n = min(len(sim_swe), len(obs_swe))
    sim, obs = sim_swe[:n], obs_swe[:n]

    # KGE cost
    kge_cost = kge(sim, obs)

    # Peak diagnostics
    sim_peak_day, sim_peak_val, sim_meltout = _peak_and_meltout(sim)
    obs_peak_day, obs_peak_val, obs_meltout = _peak_and_meltout(obs)

    if obs_peak_val == 0:
        return kge_cost  # no observed SWE peak; fall back to KGE only

    nrmse_peak = abs(sim_peak_val - obs_peak_val) / obs_peak_val
    timing_error = abs(sim_peak_day - obs_peak_day) / 30.0   # normalise by ~1 month
    meltout_error = abs(sim_meltout - obs_meltout) / 30.0

    combined_timing = 0.5 * timing_error + 0.5 * meltout_error

    cost = w_kge * kge_cost + w_peak * nrmse_peak + w_timing * combined_timing
    return float(np.clip(cost, 0.0, 5.0))


# ---------------------------------------------------------------------------
# Stage 2: Soil / ET anchor metric
# ---------------------------------------------------------------------------

def compute_anchor_et(
    sim_et_monthly: np.ndarray,
    obs_et_monthly: np.ndarray,
) -> float:
    """Anchor cost for ET stage (monthly means).  Returns 1 - KGE."""
    return kge(sim_et_monthly, obs_et_monthly)


def compute_anchor_streamflow_rising(
    sim_q: np.ndarray,
    obs_q: np.ndarray,
) -> float:
    """Anchor cost for soil stage — emphasises rising limb via log-KGE."""
    n = min(len(sim_q), len(obs_q))
    s, o = sim_q[:n], obs_q[:n]
    # Use log transform to emphasise rising-limb and low-flow timing
    eps = 1e-3 * np.nanmean(o)
    return kge(np.log(s + eps), np.log(o + eps))


# ---------------------------------------------------------------------------
# Stage 3: Groundwater / baseflow anchor metric
# ---------------------------------------------------------------------------

def compute_anchor_baseflow(
    sim_q: np.ndarray,
    obs_q: np.ndarray,
    bfi_max: float = 0.65,
    a_recession: float = 0.99,
) -> float:
    """Anchor cost for groundwater stage.

    Separates baseflow from observed Q with Eckhardt filter and compares
    simulated aquifer baseflow to that estimate.
    """
    n = min(len(sim_q), len(obs_q))
    s, o = sim_q[:n], obs_q[:n]
    obs_bf = eckhardt_baseflow(o, bfi_max, a_recession)
    return kge(s, obs_bf)


# ---------------------------------------------------------------------------
# Generic metric dispatcher
# ---------------------------------------------------------------------------

def compute_anchor_generic(
    sim: np.ndarray,
    obs: np.ndarray,
    metric: str = "KGE",
) -> float:
    """Compute anchor cost using the named metric.

    Parameters
    ----------
    sim, obs : arrays of simulated and observed values (same units, aligned).
    metric   : one of "KGE", "KGE_log", "NSE", "NRMSE".

    Returns a cost value (lower = better, 0 = perfect fit).
    """
    m = metric.upper().replace("-", "_")
    if m == "KGE":
        return kge(sim, obs)
    elif m == "KGE_LOG":
        valid = np.isfinite(obs) & (obs > 0)
        if valid.sum() < 10:
            return 1.0
        eps = 1e-3 * float(np.nanmean(obs[valid]))
        return kge(np.log(sim + eps), np.log(obs + eps))
    elif m == "NSE":
        return nse(sim, obs)
    elif m == "NRMSE":
        return nrmse(sim, obs)
    elif m == "MONTHLY_PLUS_TOTAL":
        return monthly_plus_total(sim, obs)
    else:
        raise ValueError(
            f"Unknown anchor_metric {metric!r}. "
            "Choose from: KGE, KGE_log, NSE, NRMSE, MONTHLY_PLUS_TOTAL"
        )


# ---------------------------------------------------------------------------
# Stage 3: Soil / runoff anchor metric
# ---------------------------------------------------------------------------

def compute_anchor_runoff(
    sim_q: np.ndarray,
    obs_q: np.ndarray,
    metric: str = "KGE",
) -> float:
    """Anchor cost for soil stage — total streamflow using the named metric."""
    return compute_anchor_generic(sim_q, obs_q, metric)


# ---------------------------------------------------------------------------
# Stage 4: Routing anchor metric
# ---------------------------------------------------------------------------

def compute_anchor_streamflow(
    sim_q: np.ndarray,
    obs_q: np.ndarray,
    metric: str = "KGE",
) -> float:
    """Anchor cost for routing stage — full hydrograph."""
    return compute_anchor_generic(sim_q, obs_q, metric)


# ---------------------------------------------------------------------------
# Coherence metrics (no point observations required)
# ---------------------------------------------------------------------------

def compute_coherence_snow(
    all_hru_swe: Dict[int, np.ndarray],
    hru_elevations: Dict[int, float],
    hru_aspects: Optional[Dict[int, str]] = None,
) -> float:
    """Coherence cost for snow stage.

    Handles both pure-elevation and elevation×aspect discretizations.

    For elevation×aspect runs, multiple HRUs share the same elevation band
    (one per aspect class).  The ordering is: highest elevation first, then
    aspects cycle within each band (e.g. N, E, W, S, flat).

    Penalises:
    1. Violation of *band-level* elevation ordering: mean peak SWE date in a
       higher-elevation band should be ≥ that in the next lower band.
    2. Within-band aspect ordering: south-facing HRUs should have an earlier
       melt-out date than north-facing HRUs at the same elevation.
    3. Low pairwise correlation between normalised HRU SWE shapes.
    4. SWE spread (CV of peak SWE) outside the physically plausible range.

    Parameters
    ----------
    all_hru_swe:
        Dict of hru_index → SWE timeseries array.
    hru_elevations:
        Dict of hru_index → mean elevation (m).  Multiple HRUs may share the
        same elevation value (they are in the same band, different aspects).
    hru_aspects:
        Optional dict of hru_index → aspect label string.
        Expected labels: "N", "E", "S", "W", "flat" (case-insensitive).
        When provided, within-band aspect melt ordering is penalised.
    """
    from collections import defaultdict

    hru_ids = sorted(all_hru_swe.keys())
    n = len(hru_ids)
    if n < 2:
        return 0.0

    # --- Per-HRU peak / melt-out diagnostics ---
    peak_days: Dict[int, int] = {}
    peak_vals: Dict[int, float] = {}
    meltout_days: Dict[int, int] = {}
    for hid in hru_ids:
        pd_, pv, md = _peak_and_meltout(all_hru_swe[hid])
        peak_days[hid] = pd_
        peak_vals[hid] = pv
        meltout_days[hid] = md

    # --- 1. Band-level elevation ordering ---
    # Group HRUs by their elevation value.  For pure-elevation runs, each
    # HRU has a unique elevation; for elevation×aspect, groups ≥ 1.
    bands: dict = defaultdict(list)
    for hid in hru_ids:
        bands[hru_elevations.get(hid, 0.0)].append(hid)

    # Ascending elevation order (low → high)
    sorted_elevs = sorted(bands.keys())
    band_mean_peak = [
        float(np.mean([peak_days[h] for h in bands[e]]))
        for e in sorted_elevs
    ]
    band_mean_melt = [
        float(np.mean([meltout_days[h] for h in bands[e]]))
        for e in sorted_elevs
    ]

    # Higher elevation → later peak/melt-out → ascending sequence
    n_bands = len(sorted_elevs)
    ordering_violations = sum(
        1 for i in range(n_bands - 1)
        if band_mean_peak[i] > band_mean_peak[i + 1]
        or band_mean_melt[i] > band_mean_melt[i + 1]
    )
    j_ordering = ordering_violations / max(n_bands - 1, 1)

    # --- 2. Within-band aspect ordering (only when hru_aspects provided) ---
    j_aspect = 0.0
    if hru_aspects and n_bands > 0:
        # Expected melt order within a band: S melts first (most solar), N melts last.
        # Assign a numeric rank: lower rank → earlier expected melt.
        aspect_melt_rank = {"s": 0, "e": 1, "w": 2, "flat": 3, "n": 4}
        aspect_violations = 0
        aspect_pairs = 0
        for e in sorted_elevs:
            hrus_in_band = bands[e]
            if len(hrus_in_band) < 2:
                continue
            # Sort by expected melt order (S first)
            ranked = sorted(
                hrus_in_band,
                key=lambda h: aspect_melt_rank.get(
                    str(hru_aspects.get(h, "flat")).lower(), 3
                ),
            )
            # Earlier aspect rank → earlier (smaller) observed meltout day
            for i in range(len(ranked) - 1):
                h_early, h_late = ranked[i], ranked[i + 1]
                # Allow a 7-day grace window before penalising
                if meltout_days[h_early] > meltout_days[h_late] + 7:
                    aspect_violations += 1
                aspect_pairs += 1
        j_aspect = aspect_violations / max(aspect_pairs, 1)

    # --- 3. Normalised shape correlation across all HRUs ---
    normed = []
    for hid in hru_ids:
        swe = all_hru_swe[hid]
        std = np.nanstd(swe)
        normed.append((swe - np.nanmean(swe)) / std if std > 0 else np.zeros_like(swe))

    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            length = min(len(normed[i]), len(normed[j]))
            if length < 5:
                continue
            c = np.corrcoef(normed[i][:length], normed[j][:length])[0, 1]
            if np.isfinite(c):
                corrs.append(c)
    j_shape = 1.0 - (float(np.mean(corrs)) if corrs else 0.0)

    # --- 4. SWE spread (CV of peak SWE across all HRUs) ---
    pv_arr = np.array([peak_vals[h] for h in hru_ids])
    mean_pv = np.mean(pv_arr)
    cv = float(np.std(pv_arr) / mean_pv) if mean_pv > 0 else 0.0
    # Penalise CV outside [0.2, 1.5]
    j_spread = max(0.0, 0.2 - cv) + max(0.0, cv - 1.5)

    # Weights: distribute across the four sub-costs.
    # aspect ordering only contributes when hru_aspects is provided.
    if hru_aspects:
        return float(0.35 * j_ordering + 0.20 * j_aspect + 0.25 * j_shape + 0.20 * j_spread)
    else:
        return float(0.40 * j_ordering + 0.30 * j_shape + 0.30 * j_spread)


def compute_coherence_et(
    all_hru_et: Dict[int, np.ndarray],
    hru_landcover: Dict[int, str],
    time_step_hours: float = 1.0,
) -> float:
    """Coherence cost for ET stage.

    Penalises:
    1. Inverted seasonal cycle (ET summer < ET winter)
    2. Annual total ET outside physically plausible range [200–600 mm]
    """
    steps_per_day = 24.0 / time_step_hours
    steps_per_year = 365.25 * steps_per_day

    costs = []
    for hid, et_raw in all_hru_et.items():
        # SUMMA reports ET/sublimation as negative (water leaving surface = loss).
        # Negate to get physical ET as positive flux in kg m-2 s-1.
        et = -np.where(np.isfinite(et_raw), et_raw, 0.0)
        n = len(et)
        if n < int(steps_per_year * 0.5):
            continue  # too short

        # Seasonal cycle check: summer index = days 152–243 (Jun 1–Sep 1)
        summer_start = int(152 * steps_per_day)
        summer_end = int(243 * steps_per_day)
        winter_start = int(335 * steps_per_day)   # Dec 1
        winter_end = min(int(60 * steps_per_day), n)  # end of Feb next year

        summer_et = np.mean(et[summer_start:summer_end]) if summer_end <= n else np.nan
        winter_et = (
            np.mean(np.concatenate([et[winter_start:], et[:winter_end]]))
            if winter_start < n else np.nan
        )

        season_cost = 0.0
        if np.isfinite(summer_et) and np.isfinite(winter_et) and winter_et >= 0:
            ratio = summer_et / (winter_et + 1e-12)
            if ratio < 3.0:
                season_cost = (3.0 - ratio) / 3.0  # penalise inverted/flat cycle

        # Annual total: ET is in kg m-2 s-1; sum × step_seconds → mm/yr (= kg/m²/yr)
        step_seconds = time_step_hours * 3600.0
        annual_mm = float(np.sum(et[:int(steps_per_year)])) * step_seconds
        range_cost = max(0.0, 200.0 - annual_mm) / 200.0 + max(0.0, annual_mm - 600.0) / 600.0

        costs.append(0.6 * season_cost + 0.4 * range_cost)

    return float(np.mean(costs)) if costs else 0.0


def compute_coherence_baseflow(
    all_hru_bf: Dict[int, np.ndarray],
    sim_q_total: Optional[np.ndarray] = None,
    all_hru_total_runoff: Optional[Dict[int, np.ndarray]] = None,
    target_bfi_range: Tuple[float, float] = (0.1, 1.0),
) -> float:
    """Coherence cost for groundwater stage.

    Checks that the baseflow index (BFI = baseflow/total_runoff) is within
    the configured acceptable range (default 0.1-1.0).
    """
    if not all_hru_bf:
        return 0.0

    costs = []
    for hid, bf in all_hru_bf.items():
        if all_hru_total_runoff is not None and hid in all_hru_total_runoff:
            total_series = all_hru_total_runoff[hid]
            n = min(len(bf), len(total_series))
            total = np.nansum(np.abs(total_series[:n]))
        elif sim_q_total is not None:
            n = min(len(bf), len(sim_q_total))
            total = np.nansum(np.abs(sim_q_total[:n]))
        else:
            continue
        if not np.isfinite(total) or total <= 0:
            bfi = 0.0
        else:
            bfi = np.nansum(np.abs(bf[:n])) / total
        lo, hi = target_bfi_range
        costs.append(max(0.0, lo - bfi) + max(0.0, bfi - hi))

    # Recession consistency: all HRUs should have similar log-linear slope
    if len(all_hru_bf) > 1:
        slopes = []
        for bf in all_hru_bf.values():
            bf_ = np.where(bf > 0, bf, np.nan)
            log_bf = np.where(np.isfinite(bf_), np.log(bf_), np.nan)
            if np.sum(np.isfinite(log_bf)) > 10:
                x = np.arange(len(log_bf))
                valid = np.isfinite(log_bf)
                if valid.sum() > 3:
                    slope = np.polyfit(x[valid], log_bf[valid], 1)[0]
                    slopes.append(slope)
        if len(slopes) > 1:
            slope_cv = np.std(slopes) / (abs(np.mean(slopes)) + 1e-12)
            costs.append(min(slope_cv, 1.0))

    return float(np.mean(costs)) if costs else 0.0


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------

def combined_objective(
    anchor: float,
    coherence: float,
    w_anchor: float = 0.4,
) -> float:
    """Weighted sum of anchor and coherence costs (lower = better).

    Parameters
    ----------
    w_anchor:
        Weight for the anchor metric. Default 0.4 (coherence weight = 0.6).
        Increase toward 1.0 when you have dense, reliable observations.
    """
    w_coherence = 1.0 - w_anchor
    return float(w_anchor * anchor + w_coherence * coherence)
