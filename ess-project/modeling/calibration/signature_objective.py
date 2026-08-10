#!/usr/bin/env python3
"""
signature_objective.py — Aggregate-signature objective for the memory-hypothesis calibration.

The model is an instrument for a memory hypothesis, not a water-resource predictor, so we
calibrate to AGGREGATE SIGNATURES, not daily KGE/NSE (those are reported to the SI only).
See CALIBRATION_SPEC.md.

Five equal-weighted signatures (each returned as a normalized error, 0 = perfect):
  1. center-of-mass timing   — directional t_Q (circular); mean |Δ| days, normalized by 7 d
  2. monthly volume          — Σ|sim_m − obs_m| / Σ obs_m over the 12-month mean hydrograph
  3. May–Jun–Jul volume      — mean |Δ| / obs of the freshet; replaced `concentration`
                               2026-07-19 because no existing term could see a -25 mm June
                               deficit (monthly_volume netted it against a March surplus)
  4. baseflow recession      — |τ_sim − τ_target| / τ_target  (τ_target: East 27 d, Tuo 15 d)
  5. Sep–Nov low flow        — mean |Δ log10| of the annual 7-day minimum (the memory signal),
                               in LOG space so a drying stream keeps being penalised

All series are clamped to Q_FLOOR (1e-2 m3/s), the gauge measurement floor, before scoring.

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

# Measurement floor (m3 s-1). Discharge below this is not physically resolvable by a gauge,
# so a simulated 1e-9 (or 1e-15) is not "drier" than 1e-2 in any meaningful sense -- it is the
# same answer, "no measurable flow". Without this the Tuolumne optimum drove the stream to
# ~0 for 61% of days and logNSE to -12, because unbounded-small values were treated as real.
# Applied to BOTH sim and obs at the single entry point so every signature and diagnostic sees
# the same floored series.
Q_FLOOR = 1e-2


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


def sig_monthly_volume_significant(sim: pd.Series, obs: pd.Series,
                                   rel_thresh: float = 0.10,
                                   abs_thresh_mm: float = 4.0,
                                   norm: float = 0.10,
                                   area_m2: float = None,
                                   exclude_months: tuple = ()) -> float:
    """Volume-SIGNIFICANT monthly bias: fraction of annual volume misallocated, normalized.

    A month is counted only if it fails BOTH tests:
        |bias| / obs_month  > rel_thresh      (10%)   -- proportionally wrong, AND
        |bias|              > abs_thresh_mm   (4 mm)  -- and big enough to matter
    Score = (sum of |bias| over qualifying months) / (annual obs volume) / norm, clipped at 3.
    So 10% of annual volume misallocated in months that matter scores 1.0.

    Replaces the old `monthly_dist_bias` (2026-07-19), which had three fatal flaws: it
    normalized BOTH series by their own annual total (so it was completely blind to volume
    bias -- East sat at -4.2% and it reported nothing), it averaged over all 12 months
    including six near-zero winter ones, and it expressed errors as fractions-of-annual. The
    net effect was that East's -23.8 mm June error (the single largest, a 4.7-point shift of
    annual volume) scored 0.0094 overall and never moved during calibration.

    The two-sided filter is the point: a December at 50% bias but 1 mm is noise and must not
    drive the optimizer, while a June at 18% and 24 mm must. Scored on the MEAN ANNUAL CYCLE
    (climatological monthly means), so there is no water-year boundary to wrap around.

    Units: sim/obs are m3/s. If `area_m2` is given, bias is converted to mm for the absolute
    test; otherwise the absolute test is skipped (relative test only) -- so ALWAYS pass area.
    """
    sm = sim.groupby(sim.index.month).sum()
    om = obs.groupby(obs.index.month).sum()
    common = sm.index.intersection(om.index)
    if len(common) == 0 or om.loc[common].sum() <= 0:
        return 3.0
    nyr = max(len(sim) / 365.25, 1e-9)
    to_mm = (86400.0 * 1000.0 / area_m2 / nyr) if area_m2 else None
    qual = 0.0
    for m in common:
        if m in exclude_months:
            # Months the FORCING cannot support. Scoring them makes the optimiser chase an
            # unreachable target and distort parameters that matter elsewhere. Tuolumne:
            # Dec-Mar excluded (2026-07-20) -- PRISM does not resolve the magnitude of large
            # winter rain events, leaving a ~70 mm Dec-Mar deficit no parameter can close.
            # Those months were 45% of this metric and 29% of the WHOLE objective, five times
            # the weight on the Sep-Nov low-flow period that is the scientific target.
            # NOTE: the denominator stays the FULL-year observed volume, so the score keeps
            # its meaning -- "fraction of annual water misallocated in the months we trust".
            continue
        b = float(sm[m] - om[m])
        if om[m] <= 0:
            continue
        if abs(b / om[m]) <= rel_thresh:
            continue
        if to_mm is not None and abs(b * to_mm) <= abs_thresh_mm:
            continue                      # proportionally bad but volumetrically trivial
        qual += abs(b)
    return float(np.clip(qual / float(om.loc[common].sum()) / norm, 0.0, 3.0))


def sig_mjj_volume(sim: pd.Series, obs: pd.Series, months: tuple = (5, 6, 7)) -> float:
    """May-June-July freshet volume error, per water year, mean ABSOLUTE relative error.

    Added 2026-07-19 in place of `concentration`, to attack the melt-season volume deficit
    directly. The existing signatures demonstrably cannot see it: East scores
    monthly_volume = 0.012 (near-perfect) while running a -25 mm June deficit, because a
    distribution-bias statistic nets a June shortfall against a March surplus. Forcing A/B
    tests showed the deficit is a VOLUME problem -- three separate temperature treatments left
    June at -52 to -64 mm, and only adding high-elevation precip moved it -- so it needs a
    signature that responds to volume in exactly those months.

    May/Jun/Jul of calendar year Y all fall inside water year Y, so calendar grouping is
    already water-year-correct here (unlike the Sep-Nov low-flow block). Absolute per-year
    error so a model that is high some years and low others cannot cancel to zero.
    """
    def mjj(q):
        q = q.dropna()
        q = q[q.index.month.isin(months)]
        if len(q) == 0:
            return pd.Series(dtype=float)
        return q.groupby(q.index.year).sum()
    s, o = mjj(sim), mjj(obs)
    common = s.index.intersection(o.index)
    if len(common) < 2 or o.loc[common].mean() <= 0:
        return 3.0
    err = float((s.loc[common] - o.loc[common]).abs().mean() / o.loc[common].mean())
    return float(np.clip(err, 0.0, 3.0))


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


def sig_lowflow_sepnov(sim: pd.Series, obs: pd.Series,
                       months: tuple = (9, 10, 11),
                       over_weight: float = 1.0) -> float:
    """Fall baseflow low-flow error, Sep-Nov, indexed by CALENDAR year.

    Why Sep-Nov by calendar year (not Aug-Sep, not water year):
      * By Sep all melt is done (even the highest HRUs melt out by ~mid-June), so this window
        is pure baseflow recession -- no melt tail contaminating it (Aug can still carry melt).
      * Before freeze-up, so obs are not ice-affected like the true winter minimum.
      * Sep-Nov of calendar year Y is three CONSECUTIVE months. A water-year label would split
        the block (Sep ends WY_Y, Oct-Nov start WY_(Y+1)) and glue non-consecutive months.
      * The block is the residual store draining out after WY_Y's melt -- the memory signal.

    Metric: per-calendar-year 7-day-minimum flow, then the mean over years of the absolute
    error in LOG10 space, normalised so one full decade of error = 1.0. Absolute (not pooled)
    so a model that is too high some years and too low others cannot cancel -- this catches the
    interannual OVER-variance that a pooled mean hides.

    Why log space (changed 2026-07-19): the previous RELATIVE error saturated. For a stream
    that dries up, |sim-obs|/obs -> 1.0 no matter how dry it gets, so 1e-9 and 1e-2 scored
    identically and DE had NO GRADIENT to climb back out. Tuolumne exploited exactly this: it
    drained the basin (61% of days below the measurement floor, lowflow pinned at 0.998 for 10
    generations) because going dry cost at most 1.0 while BUYING a near-perfect recession
    (0.008). In log space the error keeps growing as flow falls, so drying is properly punished
    and every step back toward observed low flow pays.
    """
    def annual_min7(q):
        q = q.dropna()
        q = q[q.index.month.isin(months)]
        if len(q) == 0:
            return pd.Series(dtype=float)
        min7 = q.rolling(7, min_periods=4).mean()
        return min7.groupby(min7.index.year).min()      # by CALENDAR year
    s = annual_min7(sim)
    o = annual_min7(obs)
    common = s.index.intersection(o.index)
    if len(common) < 2 or o.loc[common].mean() <= 0:
        return 3.0
    # log10 error, floored so log() is always defined; 1.0 == one decade off, clipped at 3
    sv = np.log10(np.maximum(s.loc[common].values, Q_FLOOR))
    ov = np.log10(np.maximum(o.loc[common].values, Q_FLOOR))
    e = sv - ov                                   # POSITIVE = model too HIGH
    # ASYMMETRIC penalty (over_weight > 1, added 2026-07-20, Tuolumne only).
    # Over-prediction of late-season low flow is the DANGEROUS error direction for the memory
    # experiments: a model that retains melt water and dribbles it out through the fall
    # manufactures exactly the storage-driven signal the imposed-IC runs are meant to detect,
    # so an over-wet basin inflates the apparent memory. Running slightly dry is conservative --
    # it understates memory, making any detected signal more credible. Tuolumne sat too high in
    # 9 of 11 years (mean x1.54, up to x3.5), which is that failure mode.
    # Normalised by the MEAN of the two weights so the asymmetry changes the DIRECTION of the
    # pull without silently inflating its magnitude on top of the per-domain weight:
    #   over_weight=1 -> exactly the old symmetric mean|e| (East is untouched)
    #   over_weight=2 -> overshoot costs 1.33x, undershoot 0.67x; optimum sits near the 33rd
    #                    percentile of observed low flow, i.e. "slightly too low".
    w = np.where(e > 0, float(over_weight), 1.0)
    err = float((w * np.abs(e)).mean() / ((float(over_weight) + 1.0) / 2.0))
    return float(np.clip(err, 0.0, 3.0))


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
    "concentration": 1.0,      # directional R (peakiness) -- restored 2026-07-20
    "recession": 1.0,
    "lowflow_sepnov": 1.0,     # Sep-Nov fall baseflow, per-calendar-year (replaced augsep_lowflow)
}


@dataclass
class SignatureResult:
    objective: float
    parts: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)  # KGE/NSE etc. for SI


def signature_objective(sim: pd.Series,
                        obs: pd.Series,
                        tau_target_days: float,
                        weights: Optional[Dict[str, float]] = None,
                        area_m2: Optional[float] = None,
                        exclude_months: tuple = (),
                        lowflow_over_weight: float = 1.0) -> SignatureResult:
    """Equal-weighted (by default) normalized-error objective; lower is better."""
    w = weights or DEFAULT_WEIGHTS
    joint = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna()
    if len(joint) < 60:
        return SignatureResult(objective=1e6, parts={"error": np.nan})
    # clamp to the gauge measurement floor before ANY signature or diagnostic is computed
    s, o = joint["s"].clip(lower=Q_FLOOR), joint["o"].clip(lower=Q_FLOOR)
    parts = {
        "com_timing":     sig_com_timing(s, o),       # directional t_Q
        "monthly_volume": sig_monthly_volume_significant(s, o, area_m2=area_m2,
                                                         exclude_months=exclude_months),
        "concentration":  sig_concentration(s, o),    # directional R (peakiness)
        "recession":      sig_recession(s, tau_target_days),
        "lowflow_sepnov": sig_lowflow_sepnov(s, o, over_weight=lowflow_over_weight),   # Sep-Nov fall baseflow, per-calendar-year
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


def min7_median(q: pd.Series) -> float:
    """Median across years of the annual 7-day-minimum flow (m3 s-1)."""
    q = q.dropna()
    if len(q) < 60:
        return float("nan")
    m7 = q.rolling(7, min_periods=4).mean()
    return float(m7.groupby(m7.index.year).min().median())


def guard_lowflow(sim_min7: float, obs_min7: float, lo: float = 0.10) -> bool:
    """Reject trials whose stream effectively dries out.

    The log-space lowflow signature gives DE a GRADIENT back toward observed low flow, but
    nothing stops it trading the stream away outright -- Tuolumne's pre-fix optimum sat below
    the measurement floor 61% of days because drying bought a near-perfect recession. This is
    the hard floor that makes a dry basin infeasible rather than merely penalised.

    Deliberately lenient (10% of the observed annual 7-day minimum): a strict threshold on a
    basin that genuinely struggles to hold water would reject the whole population and leave DE
    with no gradient at all, which is a worse failure than a soft constraint.
    """
    if not np.isfinite(sim_min7) or not np.isfinite(obs_min7) or obs_min7 <= 0:
        return True                      # cannot judge -> do not reject
    return (sim_min7 / obs_min7) >= lo


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
