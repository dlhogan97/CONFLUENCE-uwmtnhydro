"""Brutsaert-Nieber recession analysis: -dQ/dt vs Q.

Rather than fitting Q against time, plot the recession rate ``-dQ/dt`` against Q
itself on log-log axes, pooling every screened recession day, and fit

    -dQ/dt = a * Q^b        (linear in log-log:  ln(-dQ/dt) = ln a + b * ln Q)

The exponent ``b`` is the diagnostic:

- ``b ~ 1``  -> linear reservoir (constant timescale tau = 1/a; single exponential),
  an *independent* confirmation of the MRC linear-reservoir assumption.
- ``b > 1``  -> nonlinear storage-discharge; the drainage timescale depends on flow.

The physically meaningful object is the **lower envelope** of the cloud: points
above it carry ongoing input (rain/melt adds flow, pushing -dQ/dt up), so the
bottom edge is the pure-drainage signal. We fit that envelope (a low quantile per
log-Q bin) by default, and also offer a full-cloud OLS fit for comparison.

Splitting the cloud by wetness state tests the drought hypothesis: if wet-period
and drought-period recessions fall on *separate* envelopes (same b, different
intercept -> different effective tau), the lumped tau drifts with storage state
without the reservoir ceasing to look linear within any period.

Caveats (see README): ``-dQ/dt`` from daily data is finite-difference noisy near
the flat low-Q tail, and the fitted ``b`` is sensitive to envelope-extraction
choices -- sweep ``envelope.*`` the way the MRC thresholds were swept before
leaning on a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .recession_id import Recession, _water_year


@dataclass
class BNFit:
    a: float          # coefficient in -dQ/dt = a * Q^b
    b: float          # exponent (1 = linear reservoir)
    tau_ref: float    # effective timescale Q/(-dQ/dt) at the reference flow (days)
    q_ref: float      # reference flow at which tau_ref is evaluated
    r2: float         # r^2 of the log-log fit (on the fitted points)
    n_points: int     # points used in the fit (cloud points, or envelope bins)
    n_cloud: int      # cloud points in this group (before binning)
    method: str


@dataclass
class BNResult:
    cloud: pd.DataFrame            # per-pair: Q, minus_dQdt, seg_id, water_year, [wetness, group]
    overall: BNFit
    groups: Dict[str, BNFit] = field(default_factory=dict)
    envelope: pd.DataFrame = field(default_factory=pd.DataFrame)  # binned envelope pts (overall)
    q_ref: float = np.nan
    b_reference: float = 1.0       # the linear-reservoir reference exponent


# --------------------------------------------------------------------------- #
# Gauge quantization (resolution-aware noise floor)
# --------------------------------------------------------------------------- #
def infer_quantization(q_all: np.ndarray, n_bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Infer the gauge's reporting resolution as a function of Q, from the data.

    Stream gauges report ~constant significant figures, so the *absolute* discrete
    step coarsens with flow while relative precision stays roughly fixed. Within
    quantile bins of Q we take the smallest nonzero gap between distinct reported
    values as the local step.

    Returns ``(edges, steps)`` where ``steps[i]`` is the resolution for
    ``edges[i] <= Q < edges[i+1]``.
    """
    q = np.asarray(q_all, dtype=float)
    q = q[np.isfinite(q) & (q > 0)]
    if q.size < 3:
        return np.array([0.0, np.inf]), np.array([0.0])

    edges = np.unique(np.quantile(q, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return np.array([0.0, np.inf]), np.array([0.0])
    edges[0], edges[-1] = 0.0, np.inf

    steps = np.full(edges.size - 1, np.nan)
    for i in range(edges.size - 1):
        vals = np.unique(q[(q >= edges[i]) & (q < edges[i + 1])])
        if vals.size < 3:
            continue
        gaps = np.diff(vals)
        gaps = gaps[gaps > 1e-12]
        if gaps.size:
            steps[i] = gaps.min()

    # Fill any empty bins from neighbours (forward then backward).
    if np.all(np.isnan(steps)):
        return edges, np.zeros(edges.size - 1)
    idx = np.arange(steps.size)
    good = ~np.isnan(steps)
    steps = np.interp(idx, idx[good], steps[good])
    return edges, steps


def step_for(Q: np.ndarray, edges: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Look up the quantization step applicable to each flow value."""
    i = np.clip(np.searchsorted(edges, Q, side="right") - 1, 0, steps.size - 1)
    return steps[i]


# --------------------------------------------------------------------------- #
# Cloud construction
# --------------------------------------------------------------------------- #
def recession_rate_cloud(recessions: List[Recession], cfg: Config) -> pd.DataFrame:
    """Build the -dQ/dt vs Q point cloud from screened recession segments.

    Uses a backward finite difference within each segment (never across segment
    boundaries/gaps): for consecutive days, ``-dQ/dt = (Q[i-1]-Q[i]) / dt`` at a
    reference flow (mean of the pair by default). Non-positive or negligible
    rates (flat-tail FD noise, upticks) are dropped via ``bn.dqdt.min_rate``.
    """
    dt = float(cfg.io.units.timestep_days)
    mode = cfg.bn.dqdt.q_reference
    min_rate = float(cfg.bn.dqdt.min_rate)
    min_q = float(getattr(cfg.bn.dqdt, "min_q", 0.0) or 0.0)

    Qs, rates, segs, wys, dates = [], [], [], [], []
    for r in recessions:
        Q = np.asarray(r.Q, dtype=float)
        if Q.size < 2:
            continue
        minus = -(np.diff(Q)) / dt              # -(Q[i]-Q[i-1])/dt  (>0 on a recession)
        if mode == "q_start":
            qref = Q[:-1]
        elif mode == "q_end":
            qref = Q[1:]
        else:                                    # mean_pair (Brutsaert-Nieber)
            qref = 0.5 * (Q[1:] + Q[:-1])
        Qs.append(qref)
        rates.append(minus)
        segs.append(np.full(minus.size, r.seg_id))
        wys.append(np.full(minus.size, r.water_year))
        dates.append(np.asarray(r.dates[1:]))

    if not Qs:
        return pd.DataFrame(columns=["date", "Q", "minus_dQdt", "seg_id", "water_year"])

    cloud = pd.DataFrame({
        "date": np.concatenate(dates),
        "Q": np.concatenate(Qs),
        "minus_dQdt": np.concatenate(rates),
        "seg_id": np.concatenate(segs),
        "water_year": np.concatenate(wys),
    })
    # `minus_dQdt > min_rate` is the positivity guard: a rise or flat step has no
    # logarithm and cannot be placed on log-log axes. `min_q` censors the noisy
    # low-flow region on the Q axis -- symmetric in -dQ/dt, so it does not truncate
    # the lower envelope the way a rate floor would.
    cloud = cloud[(cloud["Q"] >= min_q) & (cloud["Q"] > 0) & (cloud["minus_dQdt"] > min_rate)]

    # Optional resolution-aware floor (disabled by default): require the daily change
    # to span at least `min_rate_steps` gauge quantization steps.
    n_steps = float(getattr(cfg.bn.dqdt, "min_rate_steps", 0.0) or 0.0)
    if n_steps > 0 and len(cloud):
        edges, steps = infer_quantization(np.concatenate([r.Q for r in recessions]))
        floor = n_steps * step_for(cloud["Q"].to_numpy(), edges, steps) / dt
        cloud = cloud[cloud["minus_dQdt"] >= floor]

    return cloud.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Envelope + fit
# --------------------------------------------------------------------------- #
def lower_envelope(logQ: np.ndarray, logR: np.ndarray, *, n_bins: int,
                   quantile: float, min_bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin-center logQ, low-quantile logR) tracing the cloud's lower edge."""
    if logQ.size == 0:
        return np.array([]), np.array([])
    edges = np.linspace(logQ.min(), logQ.max(), n_bins + 1)
    idx = np.clip(np.digitize(logQ, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if int(m.sum()) < min_bin_count:
            continue
        xs.append(0.5 * (edges[b] + edges[b + 1]))
        ys.append(np.quantile(logR[m], quantile))
    return np.asarray(xs), np.asarray(ys)


def _tau_at(a: float, b: float, q_ref: float) -> float:
    """Effective drainage timescale tau = Q / (-dQ/dt) = Q^(1-b) / a at q_ref (days)."""
    if a <= 0 or not np.isfinite(a):
        return np.nan
    return float(q_ref ** (1.0 - b) / a)


def fit_bn(Q: np.ndarray, minus_dQdt: np.ndarray, cfg: Config,
           q_ref: Optional[float] = None) -> BNFit:
    """Fit ``-dQ/dt = a Q^b`` by the configured method; report a, b, and tau at q_ref."""
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(minus_dQdt, dtype=float)
    good = (Q > 0) & (R > 0) & np.isfinite(Q) & np.isfinite(R)
    Q, R = Q[good], R[good]
    n_cloud = Q.size
    method = cfg.bn.envelope.method
    if q_ref is None:
        q_ref = float(np.median(Q)) if n_cloud else np.nan
    if n_cloud < 2:
        return BNFit(np.nan, np.nan, np.nan, q_ref, np.nan, 0, n_cloud, method)

    x, y = np.log(Q), np.log(R)
    if method == "ols_cloud":
        fx, fy = x, y
    elif method == "lower_quantile":
        e = cfg.bn.envelope
        fx, fy = lower_envelope(x, y, n_bins=e.n_bins, quantile=e.quantile,
                                min_bin_count=e.min_bin_count)
    else:
        raise ValueError(f"Unknown bn.envelope.method '{method}'")

    if fx.size < 2:
        return BNFit(np.nan, np.nan, np.nan, q_ref, np.nan, int(fx.size), n_cloud, method)

    b, lna = np.polyfit(fx, fy, 1)
    pred = b * fx + lna
    ss_res = float(np.sum((fy - pred) ** 2))
    ss_tot = float(np.sum((fy - fy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    a = float(np.exp(lna))
    return BNFit(a=a, b=float(b), tau_ref=_tau_at(a, float(b), q_ref), q_ref=float(q_ref),
                 r2=float(r2), n_points=int(fx.size), n_cloud=n_cloud, method=method)


# --------------------------------------------------------------------------- #
# Wetness classification (drought test)
# --------------------------------------------------------------------------- #
def wetness_per_recession(recessions: List[Recession], cfg: Config,
                          df: Optional[pd.DataFrame]) -> Dict[int, float]:
    """A per-recession wetness scalar (higher = wetter) keyed by seg_id."""
    metric = cfg.bn.wetness.metric
    if metric == "recession_q_start":
        return {r.seg_id: float(r.Q[0]) for r in recessions}

    if df is None or "Q" not in df.columns:
        # Fall back to the recession's own starting flow if no full series given.
        return {r.seg_id: float(r.Q[0]) for r in recessions}

    wy = _water_year(df.index, cfg.temporal.window.water_year_start_month)
    agg = "min" if metric == "water_year_min_flow" else "mean"
    by_wy = pd.Series(df["Q"].to_numpy(), index=wy).groupby(level=0).agg(agg)
    return {r.seg_id: float(by_wy.get(r.water_year, np.nan)) for r in recessions}


def assign_groups(cloud: pd.DataFrame, wetness: Dict[int, float], cfg: Config) -> pd.DataFrame:
    """Attach per-point wetness value and a group label.

    Split modes (``bn.wetness.split``):
      - ``median``   -> wet / dry at the recession-weighted median of the metric.
      - ``terciles`` -> wet / mid / dry at the metric's 1/3 and 2/3 quantiles.
      - ``extremes`` -> the driest ``n_extreme`` and wettest ``n_extreme`` water
                        YEARS (ranked by the per-year metric); middle years become
                        ``unknown`` and are excluded from the group fits.
      - ``era``      -> ``pre<year>`` / ``post<year>`` at ``split_year`` (a temporal
                        split; the wetness metric is ignored).
    """
    w = cfg.bn.wetness
    cloud = cloud.copy()
    cloud["wetness"] = cloud["seg_id"].map(wetness)
    split = w.split

    if split == "era":
        yr = int(getattr(w, "split_year", 2000))
        cloud["group"] = np.where(cloud["water_year"] < yr, f"pre{yr}", f"post{yr}")
        return cloud

    if split == "extremes":
        n = int(getattr(w, "n_extreme", 10))
        yr_val = cloud.groupby("water_year")["wetness"].mean().dropna().sort_values()
        dry_years = set(yr_val.index[:n])
        wet_years = set(yr_val.index[-n:]) - dry_years  # disjoint if 2n > n_years
        cloud["group"] = cloud["water_year"].map(
            lambda y: "dry" if y in dry_years else ("wet" if y in wet_years else "unknown"))
        return cloud

    vals = pd.Series(wetness).dropna()
    if split == "terciles":
        lo, hi = vals.quantile([1 / 3, 2 / 3])
        def label(v):
            if not np.isfinite(v):
                return "unknown"
            return "dry" if v <= lo else ("wet" if v >= hi else "mid")
    else:  # median
        med = vals.median()
        def label(v):
            return "unknown" if not np.isfinite(v) else ("wet" if v >= med else "dry")
    cloud["group"] = cloud["wetness"].map(label)
    return cloud


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def bn_analysis(recessions: List[Recession], cfg: Config,
                df: Optional[pd.DataFrame] = None) -> BNResult:
    """Full Brutsaert-Nieber analysis: cloud, overall fit, and per-wetness fits.

    A single reference flow (median of the whole cloud, or configured) is used for
    every group's ``tau_ref`` so timescales are compared at the same Q.
    """
    cloud = recession_rate_cloud(recessions, cfg)
    if cloud.empty:
        return BNResult(cloud=cloud, overall=fit_bn(np.array([]), np.array([]), cfg))

    if cfg.bn.reference_flow == "value" and cfg.bn.reference_flow_value:
        q_ref = float(cfg.bn.reference_flow_value)
    else:
        q_ref = float(np.median(cloud["Q"]))

    overall = fit_bn(cloud["Q"].to_numpy(), cloud["minus_dQdt"].to_numpy(), cfg, q_ref=q_ref)

    ex, ey = lower_envelope(np.log(cloud["Q"].to_numpy()), np.log(cloud["minus_dQdt"].to_numpy()),
                            n_bins=cfg.bn.envelope.n_bins, quantile=cfg.bn.envelope.quantile,
                            min_bin_count=cfg.bn.envelope.min_bin_count)
    envelope = pd.DataFrame({"Q": np.exp(ex), "minus_dQdt": np.exp(ey)})

    groups: Dict[str, BNFit] = {}
    if cfg.bn.wetness.enabled:
        wet = wetness_per_recession(recessions, cfg, df)
        cloud = assign_groups(cloud, wet, cfg)
        for label, sub in cloud.groupby("group"):
            if label == "unknown":
                continue
            groups[label] = fit_bn(sub["Q"].to_numpy(), sub["minus_dQdt"].to_numpy(),
                                   cfg, q_ref=q_ref)

    return BNResult(cloud=cloud, overall=overall, groups=groups, envelope=envelope, q_ref=q_ref)


def fits_frame(result: BNResult) -> pd.DataFrame:
    """Tidy table of B-N fits (overall + each wetness group)."""
    rows = [{"group": "overall", **_fit_row(result.overall)}]
    for label, fit in result.groups.items():
        rows.append({"group": label, **_fit_row(fit)})
    return pd.DataFrame(rows)


def _fit_row(f: BNFit) -> dict:
    return {"b": f.b, "a": f.a, "tau_ref_days": f.tau_ref, "q_ref": f.q_ref,
            "r2": f.r2, "n_fit_points": f.n_points, "n_cloud": f.n_cloud, "method": f.method}
