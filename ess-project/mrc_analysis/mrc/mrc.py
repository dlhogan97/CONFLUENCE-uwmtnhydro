"""Master recession curve construction.

Two steps:

1. Per-recession tail fit. Each recession's LATE-TIME tail is fit on semi-log
   (ln Q vs t) as a linear reservoir, Q(t) = Q0 * exp(-k t). The recession
   constant ``k`` (1/day) and storage timescale ``tau = 1/k`` (days) come from
   the tail only, not the full segment (the early limb is not linear-reservoir).

2. Automated matching strip. Conforming recessions are assembled into one
   master curve by treating each recession's time offset as a free parameter and
   optimizing all offsets jointly against a common master model (a polynomial in
   ln Q vs shifted time; order 1 = single exponential). The master recession
   constant ``k`` is read from the fitted master, with RMSE and counts reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import least_squares

from .config import Config
from .recession_id import Recession


# --------------------------------------------------------------------------- #
# Per-recession tail fit
# --------------------------------------------------------------------------- #
@dataclass
class TailFit:
    k: float          # recession constant (1/day); dlnQ/dt = -k
    tau: float        # storage timescale (days) = 1/k
    intercept: float  # ln Q at t=0 of the tail's local time axis
    r2: float         # coefficient of determination of the ln-linear fit
    n_tail: int       # points used in the tail
    conforming: bool  # passed the r2 / positivity screen


def select_tail_points(t: np.ndarray, Q: np.ndarray, *, method: str = "fraction",
                       tail_fraction: float = 0.5, tail_flow_frac: float = 0.667,
                       tail_min_points: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Select the late-time tail (t, Q) of one recession for the k fit.

    Two methods:

    - ``"flow_threshold"``: keep days with ``Q < tail_flow_frac * Q_ref``, where
      ``Q_ref`` is the mean daily flow over the recession. Because a recession
      falls monotonically, these are the low-flow, late-time days — the
      linear-reservoir tail, defined by flow magnitude rather than a point count.
      If fewer than ``tail_min_points`` fall below the threshold, fall back to the
      lowest-flow ``tail_min_points`` days.
    - ``"fraction"``: keep the last ``max(tail_min_points, ceil(tail_fraction*n))``
      points by count.

    Non-finite and non-positive flows are dropped first (log-safe).
    """
    t = np.asarray(t, dtype=float)
    Q = np.asarray(Q, dtype=float)
    good = np.isfinite(Q) & (Q > 0)
    tg, qg = t[good], Q[good]
    if tg.size == 0:
        return tg, qg

    if method == "flow_threshold":
        q_ref = float(np.mean(qg))
        thr = tail_flow_frac * q_ref
        mask = qg < thr
        if int(mask.sum()) >= max(2, tail_min_points):
            return tg[mask], qg[mask]
        # Too few below threshold -> take the lowest-flow (latest) points instead.
        n = min(max(tail_min_points, 2), tg.size)
        return tg[-n:], qg[-n:]

    if method != "fraction":
        raise ValueError(f"Unknown tail_method '{method}' (use 'fraction' or 'flow_threshold')")

    n_tail = max(tail_min_points, int(np.ceil(tail_fraction * tg.size)))
    n_tail = min(n_tail, tg.size)
    return tg[-n_tail:], qg[-n_tail:]


def fit_tail(t: np.ndarray, Q: np.ndarray, *, min_r2: float,
             method: str = "fraction", tail_fraction: float = 0.5,
             tail_flow_frac: float = 0.667, tail_min_points: int = 5) -> TailFit:
    """Fit ``ln Q = intercept - k * t`` on the late-time tail of one recession.

    ``t`` is days since the segment start; the tail is chosen by
    :func:`select_tail_points` (``method`` = ``fraction`` or ``flow_threshold``).
    """
    tt, qq = select_tail_points(t, Q, method=method, tail_fraction=tail_fraction,
                                tail_flow_frac=tail_flow_frac, tail_min_points=tail_min_points)
    if tt.size < max(2, tail_min_points):
        return TailFit(np.nan, np.nan, np.nan, np.nan, int(tt.size), False)

    lnq = np.log(qq)
    slope, intercept = np.polyfit(tt, lnq, 1)
    resid = lnq - (slope * tt + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((lnq - lnq.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    k = -slope
    tau = 1.0 / k if k > 0 else np.inf
    conforming = bool(np.isfinite(r2) and r2 >= min_r2 and k > 0)
    return TailFit(k=k, tau=tau, intercept=float(intercept), r2=float(r2),
                   n_tail=int(tt.size), conforming=conforming)


def _tail_kwargs(cfg: Config) -> dict:
    m = cfg.mrc
    return dict(method=getattr(m, "tail_method", "fraction"),
                tail_fraction=m.tail_fraction,
                tail_flow_frac=getattr(m, "tail_flow_frac", 0.667),
                tail_min_points=m.tail_min_points)


def fit_all_tails(recessions: List[Recession], cfg: Config) -> List[TailFit]:
    kw = _tail_kwargs(cfg)
    return [fit_tail(r.t, r.Q, min_r2=cfg.mrc.min_tail_r2, **kw) for r in recessions]


# --------------------------------------------------------------------------- #
# Automated matching strip
# --------------------------------------------------------------------------- #
@dataclass
class MRCResult:
    k: float                   # master recession constant (1/day)
    tau: float                 # master storage timescale (days)
    rmse: float                # RMSE of time (days) about the master t(lnQ) curve
    n_recessions: int          # recessions supplied
    n_conforming: int          # recessions that passed the tail screen
    n_used: int                # recessions actually assembled into the master
    poly_coeffs: np.ndarray    # master polynomial time = P(lnQ), np.polyfit order
    offsets: np.ndarray        # optimized time offset per used recession (days)
    used_seg_ids: List[int]    # seg_ids of the used recessions (ranked by length)
    lnq_eval: float = np.nan   # lnQ at which k is evaluated (late-time tail end)
    master_t: np.ndarray = field(default_factory=lambda: np.array([]))
    master_Q: np.ndarray = field(default_factory=lambda: np.array([]))


def _tail_points(r: Recession, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, lnQ) for the tail points used in matching (same tail selection
    as the per-recession k fit, so the master curve is built from the same days)."""
    tt, qq = select_tail_points(r.t, r.Q, **_tail_kwargs(cfg))
    return tt, np.log(qq) if qq.size else qq


def matching_strip(recessions: List[Recession], cfg: Config,
                   tails: Optional[List[TailFit]] = None) -> MRCResult:
    """Assemble conforming recessions into a master curve via optimized offsets.

    The master curve is parameterized as **time as a function of log-flow**,
    ``t = P(lnQ)`` (P a polynomial of order ``mrc.master_poly_order``). Each
    recession's tail contributes points ``(lnQ_ij, t_ij)`` and is free to shift
    along the time axis by ``offset_i``; the offsets and P are optimized jointly
    to collapse all recessions onto one curve. Parameterizing on lnQ (fixed per
    point) rather than t removes the degeneracy where free offsets could slide
    recessions apart and flatten the master.

    For a linear reservoir P is linear with slope ``dt/dlnQ = -tau``, so the
    master recession constant is ``k = -1 / P'(lnQ)`` evaluated at the late-time
    (lowest-flow) tail end. The longest recession is the reference (offset 0).
    """
    m = cfg.mrc
    if tails is None:
        tails = fit_all_tails(recessions, cfg)

    # Select conforming recessions, ranked by length descending (longest first).
    conforming = [(r, tf) for r, tf in zip(recessions, tails) if tf.conforming]
    conforming.sort(key=lambda rt: rt[0].length, reverse=True)
    if m.max_recessions_per_window is not None:
        conforming = conforming[: m.max_recessions_per_window]

    n_conf = len(conforming)
    if n_conf == 0:
        return MRCResult(np.nan, np.nan, np.nan, len(recessions), 0, 0,
                         np.array([]), np.array([]), [])

    # Per recession: x = lnQ (independent), y = t (dependent, shiftable).
    pts = [_tail_points(r, cfg) for r, _ in conforming]  # (t, lnq)
    xy = [(lnq, t) for (t, lnq) in pts]                  # (lnq, t)
    order = int(m.master_poly_order)
    n_rec = n_conf

    if n_rec == 1:
        lnq, t = xy[0]
        coeffs = (np.polyfit(lnq, t, order) if lnq.size > order
                  else np.array([-1.0 / max(conforming[0][1].k, 1e-6), 0.0]))
        return _finalize_mrc(coeffs, [0.0], conforming, xy, len(recessions), n_conf)

    # Reference is the longest recession (index 0), offset fixed to 0.
    ref_coeffs = np.polyfit(xy[0][0], xy[0][1], order)

    # Init offsets so each recession's first tail point lands on the reference
    # curve evaluated at that recession's log-flow.
    off0 = np.zeros(n_rec)
    for i in range(1, n_rec):
        lnq_i, t_i = xy[i]
        off0[i] = float(np.clip(np.polyval(ref_coeffs, lnq_i[0]) - t_i[0],
                                -m.matching.max_offset_days, m.matching.max_offset_days))

    def unpack(params):
        offs = np.concatenate([[0.0], params[: n_rec - 1]])
        coeffs = params[n_rec - 1:]
        return offs, coeffs

    def residuals(params):
        offs, coeffs = unpack(params)
        res = [(t + offs[i]) - np.polyval(coeffs, lnq) for i, (lnq, t) in enumerate(xy)]
        r = np.concatenate(res)
        delta = cfg.mrc.matching.huber_delta
        if delta:  # pseudo-Huber reweighting in the time residual
            a = np.abs(r)
            scale = np.where(a <= delta, 1.0,
                             np.sqrt(2 * delta * a - delta ** 2) / np.maximum(a, 1e-12))
            r = r * scale
        return r

    p0 = np.concatenate([off0[1:], ref_coeffs])
    lb = np.concatenate([np.full(n_rec - 1, -m.matching.max_offset_days),
                         np.full(order + 1, -np.inf)])
    ub = np.concatenate([np.full(n_rec - 1, m.matching.max_offset_days),
                         np.full(order + 1, np.inf)])
    sol = least_squares(residuals, p0, bounds=(lb, ub), method="trf", max_nfev=5000)
    offs, coeffs = unpack(sol.x)
    return _finalize_mrc(coeffs, list(offs), conforming, xy, len(recessions), n_conf)


def _finalize_mrc(coeffs, offsets, conforming, xy, n_recessions, n_conf) -> MRCResult:
    coeffs = np.asarray(coeffs, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    all_lnq = np.concatenate([lnq for lnq, _ in xy])
    lnq_lo, lnq_hi = float(all_lnq.min()), float(all_lnq.max())

    # k evaluated at the late-time (lowest-flow) tail end: k = -1 / (dt/dlnQ).
    deriv = np.polyder(coeffs) if coeffs.size > 1 else np.array([0.0])
    dtdlnq = float(np.polyval(deriv, lnq_lo))
    k = -1.0 / dtdlnq if dtdlnq < 0 else np.nan   # dt/dlnQ must be negative
    tau = 1.0 / k if (np.isfinite(k) and k > 0) else np.inf

    # RMSE of time about the master t(lnQ) curve.
    res = [(t + offsets[i]) - np.polyval(coeffs, lnq) for i, (lnq, t) in enumerate(xy)]
    res = np.concatenate(res) if res else np.array([])
    rmse = float(np.sqrt(np.mean(res ** 2))) if res.size else np.nan

    # Master curve samples over the pooled log-flow range (for plotting).
    lnq_grid = np.linspace(lnq_lo, lnq_hi, 200)
    t_master = np.polyval(coeffs, lnq_grid)
    Q_master = np.exp(lnq_grid)
    order_sort = np.argsort(t_master)

    return MRCResult(
        k=k, tau=tau, rmse=rmse,
        n_recessions=n_recessions, n_conforming=n_conf, n_used=len(conforming),
        poly_coeffs=coeffs, offsets=offsets,
        used_seg_ids=[r.seg_id for r, _ in conforming],
        lnq_eval=lnq_lo,
        master_t=t_master[order_sort], master_Q=Q_master[order_sort],
    )


def build_mrc(recessions: List[Recession], cfg: Config) -> tuple[MRCResult, List[TailFit]]:
    """Convenience: fit tails and assemble the master curve in one call."""
    tails = fit_all_tails(recessions, cfg)
    result = matching_strip(recessions, cfg, tails=tails)
    return result, tails


def recession_metrics_frame(recessions: List[Recession], tails: List[TailFit]):
    """Tidy per-recession table combining segment metadata with tail-fit metrics."""
    import pandas as pd
    cols = ["seg_id", "start", "end", "length", "doy_start", "water_year",
            "q_start", "q_end", "tail_k", "tau", "tail_r2", "n_tail",
            "conforming", "flags"]
    if not recessions:
        return pd.DataFrame(columns=cols)
    rows = []
    for r, tf in zip(recessions, tails):
        row = r.as_row()
        row.update({"tail_k": tf.k, "tau": tf.tau, "tail_r2": tf.r2,
                    "n_tail": tf.n_tail, "conforming": tf.conforming})
        rows.append(row)
    return pd.DataFrame(rows)[cols]
