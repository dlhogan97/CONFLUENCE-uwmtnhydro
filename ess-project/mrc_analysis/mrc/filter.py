"""Lyne-Hollick recursive digital baseflow filter (screening use only).

The filter is used purely to *screen* candidate recession days by their
quickflow fraction; it is not the recession estimator itself. We separate a
"quickflow" (fast) signal ``f`` from total flow ``Q`` and take baseflow
``b = Q - f``, constrained to ``0 <= b <= Q``.

Filter equation (per pass, causal direction):

    f[i] = alpha * f[i-1] + (1 + alpha) / 2 * (Q[i] - Q[i-1])
    b[i] = Q[i] - f[i],   then clip b to [0, Q]

Standard practice applies three passes: forward, backward, forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class BaseflowResult:
    baseflow: np.ndarray       # b, same length as input (NaN where Q was NaN)
    quickflow: np.ndarray      # Q - b
    quickflow_frac: np.ndarray # quickflow / Q, clipped to [0, 1]
    alpha: float
    passes: int


def _single_pass(Q: np.ndarray, alpha: float, forward: bool) -> np.ndarray:
    """One causal Lyne-Hollick pass returning baseflow b (clipped to [0, Q])."""
    q = Q if forward else Q[::-1]
    n = q.size
    f = np.zeros(n)
    b = np.empty(n)
    b[0] = q[0]
    f[0] = 0.0
    c = (1.0 + alpha) / 2.0
    for i in range(1, n):
        f[i] = alpha * f[i - 1] + c * (q[i] - q[i - 1])
        if f[i] < 0.0:
            f[i] = 0.0            # quickflow cannot be negative
        bi = q[i] - f[i]
        # Constrain baseflow to [0, Q].
        if bi > q[i]:
            bi = q[i]
        elif bi < 0.0:
            bi = 0.0
        b[i] = bi
    return b if forward else b[::-1]


def lyne_hollick(
    Q: np.ndarray,
    alpha: float = 0.925,
    passes: int = 3,
    reflect: int = 30,
) -> np.ndarray:
    """Multi-pass Lyne-Hollick baseflow on a 1-D array (may contain NaN gaps).

    NaNs are treated as segment boundaries: the filter is applied independently
    to each contiguous finite run so gaps never leak across. Edge effects within
    a run are reduced by reflecting ``reflect`` samples at each end.
    """
    Q = np.asarray(Q, dtype=float)
    b = np.full(Q.shape, np.nan)

    for s, e in _finite_runs(Q):
        seg = Q[s:e]
        b[s:e] = _filter_run(seg, alpha, passes, reflect)
    return b


def _filter_run(seg: np.ndarray, alpha: float, passes: int, reflect: int) -> np.ndarray:
    """Multi-pass filter on one contiguous finite run, with reflective padding.

    Passes alternate forward/backward/forward... Each pass filters the previous
    pass's baseflow estimate, and the result is kept <= original Q throughout
    (Nathan & McMahon multi-pass convention).
    """
    r = min(reflect, seg.size - 1) if seg.size > 1 else 0
    if r > 0:
        padded = np.concatenate([seg[1 : r + 1][::-1], seg, seg[-r - 1 : -1][::-1]])
    else:
        padded = seg

    b = padded.copy()
    forward = True
    for _ in range(passes):
        b = np.minimum(_single_pass(b, alpha, forward), padded)
        forward = not forward

    if r > 0:
        b = b[r : r + seg.size]
    return np.clip(b, 0.0, seg)


def _finite_runs(x: np.ndarray) -> List[tuple]:
    """Yield (start, end) index pairs of contiguous non-NaN runs."""
    finite = np.isfinite(x)
    runs = []
    i, n = 0, x.size
    while i < n:
        if finite[i]:
            j = i
            while j < n and finite[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def separate_baseflow(df: pd.DataFrame, cfg: Config, alpha: float | None = None,
                      passes: int | None = None, reflect: int | None = None) -> BaseflowResult:
    """Run Lyne-Hollick on ``df['Q']`` using config defaults (or overrides)."""
    bf = cfg.baseflow
    a = bf.alpha if alpha is None else alpha
    p = bf.passes if passes is None else passes
    # A single forward pass starts clean (b[0]=Q[0]); reflective padding would
    # inject a spurious peak at the leading edge, so default reflect=0 there.
    r = (0 if p == 1 else bf.reflect) if reflect is None else reflect
    Q = df["Q"].to_numpy(dtype=float)
    b = lyne_hollick(Q, alpha=a, passes=p, reflect=r)
    quick = Q - b
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(Q > 0, quick / Q, np.nan)
    frac = np.clip(frac, 0.0, 1.0)
    return BaseflowResult(baseflow=b, quickflow=quick, quickflow_frac=frac, alpha=a, passes=p)


def add_baseflow_columns(df: pd.DataFrame, cfg: Config, alpha: float | None = None) -> pd.DataFrame:
    """Return a copy of df with baseflow / quickflow / qf_frac columns added.

    ``baseflow`` / ``quickflow`` use the full multi-pass separation (default 3,
    for BFI reporting). ``qf_frac`` is the *screening* quickflow fraction from a
    single forward pass (``baseflow.screen_passes``): it is ~0 on a clean
    recession and rises on quickflow spikes, so the recession gate behaves as its
    default threshold expects. See config note on ``screen_passes``.
    """
    full = separate_baseflow(df, cfg, alpha=alpha, passes=cfg.baseflow.passes)
    screen = separate_baseflow(df, cfg, alpha=alpha, passes=cfg.baseflow.screen_passes)
    out = df.copy()
    out["baseflow"] = full.baseflow
    out["quickflow"] = full.quickflow
    out["bfi_qf_frac"] = full.quickflow_frac   # multi-pass fraction (reporting)
    out["qf_frac"] = screen.quickflow_frac      # single-pass fraction (gating)
    return out
