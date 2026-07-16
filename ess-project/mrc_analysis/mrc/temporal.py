"""Temporal / stationarity analysis of the master recession constant.

Builds MRCs on overlapping multi-year windows (and optionally per water year),
extracts the master ``k`` per window into a tidy series, and applies a
Mann-Kendall trend test with a Sen slope estimate. The trend report explicitly
caveats serial autocorrelation and unequal per-window sample sizes, which bias
the nominal p-value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import Config
from .mrc import MRCResult, build_mrc
from .recession_id import Recession


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def _window_label(y0: int, y1: int) -> str:
    return f"{y0}-{y1}"


def window_mrc_series(recessions: List[Recession], cfg: Config) -> pd.DataFrame:
    """Fit one MRC per sliding window; return a tidy per-window summary table.

    Windows are defined on water years (or calendar years) spanning
    ``length_years`` and stepped by ``step_years``. Windows with fewer than
    ``min_recessions`` conforming recessions are recorded with NaN ``k`` and
    ``n_used`` so the coverage gap is visible rather than silently dropped.
    """
    w = cfg.temporal.window
    if not recessions:
        return _empty_window_frame()

    years = np.array([r.water_year for r in recessions])
    y_min, y_max = int(years.min()), int(years.max())

    rows = []
    y0 = y_min
    while y0 + w.length_years - 1 <= y_max:
        y1 = y0 + w.length_years - 1
        in_win = [r for r in recessions if y0 <= r.water_year <= y1]
        row = _summarize_window(in_win, cfg, label=_window_label(y0, y1),
                                year_start=y0, year_end=y1,
                                year_mid=y0 + (w.length_years - 1) / 2.0)
        rows.append(row)
        y0 += w.step_years

    return pd.DataFrame(rows)


def per_year_mrc_series(recessions: List[Recession], cfg: Config) -> pd.DataFrame:
    """Fit one MRC per water year (where sample size permits)."""
    py = cfg.temporal.per_year
    if not py.enabled or not recessions:
        return _empty_window_frame()

    years = sorted({r.water_year for r in recessions})
    rows = []
    for y in years:
        in_yr = [r for r in recessions if r.water_year == y]
        rows.append(
            _summarize_window(in_yr, cfg, label=str(y), year_start=y, year_end=y,
                              year_mid=float(y), min_override=py.min_recessions)
        )
    return pd.DataFrame(rows)


def _summarize_window(recs: List[Recession], cfg: Config, *, label: str,
                      year_start: int, year_end: int, year_mid: float,
                      min_override: Optional[int] = None) -> dict:
    w = cfg.temporal.window
    min_rec = w.min_recessions if min_override is None else min_override

    base = {
        "window": label, "year_start": year_start, "year_end": year_end,
        "year_mid": year_mid, "n_recessions": len(recs),
    }
    # Count conforming without a full fit is cheap enough to just fit.
    if len(recs) < min_rec:
        base.update(_nan_mrc_fields())
        base["n_recessions"] = len(recs)
        return base

    result, _ = build_mrc(recs, cfg)
    if result.n_used < min_rec:
        base.update(_nan_mrc_fields())
        base["n_conforming"] = result.n_conforming
        return base

    base.update({
        "k": result.k, "tau": result.tau, "rmse": result.rmse,
        "n_conforming": result.n_conforming, "n_used": result.n_used,
    })
    return base


def _nan_mrc_fields() -> dict:
    return {"k": np.nan, "tau": np.nan, "rmse": np.nan,
            "n_conforming": 0, "n_used": 0}


def _empty_window_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "window", "year_start", "year_end", "year_mid", "n_recessions",
        "k", "tau", "rmse", "n_conforming", "n_used",
    ])


# --------------------------------------------------------------------------- #
# Mann-Kendall + Sen slope
# --------------------------------------------------------------------------- #
@dataclass
class TrendResult:
    n: int
    tau: float          # Kendall's tau
    s: float            # Mann-Kendall S statistic
    z: float            # normal test statistic (tie/continuity corrected)
    p_value: float      # two-sided p
    sen_slope: float    # Theil-Sen slope (units of k per x-unit)
    sen_intercept: float
    trend: str          # "increasing" | "decreasing" | "no trend" (at alpha=0.05)
    caveat: str

    def as_dict(self) -> dict:
        return {
            "n": self.n, "kendall_tau": self.tau, "S": self.s, "Z": self.z,
            "p_value": self.p_value, "sen_slope": self.sen_slope,
            "sen_intercept": self.sen_intercept, "trend": self.trend,
            "caveat": self.caveat,
        }


def mann_kendall(y: np.ndarray, x: Optional[np.ndarray] = None,
                 alpha: float = 0.05, caveat: bool = True) -> TrendResult:
    """Mann-Kendall trend test with tie/continuity correction and Sen slope.

    ``y`` is the k series (NaNs dropped); ``x`` the ordinate (e.g. window mid
    year) used only for the Sen slope. Autocorrelation is NOT corrected here —
    the caveat string flags that the nominal p-value is optimistic under the
    serial dependence induced by overlapping windows.
    """
    y = np.asarray(y, dtype=float)
    if x is None:
        x = np.arange(y.size, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = y.size

    caveat_msg = (
        "Nominal p assumes independent samples. Overlapping windows induce strong "
        "positive serial autocorrelation (inflating significance), and per-window "
        "sample sizes are unequal (heteroscedastic k estimates). Treat p as a "
        "screening indicator, not a formal test." if caveat else ""
    )

    if n < 3:
        return TrendResult(n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           "insufficient data", caveat_msg)

    # S statistic.
    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(y[i + 1:] - y[i]))
    s = float(s)

    # Variance with tie correction.
    _, counts = np.unique(y, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0

    if var_s <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    # Theil-Sen slope over all pairwise (x, y).
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        valid = dx != 0
        slopes.extend((dy[valid] / dx[valid]).tolist())
    sen = float(np.median(slopes)) if slopes else np.nan
    intercept = float(np.median(y - sen * x)) if np.isfinite(sen) else np.nan

    if p < alpha and z > 0:
        trend = "increasing"
    elif p < alpha and z < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    return TrendResult(n=n, tau=float(tau), s=s, z=float(z), p_value=float(p),
                       sen_slope=sen, sen_intercept=intercept, trend=trend,
                       caveat=caveat_msg)


def trend_on_windows(window_df: pd.DataFrame, cfg: Config) -> TrendResult:
    """Run the Mann-Kendall / Sen test on the windowed master-k series."""
    df = window_df.dropna(subset=["k"]).sort_values("year_mid")
    return mann_kendall(df["k"].to_numpy(), df["year_mid"].to_numpy(),
                        caveat=cfg.temporal.trend.autocorr_caveat)
