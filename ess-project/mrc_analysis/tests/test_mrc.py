"""Tests for tail fitting, the matching strip, and the trend test."""
import numpy as np
import pandas as pd

from mrc import mrc as M
from mrc import temporal as T
from mrc.recession_id import Recession


def _synthetic_recession(seg_id, k, q0, n, start_t=0.0, water_year=2000, noise=0.0,
                         rng=None):
    t = np.arange(n, dtype=float)
    Q = q0 * np.exp(-k * (t + start_t))
    if noise and rng is not None:
        Q = Q * (1.0 + noise * rng.standard_normal(n))
    return Recession(
        seg_id=seg_id, start=pd.Timestamp("2000-06-01"), end=pd.Timestamp("2000-07-01"),
        length=n, dates=pd.date_range("2000-06-01", periods=n, freq="D"),
        Q=Q, t=t, doy_start=152, water_year=water_year,
    )


def test_tail_fit_recovers_k():
    """Pure exponential -> tail fit recovers k, tau, r2 ~ 1."""
    t = np.arange(80.0)
    Q = 200.0 * np.exp(-0.04 * t)
    tf = M.fit_tail(t, Q, tail_fraction=0.5, tail_min_points=5, min_r2=0.8)
    assert abs(tf.k - 0.04) < 1e-6
    assert abs(tf.tau - 25.0) < 1e-3
    assert tf.r2 > 0.999
    assert tf.conforming


def test_flow_threshold_selects_low_flow_days():
    """flow_threshold keeps exactly the days with Q < frac * mean(Q)."""
    t = np.arange(80.0)
    Q = 200.0 * np.exp(-0.04 * t)
    tt, qq = M.select_tail_points(t, Q, method="flow_threshold",
                                  tail_flow_frac=0.667, tail_min_points=5)
    thr = 0.667 * Q.mean()
    assert np.all(qq < thr)                 # every selected day below threshold
    assert np.all(Q[Q < thr] == qq)         # and it is exactly that set (monotone)


def test_flow_threshold_recovers_k():
    """flow_threshold tail still recovers k on a pure exponential."""
    t = np.arange(80.0)
    Q = 200.0 * np.exp(-0.04 * t)
    tf = M.fit_tail(t, Q, min_r2=0.8, method="flow_threshold",
                    tail_flow_frac=0.667, tail_min_points=5)
    assert abs(tf.k - 0.04) < 1e-6
    assert tf.conforming


def test_flow_threshold_fallback_when_too_few():
    """A nearly flat recession has ~no days below 2/3 of its mean -> fallback
    to the lowest-flow tail_min_points days (never an empty tail)."""
    t = np.arange(20.0)
    Q = np.linspace(11.0, 10.0, 20)  # tiny range; 0.667*mean well below all values
    tt, qq = M.select_tail_points(t, Q, method="flow_threshold", tail_min_points=5)
    assert tt.size == 5
    assert np.allclose(qq, Q[-5:])


def test_tail_fit_flags_nonconforming_on_noise():
    """Heavily corrupted flat series is non-conforming (low r2 or k<=0)."""
    rng = np.random.default_rng(1)
    t = np.arange(40.0)
    Q = 10.0 + rng.standard_normal(40)  # no real recession
    tf = M.fit_tail(t, Q, tail_fraction=0.5, tail_min_points=5, min_r2=0.8)
    assert not tf.conforming


def test_matching_strip_recovers_common_k(cfg):
    """Several exponentials sharing k -> master k equals that shared k."""
    recs = [
        _synthetic_recession(0, k=0.05, q0=300, n=40),
        _synthetic_recession(1, k=0.05, q0=120, n=30),
        _synthetic_recession(2, k=0.05, q0=60, n=25),
        _synthetic_recession(3, k=0.05, q0=500, n=50),
    ]
    result, tails = M.build_mrc(recs, cfg)
    assert result.n_conforming == 4
    assert result.n_used == 4
    assert abs(result.k - 0.05) < 1e-3
    assert abs(result.tau - 20.0) < 0.5
    assert result.rmse < 1e-2


def test_matching_strip_ranks_by_length(cfg):
    """Longest recession is the reference (first in used_seg_ids)."""
    recs = [
        _synthetic_recession(0, k=0.04, q0=100, n=20),
        _synthetic_recession(1, k=0.04, q0=100, n=60),  # longest
        _synthetic_recession(2, k=0.04, q0=100, n=35),
    ]
    result, _ = M.build_mrc(recs, cfg)
    assert result.used_seg_ids[0] == 1


def test_no_hard_cap_on_conforming(cfg):
    """With max_recessions_per_window null, all conforming recessions are used."""
    rng = np.random.default_rng(3)
    recs = [_synthetic_recession(i, k=0.03, q0=100 + i, n=30, water_year=2000,
                                 noise=0.005, rng=rng) for i in range(25)]
    result, _ = M.build_mrc(recs, cfg)
    assert result.n_used == result.n_conforming
    assert result.n_used > 15  # not capped at 15


def test_mann_kendall_detects_increasing_trend():
    """Monotone increasing series -> increasing trend, positive Sen slope."""
    x = np.arange(20.0)
    y = 0.02 + 0.001 * x  # clean upward trend
    tr = T.mann_kendall(y, x)
    assert tr.trend == "increasing"
    assert tr.sen_slope > 0
    assert tr.p_value < 0.05
    assert tr.caveat  # caveat present by default


def test_mann_kendall_no_trend_flat():
    """Flat (constant) series -> no trend, slope ~ 0."""
    x = np.arange(15.0)
    y = np.full(15, 0.03)
    tr = T.mann_kendall(y, x)
    assert tr.trend == "no trend"
    assert abs(tr.sen_slope) < 1e-9


def test_window_series_and_trend(cfg):
    """End-to-end windowing on recessions with an imposed k drift is detected."""
    rng = np.random.default_rng(4)
    recs = []
    sid = 0
    for wy in range(2000, 2020):
        k_wy = 0.03 + 0.001 * (wy - 2000)  # rising k
        for _ in range(6):
            recs.append(_synthetic_recession(sid, k=k_wy, q0=100, n=30,
                                              water_year=wy, noise=0.004, rng=rng))
            sid += 1
    window_df = T.window_mrc_series(recs, cfg)
    assert window_df["k"].notna().sum() >= 10
    tr = T.trend_on_windows(window_df, cfg)
    assert tr.trend == "increasing"
    assert tr.sen_slope > 0
