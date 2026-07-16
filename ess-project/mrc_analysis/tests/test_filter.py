"""Tests for the Lyne-Hollick baseflow filter."""
import numpy as np
import pandas as pd

from mrc import filter as flt


def test_baseflow_bounds_constant_flow(cfg):
    """On constant flow, baseflow == Q and quickflow == 0."""
    Q = np.full(200, 5.0)
    b = flt.lyne_hollick(Q, alpha=cfg.baseflow.alpha, passes=cfg.baseflow.passes)
    assert np.allclose(b, Q, atol=1e-9)


def test_baseflow_constraints_hold(cfg):
    """0 <= b <= Q everywhere for an arbitrary positive series."""
    rng = np.random.default_rng(0)
    Q = np.abs(rng.gamma(2.0, 2.0, size=500)) + 0.1
    b = flt.lyne_hollick(Q, alpha=0.925, passes=3)
    assert np.all(b >= -1e-9)
    assert np.all(b <= Q + 1e-9)


def test_single_pass_recession_is_all_baseflow(cfg):
    """The single forward pass (screening filter) leaves a clean recession as
    ~100% baseflow: qf ~ 0. This is the property the recession gate relies on.
    (The multi-pass filter's backward pass legitimately carves quickflow even
    from a pure recession, which is why gating uses screen_passes=1.)"""
    t = np.arange(120.0)
    Q = 100.0 * np.exp(-0.03 * t)
    b = flt.lyne_hollick(Q, alpha=0.925, passes=1, reflect=0)
    qf_frac = (Q - b) / Q
    assert np.max(qf_frac) < 1e-9


def test_multipass_carves_more_quickflow_than_single(cfg):
    """Sanity: the multi-pass filter attributes >= as much quickflow as one pass."""
    rng = np.random.default_rng(2)
    Q = np.abs(rng.gamma(2.0, 2.0, size=400)) + 0.5
    b1 = flt.lyne_hollick(Q, alpha=0.925, passes=1)
    b3 = flt.lyne_hollick(Q, alpha=0.925, passes=3)
    assert np.sum(Q - b3) >= np.sum(Q - b1) - 1e-6


def test_quickflow_spike_detected(cfg):
    """A sharp pulse on top of baseflow shows up as elevated quickflow fraction."""
    Q = np.full(100, 10.0)
    Q[50] += 40.0  # spike
    b = flt.lyne_hollick(Q, alpha=0.925, passes=3)
    qf_frac = (Q - b) / Q
    assert qf_frac[50] > 0.3
    assert qf_frac[5] < 0.05  # away from the spike, near-zero


def test_nan_gaps_do_not_leak(cfg):
    """Filter runs independently on either side of a NaN gap."""
    Q = np.concatenate([np.full(30, 8.0), np.full(10, np.nan), np.full(30, 3.0)])
    b = flt.lyne_hollick(Q, alpha=0.925, passes=3)
    assert np.all(np.isnan(b[30:40]))
    assert np.allclose(b[:30], 8.0, atol=1e-6)
    assert np.allclose(b[40:], 3.0, atol=1e-6)


def test_separate_baseflow_frame(cfg):
    idx = pd.date_range("2000-01-01", periods=100, freq="D")
    df = pd.DataFrame({"Q": np.linspace(20, 5, 100)}, index=idx)
    res = flt.separate_baseflow(df, cfg)
    assert res.baseflow.shape == (100,)
    assert np.all(res.quickflow_frac[np.isfinite(res.quickflow_frac)] >= 0)
