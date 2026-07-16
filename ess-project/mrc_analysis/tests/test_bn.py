"""Tests for Brutsaert-Nieber recession analysis (-dQ/dt vs Q)."""
import numpy as np
import pandas as pd
import pytest

from mrc import bn
from mrc.recession_id import Recession


@pytest.fixture(autouse=True)
def _no_rate_floor(cfg):
    """These tests exercise the math on synthetic, continuously-valued series, so
    disable both noise floors: an absolute ``min_rate`` would censor low-rate
    points (biasing the fitted exponent), and ``min_rate_steps`` has no meaningful
    quantization to infer from unquantized data."""
    cfg.bn.dqdt.min_rate = 1e-9
    cfg.bn.dqdt.min_rate_steps = 0.0
    cfg.bn.dqdt.min_q = 0.0


def _rec(seg_id, Q, water_year=2000, start="2000-06-01"):
    Q = np.asarray(Q, dtype=float)
    n = Q.size
    return Recession(seg_id=seg_id, start=pd.Timestamp(start), end=pd.Timestamp(start),
                     length=n, dates=pd.date_range(start, periods=n, freq="D"),
                     Q=Q, t=np.arange(n, dtype=float), doy_start=152, water_year=water_year)


def _linear(tau, q0, n):
    t = np.arange(n, dtype=float)
    return q0 * np.exp(-t / tau)


def _nonlinear(a, b, q0, n):
    """Recession satisfying -dQ/dt = a Q^b exactly (b != 1)."""
    t = np.arange(n, dtype=float)
    return (q0 ** (1 - b) + (b - 1) * a * t) ** (1.0 / (1 - b))


def test_bn_linear_reservoir_gives_b1(cfg):
    """Linear reservoir -> b ~ 1 and tau_ref recovers the true timescale."""
    cfg.bn.envelope.method = "ols_cloud"
    recs = [_rec(i, _linear(25.0, 300 - 20 * i, 50)) for i in range(6)]
    res = bn.bn_analysis(recs, cfg)
    assert abs(res.overall.b - 1.0) < 1e-3
    assert abs(res.overall.tau_ref - 25.0) < 0.3


def test_bn_recovers_nonlinear_exponent(cfg):
    """A recession built with -dQ/dt = a Q^1.5 recovers b ~ 1.5."""
    cfg.bn.envelope.method = "ols_cloud"
    recs = [_rec(i, _nonlinear(0.02, 1.5, 200 - 20 * i, 60)) for i in range(6)]
    res = bn.bn_analysis(recs, cfg)
    assert abs(res.overall.b - 1.5) < 0.02


def test_dqdt_cloud_positive_and_paired(cfg):
    """The cloud has one point per within-segment day-pair, all -dQ/dt > 0."""
    recs = [_rec(0, _linear(20.0, 100, 30)), _rec(1, _linear(20.0, 80, 25))]
    cloud = bn.recession_rate_cloud(recs, cfg)
    assert len(cloud) == (30 - 1) + (25 - 1)
    assert (cloud["minus_dQdt"] > 0).all()
    assert (cloud["Q"] > 0).all()


def test_wetness_split_separates_two_taus(cfg):
    """Two families of linear recessions with different tau, tagged wet/dry by a
    covariate, must recover the two timescales in the split fit."""
    cfg.bn.envelope.method = "ols_cloud"
    cfg.bn.wetness.metric = "recession_q_start"   # start flow as the wetness proxy
    cfg.bn.wetness.split = "median"
    # "wet" = high start flow, slow drainage (tau=40); "dry" = low start, fast (tau=14)
    recs = []
    for i in range(6):
        recs.append(_rec(i, _linear(40.0, 400 - 10 * i, 50)))          # high Q0, tau 40
    for i in range(6):
        recs.append(_rec(100 + i, _linear(14.0, 40 - 3 * i, 50)))      # low Q0, tau 14
    res = bn.bn_analysis(recs, cfg)
    assert set(res.groups) == {"wet", "dry"}
    # Both look linear...
    assert abs(res.groups["wet"].b - 1.0) < 0.05
    assert abs(res.groups["dry"].b - 1.0) < 0.05
    # ...but the effective tau separates (wet slower than dry).
    assert res.groups["wet"].tau_ref > res.groups["dry"].tau_ref
    assert abs(res.groups["wet"].tau_ref - 40.0) < 3.0
    assert abs(res.groups["dry"].tau_ref - 14.0) < 3.0


def test_extremes_split_picks_tail_years(cfg):
    """'extremes' labels the driest/wettest n_extreme YEARS, dropping the middle."""
    cfg.bn.envelope.method = "ols_cloud"
    cfg.bn.wetness.metric = "water_year_mean_flow"  # falls back to q_start w/o df; use recession_q_start
    cfg.bn.wetness.metric = "recession_q_start"
    cfg.bn.wetness.split = "extremes"
    cfg.bn.wetness.n_extreme = 2
    # 5 water years, one recession each, monotonically increasing start flow.
    recs = [_rec(i, _linear(25.0, 20 + 30 * i, 40), water_year=2000 + i) for i in range(5)]
    res = bn.bn_analysis(recs, cfg)
    g = res.cloud.groupby("group")["water_year"].apply(lambda s: sorted(set(s))).to_dict()
    assert g["dry"] == [2000, 2001]          # two lowest-flow years
    assert g["wet"] == [2003, 2004]          # two highest-flow years
    assert g.get("unknown") == [2002]        # middle year excluded


def test_era_split_by_year(cfg):
    """'era' splits pre/post split_year regardless of the wetness metric."""
    cfg.bn.envelope.method = "ols_cloud"
    cfg.bn.wetness.split = "era"
    cfg.bn.wetness.split_year = 2000
    recs = [_rec(i, _linear(25.0, 100, 40), water_year=1997 + i) for i in range(6)]  # 1997..2002
    res = bn.bn_analysis(recs, cfg)
    assert set(res.groups) == {"pre2000", "post2000"}
    pre = sorted(set(res.cloud.loc[res.cloud.group == "pre2000", "water_year"]))
    assert pre == [1997, 1998, 1999]


def test_min_q_censors_low_flow_symmetrically(cfg):
    """The Q floor drops every point below the cutoff flow, regardless of its rate
    (symmetric in -dQ/dt), so it cannot truncate the lower envelope from below."""
    recs = [_rec(0, _linear(25.0, 10.0, 60))]        # decays from 10 down to ~0.9
    cfg.bn.dqdt.min_q = 0.0
    all_pts = bn.recession_rate_cloud(recs, cfg)
    cfg.bn.dqdt.min_q = 2.0
    kept = bn.recession_rate_cloud(recs, cfg)
    assert (kept["Q"] >= 2.0).all()
    assert len(kept) < len(all_pts)
    # nothing was removed on the rate axis: every retained-Q point survived
    assert len(kept) == int((all_pts["Q"] >= 2.0).sum())


def test_infer_quantization_recovers_flow_dependent_step():
    """Resolution inferred from the data: fine step at low Q, coarse at high Q."""
    fine = np.arange(1, 400) * 0.002832        # Q < ~1.1, step 0.002832
    coarse = np.arange(100, 400) * 0.028317    # Q ~2.8-11, step 0.028317
    q = np.concatenate([fine, coarse])
    edges, steps = bn.infer_quantization(q, n_bins=8)
    assert abs(bn.step_for(np.array([0.5]), edges, steps)[0] - 0.002832) < 1e-4
    assert abs(bn.step_for(np.array([9.0]), edges, steps)[0] - 0.028317) < 5e-3


def test_min_rate_steps_scales_with_resolution(cfg):
    """The resolution-aware floor drops only sub-resolution rates, and (unlike an
    absolute floor) does not preferentially delete well-resolved low-Q points."""
    step = 0.01
    # Recession quantized to `step`; rates span 1..N steps/day.
    Q = np.round(np.linspace(5.0, 1.0, 40) / step) * step
    recs = [_rec(0, Q)]

    cfg.bn.dqdt.min_rate = 1e-9
    cfg.bn.dqdt.min_rate_steps = 0.0
    n_all = len(bn.recession_rate_cloud(recs, cfg))

    cfg.bn.dqdt.min_rate_steps = 2.0
    kept = bn.recession_rate_cloud(recs, cfg)
    # every retained rate clears 2 quantization steps per day
    assert (kept["minus_dQdt"] >= 2 * step - 1e-9).all()
    assert len(kept) <= n_all


def test_lower_envelope_tracks_bottom(cfg):
    """The lower-quantile envelope sits at or below the cloud's bulk in each bin."""
    rng = np.random.default_rng(0)
    logQ = rng.uniform(0, 3, 2000)
    logR = logQ + rng.uniform(0, 1.5, 2000)   # one-sided scatter ABOVE the b=1 line
    xb, yb = bn.lower_envelope(logQ, logR, n_bins=10, quantile=0.05, min_bin_count=5)
    assert xb.size >= 5
    b, lna = np.polyfit(xb, yb, 1)
    assert abs(b - 1.0) < 0.15         # envelope recovers the true b=1 lower bound
