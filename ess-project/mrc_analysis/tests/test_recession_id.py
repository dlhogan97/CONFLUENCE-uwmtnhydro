"""Tests for recession segment detection and screening."""
import numpy as np
import pandas as pd

from mrc import filter as flt
from mrc import recession_id as rid


def _make_df(Q, start="2001-06-01", **cols):
    idx = pd.date_range(start, periods=len(Q), freq="D")
    data = {"Q": np.asarray(Q, dtype=float)}
    data.update({k: np.asarray(v, dtype=float) for k, v in cols.items()})
    return pd.DataFrame(data, index=idx)


def test_single_clean_recession_detected(cfg):
    """One long exponential recession -> exactly one segment.

    Season filter disabled so the test is about falling/gate/length only,
    independent of the configured recession-season window.
    """
    cfg.recession.season.enabled = False
    t = np.arange(60.0)
    Q = 100.0 * np.exp(-0.05 * t)
    df = flt.add_baseflow_columns(_make_df(Q), cfg)
    recs = rid.identify_recessions(df, cfg)
    assert len(recs) == 1
    r = recs[0]
    # Front removal = the peak day (no dQ/dt, not "falling") + drop_days_after_peak.
    assert r.length == 60 - cfg.recession.drop_days_after_peak - 1
    assert np.all(np.diff(r.Q) < 1e-9)  # monotone non-increasing


def test_rising_limb_excluded(cfg):
    """A pure rising limb yields no recession segments."""
    t = np.arange(60.0)
    Q = 5.0 + 0.5 * t  # strictly increasing
    df = flt.add_baseflow_columns(_make_df(Q), cfg)
    recs = rid.identify_recessions(df, cfg)
    assert len(recs) == 0


def test_short_segment_rejected(cfg):
    """A falling run shorter than min_segment_length is dropped."""
    t = np.arange(8.0)
    Q = 50.0 * np.exp(-0.05 * t)
    df = flt.add_baseflow_columns(_make_df(Q), cfg)
    recs = rid.identify_recessions(df, cfg)
    assert len(recs) == 0


def test_gap_breaks_segment(cfg):
    """A NaN gap splits one recession into (at most) two; none span the gap."""
    cfg.recession.season.enabled = False
    t = np.arange(80.0)
    Q = 100.0 * np.exp(-0.04 * t)
    Q[40:45] = np.nan
    df = flt.add_baseflow_columns(_make_df(Q), cfg)
    recs = rid.identify_recessions(df, cfg)
    for r in recs:
        assert not r.dates.isin(df.index[40:45]).any()


def test_forcing_precip_breaks_segment(cfg):
    """A rain day above threshold breaks the segment (forcing precedence)."""
    cfg.recession.season.enabled = False
    t = np.arange(80.0)
    Q = 100.0 * np.exp(-0.03 * t)
    precip = np.zeros(80); precip[40] = 20.0  # big rain day mid-recession
    swe = np.zeros(80); tair = np.full(80, -5.0)  # no melt
    df = _make_df(Q, precip=precip, swe=swe, tair=tair)
    df = flt.add_baseflow_columns(df, cfg)
    recs = rid.identify_recessions(df, cfg)
    # The rain day should not appear inside any recession.
    rain_day = df.index[40]
    for r in recs:
        assert rain_day not in r.dates


def test_seasonal_filter_excludes_offseason(cfg):
    """A falling limb entirely outside the season window is excluded."""
    t = np.arange(60.0)
    Q = 100.0 * np.exp(-0.05 * t)
    # Start Jan 1 -> doy 1..60, well before doy_start=152.
    df = flt.add_baseflow_columns(_make_df(Q, start="2001-01-01"), cfg)
    recs = rid.identify_recessions(df, cfg)
    assert len(recs) == 0


def _swe_water_year(peak_doy, peak_val, meltout_doy, year=2001, wy_start="2000-10-01"):
    """Build a daily SWE series for one water year: 0 in fall, ramp to a peak,
    melt to 0 at meltout_doy, 0 after."""
    idx = pd.date_range(wy_start, periods=365, freq="D")
    doy = idx.dayofyear.values
    # Triangular snowpack: rise to peak at peak_doy, linear melt to 0 at meltout_doy.
    swe = np.zeros(len(idx))
    # Accumulation from ~doy 300 (prev year) wrapping; approximate with a peak at peak_doy.
    for i, d in enumerate(doy):
        if d <= peak_doy:
            swe[i] = peak_val * max(0.0, (d) / peak_doy)
        elif d <= meltout_doy:
            swe[i] = peak_val * max(0.0, (meltout_doy - d) / (meltout_doy - peak_doy))
        else:
            swe[i] = 0.0
    return pd.Series(swe, index=idx)


def test_meltout_dates_after_peak():
    """Meltout is the first snow-free day AFTER the peak, not incidental fall zeros."""
    swe = _swe_water_year(peak_doy=100, peak_val=20.0, meltout_doy=160)
    melt = rid._io.meltout_dates(swe, water_year_start_month=10,
                                 swe_threshold=0.0, min_peak_swe=2.0)
    # One water year present; meltout near doy 161 (first zero after linear melt).
    d = melt.dropna().iloc[0]
    assert 158 <= d.dayofyear <= 163


def test_meltout_skips_low_snow_years():
    """A year whose peak SWE is below min_peak_swe yields NaT (no reliable meltout)."""
    swe = _swe_water_year(peak_doy=100, peak_val=0.5, meltout_doy=160)
    melt = rid._io.meltout_dates(swe, min_peak_swe=2.0)
    assert melt.dropna().empty


def test_dynamic_season_start_is_meltout_plus_buffer(cfg, tmp_path):
    """swe_meltout mode: season start = meltout + buffer_days, per calendar year."""
    swe = _swe_water_year(peak_doy=100, peak_val=20.0, meltout_doy=160)
    p = tmp_path / "swe.csv"
    swe.rename("SWE").rename_axis("datetime").to_csv(p, header=True)

    cfg.recession.season.mode = "swe_meltout"
    cfg.recession.season.meltout.swe_path = str(p)
    cfg.recession.season.meltout.buffer_days = 28
    # Pin thresholds: this synthetic pack peaks at 20 (unit-agnostic), so it must not
    # be judged against the mm-scale basin defaults.
    cfg.recession.season.meltout.swe_threshold = 0.0
    cfg.recession.season.meltout.min_peak_swe = 2.0
    cfg.recession.season.meltout.search_start_month = 1
    rid._MELTOUT_CACHE.clear()

    starts = rid.dynamic_season_starts(cfg)
    melt_doy = 161
    # season start ~ meltout + 28 days.
    start_doy = int(starts.dropna().iloc[0].dayofyear)
    assert abs(start_doy - (melt_doy + 28)) <= 3


def test_fallback_mean_meltout(cfg, tmp_path):
    """Years with no reliable meltout inherit the mean meltout doy + buffer,
    not an arbitrary fixed calendar date."""
    frames = []
    for yr, peak in [(2001, 300.0), (2002, 300.0), (2003, 10.0)]:  # 2003: no real pack
        idx = pd.date_range(f"{yr}-03-01", f"{yr}-09-30", freq="D")
        s = pd.Series(0.0, index=idx)
        s.loc[f"{yr}-03-01":f"{yr}-05-31"] = peak      # melts out ~Jun 1
        frames.append(s)
    swe = pd.concat(frames)
    p = tmp_path / "swe.csv"
    swe.rename("SWE").rename_axis("datetime").to_csv(p, header=True)

    m = cfg.recession.season.meltout
    m.swe_path = str(p); m.swe_threshold = 25.0; m.min_peak_swe = 100.0
    m.buffer_days = 14; m.search_start_month = 3; m.fallback = "mean_meltout"
    rid._MELTOUT_CACHE.clear()

    starts = rid.dynamic_season_starts(cfg)
    assert pd.isna(starts.get(2003))                    # low-peak year unresolved
    resolved_doy = pd.DatetimeIndex(starts.dropna()).dayofyear.to_numpy().mean()
    assert rid.fallback_start_doy(cfg) == int(round(resolved_doy))

    m.fallback = "fixed"; m.fallback_doy_start = 182
    assert rid.fallback_start_doy(cfg) == 182


def test_alpha_sensitivity_monotone(cfg):
    """Higher alpha attributes more quickflow -> fewer/shorter recession days."""
    from mrc import io
    df = io.generate_synthetic_record(cfg, n_years=8)
    table = rid.alpha_sensitivity(df, cfg, alphas=[0.9, 0.925, 0.95])
    days = table.sort_values("alpha")["total_recession_days"].to_numpy()
    assert days[0] >= days[-1]  # non-increasing with alpha
