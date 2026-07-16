"""Tests for data loading and gap handling."""
import numpy as np
import pandas as pd

from mrc import io


def test_mixed_time_of_day_does_not_null_days(cfg, tmp_path):
    """Daily records with a shifting time-of-day (e.g. MST 07:00 vs MDT 06:00)
    must key by calendar day, not misalign the grid and null out off-grid days."""
    dates = pd.date_range("2000-01-01", periods=100, freq="D")
    # Alternate the timestamp between 06:00 and 07:00, as real USGS exports do.
    times = ["06:00:00" if i % 2 == 0 else "07:00:00" for i in range(len(dates))]
    stamps = [f"{d.date()} {t}" for d, t in zip(dates, times)]
    raw = pd.DataFrame({"date": stamps, "Q": np.linspace(20, 5, len(dates))})
    p = tmp_path / "q.csv"
    raw.to_csv(p, index=False)

    cfg.io.streamflow.path = str(p)
    cfg.io.streamflow.format = "generic"
    cfg.io.streamflow.date_col = "date"
    cfg.io.streamflow.flow_col = "Q"
    df = io.load_streamflow(cfg)

    assert len(df) == 100
    assert int(df["Q"].isna().sum()) == 0          # no spurious NaNs
    assert len(io.detect_gaps(df, "Q")) == 0        # and no phantom gaps


def test_min_flow_masks_low_flows_globally(cfg, tmp_path):
    """preprocess.min_flow masks Q <= threshold at LOAD, so low flows (and zeros)
    never reach any downstream stage, and show up in the gap report."""
    dates = pd.date_range("2000-01-01", periods=20, freq="D")
    q = np.full(20, 5.0)
    q[10] = 0.0      # exact zero -> would break a log transform
    q[11] = 0.05     # below the screen
    raw = pd.DataFrame({"date": dates.astype(str), "Q": q})
    p = tmp_path / "q.csv"
    raw.to_csv(p, index=False)

    cfg.io.streamflow.path = str(p)
    cfg.io.streamflow.format = "generic"
    cfg.io.streamflow.date_col = "date"
    cfg.io.streamflow.flow_col = "Q"
    cfg.preprocess.min_flow = 0.1

    df = io.load_streamflow(cfg)
    assert df["Q"].isna().sum() == 2               # both masked
    assert np.isnan(df["Q"].iloc[10]) and np.isnan(df["Q"].iloc[11])
    assert (df["Q"].dropna() > 0.1).all()          # nothing at/below the screen survives
    gaps = io.detect_gaps(df, "Q")                 # and they surface as a gap
    assert len(gaps) == 1 and gaps[0].length_days == 2


def test_min_flow_disabled_keeps_low_flows(cfg, tmp_path):
    """With min_flow effectively off, low positive flows are retained."""
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    raw = pd.DataFrame({"date": dates.astype(str), "Q": np.full(10, 0.05)})
    p = tmp_path / "q.csv"
    raw.to_csv(p, index=False)
    cfg.io.streamflow.path = str(p)
    cfg.io.streamflow.format = "generic"
    cfg.io.streamflow.date_col = "date"
    cfg.io.streamflow.flow_col = "Q"
    cfg.preprocess.min_flow = 1e-6
    df = io.load_streamflow(cfg)
    assert df["Q"].notna().all()


def test_date_end_truncates_before_grid(cfg, tmp_path):
    """date_end cuts the record at load, so an excluded tail is not a giant gap."""
    dates = pd.date_range("2000-01-01", periods=100, freq="D")
    raw = pd.DataFrame({"date": dates.astype(str), "Q": np.full(100, 5.0)})
    p = tmp_path / "q.csv"
    raw.to_csv(p, index=False)
    cfg.io.streamflow.path = str(p)
    cfg.io.streamflow.format = "generic"
    cfg.io.streamflow.date_col = "date"
    cfg.io.streamflow.flow_col = "Q"
    cfg.preprocess.date_end = "2000-02-01"

    df = io.load_streamflow(cfg)
    assert df.index.max() == pd.Timestamp("2000-02-01")
    assert len(df) == 32
    assert len(io.detect_gaps(df, "Q")) == 0     # truncation is not a gap


def _write_q(tmp_path, start, n, name="q.csv"):
    dates = pd.date_range(start, periods=n, freq="D")
    p = tmp_path / name
    pd.DataFrame({"date": dates.astype(str), "Q": np.full(n, 5.0)}).to_csv(p, index=False)
    return p


def _write_swe(tmp_path, start, n, name="swe.csv"):
    dates = pd.date_range(start, periods=n, freq="D")
    p = tmp_path / name
    pd.DataFrame({"datetime": dates.astype(str), "SWE": np.full(n, 300.0)}).to_csv(p, index=False)
    return p


def _wire(cfg, qp, swep):
    cfg.io.streamflow.path = str(qp)
    cfg.io.streamflow.format = "generic"
    cfg.io.streamflow.date_col = "date"
    cfg.io.streamflow.flow_col = "Q"
    cfg.recession.season.mode = "swe_meltout"
    cfg.recession.season.meltout.swe_path = str(swep)
    cfg.recession.season.meltout.date_col = "datetime"
    cfg.recession.season.meltout.swe_col = "SWE"
    cfg.preprocess.align_start_to_swe = True
    cfg.preprocess.date_start = None


def test_align_start_to_swe_when_swe_starts_later(cfg, tmp_path):
    """Q starts first -> analysis begins at the SWE record start."""
    qp = _write_q(tmp_path, "2000-01-01", 800)
    swep = _write_swe(tmp_path, "2001-06-01", 300)
    _wire(cfg, qp, swep)
    df = io.load_streamflow(cfg)
    assert df.index.min() == pd.Timestamp("2001-06-01")


def test_align_start_to_swe_when_q_starts_later(cfg, tmp_path):
    """SWE starts first -> the later Q start wins (alignment is a lower bound)."""
    qp = _write_q(tmp_path, "2002-01-01", 300)
    swep = _write_swe(tmp_path, "2000-01-01", 900)
    _wire(cfg, qp, swep)
    df = io.load_streamflow(cfg)
    assert df.index.min() == pd.Timestamp("2002-01-01")


def test_manual_date_start_overrides_swe_alignment(cfg, tmp_path):
    """An explicit date_start always wins, so gauges can be forced onto one window."""
    qp = _write_q(tmp_path, "2000-01-01", 800)
    swep = _write_swe(tmp_path, "2001-06-01", 300)
    _wire(cfg, qp, swep)
    cfg.preprocess.date_start = "2000-07-01"
    df = io.load_streamflow(cfg)
    assert df.index.min() == pd.Timestamp("2000-07-01")


def test_align_start_disabled(cfg, tmp_path):
    """With alignment off, the full Q record is kept."""
    qp = _write_q(tmp_path, "2000-01-01", 800)
    swep = _write_swe(tmp_path, "2001-06-01", 300)
    _wire(cfg, qp, swep)
    cfg.preprocess.align_start_to_swe = False
    df = io.load_streamflow(cfg)
    assert df.index.min() == pd.Timestamp("2000-01-01")


def test_load_swe_converts_inches_to_mm(tmp_path):
    """A SNOTEL export carrying SWE_units='in' is converted to mm automatically."""
    dates = pd.date_range("2001-01-01", periods=5, freq="D")
    raw = pd.DataFrame({"datetime": dates.astype(str), "SWE": [1.0, 2.0, 0.0, 10.0, 26.5],
                        "SWE_units": ["in"] * 5})
    p = tmp_path / "swe.csv"
    raw.to_csv(p, index=False)
    s = io.load_swe(p, date_col="datetime", swe_col="SWE")
    assert np.allclose(s.to_numpy(), np.array([1.0, 2.0, 0.0, 10.0, 26.5]) * 25.4)


def test_load_swe_assumes_mm_without_units_column(tmp_path):
    """No units column -> values are taken as mm, unchanged."""
    dates = pd.date_range("2001-01-01", periods=3, freq="D")
    raw = pd.DataFrame({"date": dates.astype(str), "value": [0.0, 27.94, 300.0]})
    p = tmp_path / "swe.csv"
    raw.to_csv(p, index=False)
    s = io.load_swe(p, date_col="date", swe_col="value")
    assert np.allclose(s.to_numpy(), [0.0, 27.94, 300.0])


def test_load_swe_explicit_units_override(tmp_path):
    """An explicit units= wins over any units column."""
    dates = pd.date_range("2001-01-01", periods=2, freq="D")
    raw = pd.DataFrame({"date": dates.astype(str), "swe": [1.0, 2.0],
                        "swe_units": ["mm", "mm"]})
    p = tmp_path / "swe.csv"
    raw.to_csv(p, index=False)
    s = io.load_swe(p, date_col="date", swe_col="swe", units="in")
    assert np.allclose(s.to_numpy(), [25.4, 50.8])


def test_meltout_ignores_oct_dec_peak():
    """An early-season Oct-Dec storm never defines the meltout: the search is confined
    to the water year's calendar year, so a November peak cannot produce a fall date."""
    idx = pd.date_range("2014-10-01", "2015-09-30", freq="D")
    swe = pd.Series(0.0, index=idx)
    swe.loc["2014-11-01":"2014-11-20"] = 300.0    # biggest pack of the water year
    swe.loc["2015-03-01":"2015-04-15"] = 150.0    # modest spring pack
    swe.name = "swe"
    m = io.meltout_dates(swe, swe_threshold=25.0, min_peak_swe=100.0, search_start_month=1)
    assert m.dropna().iloc[0] == pd.Timestamp("2015-04-16")   # spring, not November


def test_meltout_search_start_month_rejects_midwinter_disappearance():
    """A pack that melts out in February (an anomalous year) is resolved when the
    window opens in January, but with a March-onward window the in-window peak falls
    below min_peak_swe and the year is left unresolved for the caller's fallback."""
    idx = pd.date_range("2014-10-01", "2015-09-30", freq="D")
    swe = pd.Series(0.0, index=idx)
    swe.loc["2015-01-01":"2015-02-05"] = 300.0    # melts out in early February
    swe.name = "swe"

    jan = io.meltout_dates(swe, swe_threshold=25.0, min_peak_swe=100.0, search_start_month=1)
    mar = io.meltout_dates(swe, swe_threshold=25.0, min_peak_swe=100.0, search_start_month=3)
    assert jan.dropna().iloc[0].month == 2        # mid-winter "meltout"
    assert pd.isna(mar.iloc[0])                   # rejected -> falls back


def test_detect_gaps_reports_true_gaps(cfg, tmp_path):
    """A genuinely missing block of days is reported as one gap (not interpolated)."""
    dates = pd.date_range("2000-01-01", periods=60, freq="D")
    q = np.linspace(30, 10, 60)
    raw = pd.DataFrame({"date": dates.astype(str), "Q": q}).drop(index=range(20, 30))
    p = tmp_path / "q.csv"
    raw.to_csv(p, index=False)

    cfg.io.streamflow.path = str(p)
    cfg.io.streamflow.format = "generic"
    cfg.io.streamflow.date_col = "date"
    cfg.io.streamflow.flow_col = "Q"
    df = io.load_streamflow(cfg)
    gaps = io.detect_gaps(df, "Q")
    assert len(gaps) == 1
    assert gaps[0].length_days == 10
