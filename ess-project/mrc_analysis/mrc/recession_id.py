"""Recession segment identification and screening.

A *recession segment* is a contiguous run of falling flow that survives a
sequence of screens designed to isolate clean baseflow recession:

1. dQ/dt < tol       - falling limb (small +tol allows for measurement noise)
2. drop N days after each peak - remove the quickflow-dominated early limb
3. quickflow-fraction gate      - require f/Q below a threshold each day
4. seasonal window   - restrict to a post-peak day-of-year range (no freshet limb)
5. forcing screen (if available) - break on rain / snowmelt input days
6. gap break         - never span a missing-data gap
7. minimum length    - keep only segments >= min_segment_length days

The forcing screen, when forcing is provided, takes precedence over the
quickflow-fraction screen (a physically identified input event overrides the
digital-filter heuristic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .filter import add_baseflow_columns
from . import io as _io


@dataclass
class Recession:
    """One screened recession segment (a view into the daily record)."""
    seg_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    length: int
    dates: pd.DatetimeIndex
    Q: np.ndarray
    t: np.ndarray                 # days since segment start (0..length-1)
    doy_start: int
    water_year: int
    flags: List[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "seg_id": self.seg_id,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "doy_start": self.doy_start,
            "water_year": self.water_year,
            "q_start": float(self.Q[0]),
            "q_end": float(self.Q[-1]),
            "flags": ";".join(self.flags),
        }


# --------------------------------------------------------------------------- #
# Eligibility masks
# --------------------------------------------------------------------------- #
def _water_year(dates: pd.DatetimeIndex, start_month: int) -> np.ndarray:
    return dates.year.values + (dates.month.values >= start_month).astype(int)


# --- Seasonal window (fixed day-of-year, or dynamic SWE-meltout driven) ------ #
_MELTOUT_CACHE: dict = {}


def dynamic_season_starts(cfg: Config) -> pd.Series:
    """Per-water-year recession-season start = SWE meltout date + buffer_days.

    Reads the SNOTEL-style SWE file named in
    ``recession.season.meltout.swe_path``, finds each water year's snow-free
    (meltout) date after the peak, and adds ``buffer_days`` (default 4 weeks).
    Water years without a reliable meltout get NaT (caller applies a fallback).
    Cached by the meltout parameters so repeated calls (e.g. alpha-sensitivity)
    don't re-read the file.
    """
    mo = cfg.recession.season.meltout
    if mo.swe_path is None:
        raise ValueError(
            "season.mode == 'swe_meltout' requires recession.season.meltout.swe_path"
        )
    wy_start = cfg.temporal.window.water_year_start_month
    search_start = int(getattr(mo, "search_start_month", 3))
    swe_units = getattr(mo, "swe_units", "auto")
    swe_units_col = getattr(mo, "swe_units_col", None)
    key = (str(mo.swe_path), mo.date_col, mo.swe_col, float(mo.swe_threshold),
           float(mo.min_peak_swe), int(mo.buffer_days), int(wy_start), search_start,
           str(swe_units), str(swe_units_col))
    if key in _MELTOUT_CACHE:
        return _MELTOUT_CACHE[key]

    swe = _io.load_swe(mo.swe_path, date_col=mo.date_col, swe_col=mo.swe_col,
                       units=swe_units, units_col=swe_units_col)
    melt = _io.meltout_dates(swe, water_year_start_month=wy_start,
                             swe_threshold=mo.swe_threshold, min_peak_swe=mo.min_peak_swe,
                             search_start_month=search_start)
    starts = melt + pd.to_timedelta(mo.buffer_days, unit="D")
    _MELTOUT_CACHE[key] = starts
    return starts


def fallback_start_doy(cfg: Config) -> int:
    """Season-start day-of-year for water years with no reliable meltout.

    ``fallback: mean_meltout`` uses the mean meltout day-of-year across the resolved
    years plus ``buffer_days`` -- so a year we cannot date inherits the basin's
    typical snow-disappearance timing rather than an arbitrary calendar date.
    ``fallback: fixed`` uses the static ``fallback_doy_start``.
    """
    mo = cfg.recession.season.meltout
    if getattr(mo, "fallback", "fixed") != "mean_meltout":
        return int(mo.fallback_doy_start)

    starts = dynamic_season_starts(cfg)          # already meltout + buffer
    doys = starts.dropna()
    if doys.empty:
        return int(mo.fallback_doy_start)
    return int(round(float(pd.DatetimeIndex(doys).dayofyear.to_numpy().mean())))


def _fixed_season_mask(dates: pd.DatetimeIndex, doy_start: int, doy_end: int) -> np.ndarray:
    doy = dates.dayofyear.values
    if doy_start <= doy_end:
        return (doy >= doy_start) & (doy <= doy_end)
    return (doy >= doy_start) | (doy <= doy_end)   # wrap-around window


def _season_mask(dates: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """Per-day in-season boolean.

    fixed mode       -> static [doy_start, doy_end] window.
    swe_meltout mode -> per-water-year start = meltout+buffer (fallback to
                        fallback_doy_start), with the same doy_end upper bound.
    """
    rec = cfg.recession
    if not rec.season.enabled:
        return np.ones(len(dates), dtype=bool)

    mode = getattr(rec.season, "mode", "fixed")
    if mode == "fixed":
        return _fixed_season_mask(dates, rec.season.doy_start, rec.season.doy_end)

    if mode != "swe_meltout":
        raise ValueError(f"Unknown season.mode '{mode}' (use 'fixed' or 'swe_meltout')")

    starts = dynamic_season_starts(cfg)   # indexed by meltout calendar year
    fb_doy = fallback_start_doy(cfg)

    # Map each day to a season start by CALENDAR year: the recession season runs
    # from that year's spring meltout (+buffer) through doy_end (~Dec 1), all
    # within one calendar year. (Water-year labelling would split the Oct-Dec
    # tail into the next water year, away from its own spring meltout.)
    start_per_day = np.empty(len(dates), dtype="datetime64[ns]")
    cal_year = dates.year.values
    for i, y in enumerate(cal_year):
        ts = starts.get(int(y), pd.NaT)
        if pd.isna(ts):
            ts = _fallback_start_date(int(y), fb_doy)
        start_per_day[i] = np.datetime64(ts)

    after_start = dates.values >= start_per_day
    below_end = dates.dayofyear.values <= rec.season.doy_end
    return after_start & below_end


def _fallback_start_date(calendar_year: int, fallback_doy: int) -> pd.Timestamp:
    """Calendar date of ``fallback_doy`` within the given calendar year."""
    return pd.Timestamp(year=calendar_year, month=1, day=1) + pd.to_timedelta(fallback_doy - 1, unit="D")


def season_starts_frame(cfg: Config) -> pd.DataFrame:
    """Tidy table of dynamic season starts for inspection (swe_meltout mode).

    Columns: calendar_year, meltout_date, season_start (meltout+buffer), doy_start,
    used_fallback. Useful for QC of the meltout detection before running.
    """
    mo = cfg.recession.season.meltout
    starts = dynamic_season_starts(cfg)
    fb_doy = fallback_start_doy(cfg)
    rows = []
    for year, ts in starts.items():
        if pd.isna(ts):
            start = _fallback_start_date(int(year), fb_doy)
            rows.append({"calendar_year": int(year), "meltout_date": pd.NaT,
                         "season_start": start, "doy_start": int(start.dayofyear),
                         "used_fallback": True})
        else:
            rows.append({"calendar_year": int(year), "meltout_date": ts - pd.to_timedelta(mo.buffer_days, unit="D"),
                         "season_start": ts, "doy_start": int(ts.dayofyear),
                         "used_fallback": False})
    return pd.DataFrame(rows).sort_values("calendar_year").reset_index(drop=True)


def build_eligibility(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Annotate the daily record with per-day eligibility flags used for cutting.

    Adds boolean columns:
      falling      - dQ/dt below the noise tolerance
      season_ok    - day-of-year within the recession season
      qf_ok        - quickflow fraction below threshold
      forcing_ok   - no rain/melt input on this day (True if no forcing)
      flag_ok      - not dropped by qualification flags
      break_here   - a hard segment boundary starts at this day (gap / input event)
    """
    rec = cfg.recession
    out = df.copy()
    Q = out["Q"].to_numpy(dtype=float)

    # dQ/dt with a small positive tolerance (fraction of local Q).
    dQ = np.diff(Q, prepend=np.nan)
    tol = rec.dqdt_tolerance * np.where(np.isfinite(Q), np.abs(Q), 0.0)
    out["falling"] = dQ < tol   # NaN comparisons -> False (gap days excluded)

    # Seasonal window.
    out["season_ok"] = _season_mask(out.index, cfg)

    # Quickflow-fraction gate (requires baseflow columns; add if missing).
    if "qf_frac" not in out.columns:
        out = add_baseflow_columns(out, cfg)
    out["qf_ok"] = out["qf_frac"].to_numpy() <= rec.quickflow_frac_max

    # Qualification-flag screen.
    out["flag_ok"] = _flag_ok(out, cfg)

    # Forcing screen -> per-day "input event" that forces a break.
    forcing_ok, input_event = _forcing_screen(out, cfg)
    out["forcing_ok"] = forcing_ok

    # Hard breaks: missing data (gap) or an identified input event.
    out["break_here"] = (~np.isfinite(Q)) | input_event
    return out


def _flag_ok(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    flags = cfg.flags
    ok = np.ones(len(df), dtype=bool)
    if "flag" not in df.columns:
        return ok
    col = df["flag"].astype("string").fillna("")
    if flags.drop_ice_affected:
        for code in flags.ice_affected:
            ok &= ~col.str.contains(code, case=False, na=False).to_numpy()
    if flags.drop_estimated:
        for code in flags.estimated:
            ok &= ~col.str.contains(code, case=False, na=False).to_numpy()
    return ok


def _forcing_screen(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (forcing_ok, input_event) per day.

    An input event is a rain day above the precip threshold, or a melt day
    (SWE decrease beyond a threshold, or a temperature-index proxy when SWE is
    absent). Days with an input event get forcing_ok=False and force a break.
    Returns all-True / all-False if no forcing columns are present.
    """
    n = len(df)
    fs = cfg.recession.forcing_screen
    input_event = np.zeros(n, dtype=bool)

    has_precip = "precip" in df.columns
    has_swe = "swe" in df.columns
    has_temp = "tair" in df.columns
    if not (has_precip or has_swe or has_temp):
        return np.ones(n, dtype=bool), input_event

    if has_precip:
        p = df["precip"].to_numpy(dtype=float)
        input_event |= np.nan_to_num(p, nan=0.0) > fs.precip_threshold

    if has_swe:
        swe = df["swe"].to_numpy(dtype=float)
        dswe = np.diff(swe, prepend=swe[:1])
        # Melt = SWE decrease beyond threshold (negative change).
        input_event |= (-np.nan_to_num(dswe, nan=0.0)) > fs.melt_swe_drop
    elif has_temp and fs.use_temp_index:
        t = df["tair"].to_numpy(dtype=float)
        input_event |= np.nan_to_num(t, nan=-999.0) > fs.melt_temp_index

    return ~input_event, input_event


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def identify_recessions(df: pd.DataFrame, cfg: Config) -> List[Recession]:
    """Cut the record into screened recession segments.

    A day is *inside a recession* when it is falling, in-season, flag-ok, and
    (forcing_ok OR qf_ok) — with forcing taking precedence: if forcing is present
    a forcing input event breaks the segment regardless of the filter; if forcing
    is absent, the qf_frac gate governs. Breaks also occur on gaps/input events.
    """
    rec = cfg.recession
    ann = build_eligibility(df, cfg)
    forcing_present = any(c in df.columns for c in ("precip", "swe", "tair"))

    inside = ann["falling"].to_numpy() & ann["season_ok"].to_numpy() & ann["flag_ok"].to_numpy()
    if forcing_present:
        # Forcing precedence: clean day = no input event; qf gate advisory.
        inside &= ann["forcing_ok"].to_numpy()
    else:
        inside &= ann["qf_ok"].to_numpy()

    break_here = ann["break_here"].to_numpy()
    Q = ann["Q"].to_numpy(dtype=float)
    dates = ann.index

    wy = _water_year(dates, cfg.temporal.window.water_year_start_month)
    doy = dates.dayofyear.values

    # Walk the record, opening/closing runs of `inside` and cutting on breaks.
    raw_runs: List[tuple] = []
    i, n = 0, len(ann)
    while i < n:
        if inside[i] and not break_here[i] and np.isfinite(Q[i]):
            j = i
            while j < n and inside[j] and not break_here[j] and np.isfinite(Q[j]):
                j += 1
            raw_runs.append((i, j))
            i = j
        else:
            i += 1

    # Post-process: drop N days after the peak, enforce monotone strictness,
    # enforce minimum length.
    recessions: List[Recession] = []
    seg_id = 0
    drop = rec.drop_days_after_peak
    for (s, e) in raw_runs:
        s2 = s + drop
        if e - s2 < rec.min_segment_length:
            continue
        qseg = Q[s2:e]
        dseg = dates[s2:e]
        recessions.append(
            Recession(
                seg_id=seg_id,
                start=dseg[0],
                end=dseg[-1],
                length=len(qseg),
                dates=dseg,
                Q=qseg,
                t=np.arange(len(qseg), dtype=float),
                doy_start=int(doy[s2]),
                water_year=int(wy[s2]),
            )
        )
        seg_id += 1

    return recessions


def recessions_to_frame(recessions: List[Recession]) -> pd.DataFrame:
    """Tidy per-recession summary table (pre-fit; fit metrics added later)."""
    if not recessions:
        return pd.DataFrame(
            columns=["seg_id", "start", "end", "length", "doy_start",
                     "water_year", "q_start", "q_end", "flags"]
        )
    return pd.DataFrame([r.as_row() for r in recessions])


# --------------------------------------------------------------------------- #
# Alpha-sensitivity utility
# --------------------------------------------------------------------------- #
def alpha_sensitivity(df: pd.DataFrame, cfg: Config,
                      alphas: Optional[List[float]] = None) -> pd.DataFrame:
    """Re-run recession selection across filter alphas; report how it changes.

    Only meaningful when the quickflow-fraction gate is active (no forcing), but
    it still runs with forcing present to show robustness. Returns one row per
    alpha with segment count, total recession days, and median segment length.
    """
    if alphas is None:
        alphas = list(cfg.baseflow.sensitivity_alphas)

    rows = []
    for a in alphas:
        annotated = add_baseflow_columns(df, cfg, alpha=a)
        recs = identify_recessions(annotated, cfg)
        lengths = [r.length for r in recs]
        rows.append(
            {
                "alpha": a,
                "n_segments": len(recs),
                "total_recession_days": int(sum(lengths)),
                "median_length": float(np.median(lengths)) if lengths else 0.0,
                "max_length": int(max(lengths)) if lengths else 0,
            }
        )
    return pd.DataFrame(rows)
