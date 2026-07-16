"""Data loading, gap detection, and a synthetic-record generator.

The pipeline operates on a single daily DataFrame indexed by date with, at
minimum, a ``Q`` column. Optional columns: ``flag`` (qualification code) and
forcing (``precip``, ``swe``, ``tair``). All parsing is driven by the config.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class GapInfo:
    """A run of missing daily timesteps between two present dates."""
    start: pd.Timestamp   # first missing day
    end: pd.Timestamp     # last missing day
    length_days: int


# --------------------------------------------------------------------------- #
# Streamflow loading
# --------------------------------------------------------------------------- #
def load_streamflow(cfg: Config) -> pd.DataFrame:
    """Load streamflow to a daily DataFrame with columns ``Q`` (+ optional ``flag``).

    If ``io.streamflow.path`` is null, a reproducible synthetic 50-year record
    is generated instead (useful for tests / demos).
    """
    sf = cfg.io.streamflow
    if sf.path is None:
        return generate_synthetic_record(cfg)

    path = Path(sf.path)
    if sf.format == "usgs_rdb":
        df = _read_usgs_rdb(path, cfg)
    else:
        df = _read_generic_csv(path, cfg)

    df = _finalize_daily(df, cfg)
    return df


def _read_generic_csv(path: Path, cfg: Config) -> pd.DataFrame:
    sf = cfg.io.streamflow
    raw = pd.read_csv(
        path,
        comment=sf.comment or None,
        na_values=list(sf.na_values),
        keep_default_na=True,
    )
    if sf.date_col not in raw.columns:
        raise KeyError(f"date_col '{sf.date_col}' not in {list(raw.columns)}")
    if sf.flow_col not in raw.columns:
        raise KeyError(f"flow_col '{sf.flow_col}' not in {list(raw.columns)}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[sf.date_col], format=sf.date_format, errors="coerce")
    out["Q"] = pd.to_numeric(raw[sf.flow_col], errors="coerce")
    if sf.flag_col and sf.flag_col in raw.columns:
        out["flag"] = raw[sf.flag_col].astype("string")
    return out


def _read_usgs_rdb(path: Path, cfg: Config) -> pd.DataFrame:
    """Read a USGS RDB (tab-delimited) daily-values export.

    RDB files have '#' comment lines, a header row, then a format row
    ('5s', '15s', ...). Discharge columns contain the parameter-code hint
    (default '00060'); each value column ``X`` has a companion flag ``X_cd``.
    """
    sf = cfg.io.streamflow
    hint = sf.usgs.value_col_hint

    # Locate the header line (first non-comment line).
    header_idx = None
    with path.open() as fh:
        for i, line in enumerate(fh):
            if not line.startswith("#"):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"No header row found in RDB file {path}")

    raw = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=0,
        skiprows=[header_idx + 1],  # drop the RDB format row
        dtype=str,
    )

    date_col = "datetime" if "datetime" in raw.columns else sf.date_col
    value_cols = [c for c in raw.columns if hint in c and not c.endswith("_cd")]
    if not value_cols:
        raise KeyError(
            f"No discharge column containing '{hint}' in {list(raw.columns)}"
        )
    vcol = value_cols[0]
    fcol = f"{vcol}_cd"

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    out["Q"] = pd.to_numeric(raw[vcol], errors="coerce")
    if fcol in raw.columns:
        out["flag"] = raw[fcol].astype("string")
    return out


def swe_record_start(cfg: Config) -> Optional[pd.Timestamp]:
    """First date of the SWE record backing the swe_meltout season, if any."""
    season = cfg.recession.season
    if getattr(season, "mode", "fixed") != "swe_meltout":
        return None
    mo = season.meltout
    if not mo.swe_path:
        return None
    swe = load_swe(mo.swe_path, date_col=mo.date_col, swe_col=mo.swe_col,
                   units=getattr(mo, "swe_units", "auto"),
                   units_col=getattr(mo, "swe_units_col", None))
    return swe.index.min() if len(swe) else None


def _resolve_start_date(cfg: Config) -> Optional[pd.Timestamp]:
    """Analysis start date.

    ``preprocess.date_start`` always wins (a manual override, so several gauges can
    be forced onto an identical window). Otherwise, when
    ``preprocess.align_start_to_swe`` is set and the season is meltout-driven, the
    record starts at the beginning of the SWE record — so every analysis year has a
    real meltout rather than a fallback. A later streamflow start simply wins, since
    the filter is a lower bound.
    """
    pp = cfg.preprocess
    manual = getattr(pp, "date_start", None)
    if manual:
        return pd.Timestamp(manual)
    if getattr(pp, "align_start_to_swe", False):
        return swe_record_start(cfg)
    return None


def _finalize_daily(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sort, dedupe, scale, floor/negative-handle, and reindex to a full daily grid."""
    pp = cfg.preprocess
    units = cfg.io.units

    df = df.dropna(subset=["date"]).sort_values("date")
    # Key by calendar day: daily records often carry a time-of-day that shifts
    # (e.g. MST 07:00 vs MDT 06:00), which would misalign the daily grid and
    # spuriously null out every off-grid day. Normalize to midnight first.
    df["date"] = df["date"].dt.normalize()
    df = df.drop_duplicates(subset="date", keep="first").set_index("date")

    df["Q"] = df["Q"] * units.flow_scale
    if pp.drop_negative:
        df.loc[df["Q"] < 0, "Q"] = np.nan

    # Global low-flow screen. Flows at or below `min_flow` are masked to NaN, so
    # they are excluded everywhere downstream (recession segments break on them,
    # and detect_gaps() reports them) rather than being silently filtered later.
    # This also guards the log transform against zeros. Units are the analysis
    # units, i.e. after `flow_scale`.
    if pp.min_flow and pp.min_flow > 0:
        df.loc[df["Q"] <= pp.min_flow, "Q"] = np.nan

    # Optional record truncation, applied before the daily grid is built so an
    # excluded tail does not appear as a giant gap (e.g. Tuolumne's incomplete
    # WY2024 -> set date_end: "2023-09-30" to end at the close of WY2023).
    start = _resolve_start_date(cfg)
    if start is not None:
        df = df[df.index >= start]      # a later Q start simply wins
    if getattr(pp, "date_end", None):
        df = df[df.index <= pd.Timestamp(pp.date_end)]
    if df.empty:
        raise ValueError("No streamflow rows remain after preprocess.date_start/date_end")

    # Reindex onto a gap-free daily grid so downstream code can trust spacing;
    # genuinely missing days become NaN (and are reported by detect_gaps()).
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df.index.name = "date"
    if "flag" in df.columns:
        df["flag"] = df["flag"].astype("string")
    return df


# --------------------------------------------------------------------------- #
# Forcing loading
# --------------------------------------------------------------------------- #
def load_forcing(cfg: Config) -> Optional[pd.DataFrame]:
    """Load optional daily forcing (precip / swe / tair). Returns None if absent."""
    fc = cfg.io.forcing
    if fc.path is None:
        return None
    raw = pd.read_csv(fc.path, comment=fc.comment or None)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[fc.date_col], format=fc.date_format, errors="coerce").dt.normalize()
    for src, dst in [(fc.precip_col, "precip"), (fc.swe_col, "swe"), (fc.temp_col, "tair")]:
        if src and src in raw.columns:
            out[dst] = pd.to_numeric(raw[src], errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
    return out


def merge_forcing(flow: pd.DataFrame, forcing: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Left-join forcing onto the daily flow grid (missing forcing -> NaN)."""
    if forcing is None:
        return flow
    return flow.join(forcing.reindex(flow.index), how="left")


# --------------------------------------------------------------------------- #
# SWE observations (for dynamic, meltout-driven recession season)
# --------------------------------------------------------------------------- #
#: Multipliers converting a SWE unit label to millimetres.
SWE_TO_MM = {"mm": 1.0, "millimeter": 1.0, "millimetre": 1.0,
             "cm": 10.0, "centimeter": 10.0, "centimetre": 10.0,
             "m": 1000.0, "meter": 1000.0, "metre": 1000.0,
             "in": 25.4, "inch": 25.4, "inches": 25.4}


def _swe_unit_factor(raw: pd.DataFrame, swe_col: str, units: str,
                     units_col: Optional[str]) -> tuple[float, str]:
    """Resolve the multiplier that converts the SWE column to millimetres.

    ``units="auto"`` looks for an explicit units column (``units_col``, else
    ``"<swe_col>_units"``, as SNOTEL exports carry). If none exists, the data is
    assumed already in mm.
    """
    if units and units != "auto":
        label = str(units).strip().lower()
    else:
        col = units_col or f"{swe_col}_units"
        if col not in raw.columns:
            return 1.0, "mm (assumed; no units column)"
        vals = raw[col].dropna().astype(str).str.strip().str.lower().unique()
        if len(vals) == 0:
            return 1.0, "mm (assumed; units column empty)"
        if len(vals) > 1:
            raise ValueError(f"SWE units column '{col}' has mixed units: {list(vals)}")
        label = vals[0]

    if label not in SWE_TO_MM:
        raise ValueError(f"Unrecognized SWE unit {label!r}; known: {sorted(SWE_TO_MM)}")
    return SWE_TO_MM[label], label


def load_swe(path, date_col: str = "datetime", swe_col: str = "SWE",
             comment: str = "#", units: str = "auto",
             units_col: Optional[str] = None) -> pd.Series:
    """Load a SNOTEL-style daily SWE record to a date-indexed Series, **in mm**.

    Datetimes may be timezone-aware (e.g. ``...+00:00``); they are parsed to UTC,
    stripped of tz, and floored to the day. Duplicate days keep the first value.

    SWE is always returned in millimetres so thresholds (``swe_threshold``,
    ``min_peak_swe``) mean the same thing across stations. With ``units="auto"``
    the unit is read from ``<swe_col>_units`` when present (SNOTEL exports carry
    ``SWE_units='in'``); otherwise mm is assumed. Pass ``units="in"`` etc. to force.
    """
    raw = pd.read_csv(path, comment=comment or None)
    if date_col not in raw.columns:
        raise KeyError(f"date_col '{date_col}' not in {list(raw.columns)}")
    if swe_col not in raw.columns:
        raise KeyError(f"swe_col '{swe_col}' not in {list(raw.columns)}")

    factor, _label = _swe_unit_factor(raw, swe_col, units, units_col)
    dt = pd.to_datetime(raw[date_col], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    swe = pd.to_numeric(raw[swe_col], errors="coerce") * factor
    s = pd.Series(swe.values, index=dt).dropna()
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s.index.name = "date"
    s.name = "swe_mm"
    return s


def meltout_dates(swe: pd.Series, *, water_year_start_month: int = 10,
                  swe_threshold: float = 0.0, min_peak_swe: float = 2.0,
                  search_start_month: int = 3) -> pd.Series:
    """Per-water-year snow-disappearance (meltout) date.

    For each water year: locate the peak SWE, then the FIRST day at/after the peak
    where ``SWE <= swe_threshold`` (the spring snow-free date — not the incidental
    zero SWE of early fall). Water years whose peak SWE is below ``min_peak_swe``
    (no real snowpack) or that never reach the threshold after the peak yield NaT.

    ``search_start_month`` (default 3 = March) confines both the peak and the
    disappearance to that month onward within the water year's calendar year. This
    guards two failure modes: (a) an early-season storm setting the annual peak in
    Oct-Dec, so the first sub-threshold day after it lands in the fall — Tuolumne
    WY2015, an extreme dry year, yields a 2014-11-30 meltout without this; and (b)
    a mid-winter melt-out in an anomalous year producing a Feb "disappearance" that
    is not a spring signal. Years whose in-window peak falls below ``min_peak_swe``
    (no credible spring snowpack) yield NaT and are left to the caller's fallback.

    Returns a Series indexed by integer water year, values = meltout Timestamp.
    """
    if swe.empty:
        return pd.Series(dtype="datetime64[ns]")
    wy = swe.index.year + (swe.index.month >= water_year_start_month).astype(int)
    out = {}
    for year, grp in swe.groupby(wy):
        grp = grp.sort_index()
        # Confine to <search_start_month> 1 .. Sep 30 of the water year's calendar year.
        grp = grp[(grp.index.year == int(year)) & (grp.index.month >= search_start_month)]
        if grp.empty:
            out[int(year)] = pd.NaT
            continue
        peak_val = grp.max()
        if not np.isfinite(peak_val) or peak_val < min_peak_swe:
            out[int(year)] = pd.NaT
            continue
        peak_date = grp.idxmax()
        after = grp.loc[peak_date:]
        snow_free = after[after <= swe_threshold]
        out[int(year)] = snow_free.index[0] if len(snow_free) else pd.NaT
    return pd.Series(out).sort_index()


# --------------------------------------------------------------------------- #
# Gap detection
# --------------------------------------------------------------------------- #
def detect_gaps(df: pd.DataFrame, col: str = "Q") -> List[GapInfo]:
    """Return runs of consecutive missing days in ``col`` on the daily grid.

    We never interpolate across these; recession segmentation breaks on them.
    """
    present = df[col].notna().values
    idx = df.index
    gaps: List[GapInfo] = []
    i = 0
    n = len(present)
    while i < n:
        if not present[i]:
            j = i
            while j < n and not present[j]:
                j += 1
            gaps.append(GapInfo(start=idx[i], end=idx[j - 1], length_days=j - i))
            i = j
        else:
            i += 1
    return gaps


def summarize_gaps(gaps: List[GapInfo]) -> pd.DataFrame:
    """Tidy table of gaps for logging/reporting."""
    if not gaps:
        return pd.DataFrame(columns=["start", "end", "length_days"])
    return pd.DataFrame(
        {
            "start": [g.start for g in gaps],
            "end": [g.end for g in gaps],
            "length_days": [g.length_days for g in gaps],
        }
    )


# --------------------------------------------------------------------------- #
# Synthetic record (reproducible) — snow-dominated mountain hydrograph
# --------------------------------------------------------------------------- #
def generate_synthetic_record(
    cfg: Config,
    *,
    n_years: int = 50,
    start: str = "1970-10-01",
    k_true: float = 0.035,
    k_trend_per_year: float = 0.0,
    with_forcing: bool = False,
) -> pd.DataFrame:
    """Generate a reproducible daily hydrograph with snowmelt freshet + recessions.

    Each water year: low winter baseflow, a spring/summer melt freshet peak, then
    a long summer/fall recession that decays exponentially at rate ``k_true``
    (optionally drifting by ``k_trend_per_year`` to exercise the trend test).
    Superimposed summer rain pulses create quickflow spikes to be screened out.

    Returns a DataFrame on a daily grid with ``Q`` (+ forcing columns if requested).
    Not physically calibrated — it exists to exercise the pipeline deterministically.
    """
    rng = np.random.default_rng(cfg.run.seed)
    idx = pd.date_range(start=start, periods=n_years * 365, freq="D")
    doy = idx.dayofyear.values
    year0 = idx[0].year
    years_elapsed = (idx.year.values - year0) + (idx.month.values >= 10).astype(int)

    Q = np.full(len(idx), np.nan)
    precip = np.zeros(len(idx))
    swe = np.zeros(len(idx))
    tair = np.zeros(len(idx))

    base = 1.5  # winter baseflow floor (analysis units)
    for wy in np.unique(years_elapsed):
        mask = years_elapsed == wy
        if not mask.any():
            continue
        d = doy[mask]
        # Snowpack: accumulates in winter, melts out through spring.
        peak_doy = 150 + rng.integers(-15, 15)          # ~late May freshet peak
        amp = 40.0 * (1.0 + 0.25 * rng.standard_normal())  # interannual variability
        amp = max(amp, 10.0)
        # Rising limb (accumulation->melt) then exponential summer recession.
        k_wy = k_true + k_trend_per_year * wy
        q = np.empty(d.shape)
        rising = d <= peak_doy
        q[rising] = base + amp * np.exp(-((d[rising] - peak_doy) ** 2) / (2 * 45.0 ** 2))
        t_since = d[~rising] - peak_doy
        q[~rising] = base + amp * np.exp(-k_wy * t_since)
        # Temperature-index-ish SWE proxy + air temp seasonal cycle.
        sw = np.clip(amp * 3 * (peak_doy - d) / peak_doy, 0, None)
        ta = 5 + 12 * np.sin(2 * np.pi * (d - 100) / 365.0)
        Q[mask] = q
        swe[mask] = sw
        tair[mask] = ta

    # Summer rain pulses -> quickflow spikes (things the screen must reject).
    n_pulses = int(0.02 * len(idx))
    pulse_idx = rng.integers(0, len(idx), size=n_pulses)
    for p in pulse_idx:
        if 160 <= doy[p] <= 330:  # only during recession season
            mag = rng.uniform(3, 25)
            precip[p] += rng.uniform(5, 30)
            decay = mag * np.exp(-np.arange(6) * 0.6)
            end = min(p + 6, len(Q))
            Q[p:end] += decay[: end - p]

    # Small multiplicative measurement noise (daily gauge records are smooth).
    Q *= 1.0 + 0.005 * rng.standard_normal(len(Q))
    Q = np.clip(Q, cfg.preprocess.min_flow, None)

    df = pd.DataFrame({"Q": Q}, index=idx)
    df.index.name = "date"
    if with_forcing:
        df["precip"] = precip
        df["swe"] = swe
        df["tair"] = tair
    return df
