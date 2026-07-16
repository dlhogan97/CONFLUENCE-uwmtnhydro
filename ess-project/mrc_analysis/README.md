# MRC Analysis — Master Recession Curve stationarity testing

A config-driven, modular pipeline that identifies clean baseflow recessions in a
long daily streamflow record, builds **master recession curves (MRCs)** on
moving multi-year windows, and tests whether the baseflow recession constant
`k` (equivalently the storage timescale `τ = 1/k`) is **stationary** over the
record. Built for snow-dominated mountain basins (East River, CO; Tuolumne, CA).

This is a plain module folder (not an installed package). Run it in place with
`python -m mrc.cli ...` from this directory.

---

## Quick start

```bash
cd ess-project/mrc_analysis

# Full pipeline on a synthetic 50-year record (no input data needed):
python -m mrc.cli run --config config/default.yaml --output-dir outputs --label demo

# Alpha-sensitivity of recession selection only:
python -m mrc.cli alpha --config config/default.yaml

# Run the tests:
python -m pytest
```

With no `io.streamflow.path` in the config, a reproducible synthetic hydrograph
is generated so the whole pipeline can be exercised out of the box. Point
`io.streamflow.path` at a real CSV to analyze observations.

---

## Layout

```
mrc_analysis/
├── config/default.yaml     # every threshold, documented inline
├── mrc/
│   ├── config.py           # lightweight dot-access config + YAML deep-merge loader
│   ├── io.py               # CSV/USGS-RDB loading, gaps, synthetic generator
│   ├── filter.py           # Lyne-Hollick recursive digital baseflow filter
│   ├── recession_id.py     # recession segment detection + screening
│   ├── mrc.py              # tail k fit + automated matching strip
│   ├── temporal.py         # windowing/per-year MRC series + Mann-Kendall/Sen
│   ├── plotting.py         # 3 standard figures + optional QC viewer
│   └── cli.py              # end-to-end orchestrator
├── tests/                  # filter, segment detection, tail-fit, matching, trend
└── pytest.ini
```

Module dependency order: `io → filter → recession_id → mrc → temporal → plotting`,
orchestrated by `cli`.

---

## Inputs

**Streamflow** (required): a daily CSV. Two readers, chosen by
`io.streamflow.format`:
- `generic` — set `date_col`, `flow_col`, and optional `flag_col`.
- `usgs_rdb` — tab-delimited USGS daily-values export; the discharge column is
  found by parameter-code hint (default `00060`) and paired with its `_cd` flag.

`io.units.flow_scale` rescales raw flow into your analysis units; column names
and date formats are all configurable.

**Forcing** (optional): a daily CSV with any of `precip`, `swe`, `tair`
(`io.forcing.path`). If absent, recession screening falls back to the
streamflow-only (digital-filter) screen.

---

## What the pipeline does

### 1. Preprocessing (`io`)
- Robust date parsing; the record is reindexed onto a **gap-free daily grid** so
  downstream code can trust spacing. Genuinely missing days become `NaN`.
- **Gaps are detected and reported, never interpolated.** Recession segments
  break at gaps.
- Negative flows are dropped; a `min_flow` floor guards the log transform.
- If USGS qualification flags are present, ice-affected / estimated days are
  flagged and (configurably) dropped, with a count reported.

### 2. Baseflow separation (`filter`) — screening only
Lyne-Hollick recursive digital filter:

```
f[i] = α·f[i-1] + (1+α)/2·(Q[i] − Q[i-1]),   b = Q − f,   0 ≤ b ≤ Q
```

- `passes` (default **3**, forward-backward-forward) drives the **reported**
  baseflow / quickflow / BFI separation.
- **`screen_passes` (default 1)** drives the quickflow fraction used by the
  recession *gate*. This matters: a single forward pass leaves a clean recession
  as ~100% baseflow (`f/Q ≈ 0`), whereas the multi-pass backward filter
  legitimately carves substantial quickflow **even from a pure recession**. The
  gate's default threshold assumes a near-zero recession baseline, so it must use
  the single-pass fraction; the full multi-pass separation is still reported.
- An α-sensitivity utility reruns recession *selection* at α ∈ {0.9, 0.925, 0.95}
  and reports how segment count / total recession days change.

### 3. Recession identification (`recession_id`)
A day is inside a recession when **all** hold, and segments are cut at breaks:
1. **Falling:** `dQ/dt < tol` (`dqdt_tolerance` allows small noise).
2. **Post-peak drop:** the first `drop_days_after_peak` falling days are removed
   (the peak day itself is not "falling", so front removal is `N+1`).
3. **Quickflow gate:** screening `f/Q ≤ quickflow_frac_max`; a spike breaks it.
4. **Season window:** either a fixed `[doy_start, doy_end]` day-of-year band, or
   a **dynamic snowmelt-driven start** (see below). Excludes the rising freshet
   limb.
5. **Forcing screen (takes precedence):** if forcing is provided, a rain day
   (`precip > precip_threshold`) or melt day (SWE drop, or a temperature-index
   proxy when SWE is absent) breaks the segment regardless of the filter gate.
6. **Gap break:** segments never span missing data.
7. **Minimum length:** `min_segment_length` days.

#### Dynamic snowmelt-driven season start (`season.mode: swe_meltout`)

For snow-dominated basins the recession season should begin *after the snowpack
is gone*, which varies year to year. Set `recession.season.mode: swe_meltout`
and point `recession.season.meltout.swe_path` at a SNOTEL-style daily SWE CSV.
For each water year the pipeline finds the **meltout date** — the first snow-free
day (`SWE ≤ swe_threshold`) *after* the winter peak (not the incidental zero SWE
of early fall) — and sets that year's season start to **meltout + `buffer_days`**
(default 28 = 4 weeks). Years whose peak SWE is below `min_peak_swe` (no real
snowpack) or with no post-peak meltout fall back to `fallback_doy_start`. The
same `doy_end` upper bound applies. Per-year starts are logged and written to
`tables/season_starts.csv`; inspect them with
`recession_id.season_starts_frame(cfg)`.

For the East River, the Butte SNOTEL record resolves all 45 water years (meltout
April–June, mostly May), giving a mean season start near day-of-year 170.

### 4. MRC construction (`mrc`)
- **Per-recession tail fit:** `ln Q = intercept − k·t` on the **late-time tail**
  — a linear-reservoir fit to the tail, not the whole limb. Reports `k`, `τ`,
  `r²`; a recession *conforms* if `r² ≥ min_tail_r2` and `k > 0`. The tail is
  chosen by `mrc.tail_method`:
  - `flow_threshold` (default): keep days with `Q < tail_flow_frac · Q_ref`
    where `Q_ref` is the **mean daily flow over the recession** (default
    `tail_flow_frac = 0.667`, i.e. 2/3 of the mean). Because a recession falls
    monotonically, these are the low-flow late-time days — the tail is defined by
    flow magnitude, not a point count. If too few days fall below the threshold,
    it falls back to the lowest-flow `tail_min_points` days.
  - `fraction`: keep the last `tail_fraction` of points by count.

  The matching strip uses the **same** tail selection, so the master curve is
  built from the same days as the per-recession `k`.
- **Automated matching strip:** conforming recessions are ranked longest-first
  (no hard cap — however many conform are used, and the count is reported) and
  assembled into one master curve. The master is parameterized as **time as a
  function of log-flow**, `t = P(lnQ)`, with each recession free to shift along
  the time axis (`offset_i`); offsets and `P` are optimized jointly to collapse
  the recessions onto one curve.

  > **Why `t = P(lnQ)` and not `lnQ = P(t)`?** With free time-offsets, an
  > `lnQ = P(t)` objective is degenerate — the optimizer slides recessions apart
  > along `t` to flatten the master. Making `lnQ` (fixed per point) the
  > independent variable removes that freedom: offsets can only shift `t`, so the
  > curve's slope is pinned by within-recession structure. For a linear reservoir
  > `t = −τ·lnQ + c`, and the master constant is `k = −1 / (dt/dlnQ)` evaluated at
  > the late-time (lowest-flow) tail end. Diagnostics: master `k`, `τ`, RMSE (in
  > days), n conforming, n used.

### 5. Temporal / stationarity (`temporal`)
- MRCs are built on **overlapping windows** (`length_years`, `step_years`,
  default 5-year / 1-year) and **per water year** where sample size permits.
  Windows below `min_recessions` conforming are recorded with `NaN k` so
  coverage gaps stay visible.
- The windowed `k` series is tested with **Mann-Kendall** (tie/continuity
  corrected) + **Theil-Sen slope**, reporting `p`, `S`, Kendall's τ, and slope.

  > **Trend caveat (always emitted):** overlapping windows induce strong positive
  > serial autocorrelation, which **inflates** nominal significance, and
  > per-window sample sizes are unequal (heteroscedastic `k`). Treat `p` as a
  > screening indicator, not a formal test. For a defensible test, use
  > non-overlapping windows or per-year `k`, and consider a
  > variance-/autocorrelation-corrected variant.

### 6. Brutsaert–Nieber diagnostic (`bn`)
An **independent** check of the linear-reservoir assumption, run on the *same*
screened recession segments. Instead of fitting Q vs time, it pools every
recession day and fits the recession rate against flow:

```
-dQ/dt = a · Q^b        (log-log:  ln(-dQ/dt) = ln a + b·ln Q)
```

- **`b` is the diagnostic:** `b ≈ 1` is a linear reservoir (constant `τ = 1/a`,
  the MRC assumption); `b > 1` is nonlinear storage–discharge. If B-N returns
  `b ≈ 1`, that corroborates the poly-order (linear-master) argument by a second,
  independent method.
- **Lower envelope:** the physical signal is the cloud's lower edge — points above
  it carry ongoing rain/melt input. Default fit is a low quantile per log-Q bin
  (`envelope.method: lower_quantile`); `ols_cloud` fits the full cloud instead.
- **`dQdt.q_reference`** (`mean_pair`/`q_start`/`q_end`) sets the flow paired with
  each backward finite difference; differences never cross segment boundaries/gaps.
- **Wetness split (`bn.wetness`)** — the drought test. Classify recessions by a
  wetness metric (`water_year_mean_flow` / `water_year_min_flow` /
  `recession_q_start`) and fit each group. **Separated** envelopes (same `b`,
  different intercept → different effective `τ`) mean the lumped `τ` drifts with
  storage state without breaking linearity; **overlapping** envelopes mean `τ` is
  stationary w.r.t. wetness. `bn_analysis(...)` reports `tau_ref` for every group
  at a common reference flow so timescales are comparable.

  > **Caveats:** `-dQ/dt` from daily data is finite-difference noisy near the flat
  > low-Q tail (discretization stripes, envelope flattening), and the fitted `b`
  > shifts with envelope choice. Sweep `envelope.method`/`quantile` (as the MRC
  > thresholds were swept): in practice `b` stays near 1 robustly while the
  > envelope `τ` magnitude moves a lot — trust the `b`-vs-1 comparison over the `τ`.

API: `mrc.bn.bn_analysis(recessions, cfg, df)` → `BNResult` (cloud, overall +
per-group `BNFit`, envelope points); `mrc.bn.fits_frame(result)` for a tidy table;
`mrc.plotting.plot_bn` / `plot_bn_split` for the two panels.

---

## Outputs

Written under `<output_dir>/<label>/`:

- `config_used.yaml` — the fully-resolved config snapshot (reproducibility).
- `tables/`
  - `per_recession_metrics.csv` — start/end, length, tail `k`, `τ`, `r²`, season
    day-of-year, water year, conforming flag, qualification flags.
  - `per_window_mrc.csv`, `per_year_mrc.csv` — `k`, `τ`, RMSE, n conforming/used.
  - `alpha_sensitivity.csv`, `gaps.csv`, `trend_test.csv`.
  - `season_starts.csv` (swe_meltout mode); `bn_fits.csv` + `bn_cloud.csv` (when
    `bn.enabled`).
- `figures/`
  1. `fig1_ranked_recessions` — ranked recession traces on semi-log, staggered.
  2. `fig2_matched_mrc` — matched master curve with `k`/`τ`/RMSE annotation.
  3. `fig3_k_vs_time` — master `k` vs time with Sen trend line and a ±1σ band.
  4. `fig4_brutsaert_nieber` — `-dQ/dt` vs `Q` cloud, lower envelope, `b`=1 ref.
  5. `fig5_bn_wetness_split` — B-N envelopes split by wetness state (drought test).

### Optional interactive QC viewer
Off by default (`output.qc_viewer.enabled: false`). When enabled it opens an
interactive tool (matplotlib sliders, or a plotly HTML) to manually time-shift
recessions for one window. **QC only — adjustments are not persisted.** Requires
an interactive backend/display (won't work headless).

---

## Reproducibility

- One RNG `seed` (config) seeds NumPy and the synthetic generator.
- The resolved config is snapshotted to every run directory.
- No hidden state: given the same config and inputs, results are deterministic.

---

## Key assumptions & limitations

- **Linear-reservoir late-time recession.** `k` is a tail property; early-limb
  and non-linear-storage behavior are deliberately excluded via the tail fit and
  the post-peak drop.
- **The digital filter is a screen, not the estimator.** Recession `k` comes from
  the semi-log tail fit, never from the filter.
- **Snowmelt is the dominant confounder.** Without forcing, the day-of-year
  season window is the main guard against melt-driven flow; with forcing, an
  explicit melt/rain screen takes precedence. Set the season window to the
  post-freshet recession season for your basin.
- **Stationarity `p`-values are indicative.** See the trend caveat above.
- **Synthetic data are for exercising the code**, not calibrated hydrology; the
  synthetic recessions sit on a constant baseflow floor, so their tails flatten
  and the recovered master `k` is biased low relative to the injected rate — a
  real feature of offset (two-store) systems, useful for testing robustness.
```
