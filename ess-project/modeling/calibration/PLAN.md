# Calibration Plan — East River & Tuolumne (distributed elevAspect)

Status: draft. Two decisions still open (see "Blocking decisions").

## Principles

1. **Calibrate to streamflow only.** SWE (ASO, CDEC), ET (OpenET), and all multi-year
   memory signatures are held out. Never in the objective.
2. **Routing translates, physics remembers.** Routing is capped at a few days. Any
   timescale longer than that must be produced by the model physics, not by a
   downstream reservoir.
3. **Spatial structure is a prior, not a target.** The optimizer tunes a global
   magnitude; the per-HRU shape comes from a physically-motivated hypothesis.
4. **Hypotheses are dials with null values**, not separate models. One calibration
   per basin.
5. **Keep the behavioral ensemble, not the winner.** The trial log is the product.

## Windows

Both basins have `basin_averaged_data` covering 1981-01 → 2025-12. `SUMMA_input`
must be relinked to it first (currently stale: Tuolumne 1990–2021, East 1999–2024).

| Phase | Water years | Use |
|---|---|---|
| Spinup | WY1991–2000 | Discarded. 10 yr: East `bigBuckt` must forget `scalarAquiferStorage=2.5 m`. |
| Calibration | WY2001–2014 | Objective evaluated here. Only here. |
| Selection | WY2015–2019 | Prune the behavioral ensemble. Modern era: ASO, OpenET, CDEC. |
| Final test | WY2020–2024 | Untouched. Reported skill comes from here. |

Sim per trial = spinup + calibration = 24 yr. At ~74 s/simulated-year (26 HRU,
minimal output, measured) that is **~30 min/trial**.

Three-way split exists so the ensemble can be pruned for the forecast product on
WY2015–2019 without contaminating the reported out-of-sample skill.

## Objective (calibration period only)

```
J = w1 * (1 - KGE(Q_daily))          # volume + variance + correlation
  + w2 * (1 - KGE(log Q_daily))      # recession, baseflow, Aug-Sep low flow
  + w3 * min(CoM_error_days / 30, 3) # snowmelt centre-of-mass timing
```
Default `w = (0.4, 0.3, 0.3)`. **Each component is logged separately per trial.**

Total volume is *not* a separate term — KGE's beta term is already volume bias.

## Parameters

Multiplier on the a-priori per-HRU value, with a hypothesis-derived spatial weight:
`param_i = clamp(base_i * W_i * M)`. The optimizer tunes `M` and the hypothesis
dials that define `W`.

### Physics multipliers (candidates; the Morris screen decides which stay free)

| Group | Parameters |
|---|---|
| Snow | `frozenPrecipMultip`, `tempCritRain`, `albedoDecayRate` |
| ET | `rootingDepth`, `minStomatalResistance`, `summerLAI` |
| Soil | `k_soil`, `theta_sat`, `theta_res`, `vGn_alpha`, `vGn_n`, `qSurfScale`, `fieldCapacity` |
| Routing | `tau` (= shape x scale), `routingGammaShape` |
| East only (`bigBuckt`) | `aquiferScaleFactor`, `aquiferBaseflowExp`, `aquiferBaseflowRate` |
| Tuolumne only (`qTopmodl`) | `zScale_TOPMODEL`, `kAnisotropic` |

`routingGammaScale` is derived as `tau / shape`, never searched directly.
Bounds: `tau in [6h, 72h]`, `shape in [1.5, 4]`. This is the memory cap.

### Hypothesis dials

| Dial | Applies to | Form | Null | Hypothesis |
|---|---|---|---|---|
| `gamma` | `k_soil` | `W_i = (z_i/z_mean)^gamma` | 0 | `> 0` (K increases with elevation) |
| `eps` | `rootingDepth` | `W_i = (z_i/z_mean)^eps` | 0 | `< 0` (deeper roots at low elevation) |
| `delta` | soil depth (coldState) | `W_i = (z_i/z_mean)^delta` | 0 | `> 0` (deeper soils at high elevation) |
| `lambda_d` | soil depth, flat HRUs | multiplier | 1 | `> 1` (flat = deepest in band) |
| `lambda_t` | `theta_sat`, flat HRUs | multiplier | 1 | `< 1` (flat = less porous) |
| `phi` | `aquiferBaseflowRate`, flat (East) | multiplier | 1 | `< 1` (flat drains slower) |
| `psi` | `aquiferScaleFactor`, flat (East) | multiplier | 1 | `> 1` (flat stores more) |
| `S0` | initial SWE, snowfield HRU (Tuolumne) | mm w.e. | 0 | `> 0` (persistent pile) |

`gamma`, `eps`, `delta` are exponents whose null is **zero**, not one. If the
behavioral distribution of a dial straddles its null, the data do not support the
hypothesis — that is a result, not a failure.

**Note:** soil depth is NOT in `trialParams.nc`. `delta` and `lambda_d` are applied
via `mLayerDepth`/`iLayerHeight` in a per-trial `coldState.nc`. Same writer as `S0`.

### Known SUMMA constraints (measured, not assumed)

- `mLayerVolFracIce` must stay below ~0.6 (~550 kg/m3). True glacier density
  (800-900) is NOT representable — carry the mass as extra depth instead.
- Top snow layer must be >= 0.05 m. A 0.02 m surface layer over a deep pack
  diverges in the energy solver.
- The bottom snow layer is unbounded. Verified to 8 m thick / 9.75 m total.

## Held-out signatures (never in the objective)

**Memory** — the discriminating diagnostics for the storage hypotheses:
- lag-1 and lag-2 autocorrelation of annual runoff
- multi-year recession slope through the 2012-2015 drought
- September flow regressed on *previous* year's peak SWE
- storage-discharge relationship / hysteresis

**Snow** — ASO SWE hypsometry and magnitude; CDEC point SWE; snowpatch + glacier
area decline (1988-2021 literature record).

**ET** — OpenET monthly.

ASO caveat to declare in writing: the a-priori `frozenPrecipMultip` spatial weights
are ASO-derived. Treat the *normalized median shape* as a static physiographic prior
(same status as a soil map) and evaluate against ASO **magnitude and interannual
variability**, never against the mean shape.

## Sequence

### Phase 0 — Trust (1 day, no compute)
- Hard-fail when sim/obs overlap is below threshold. (The ET stage silently no-op'd
  for want of this: OpenET 2018-2020 vs. a 2013-2017 sim window, zero overlap,
  `kge()` returned its `len<10` fallback of 1.0, and the anchor became a constant.)
- Run manifest per results dir: git SHA, resolved config, obs file hashes, window.
- Remove the `999.0` failure sentinel — it is a cliff in the DE landscape, not a
  gradient. (71/378 trials hit it last run.)
- Trim `outputControl.txt` to ~5 variables. Measured: 30x less I/O, 16% faster,
  5x less RSS. At 22 concurrent writers this is the difference between CPU-bound
  and filesystem-bound.
- Audit `PHYSICAL_BOUNDS`. `aquiferScaleFactor` was *binding* last run — a bound,
  not the data, was setting the answer.
- Relink `SUMMA_input` to `basin_averaged_data` (1981-2025) for both basins.
- Decide gradient-corrected vs. uncorrected Tuolumne forcing. **The alpha=0.5
  correction is ASO-derived; using it forfeits the independence of the ASO
  validation.**

### Phase 1 — Screen (~3 h)
- Morris elementary-effects, 8-HRU domain, ~140 runs -> which multipliers move J.
- Morris, 26-HRU, ~140 runs -> aspect-dependent dials (the 8-HRU domain has no
  aspect and cannot screen `lambda_*`, `phi`, `psi`).
- Fix the inert parameters at their a-priori values. This is the lever that makes
  a simultaneous DE affordable, and it is the defensible answer to "why these
  parameters and not those?"

### Phase 2 — Calibrate (~16 h per basin)
- One DE per basin. `popsize * n_params ~= 22` so the population equals the core
  count and each generation costs exactly one trial-time. ~40 generations,
  ~840 evaluations, `updating="deferred"`, `polish=False`, seeded.
- Every trial -> CSV with all objective components. This log *is* the behavioral
  ensemble; it is not a diagnostic byproduct.

### Phase 3 — Behavioral ensemble (~4 h)
- Filter the trial log on **calibration-period** skill only (e.g. within 10% of
  best J, or daily KGE > 0.7) -> ~150 sets. Zero extra model runs.
- Run those 150 over the full record (WY1991-2024).

### Phase 4 — Hypothesis tests (~12 h)
- **Continuous dials:** does the behavioral distribution of `gamma`/`eps`/`delta`/
  `lambda_*`/`phi`/`psi` exclude its null? Discriminate using the held-out memory
  signatures: among parameter sets that fit within-year streamflow *identically
  well*, do the ones reproducing multi-year memory have systematically different
  dial values?
- **Snow pile:** cross the 150 behavioral sets against `S0 in {0, lit, 2*lit}`.
  Paired comparison — same theta, only the pile differs — then Wilcoxon signed-rank
  on the 150 paired differences. Two separate calibrations would be both costlier
  and confounded; this is the controlled experiment.
- Validate the survivors against ASO SWE and OpenET.

Three outcomes, all publishable:
1. Memory-matching members cluster away from the null -> **hypothesis supported**,
   and compensation is controlled by construction (every member fit Q equally).
2. Memory reproduced across all dial values -> **hypothesis unnecessary**.
3. No member reproduces memory at any dial value -> **the structure is inadequate**.
   This is the only case that justifies a new model structure.

### Phase 5 — Supervisor
- Deterministic restart/resume. Agent invoked only at decision points (plateau,
  >20% trial failure, a dial pinned to a bound). Every intervention -> a
  `decisions.jsonl` entry plus a git-tracked config diff.
- Guardrails: never touch the holdout, never widen `PHYSICAL_BOUNDS`, capped number
  of interventions per run.

**Total: ~45-55 h compute (~2 days).** Budget 2-3x in practice for reruns.

## Architecture

```
calibration/
  config/{east_river,tuolumne}.yaml  # window, physics, dials, bounds, weights
  spatial_weights.py   # dials -> per-HRU weight arrays
  coldstate.py         # per-trial coldState: soil depth (delta, lambda_d) + S0
  objective.py         # 3-component score, components returned separately
  signatures.py        # HELD-OUT diagnostics. Imports nothing from objective.py.
  screen.py            # Morris
  calibrate.py         # one DE, popsize ~ n_cores, logs every trial
  behavioral.py        # behavioral set -> apply signatures -> paired tests
  manifest.py          # git SHA + config + obs hashes
  runner.py            # kept from summa_runner.py
  parameters.py        # kept from parameter_manager.py
```

**Delete:** `staged_optimizer.py` staging machinery, `stage_configs/`, the coherence
metrics in `objective_functions.py`, `pre_calibrate_routing.py` + the offline
two-reservoir, the five `optimization_config_*.yaml` variants.

`forcing_adjuster.py` moves to the seasonal-ensemble code, which still needs it.

### Why the offline routing must go

`routing_params_shared.json` is a **262-day** linear reservoir holding 30% of the
flow, fit against an *uncalibrated* baseline to NSE = -0.19 (worse than the mean),
with `f_fast` pinned at its bound. That is not routing — a 750 km2 basin translates
water in 9-28 hours. It is a second aquifer bolted downstream of SUMMA.

It mechanically explains the pinned aquifer parameters in the last run:
`aquiferScaleFactor` = 0.219 (lower bound) and `aquiferBaseflowRate` = 9.976 (upper
bound) — minimum storage, maximum drainage. The offline reservoir was already
producing all the recession, so DE drained SUMMA's aquifer as fast as it could.

`pre_calibrate_routing.py` claims a shared routing "means differences across physics
configs reflect actual subsurface physics — not compensating routing." The opposite
is true: a shared 262-day store *erases* the subsurface signal.

**Any store that can absorb the memory signal makes `phi`, `psi` and `S0`
unidentifiable by construction.** Fixing routing is a precondition for the entire
experiment, not a side quest.

## Domain rebuild — 2026-07-14 (COMPLETE)

Settings and forcing were on *different discretizations*: the July METSIM run
remapped onto the canonical catchment shapefiles (Tuolumne 28 HRUs, East 18),
but the settings were still the April versions (26 / 25).  The 26 was a *merge*
of the 28, so `hruId k` meant different ground in each file.  Both basins are
now rebuilt and consistent: attributes / coldState / trialParams / forcing /
forcingFileList all agree on HRU count and hruId.

### Bugs found and fixed (in order of consequence)

1. **`tmZoneInfo 'utcTime'` on local-solar-time forcing.**  Forcing SWRadAtm
   peaks at hour 11-12 in every month, i.e. local solar time; SUMMA was told
   UTC, so `derivforce.f90` applied `timeOffset = longitude/360` = -7.1 h.  At
   the 930 W/m2 midday peak SUMMA placed the sun on the horizon, `cosZenith = 0`,
   `scalarFractionDirect = 0`, and **treated all shortwave as diffuse for the
   entire run**.  Snow's high diffuse albedo then prevented melt-out ->
   9 of 26 HRUs (43% of basin) grew a permanent 14 m snowpack.
   FIX: `tmZoneInfo 'localTime'`.  Runaway is gone: every high HRU now melts out
   annually, snow persists into September only in big years (1993, 1995).

   This also explains the old calibration's pinned forcing knobs
   (`frozenPrecipMultip` -> 0.30, `lw_mult` -> 1.195): they were compensating for
   a snowpack that could not melt.  Those knobs stay deleted.

2. **`data_step` missing from the regenerated forcing.**  SUMMA reported
   "number of time steps = 1", the time-delay routing histogram collapsed to a
   zero fraction, and the solver failed on step 1.  FIX: `add_data_step.py`
   (540 files x 2 basins).

3. **`compactedDepth` (1.0 m) exceeded the shallowest soil column.**
   `satHydCond` computes `(1 - compactedDepth/soilDepth)**(zScale_TOPMODEL - 1)`;
   with Tuolumne's 0.33 m alpine soils the base goes negative and the fractional
   power yields NaN.  FIX: set `compactedDepth` to half the shallowest column
   (Tuolumne 0.17 m, East 0.60 m).

4. **Land cover came from an ~886 m raster** that resolved ~1% barren, smearing
   alpine terrain into grassland/savanna.  Both basins now use the 30 m
   NLCD->IGBP product: barren on the peaks, evergreen mid-slope, shrub low.
   (The correct East raster lived in a *different domain directory* and was
   named `..._land_class_modis.tif`, singular.)

5. **`trialParams.nc` held frozen calibration output, not a-priori values.**  The
   `k_soil` ratios were exactly the old `stage3_soil` spatial weights
   (0.55 / 1.50).  Only the 14 parameters the staged calibration touched are
   reset to defaults; structural parameters (`zScale_TOPMODEL`, `kAnisotropic`,
   `theta_sat`, ...) are preserved -- resetting `zScale_TOPMODEL` to its table
   default of 15.4 amplifies near-surface conductivity ~7000x and stalls the
   solver.

6. **No terrain correction on shortwave.**  `sunGeomtry.f90` computes the
   slope/aspect radiation index `hri`; `derivforce.f90` receives it and never
   uses it.  Only `cosZenith` survives, and it only sets the direct/diffuse
   split -- the four spectral components sum back to exactly `SWRadAtm`.  So the
   correction MUST live in the forcing.  `apply_terrain_sw.py` now applies it:
   S-facing minus N-facing = +50.7 W/m2 (Tuolumne), +31.7 W/m2 (East).
   Note the forcing timestamps are interval-START (SW is symmetric about hour
   11.5), so the solar position needs a half-timestep offset as well.

### localParamInfo is unreliable — do not trust its defaults

Several defaults sit outside their own bounds, and some bounds are junk:
`aquiferBaseflowRate` default 2.0 vs upper bound 1e-5; `qSurfScale` default 50
vs upper bound 10; Tuolumne's `vGn_alpha` lower bound is **-70 m-1** (its
midpoint, -35.5, makes the van Genuchten curve singular).  `specificStorage`
has *inverted* bounds.  The rebuild clamps out-of-bounds defaults into range and
warns; it never midpoints.

### Post-fix validation (WY1991-95, a-priori parameters)

| | Sept SWE > 10 mm | basin-mean April-1 SWE |
|---|---|---|
| Tuolumne (28 HRU) | 0, 0, 1, 0, 5 of 28 | 490 / 435 / 994 / 327 / 1141 mm |
| East (18 HRU)     | 0, 0, 0, 0, 0 of 18 | 252 / 236 / 538 / 214 / 392 mm |

Both track Sierra/Colorado climatology (1992 drought, 1993 and 1995 big years).

### New tooling

- `ess-project/modeling/baseline/regen_intersections.py` — rebuild soil/land
  intersections against the current catchment; picks rasters by *validity over
  the catchment* then *finest resolution*, and reprojects (rasterstats does not).
- `ess-project/1_forcing/apply_terrain_sw.py` — slope/aspect shortwave correction.
- `ess-project/1_forcing/add_data_step.py` — restore the `data_step` variable.
- `update_settings_for_hrus.py` — patched (catchment selection, soil-depth ramp
  re-fit, calibrated-vs-structural parameter split, bounds clamping).

## A-priori streamflow check — 2026-07-14

First look at streamflow since the domain rebuild. WY2008–12, a-priori parameters,
no calibration; WY2006–07 discarded as spin-up. **Verdict: timing is broadly right,
volume is not, and the two basins fail in opposite directions.**

### Observed data was wrong for the Tuolumne

The Tuolumne was being compared against **Hetch Hetchy unimpaired inflow**
(`Tuolumne_River_lumped_streamflow_processed.csv`), which drains ~1189 km². The model
domain is 774.5 km². Over the model area that series implies a **runoff ratio of 1.13** —
more water out than falls in, with zero ET. Impossible.

The correct gauge is **USGS 11274790** (Tuolumne R. a Grand Canyon of the Tuolumne, above
Hetch Hetchy), drainage 301 mi² = **780 km²**, matching the model's 774.5 km². Over the
2007–2024 overlap the HH series carries **2.01×** the flow of the true gauge (area ratio
alone is only 1.52×). Any calibration against the HH series was chasing ~2× too much water.

- East gauge is fine: **USGS 09112500** (Almont), 748.5 km² vs model 748.3 km².
- Installed: `East_River_USGS09112500_dailyQ_WY1970_2025.csv` (WY1970–2025),
  `Tuolumne_River_USGS11274790_dailyQ_WY2007_2025.csv` (WY2007–2025).
- The Tuolumne record starts **2006-10-13**, so calibration/evaluation windows for the
  Tuolumne cannot begin before WY2007 unless a longer proxy is reconstructed.

### Extending the Tuolumne record (DECIDED)

The true gauge starts 2006-10-13 — too short to calibrate on. The HH series is rescaled to
gauge-equivalent and used before that.

Scaling is by **month**, not a single scalar. The HH/gauge ratio is strongly seasonal
(Dec 2.34, Aug 1.58; annual mean 2.00, sd 0.25):

| Oct | Nov | Dec | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|1.90 |1.86 |2.34 |2.40 |2.41 |2.25 |2.07 |1.98 |1.90 |1.88 |1.58 |1.69 |

Leave-one-year-out, out-of-sample (15 complete WYs):

| method | annual volume, mean abs err | seasonal shape bias |
|--------|---------------------------|---------------------|
| flat ÷2 | **8.2%** | Dec–Feb **+17…+20%**, Aug **−21%**, Sep −15% |
| monthly factors | 10.0% | within ±3% Mar–Sep; +16% Oct, +11% Dec |

Flat ÷2 is marginally *better* on annual volume (year-to-year ratio scatter dominates and
monthly factors add noise), but it carries a systematic **+20% winter / −21% August** bias —
precisely the two signatures the objective targets. Monthly factors are chosen because their
errors are unbiased where it counts, not because they are more accurate overall.

- File: `Tuolumne_River_merged_dailyQ_WY1971_2025.csv` — `source` column marks
  `gauge` (WY2007–2025, 6793 d) vs `HH_scaled` (before, 13569 d).
- **Carry this caveat:** pre-2007 years have ~10% irreducible annual-volume uncertainty.
  Down-weight the *volume* term (not the timing terms) on those years.

### Groundwater structure (DECIDED)

- East: `groundwatr = bigBuckt` — explicit aquifer; deep colluvium/shale, real groundwater.
- Tuolumne: `groundwatr = qTopmodl` — **deliberate**. Thin soils on granite; an explicit
  bucket aquifer is not the right structure for this basin. Storage lives in the soil column
  with a TOPMODEL saturated zone.

The basins are therefore *intentionally* different. Consequence: the Tuolumne memory/storage
knobs are **soil depth, `k_soil`, `zScale_TOPMODEL`** — *not* aquifer parameters. Any
cross-basin storage comparison must be framed in terms of behaviour (recession, memory), not
shared parameters.

**Open problem:** Tuolumne simulated December flow is **0.00** vs 0.40 mm/day observed. The
basin does sustain winter baseflow, so under qTopmodl the soil column must supply it. It
currently cannot. This is the first thing calibration has to fix in the Tuolumne.

### Monthly mean streamflow, mm/day (WY2008–12)

East (obs = Almont):

|      | Oct | Nov | Dec | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| obs  |0.34 |0.30 |0.26 |0.23 |0.21 |0.25 |0.98 |3.10 |4.38 |1.91 |0.62 |0.33 |
| sim  |0.02 |0.03 |0.01 |0.01 |0.00 |0.07 |0.18 |2.19 |2.07 |0.36 |0.04 |0.01 |

Tuolumne (obs = 11274790):

|      | Oct | Nov | Dec | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| obs  |0.41 |0.37 |0.40 |0.49 |0.57 |1.00 |2.43 |5.15 |6.19 |2.52 |0.45 |0.12 |
| sim  |0.36 |0.16 |0.00 |0.02 |0.04 |0.13 |0.75 |4.92 |7.90 |5.10 |1.11 |0.29 |

### Water balance (mm/yr, 5-yr mean)

| basin    |  P   | obs Q | sim Q | sim/obs | sim ET | ET/P | obs runoff ratio | sim runoff ratio |
|----------|------|-------|-------|---------|--------|------|------------------|------------------|
| East     |  890 |  393  |  153  | **0.39**|  610   | 0.69 | 0.44             | 0.17             |
| Tuolumne | 1093 |  613  |  635  | **1.04**|  368   | 0.34 | 0.56             | 0.58             |

### What works

**Peak timing, East.** Hydrograph centroid within +1.5 to +8 days in 4 of 5 years; peak
day within 1–2 days in WY2008/09/11. Peak month May vs observed Jun (one month early).
This is good for an uncalibrated model and says the snow/melt energetics are now sound.

### What is broken

1. **East produces 39% of observed runoff.** ET is 69% of P (610 mm/yr); the observed
   balance implies ~500 mm/yr. Roughly half the deficit is excess ET, half is water
   entering storage and never leaving (P − ET − Q leaves a +127 mm/yr residual).

2. **Neither basin produces baseflow — but for different reasons, and only one is a bug.**
   - East `bigBuckt` with a-priori `aquiferBaseflowRate = 1e-5 m/s`,
     `aquiferScaleFactor = 0.35 m`, `aquiferBaseflowExp = 2.0` gives
     `Qb = 1e-5·(S/0.35)²`. At 1 mm/day recharge this equilibrates at **S ≈ 12 mm with a
     ~12-day residence time**. Mean simulated aquifer storage over WY2008–12 is **2.4 mm**.
     A reservoir that small cannot sustain the observed 0.2–0.3 mm/day winter flow, let
     alone multi-year memory. Winter sim is 0.00–0.03 vs 0.21–0.26 observed.
     `aquiferBaseflowRate = 1e-5` is itself an artifact of clamping to
     `localParamInfo.txt`'s junk upper bound (default 2.0, upper bound 1e-5).
     **This one is a bug.**
   - Tuolumne `qTopmodl`: `scalarAquiferStorage` inert (min = max = 2500 mm) is *expected* —
     under qTopmodl there is no explicit aquifer by design. The failure is that the soil
     column, which is supposed to carry the store, cannot: December flow is **0.00** vs 0.40
     observed. **Not a bug — an under-parameterized soil profile.** Knobs: soil depth,
     `k_soil`, `zScale_TOPMODEL`.

3. **Tuolumne volume is right for the wrong reason.** Total is 1.04× observed, but the
   seasonal shape is wrong in both directions: too little Dec–Apr (Dec 0.00 vs 0.40) and
   far too much Jun–Aug (Jul 5.10 vs 2.52). Centroid is **17–46 days late**. The basin
   holds nearly all its water as snow and dumps it in summer. The correct annual total is
   masking a large timing error — a caution against any objective weighted mainly on volume.

4. **Cold-state aquifer storage is 2.5 m.** In East that releases 44,000 mm/day on step 1;
   the model dumps **2359 mm on day 1** and drains the whole 2.5 m within October
   (WY1991 total 2805 mm vs 337 observed). Every run needs either a sane `S₀` or ≥1 yr of
   discarded spin-up. This is what corrupted the WY1991 numbers in the earlier 5-yr check.

### Implications for the plan

- The low-flow / Aug–Sep term in the objective is currently unreachable: the model has no
  mechanism to produce late-summer baseflow. **Aquifer parameters must be freed and given
  sane bounds before calibration, not treated as a fine-tuning knob.**
- The East memory hypothesis ("flat HRUs drain slowly, multi-year storage") is not a
  refinement of a working model — the baseline has *no* storage at all. Test it against a
  baseline whose aquifer is at least capable of producing observed baseflow.
- Redo the Tuolumne S₀ snow-pile sweep: the earlier result ("streamflow cannot see the
  pile") was obtained on the broken model and is void.

## East aquifer fix — 2026-07-14 (DONE)

### Parameters derived from the observed recession, not guessed

Deep-winter (Dec 1 – Mar 31) drawdown of the East River, WY1971–2025, when snow is
accumulating and there is no melt or rain — so all flow is storage release:

- water released Dec–Mar: **median 27 mm** (10–90%: 21–33)
- water released Aug–Mar: **median 77 mm** — lower bound on active storage
- recession time constant: **median 512 d** (dry winters 744 d, wet 442 d)
- **implied active storage ≈ 110–205 mm**

The **exponent is not identifiable** from this record. Brutsaert–Nieber gives b ≈ 1.03 but
with r = 0.29 — daily noise dominates — and the cross-year τ-vs-Q regression returns a
nonsensical negative n (wet winters carry mid-winter recharge that contaminates the
recession). The τ ordering (dry 744 d > wet 442 d) says exponent > 1, but not what it is.
**Leave `aquiferBaseflowExp` free for calibration; do not claim it is constrained.**

### Values set (East, `bigBuckt`)

| param | old | new | basis |
|-------|-----|-----|-------|
| `aquiferBaseflowRate` | 1e-5 m/s | **5.14e-7** | hits Q = 0.25 mm/d at S = 150 mm |
| `aquiferScaleFactor` | 0.35 m | **2.0 m** | keeps rate mid-range, not at a bound |
| `aquiferBaseflowExp` | 2.0 | **2.0** | unconstrained; free in calibration |
| `coldState` S₀ | 2.5 m | **0.15 m** | ≈ observed active store |

`localParamInfo.txt` bounds fixed: `aquiferBaseflowRate` had **default 2.0 against an upper
bound of 1e-5** — 2×10⁵ times above its own maximum. That default got clamped to 1e-5, which
is where the 12 mm / 12-day bucket came from. Bounds widened to [1e-10, 1e-4] so calibration
can actually reach slow baseflow.

### Result (WY2008–12, a-priori, no calibration)

| mm/day | Oct | Nov | Dec | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| obs    |0.34 |0.30 |0.26 |0.23 |0.21 |0.25 |0.98 |3.10 |4.38 |1.91 |0.62 |0.33 |
| old    |0.02 |0.03 |0.01 |0.01 |0.00 |0.07 |0.18 |2.19 |2.07 |0.36 |0.04 |0.01 |
| **new**|0.37 |0.31 |0.26 |0.23 |0.20 |0.20 |0.22 |0.79 |1.13 |0.74 |0.53 |0.41 |

- Aquifer storage: **86–241 mm** (mean 141) vs observed 110–205 mm. Was 2.4 mm.
- Winter DJFM: **0.22** vs obs 0.24 (was 0.03).
- Aug–Sep: **0.47** vs obs 0.48 (was 0.03).
- **Genuine interannual carryover**: storage declines 146 → 90 mm through the WY2010–12 dry
  run and does not reset annually. Multi-year memory now exists.

**Honest accounting of what is fitted vs emergent.** Rate and scale were tuned to hit one
operating point (S = 150 mm → Q = 0.25 mm/d), so matching winter baseflow is *fitted*, not
validated. What is *emergent* and was not fitted: the full Oct→Mar monthly recession shape,
the storage range landing inside the observed 110–205 mm window, and the interannual carryover.

### What this exposed: the melt peak collapsed

May 3.10 → 0.79; Jun 4.38 → 1.13 (old was 2.19 / 2.07). The aquifer now *swallows* the melt
pulse and dribbles it out. Annual volume is essentially unchanged (0.42 vs 0.39 of observed).

The water balance now closes and attributes the deficit exactly:

| | mm/yr |
|---|---|
| P | 890 |
| model ET | 610 |
| model Q | 164 |
| ΔAquifer | −11 |
| **unexplained residual (soil storage)** | **+127** |

Observed steady state implies ET ≈ P − Q_obs = 890 − 393 = **497 mm/yr**, so model ET is
**+113 mm/yr too high**. Deficit = 113 (excess ET) + 127 (soil still filling) = 240 ≈ the
229 mm/yr shortfall. **The balance closes — every remaining error is now attributable.**

Cause: with `k_soil = 1e-4 m/s` (8.6 m/day) and `bcLowrSoiH = drainage`, the soil never
saturates — melt water percolates straight to the aquifer instead of leaving as
saturation-excess surface runoff or interflow. There is no fast path.

**Next levers (this is the calibration):**
- fast/slow partition: `k_soil`, `theta_sat`, soil depth, `zScale_TOPMODEL`, `qSurfScale`
- ET (−113 mm/yr): `rootingDepth`, `minStomatalResistance`, LAI

## ET diagnosis — 2026-07-14

Both statements from the streamflow check were correct about *different* basins, and the
apparent contradiction is resolved: I had **undercounted evaporation by omitting
sublimation** (`scalarTotalET` = ground evap + canopy evap + transpiration only; snow
sublimation is a separate flux). With sublimation added and cross-checked against latent
heat flux, both water balances close.

| WY2008–12, mm/yr | East | Tuolumne |
|---|---|---|
| P | 890 | 1093 |
| model Q | 164 (**42% of obs**) | 635 (**104% of obs**) |
| total evaporative loss | 670 (**75% of P**) | 420 (38% of P) |
| observed evap need (P − Q_obs) | 497 | 480 |
| **evap error** | **+174 too much** | −60 (about right) |

East Q is low *because* East ET is high. Tuolumne evaporates about right, so its volume
lands at 104%. No contradiction.

### ET partition (mm/yr) — where East's excess sits

| component | East | Tuolumne | Δ |
|---|---|---|---|
| ground evaporation | 80 | 64 | +16 |
| **canopy (interception) evaporation** | **235** | **73** | **+162** |
| canopy transpiration | 295 | 231 | +64 |
| snow sublimation | 60 | 52 | +8 |
| **total** | **670** | **420** | **+250** |

**The +162 mm/yr canopy interception term is ~65% of the whole ET gap.**

### What differs between the basins (the answer to "ET initialization differences")

- **Trial parameters: identical.** `rootingDepth = 2.495`, `minStomatalResistance = 50`,
  `theta_sat = 0.45` in both. Interception params (`maxCanopyIce`, `canopyWettingFactor`,
  `throughfallScaleSnow`, `leafDimension`) are not in `trialParams` in either — both use
  `localParamInfo`/MPTABLE defaults.
- **MPTABLE vegetation tables: byte-identical** (same md5). Ruled out.
- **Only one ET-relevant decision differs:** `stomResist` — East `Jarvis`, Tuolumne
  `BallBerry`. (Clark et al. 2015b found BallBerry outperforms Jarvis; see CLAUDE.md.)
  But this controls *transpiration*, not interception, so it is **not** the main driver of
  the gap.
- Veg composition differs only modestly: 51% tall-canopy (East) vs 42% (Tuolumne) — a 1.2×
  ratio that cannot explain a 3.2× canopy-evaporation ratio.

### The actual driver: summer rain interception (climate, not settings)

Canopy evaporation by month (mm/day): East peaks **Jul 1.41, Aug 1.36, Sep 0.98**; Tuolumne
stays at **0.17 / 0.23 / 0.13**. East's excess is entirely the **North American monsoon** —
summer convective rain intercepted by the canopy and evaporated before it reaches the ground.
Tuolumne's dry Mediterranean summers produce almost no interception. 235 mm/yr of
interception loss ≈ 52% of the precip falling on East's forest, which is **too high**
(literature: 15–25% for conifer). The model is over-intercepting and over-evaporating summer
rain.

### Levers for East ET (−174 mm/yr target)

1. **Interception (primary, ~summer):** raise `throughfallScaleRain`, lower canopy storage
   capacity (`refInterceptCapRain`), raise `canopyDrainageCoeff` — let more summer rain drip
   to the ground instead of evaporating off wet leaves.
2. **Stomatal scheme:** switch East to `BallBerry` for consistency with Tuolumne and the
   literature; affects transpiration (295 mm/yr), a secondary term.
3. **Transpiration limits:** `rootingDepth`, `minStomatalResistance` — shared with Tuolumne,
   so move cautiously (do not re-introduce the compensating-error problem).

Do interception first: it is the largest term, it is basin-specific (summer rain), and it
does not touch parameters shared with Tuolumne.

### Interception test (2026-07-14) — FAILED, and the failure is the key finding

Pushed East hard: `throughfallScaleRain` 0.5→0.9, `refInterceptCapRain` 1.0→0.2,
`canopyDrainageCoeff` 0.005→0.01. Result:

| ET component, mm/yr | base | test | Δ |
|---|---|---|---|
| ground evap | 80 | 84 | +4 |
| canopy (interception) evap | 235 | 191 | **−44** |
| transpiration | 295 | 334 | **+39** |
| snow sublimation | 60 | 61 | +2 |
| **total** | **670** | **671** | **+0** |

Interception evaporation fell 44 mm/yr — but transpiration rose 39 and **total ET did not
move**. The water that no longer evaporates off wet leaves is transpired through the stomata
instead, because the available *energy* is unchanged.

**East ET is energy-limited, not pathway-limited.** Rerouting water between evaporative
pathways cannot reduce the total — only cutting the transpiration/evaporation *capacity* or
the available energy can. This is the compensating-error problem appearing at the process
level inside a single basin.

**Revised lever order (interception is NOT the fix):**
1. **Transpiration capacity** — LAI (monthly veg table), `rootingDepth`,
   `minStomatalResistance`, and the `stomResist` scheme itself. This is the real knob.
2. **`stomResist` Jarvis→BallBerry** on East — the one decision that differs from Tuolumne;
   test whether it changes the ET *magnitude*, not just the partition.
3. Reducing transpiration means touching `rootingDepth` / `minStomatalResistance`, which are
   **shared with Tuolumne** — so either make them basin-specific or verify Tuolumne (already
   at 104% volume, ET about right) is not degraded. The compensating-error risk the user
   flagged is real here.

Caveat worth checking before spending calibration effort: confirm East's excess ET is not
partly a *forcing* problem (net radiation / VPD too high) rather than purely a vegetation
parameterization problem — energy-limited ET points at the energy input as much as the
canopy.

## Forcing energy-balance check — 2026-07-14 (found the real ET cause)

Before tuning vegetation, checked whether the forcing energy balance is sane. **It is not,
for East — and it explains the excess ET without touching any vegetation parameter.**

### Shortwave: fine, both basins

Annual basin-mean SW: East 209, Tuolumne 223 W/m². Summer monthly peak ~340; 99th-pct
hourly ~935; max ~1018 W/m². Physical for 3000 m sites. Terrain correction working (see
earlier N<S check). No action.

### Longwave: East is physically impossible; Tuolumne is correct

Effective atmospheric emissivity ε = LW↓ / (σ·Tair⁴):

| ε | Oct–May | Jul | Aug | Sep | annual | max hourly |
|---|---------|-----|-----|-----|--------|-----------|
| East | ~0.90–0.95 | **1.02** | **1.05** | **1.01** | **0.96** | **1.30** |
| Tuolumne | 0.74–0.78 | 0.74 | 0.75 | 0.74 | 0.76 | 0.93 |

ε > 1 means the atmosphere radiates more than a blackbody at air temperature — impossible.
Two adjacent high-elevation basins with near-identical RH (East 43%, Tuolumne 38%) and
temperature must have near-identical LW↓; East's is **61 W/m² higher**.

### Provenance: East's longwave was overwritten; Tuolumne's was not

| July 2010 | METSIM native | after remap → SUMMA_input |
|-----------|---------------|---------------------------|
| East | 302 W/m² (ε 0.78) | **364 W/m² (ε 0.94)** |
| Tuolumne | 286 W/m² (ε 0.74) | 288 W/m² (ε 0.74) |

**METSIM's own longwave is correct for both** (East 302 ≈ Tuolumne 286, as expected). East's
was replaced by the **Dilley–O'Brien scheme** during the remap ("PRISM VPD + Dilley-O'Brien
LW" commit), inflating it. Tuolumne kept METSIM native. Over WY2008–12 (60 months) the East
inflation is **+61 W/m² mean** (range +25 to +141).

The user had already said "hold off on Dilley-O'Brien." This confirms it: **the D-O override
is a bug on East and must be reverted to METSIM-native longwave.**

### Why this is *the* ET fix (not vegetation)

East ET is energy-limited (interception test proved it). A +61 W/m² surplus in net radiation
is up to ~779 mm/yr of extra evaporative energy — far more than enough to source the observed
+174 mm/yr ET excess. Reverting East LW to METSIM native drops ε to ~0.78, matching Tuolumne.

**Do this before any vegetation calibration.** Tuning `rootingDepth` / `minStomatalResistance`
to compensate for a forcing bias would be exactly the compensating-error trap — and it would
corrupt the parameters that are shared with (correctly-forced) Tuolumne.

### Action
1. Regenerate East `basin_averaged_data` + `SUMMA_input` LWRadAtm from METSIM native (drop the
   Dilley–O'Brien override), matching the Tuolumne pipeline.
2. Re-run East WY2006–12; confirm total ET falls toward ~497 mm/yr and Q rises toward obs.
3. Only then revisit vegetation (BallBerry, transpiration) for any residual.

### LW-correction test (2026-07-14) — bug confirmed, but NOT the ET fix

Swapped East SUMMA_input LWRadAtm → METSIM native (grid nearest-cell per HRU), WY2006–12.
Basin-mean LW 316 → 256 W/m² (−60, ε → 0.78, matching Tuolumne). Result:

| ET component, mm/yr | base | LW-fix | Δ |
|---|---|---|---|
| ground evap | 80 | 57 | −23 |
| canopy (interception) evap | 235 | 205 | −30 |
| snow sublimation | 60 | 35 | −25 |
| transpiration | 295 | **342** | **+47** |
| **total** | **670** | **639** | **−31** |

**Total ET fell only 31 mm/yr, not the ~174 predicted.** The non-productive losses
(sublimation + soil evap + interception) dropped 78 mm/yr, but transpiration rose 47 and
absorbed most of it.

Mechanism (from `scalarTranspireLim`, 1.0 = no water stress):
- base: growing-season transpire-limit **0.49** — summer transpiration is water-*limited*
  because the inflated LW drove heavy winter/spring sublimation that dried the soil.
- LW-fix: **0.71** — less sublimation leaves more soil water, so transpiration is less
  limited and rises to fill the gap.

**Conclusion:** the LW bias is real and unphysical and must still be fixed — but East's ET
excess is NOT primarily an energy problem. Total ET is anchored near ~640–670 mm/yr
(72–75% of P) because water is abundant and summer energy meets demand through whichever
pathway is open. Removing energy just re-routes the loss.

**This overturns "LW is the ET fix."** The remaining +142 mm/yr excess (639 vs needed 497) is
not closed by radiation. Two live hypotheses, in priority order:
1. **Precipitation is underestimated.** If true East P > 890, observed ET = P − 393 > 497 and
   the model ET may be ~right — the streamflow deficit would then be a *precip* problem, not
   ET. **Check East P against PRISM / gauge-based estimates next.** This is the cleaner test.
2. **Transpiration capacity too high** (LAI / rootingDepth) and/or soil never water-stressed
   because melt drains straight through (the "no fast path" issue). Would require touching
   params shared with Tuolumne — do only after precip is ruled out.

## Per-HRU ET distribution — 2026-07-14

Which HRUs carry East's excess ET? Answer: **vegetation sets the level (first-order),
aspect modulates within it (second-order).** Elevation, on its own, does not order ET.

### First-order: vegetation / LAI

Area-weighted ET by veg type (mm/yr): evergreen forest **739**, deciduous 690, open shrub
513, wetland 472, barren 391. Per-HRU range 367 → 850. The five evergreen-forest HRUs (5, 7,
8, 10, 13) are the high-ET group; the three high-elevation barren HRUs (1, 3, 4) are the low.

Root cause: the evergreen HRUs carry **summer LAI = 5.31** from the MODIFIED_IGBP_MODIS_NOAH
monthly table — dense-boreal territory, too high for open subalpine East River stands
(realistic ~2.5–3.5). High LAI inflates transpiration *and* interception capacity together.
Reducing LAI cuts both at once, so unlike the interception-only test it should not be undone
by compensation.

### Second-order: aspect (this was real; my first pass mis-analyzed it)

**Correction to an earlier claim.** I first reported "ET tracks veg, not aspect (corr with
elevation ≈ 0)." That was wrong on method: it correlated against *elevation* not aspect,
pooled all veg types (whose 2.3× spread buries a ~15% within-type effect), and used a linear
correlation on a circular variable. Done correctly — within veg type, against per-HRU
terrain-corrected incident SW — aspect clearly drives ET, exactly as the energy balance
requires:

| veg | corr(SW, ET) | S-facing ET | N-facing ET | ΔET | ΔSW |
|-----|--------------|-------------|-------------|-----|-----|
| evergreen | +0.50 | 850 | 710 | **+140 (+20%)** | +32 W/m² |
| open shrub | +0.36 | 516 | 466 | +50 | +19 |
| barren | +0.96 | 407 | 374 | +34 | +42 |

Within-veg partial corr(SW, ET) = **+0.51**. The terrain SW correction is working and moving
ET by aspect. **This is correct physics and must be preserved** — do not flatten it while
fixing the LAI level. Figure: `scratchpad/east_et_distribution.png`.

### Implication

The East ET lever is **evergreen-forest LAI** (and possibly canopy interception capacity),
not aspect and not radiation. Caveat: LAI comes from the veg table shared with Tuolumne, whose
ET is about right (104% volume). Check Tuolumne's evergreen HRUs before any global table edit,
or make LAI basin-specific — else fixing East over-dries Tuolumne (the compensating-error trap
again, now across basins).

## Cross-basin energy balance + the soil-depth cause — 2026-07-14

Followed the user's lead: barren HRUs (LAI≈0, no canopy → **no transpiration to compensate**)
isolate the physics. East's barren ET is genuinely too high, and the cause is **not** energy.

### Barren ET (mm/yr, ground evap + snow sublimation)

| run | ground evap | snow sublim |
|-----|-------------|-------------|
| East base (LW inflated) | 391 | 141 |
| East LW-native | 298 | 87 |
| Tuolumne base | 207 | 102 |

LW fix cuts East barren ground evap 391→298 with no compensation (this is the LW bug's true,
unmasked effect). But East barren is **still +91 above Tuolumne** after the fix.

### Energy drivers over barren HRUs are matched after the LW fix

| | SW | LW | Rn* | Tair | wind | VPD |
|---|----|----|-----|------|------|-----|
| East (LW native) | 204 | 251 | 116 | 0.4°C | 1.5 | 0.47 kPa |
| Tuolumne | 208 | 249 | 109 | 2.3°C | 1.6 | 0.59 kPa |

Same SW, same LW, same wind. East is **colder and lower-VPD** — by the forcing it should
evaporate *less*. **Energy does not explain the residual.** The user's premise was right.

### The residual is soil water supply, not energy

| barren HRUs | soil depth | mean soil water | field capacity |
|-------------|-----------|-----------------|----------------|
| East | ~1.2 m | **304 mm** | 0.25 |
| Tuolumne | ~0.4 m | 105 mm | 0.20 |

East barren soil is **3× deeper and holds 3× the water**, so bare-soil evaporation never runs
out of supply and sits near potential; Tuolumne's thin alpine soil dries and limits it.

Basin-wide (same ramp shape, corr(elev,depth) = −1.00 both, but different magnitude):

| | depth range | area-mean | highest barren |
|---|-------------|-----------|----------------|
| East | 1.20–2.00 m | **1.65 m** | 1.20 m @ 3735 m |
| Tuolumne | 0.33–2.00 m | 0.97 m | 0.33 m @ 3686 m |

East's ramp bottoms out at 1.2 m; alpine barren peaks physically should be thin soil over
bedrock (like Tuolumne's 0.33 m). The deep East soil is inherited from the original East
setup (ramp re-fit preserved old East depths) and was never reconciled with Tuolumne. Some
extra East depth is defensible (Colorado colluvium/shale vs Sierra granite), but 1.2 m on a
3735 m barren peak is not.

### Full ET diagnosis for East (three compounding causes, in order of leverage)

1. **Soil too deep / holds too much water** → surface never dries → ET pinned near potential
   (energy-limited, never water-limited). Explains the barren residual *and* why removing
   energy (LW) just re-routes rather than reduces. **Primary lever.**
2. **Inflated LW** (Dilley–O'Brien bug) → ~half the barren excess; removable, unphysical,
   fix regardless. Masked in forest by transpiration compensation.
3. **Forest LAI = 5.31** → inflates forest transpiration + interception on top of 1–2.

Caveat: soil depth also sets storage/recession (the memory hypothesis) and interacts with the
just-fixed aquifer. Thinning high-elevation soil to limit ET must be checked against baseflow
and against Tuolumne (whose ET is already right). Candidate refinements that reduce
evaporative *supply* without gutting deep storage: thinner **surface** layer, higher soil
evaporation resistance, lower `fieldCapacity` — rather than a blunt total-depth cut.

## East discretization — flat HRUs (for the storage hypothesis)

Tally requested for the "flat HRUs drain slowly" hypothesis. **East has essentially no flat
HRUs:** 18 total, slope range 1.9–25.7°. Only HRU 14 is < 2° (7.0% of area); HRU 9 joins < 5°
(8.7% total). Every HRU has a real aspect (46–318°); there is no flat class. **The East flat-
storage hypothesis has almost no area to act on** — reconsider it or re-discretize if it
matters.

## East fixes APPLIED — 2026-07-15

Four decisions from the user, three applied (LAI deferred):

### 1. Soil ramp matched to Tuolumne (APPLIED, permanent)

Both basins had the same ramp *shape* (15/30/55 split, corr(elev,depth) = −1.00) and nearly
the same *slope* (East −0.000743, Tuolumne −0.000772); the difference was the **intercept**
(East 3.9746 vs Tuolumne 3.1718) — East shifted ~0.80 m deeper everywhere. Applied Tuolumne's
`depth = 3.1718 − 0.000772·elev` (clamped [0.30, 2.0]) to East:
- East soil depth 1.20–2.00 m → **0.30–1.13 m**; barren peaks 1.20 → 0.30 m (Tuolumne 0.32).
- `compactedDepth` 0.60 → **0.15 m** (half new shallowest; avoids the `satHydCond` NaN).
- `rootingDepth` kept at 2.495 m — Tuolumne runs the same over 0.33 m soil and is fine.
- Written to real `coldState.nc` + `localParamInfo.txt`; backups `_20260715_105322`.

### 2. Longwave reverted to METSIM native (APPLIED, permanent)

All **540** East `SUMMA_input` files: `LWRadAtm` replaced with METSIM native (grid nearest-cell
per HRU), dropping the Dilley–O'Brien override. Record-mean LW 315 → 281 W/m²; July emissivity
0.94 → **0.78** (matches Tuolumne). Old inflated forcing preserved in `SUMMA_input_stale_*`.

### 3. Forest LAI — deferred (unchanged, still 5.31)

### 4. Flat HRUs — no fix needed (I was wrong earlier)

The flat class **exists and works**: `aspectClas = 0` on East HRU 9 (slope 3.0°, 13.0 km²) and
HRU 14 (slope 1.9°, 52.0 km²). My earlier "no flat class" was a misread of the continuous
`aspect` field in attributes.nc (gradient azimuth 159/169°) instead of `aspectClas` in the
shapefile. **Total flat area = 65.0 km² = 8.7%** of basin (not 15%, but real). The East
flat-storage hypothesis does have area to act on.

### Result (WY2008–12, a-priori otherwise; verified from real settings)

| | base | **fixed** | target |
|---|------|-----------|--------|
| total ET (mm/yr) | 610 | **507** | 497 ✓ |
| barren ground evap | 392 | **209** | 207 (Tuolumne) ✓ |
| annual Q (mm/yr) | 164 | **295** | 393 |
| runoff ratio | 0.18 | 0.33 | 0.44 |

**ET is essentially solved** and barren matches Tuolumne to 2 mm/yr. Streamflow volume rose
from 42% → 75% of observed. Verified: real settings + real forcing reproduce Q=295, ET=507,
emissivity 0.78, soil 0.30–1.13 m, 540 files.

### Emergent next problem: fast/slow partition (timing)

Monthly Q (mm/day): winter now **too high** (Dec 0.47 vs 0.26 obs), melt peak still **too low**
(Jun 2.19 vs 4.38). Thinner soil routes more water through the aquifer as slow year-round
baseflow instead of a sharp spring freshet — the **"no fast path"** problem. Snowmelt
infiltrates and drains slowly rather than generating saturation-excess/interflow runoff.
Levers: `k_soil` (currently 1e-4 m/s = 8.6 m/day, far too conductive), `qSurfScale`,
`aquiferBaseflowRate`, layer-1 depth. **This is calibration, not a bug** — the first real
calibration target.

## ET aspect anomaly — majority-vote veg artifact (2026-07-15)

User flagged: East SE has much less ET than other aspects at all elevations; Tuolumne NW has
more ET than its (low, north-facing) energy input warrants. **Neither is an energy error.**

### Forcing SW is correct

Incident SW by aspect class (area-weighted):

| | NE | SE | SW | NW | S−N | W−E |
|---|----|----|----|----|-----|-----|
| East | 190 | 223 | 223 | 192 | +32 | **+0.9** |
| Tuolumne | 200 | 243 | 242 | 196 | +44 | **−2.5** |

E–W is symmetric (no phase bias in the terrain SW correction), S–N is correctly positive.
East SE and SW get **identical** SW (223), yet SE ET = 358 vs SW ET = 569. Same energy,
211 mm different ET → not energy.

### Cause: single-veg-per-HRU majority vote

ET tracks forest fraction, and the forest fraction is distorted by the categorical HRU
assignment (one `vegTypeIndex` per HRU = the areal-majority class):

- East SE HRUs: HRU 6 is **35% forest** (53% shrub) → assigned **100% shrub**; HRU 11 is
  **41% forest** → **100% shrub**. SE forest fraction collapses to 0%, ET drops to 358.
- Tuolumne NW HRU 12 is **42% evergreen / 38% shrub** → assigned **100% forest**,
  over-representing it; NW reads 85% forest, ET highest at 391 despite lowest SW.

Net distortion:

| | true fractional forest | majority-vote forest | HRUs 20–80% mixed |
|---|-----------------------|----------------------|-------------------|
| East | 44% | 51% (+7) | **10 / 18** |
| Tuolumne | 37% | 42% (+5) | **17 / 28** |

Over half the HRUs are substantially mixed but forced to one class. There *is* a real
signal underneath (north-facing slopes genuinely more forested — moisture retention), but
majority vote **amplifies** it into the stark 0%-vs-100% aspect contrast. The per-HRU/aspect
ET pattern is therefore not trustworthy; net basin ET is only +5–7 pts biased.

### Options (not yet decided)

1. **Accept + caveat** — net water balance barely affected; treat aspect ET as unreliable
   at HRU scale. Cheapest.
2. **Effective LAI per HRU** — keep one veg type but scale LAI (and ideally canopy params) by
   the fractional composition, so a 35%-forest HRU gets intermediate LAI. Principled, partial
   (SUMMA keys many canopy params off veg type, not just LAI).
3. **Mosaic / composite HRUs** — split mixed HRUs into forest + shrub sub-HRUs (area-weighted).
   Physically correct; ~doubles HRU count.
4. **Add land cover as a discretization dimension** (elevation × aspect × landclass). Most
   HRUs; most faithful.

Connects to the deferred forest-LAI question — both are canopy-representation issues.

## Effective-LAI via `specified` — tested 2026-07-15

Implemented the per-HRU effective-LAI fix for the majority-vote artifact: `LAI_method =
'specified'`, per-HRU `summerLAI` (full transpiring leaf area) + `winterSAI` (persistent
canopy), both weighted by each HRU's true land-cover fractions. Ran East WY2006–12 vs the
`monTable` baseline (east_verify).

### The winterSAI insight works — winter canopy preserved ✓

User's key realization: since interception and shortwave both scale with VAI = LAI + SAI, and
SAI is year-round, put the persistent conifer canopy in `winterSAI`. Confirmed:

| | monTable | specified |
|---|---|---|
| mean peak SWE | 461 | 463 |
| snow sublimation | 34 | 35 mm/yr |
| evergreen-HRU Feb SWE | 283 | 284 mm |

Switching to `specified` did **not** damage evergreen winter snow. The winterSAI approach
avoids the phenology cost cleanly, no recompile.

### But the aspect artifact only partly closes

| aspect ET (mm) | monTable | specified | | transpiration | monTable | specified |
|---|---|---|---|---|---|---|
| NE | 543 | 506 | | NE | 293 | 258 |
| **SE** | **359** | **351** | | **SE** | **144** | **139** |
| SW | 570 | 520 | | SW | 292 | 239 |
| NW | 568 | 531 | | NW | 305 | 270 |

Aspect ET spread 211 → 180 mm — a **modest** reduction, and it comes almost entirely from the
**forest HRUs coming down** (LAI 5.31 → ~4.5), *not* from SE coming up. **SE barely moved**
despite its LAI rising 2.6 → 3.5.

**Why: `specified` changes leaf area, not veg-type physiology.** The SE HRUs are still
`vegTypeIndex = 7` (shrub), so they keep shrub stomatal resistance, rooting depth, canopy
height, and root profile — all keyed off veg type, not LAI. More leaves on a shrub still
transpire like a shrub. Effective LAI cannot fix the aspect artifact because the artifact is
mostly *physiology* (shrub vs forest), not leaf area. **Fully fixing it needs mosaic/composite
HRUs** (split mixed cells into forest + shrub sub-HRUs with full physiology) — a
re-discretization, not a parameter change.

### Net effect is still a genuine improvement (independent of the artifact)

- Forest LAI drops 5.31 → ~4.5 (its true mixed value) — resolves the deferred "LAI too high."
- Total ET 507 → **475** (closer to obs 497); Q 295 → **330** (closer to obs 393, 75%→84%).
- Winter snow preserved.

**Decision pending:** keep `specified` effective-LAI (net-positive: better forest LAI, better
water balance, snow intact) even though it only partly addresses the aspect pattern? The full
aspect fix (mosaic HRUs) is a separate, larger structural choice.

### APPLIED to both basins — 2026-07-15 (permanent)

Written per-HRU `summerLAI`+`winterSAI` to both trialParams; `LAI_method → specified` in both
modelDecisions. Backups `_20260715_121054`. Verified from real settings (WY2008–12):

| | ET before→after | Q before→after | Q/obs | peak SWE |
|---|---|---|---|---|
| East | 507 → **475** (need 497) | 295 → **330** (obs 393) | 0.75 → **0.84** | 463 (kept) |
| Tuolumne | 368 → **321** (need 481) | 635 → **685** (obs 612) | 1.04 → **1.12** | 772 (kept) |

**Opposite responses, and both are diagnostic:**
- East a-priori ET was too high → trimming over-represented forest LAI moves it toward obs. ✓
- Tuolumne a-priori ET was already too *low* (Q already 104%, and recall its 104% masked a
  17–46 day timing error) → reducing LAI pushes ET further below the 481 target, Q to 1.12.

This is not a method failure: effective-LAI gives a **physically honest a-priori** (true
land-cover fractions, not majority-vote binaries) for both basins. The residual water-balance
gaps (East 84%, Tuolumne 112%) are **calibration targets**. Tuolumne now visibly *needs more
ET* — a clear signal for the rooting-depth / stomatal calibration, previously masked.

Snow preserved in both. Aspect ET spread reduced in both. Kept for cross-basin consistency
(user directive) and physical honesty. Held-out OpenET evaluation should show the same SE-vs-
SW / NW aspect ET signature if the land-cover contrast is real — a validation opportunity.

## Blocking decisions

1. **Lateral flow.** `downHRUindex` is populated in both domains, but
   `spatial_gw = localColumn` renders it inert — each HRU receives only its own
   precipitation. The East hypothesis as stated ("flat areas collect water from
   upslope and release it slowly") is **not representable**. Either:
   - enable `distributed` lateral redistribution (costs runtime; requires real
     geometry — Tuolumne's `contourLength` is a placeholder 100.0 on every HRU), or
   - reframe to what `localColumn` can test: `phi`/`psi` give "deep and slow", but
     not "receives more".

2. **Which HRU hosts the snowfield.** There is no NE HRU. The highest (hruId 1,
   3681 m) is **west**-facing; the north-facing high HRU is hruId 2 (3394 m). The
   running S0 sweep uses hruId 1 and may therefore be melting the pile too fast.

3. `w1, w2, w3` — pick, or set by ablation.
