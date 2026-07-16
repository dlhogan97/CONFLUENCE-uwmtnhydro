# Calibration & Experiment Specification — 2026-07-15 (rev. 2)

## Framing (this drives every choice below)

The model is an **instrument for a hypothesis test**, not a calibrated water-resource model.
The question: *do specific long-term stores explain observed late-season memory?* The argument
rests on **aggregate signatures**, not daily fit. Every simplification is stated plainly so a
model-inclined reviewer judges the work on the memory axis, not on daily-hydrograph replication.

Two consequences:
1. **Objective = aggregate signatures**, not KGE/NSE. Daily KGE/NSE are computed and reported
   in the SI only — never optimized.
2. **Calibration establishes a "no long-term storage" baseline** (parameters + a memory-free
   initial state). Any memory in the experiments must come from the **initial conditions we
   impose**, not from the calibration. So the calibrated baseline must reproduce the signatures
   *without* leaning on a deep persistent store.

## Objective — aggregate signatures (both basins)

Minimize a weighted sum of normalized signature errors over the analysis window:

| # | signature | definition | why |
|---|-----------|------------|-----|
| 1 | **center-of-mass timing** | day-of-WY of 50% cumulative annual flow; error = mean \|Δ\| days | melt timing |
| 2 | **monthly volume** | 12-month mean hydrograph; error = Σ\|sim−obs\|/Σobs | seasonal shape |
| 3 | **AMJJ runoff** | Apr–Jul total volume; error = \|Δ\|/obs | freshet magnitude |
| 4 | **baseflow recession** | master recession constant τ; error = \|τ_sim − τ_target\|/τ_target | store drainage rate |
| 5 | **Aug–Sep low flow** | late-season volume; error = \|Δ\|/obs | the memory signal itself |

Start equal-weighted; low-flow/recession (4,5) may be up-weighted since they carry the memory
story. τ targets: **East 30 d, Tuolumne 15 d** — the user's targets, *independently confirmed*
by the observed master recession (East 26 d, Tuolumne 14 d). Routing mean lag is centered on the
**Lundquist et al. (2005)** Tuolumne travel-time proxy: **Tuolumne ≈20 h, East ≈24 h** (slightly
slower). Set as the a-priori (`routingGammaScale` = mean/`shape`, shape 2.5 → Tuo 28800 s, East
34560 s) and optimized in a narrow band around it. Lundquist = **routing**, not recession.

**Guards (hard — reject trial):** snow not broken (peak SWE within [0.6,1.4]× a-priori, HRUs
still melt out); 0.2 < runoff ratio < 0.9; run completed, obs/sim aligned.

**Reported to SI (not optimized):** daily KGE, NSE, log-NSE.

## Calibration = memory-free baseline

- **Wet-soil initialization** (Bart): start soils near **field capacity**, not the current 0.2
  VolFracLiq. Removes the artificial lag of snow/precip first wetting dry soil and lets the
  baseline aquifer equilibrate faster — so the baseline carries no spurious "dry-start" memory.
- **Aquifer at seasonal equilibrium**, not an imposed pile — the baseline drains and refills
  within the year; it must not itself encode multi-year carryover.
- Calibration finds the parameter set that hits the five signatures from this memory-free state.
- The end-of-spin-up state becomes the **baseline IC** that every experiment perturbs.

### Windows

**Calibration and experiments must not overlap** — the IC experiments + snow/ET evaluation own
WY2013–2024, so calibration lives entirely before it.

- **Calibration spin-up:** WY1990–1992 (3 yr; dry years → drains the aquifer to a low-storage
  baseline, consistent with the memory-free framing). Wet-soil (field-capacity) start.
- **Calibration window:** WY1993–2002 (10 yr), spin-up dropped. **Same window both basins**
  (simpler manuscript; Tuolumne pre-2007 is scaled-HH, consistent with the merged long-record
  lag analysis). Chosen for a wet/dry/average mix in both basins:
  - East: wet 1993/95/97, dry 1994/**2002 (−30%)**, avg 1996–2001. P 785–1311 mm.
  - Tuolumne: wet 1993/95/96/97/98, dry 1994/2001/2002, avg 1999/2000. P 672–1770 mm.
  - Passed on WY2003–2012 (East too dry-leaning) and WY1985–94 (spin-up would be the 3 wettest
    years on record → aquifer starts full, wrong baseline).
- **Experiment/evaluation window:** WY2015–2024 (last ~10 yr) — reserved, never calibrated on.
- **Long record:** WY1981–2024 for the observed-vs-model **lag-correlation** (behavioral
  comparison, not a fit — may span all years).

## Parameters — multipliers on the a-priori per-HRU fields

(Preserve the elevation/aspect/soil-depth structure; scale, don't flatten.)

### Tuolumne — `qTopmodl`
`k_soil` ×[0.1,10] · `zScale_TOPMODEL` ×[0.2,5] · `theta_sat` ×[0.7,1.15] ·
`aquiferBaseflowExp` [1,5] · `frozenPrecipMultip` [0.85,1.25] · `routingGammaScale` [12–30 h].
Drop `vGn_alpha`/`vGn_n`. Fix `tempCritRain = 273.16`. τ→15 d.

### East — `bigBuckt`
`k_soil` ×[0.1,10] · `theta_sat` ×[0.7,1.15] · `aquiferBaseflowRate` [1e-8,1e-5] ·
`aquiferScaleFactor` [0.5,5] · `aquiferBaseflowExp` [1,5] · `frozenPrecipMultip` [0.85,1.25] ·
`routingGammaScale` [16–36 h]. No `zScale_TOPMODEL` emphasis. Fix `tempCritRain = 273.16`. τ→30 d.

## The experiment — vary initial conditions, measure late-season flow

Not one long run — an **ensemble over initial states** on the last ~10 yr. The **spread in
Aug–Sep flow across states is the signal.**

- **Tuolumne:** vary (snowpatch volume) × (soil-moisture init) → relate the combination to
  late-season flow.
- **East:** vary (soil-moisture init) × (aquifer-drainage init) — a **deep store in the flat
  valley HRUs vs. uniform-emptying stores** — to test whether that structure produces memory.
  (This is the flat-HRU hypothesis, now framed as an IC experiment on the fixed calibrated
  parameters — no recalibration, so no compensating-error risk.)

## Memory test — does memory exist, and what carries it

1. **Does it exist:** lag-1/lag-2 autocorrelation of Aug–Sep low-flow volume, **controlling for
   concurrent late-summer precip** (separate carryover from current-year weather; late-summer P
   should not be year-to-year autocorrelated anyway).
2. **Obs vs model:** compute the same lag structure on the **observed 1981–2024** record and on
   the long model run; do they match?
3. **What carries it:** correlate late-season flow against modeled **prior-year aquifer storage**
   (East) / **residual SWE** (Tuolumne). Test whether the store reproduces the observed lag
   structure while the **uniform config does not**.

## Evaluation — behavioral checks, not the focus

Neither is where we think the missing water goes; both are fidelity checks.
- **Spatial ET (OpenET):** does the model capture aspect-driven energy-balance + land-cover
  differences (the SE-shrub / SW-forest signature — see the aspect-ET note)?
- **Spatial SWE (ASO):** pattern fidelity via the radial diagrams, then a mean difference — not
  point accuracy.

## Algorithm & compute (20 cores)

- `differential_evolution`, `workers=20`, ~7 params. ~74 s/sim-yr × 10 yr ≈ 12 min/eval serial
  → **~35 s/eval effective on 20 cores**. popsize 15×7 = 105, ~20 gen ≈ 2000 evals ≈ **~1 day**
  per basin. Feasible.
- Signature objective is smooth-ish and low-dimensional; DE is robust here.

## Reproducibility

Fixed seed; per-calibration manifest (params, window, git SHA, obs checksum); every trial logged
(params → 5 signature errors → guards → KGE/NSE for SI → runtime). Remove the `999.0` sentinel
(failed run → real penalty + logged reason). Delete `pre_calibrate_routing.py` +
`routing_params_shared.json` (offline two-reservoir; it was a 262-day groundwater model, not routing).

## Decisions (resolved 2026-07-15)

1. **Signature weights: equal** (all five). May revisit after seeing results.
2. **Wet-soil init = `fieldCapacity`** (East 0.25, Tuolumne 0.20) applied to all soil layers in
   coldState. Field capacity, not 0.75·θ_sat: it is the drained equilibrium, so no artificial
   first-days drainage pulse. East moves 0.20→0.25; Tuolumne already there.
3. **One baseline calibration per basin.** Experiments run on the fixed calibrated parameters.
4. **Best match to the five signatures** (not a memory-free straitjacket on the fit) — the user
   wants the closest-to-obs model *with respect to these functions*; decent KGE/NSE should fall
   out and is reported (SI). "Memory-free" refers to the baseline **state** (wet soils,
   equilibrium aquifer, no imposed pile), not to hobbling the parameter fit.

## IC response-surface experiment (the deliverable figure)

Vary initial state, run the ~10 yr window, color each run by its **Aug–Sep runoff anomaly vs
baseline**. Read the color gradient: which axis it aligns with is which store carries memory.

- **Tuolumne axes:** initial SWE / snowpatch volume  ×  initial soil moisture.
- **East axes:** initial **aquifer storage**  ×  aquifer **drainage rate** (soil-moisture init
  as a third dim or held fixed). Modulating drainage is allowed — it just sharpens the claim
  from "a store initialized differently" to "a *slower* store produces more memory." **Report
  which knob moved** so the attribution stays clean.
- **Grid, not 4 points:** ~4×4 (or Latin hypercube, 16–25 runs). Cheap at ~35 s/eval × 20 cores.
- **Companion time-series panel:** modeled aquifer storage (East) / residual SWE (Tuolumne) and
  baseflow vs **observed**, for a few IC states — shows the *mechanism* (slow drawdown feeding
  late-season flow), not just the *magnitude*.
