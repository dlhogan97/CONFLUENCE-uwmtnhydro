# Single-basin aquifer (leak to a basin-wide store): overnight calibration, East River

Written 2026-10-14 from the overnight run. Everything here was produced unattended; decisions are listed so they can be overridden.

![summary](../../figures/draft/singlebasin_calibration_summary.png)

## What was done

1. **SUMMA change** (`/home/dlhogan/summa_deepstore`, a copy: the original `/home/dlhogan/summa` build is untouched). `spatial_gw = singleBasin` now keeps the local HRU aquifers
   (plants access them; they drain to the channel) and adds a state-dependent leak from each to one basin-wide store (no plant access; drains only to the channel):
   `leak = aquiferLeakRate*(S/Sf)^aquiferLeakExp`, basin store `Qb = basin__aquiferHydCond*(S/basin__aquiferScaleFactor)^basin__aquiferBaseflowExp`.
   New parameters `aquiferLeakRate`, `aquiferLeakExp`, `basin__aquiferInitStorage`; new flux `scalarAquiferLeak`; new basin outputs `basin__AquiferLeak`, `basin__DeepBaseflow`, `basin__AquiferStorage`.
   Files edited: var_lookup, popMetadat, get_ixname, flxMapping, bigAquifer, computFlux, systemSolv, vegNrgFlux, coupled_em, run_oneGRU, summa_modelRun, summa_restart. Not committed.
   Tests: leak off reproduces the old build bit-for-bit (1981-82, 10 variables); mass balance closes in all three single-basin full runs (leak in = deep baseflow out to 4 decimals of a metre; area-weighted HRU leak = basin leak).
2. **Calibration design.** Snow/soil/ET parameters are frozen at the earlier calibration, so recharge is the same for every model. Only aquifer and routing parameters are fitted.
   Because the leak is one-way, a fast emulator (`emu_core.py`) reproduces SUMMA: the four full-window SUMMA runs match the emulator to 3 decimals (table at the end).
   Same objective for all models: `(1-KGE) + (1-logNSE) + |Aug-Nov bias|/100 + |total bias|/100`.
   Models: **M0** baseline (3 free params), **M1** single-basin full (7), **M2** single-basin reduced: linear leak + linear deep store (5), **M3** = M1 plus a penalty for mismatching the *observed* residual lag-1 and raw lag-1 in the same window (7).
   Each is fitted on three windows: full (WY1987-2024), A (1986-2005), B (2006-2024); fits on A and B are validated on the other half. Differential evolution, one seed (21), emulator.
3. **Verification in SUMMA.** The full-window fits were run for 1981-2025 in the new build (`deepstore_dev/overnight/summa/M*_full`).

## Headline results

* **M1 (single-basin, skill-only fit) beats the recalibrated baseline M0 on skill, in the full fit and in both held-out directions.** Full: KGE 0.81 vs 0.77, logNSE 0.83 vs 0.73, objective 0.41 vs 0.56, Aug-Nov bias about 0 for both.
  Held out (fit one half, test the other): KGE 0.85 vs 0.82 (fit 1986-2005, test 2006-2024) and 0.75 vs 0.69 (reverse); logNSE 0.85 vs 0.73 and 0.77 vs 0.69. The loss from calibration to validation is similar for M0 and M1, so M1's extra four parameters are not overfitting more than M0 here.
* **M1 carries multi-year memory without being fitted to it.** Residual lag-1 +0.31 (observed +0.39), raw lag-5 -0.43 (observed -0.40), residual lag-5 -0.24 (observed -0.30), prior-year deep storage vs low flow +0.51, 33% of baseflow routed through the basin store. M0 gives +0.07, -0.30 (raw lag-5), +0.03.
* **The gain depends on nonlinearity.** M2 (linear leak and deep store) is no better than M0 (objective 0.560 vs 0.560) and its leak coefficient went to its lower bound.
* **Fitting the memory directly (M3) did not help.** It matched the raw lag-1 (0.16 vs observed 0.14) but then had residual lag-1 only +0.07 (observed +0.39); its time-split memory is erratic. Observed raw lag-1 is much lower than residual lag-1, and none of these structures reproduces both at once.
* **Memory is not stable across halves, in the data or the models.** Observed residual lag-1 is -0.01 (1986-2005, n=20) and +0.56 (2006-2024, n=19); sampling noise at n of about 20 is roughly +/-0.45, so this may be noise. M1 gives about +0.3 whichever half it is fitted on, so it overshoots the first half (+0.29 vs -0.01) and undershoots the second (+0.35 vs +0.56).

## Things to know before trusting this

* **Objective weights are my choice.** The Aug-Nov bias term pushes the recalibrated baseline to Aug-Nov bias +1% but costs it KGE: the original baseline was KGE 0.86 / logNSE 0.70 / Aug-Nov bias +34.5%, and M0 is 0.77 / 0.73 / +1.3%. The original and the recalibrated numbers are not comparable. A different weighting could change the ranking of KGE.
* **Pegged and near-bound parameters.** M0's `aquiferBaseflowExp` sits at its upper bound of 8.0 (it was at 5.0 in a first run, objective 0.585; widened to 8, objective 0.560; the first run is kept as `fit_M0_full_exp5bound_superseded.json`). M2's leak coefficient is at its lower bound. M1 and M3 routing scale (1.43e5, 1.45e5 s) is within about 10% of the upper bound (1.58e5).
  M0 is therefore structurally limited, which is itself part of the result, but it may be handicapped by the bounds.
* **Upstream parameters are frozen.** A joint recalibration of soil/snow/ET with the aquifer parameters could narrow the gap. Not done.
* **One site, 39 years, two halves of about 20.** Differences in lag correlations of 0.1-0.2 are within noise. One optimiser seed per fit.
* **Emulator vs SUMMA.** The emulator reuses the baseline's recharge (net of aquifer transpiration). It matches SUMMA closely for all four fits, so aquifer-transpiration feedback is negligible in these runs.
* **A Fortran formatting trap.** The parameter files are read with a fixed-width real format: a value written without a decimal point (e.g. `1`) is read as 1e-4. M2's first SUMMA run failed because of this; it was rebuilt with decimals and rerun. `summa_runs.py` now always writes a decimal point.
* **Two runs were redone.** The pipeline was restarted once after the first M0 fit hit its bound (the superseded files are labelled). M2_full was relaunched once (the formatting issue above).

## Files

* Code (this folder): `emu_core.py` (emulator + metrics), `fit_models.py`, `summa_runs.py` (build/launch/score SUMMA runs), `driver.py` (the overnight pipeline), `finalize.py` (tables), `make_figure.py`. The recharge cache `recharge_cache.npz` is regenerated from the baseline run if missing.
* Results: `/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/overnight/` (`fit_<model>_<window>.json`, `summa_scores.json`, `results_tables.md`, `driver.log`); SUMMA runs in `.../overnight/summa/M*_full/` (each with `fileManager.txt`, ready to rerun with `/tmp/claude-1008/summa_ds -m`, a wrapper that points LD_LIBRARY_PATH at the new `libsumma.so`; recreate it if /tmp was cleaned).
* SUMMA source: `/home/dlhogan/summa_deepstore` (build dir `build/cmake/cmake_build_ds`, executable `bin/summa.exe`).

## Suggested next steps

1. Decide whether M1's skill gain is worth the change, then calibrate with the project's runner (joint with soil/snow parameters) using the new build, with M1's fitted aquifer parameters as the starting point.
2. Rerun the fits with 2-3 seeds and wider routing and baseflow-exponent bounds to see whether the ranking is stable.
3. Look at why raw lag-1 is overshot: the models' low flow is probably tied to same-year peak flow more tightly than the observations are.
4. Extend the forcing back to 1970 to get about 11 more years of observations for the memory comparison.

---

# Result tables (auto-generated by `finalize.py`)

## Observed memory by window

| window | n | raw lag1 | resid lag1 | raw lag5 | resid lag5 |
|---|---|---|---|---|---|
| 1986-2024 | 39 | 0.14 | 0.39 | -0.40 | -0.30 |
| 1986-2005 | 20 | 0.14 | -0.01 | -0.55 | -0.18 |
| 2006-2024 | 19 | 0.06 | 0.56 | -0.12 | -0.41 |

## Full-window fits (calibrated and scored on WY1987-2024; memory 1986-2024)

| model | free params | objective J | KGE | logNSE | Aug-Nov bias % | total bias % | raw lag1 | resid lag1 | raw lag5 | resid lag5 | prior-yr deep storage link | leak share | pegged at bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 baseline (independent buckets) | 3 | 0.560 | 0.77 | 0.73 | +1.3 | +4.4 | 0.16 | 0.07 | -0.30 | 0.03 | - | - | exp_h |
| M1 single-basin, full | 7 | 0.410 | 0.81 | 0.83 | -0.3 | +4.8 | 0.45 | 0.31 | -0.43 | -0.24 | 0.51 | 0.33 | none |
| M2 single-basin, reduced (linear) | 5 | 0.560 | 0.79 | 0.76 | +6.4 | +4.4 | 0.07 | -0.01 | -0.23 | 0.06 | -0.08 | 0.30 | llam |
| M3 single-basin, memory-informed | 7 | 0.767 | 0.80 | 0.83 | +0.2 | +4.8 | 0.16 | 0.07 | -0.33 | -0.09 | 0.04 | 0.27 | none |

### Fitted parameters (full window)

| model | aquiferScaleFactor | aquiferBaseflowExp | routingGammaScale | leak coef (x rate) | leak exp | deep scale factor | deep exp |
|---|---|---|---|---|---|---|---|
| M0 baseline (independent buckets) | 0.594 | 8 | 92492 | - | - | - | - |
| M1 single-basin, full | 0.704 | 2.79 | 142732 | 0.0187 | 1.35 | 12.4 | 2.41 |
| M2 single-basin, reduced (linear) | 0.62 | 3.1 | 91078 | 0.00513 | 1 | 70.7 | 1 |
| M3 single-basin, memory-informed | 0.425 | 6.27 | 144953 | 0.00935 | 2.76 | 16.7 | 6.81 |

## Time-split validation (fit on one half, evaluated on the other)

| model | calibrated on | cal KGE | valid KGE | cal logNSE | valid logNSE | cal AugNov bias | valid AugNov bias | cal resid lag1 | valid resid lag1 | valid OBS resid lag1 | valid raw lag1 | valid OBS raw lag1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 baseline (independent buckets) | 1986-2005 | 0.77 | 0.82 | 0.77 | 0.73 | +0 | +18 | -0.15 | 0.02 | 0.56 | 0.10 | 0.06 |
| M0 baseline (independent buckets) | 2006-2024 | 0.79 | 0.69 | 0.69 | 0.69 | +3 | -10 | 0.12 | -0.21 | -0.01 | 0.15 | 0.14 |
| M1 single-basin, full | 1986-2005 | 0.79 | 0.85 | 0.82 | 0.85 | -0 | +13 | -0.02 | 0.35 | 0.56 | 0.14 | 0.06 |
| M1 single-basin, full | 2006-2024 | 0.86 | 0.75 | 0.86 | 0.77 | +1 | -5 | 0.55 | 0.29 | -0.01 | 0.40 | 0.14 |
| M2 single-basin, reduced (linear) | 1986-2005 | 0.75 | 0.82 | 0.78 | 0.76 | +0 | +14 | -0.23 | -0.05 | 0.56 | -0.06 | 0.06 |
| M2 single-basin, reduced (linear) | 2006-2024 | 0.78 | 0.67 | 0.72 | 0.70 | +1 | -8 | -0.01 | -0.30 | -0.01 | 0.07 | 0.14 |
| M3 single-basin, memory-informed | 1986-2005 | 0.80 | 0.87 | 0.80 | 0.85 | +4 | +16 | -0.03 | 0.43 | 0.56 | 0.28 | 0.06 |
| M3 single-basin, memory-informed | 2006-2024 | 0.86 | 0.78 | 0.85 | 0.80 | +10 | -1 | 0.43 | 0.02 | -0.01 | 0.30 | 0.14 |

## Real SUMMA runs (new singleBasin build) vs emulator, full window

| model | KGE SUMMA/emulator | logNSE | AugNov bias | resid lag1 | raw lag1 |
|---|---|---|---|---|---|
| M0 baseline (independent buckets) | 0.771 / 0.771 | 0.726 / 0.726 | +1.3 / +1.3 | 0.075 / 0.075 | 0.161 / 0.159 |
| M1 single-basin, full | 0.812 / 0.812 | 0.829 / 0.829 | -0.3 / -0.3 | 0.312 / 0.312 | 0.446 / 0.446 |
| M2 single-basin, reduced (linear) | 0.787 / 0.787 | 0.761 / 0.761 | +6.4 / +6.4 | -0.002 / -0.006 | 0.075 / 0.075 |
| M3 single-basin, memory-informed | 0.796 / 0.796 | 0.828 / 0.828 | +0.2 / +0.2 | 0.073 / 0.068 | 0.162 / 0.162 |


---

# Addendum (2026-10-14): signature-based evaluation and SWE check

Scripts: `signature_eval.py` (project signature objective + extra diagnostics, WY1987-2024; csv outputs alongside), `aso_east_swe_check.py` (ASO SWE rasters averaged within each HRU polygon vs model SWE on the flight date; uses `snotel_hru_swe_metrics.csv` from `utils/snotel_metrics.py` for Butte).
Signature errors are 0 = perfect; com_timing is normalized by 7 d, monthly_volume is the volume-significant monthly bias, recession by tau target 27 d, lowflow_sepnov is the log10 error of the annual Sep-Nov 7-day minimum.

| | original baseline | M0 recal | M1 single-basin | M2 linear | M3 memory-fit |
|---|---|---|---|---|---|
| signature objective (equal-weight mean) | 0.550 | 0.432 | **0.353** | 0.392 | 0.364 |
| com_timing | 1.62 | 1.40 | 1.35 | 1.40 | 1.36 |
| monthly_volume | 0.81 | 0.31 | **0.22** | 0.24 | 0.24 |
| recession | 0.17 | 0.24 | **0.05** | 0.18 | 0.08 |
| lowflow_sepnov | 0.07 | 0.10 | 0.08 | 0.06 | 0.07 |
| total volume bias % | +4.4 | +4.4 | +4.8 | +4.4 | +4.8 |
| COM timing, mean abs error (d) | 11.3 | 9.8 | 9.4 | 9.8 | 9.5 |
| annual 7-day peak bias % | +9.7 | +29.9 | +21.9 | +27.8 | +25.7 |
| May-Jul volume bias % | +1.7 | +8.8 | +5.1 | +6.6 | +5.6 |
| Sep-Nov 7-day-min bias % (median year) | -12.0 | -23.8 | +0.8 | -10.2 | +2.0 |

SWE vs observations (snow/soil parameters are frozen, so snow is essentially the same in every run: the largest hourly HRU SWE difference between the new runs and the original baseline run is 8.7 mm; the original baseline run was used for the comparison):
* Butte SNOTEL vs the HRU containing it (HRU 13, 3016 m vs the 3106 m pillow), n=42 years: peak SWE -22% (r=0.93), peak 9 d late, melt-out 11 d late.
* ASO basin-wide, area-weighted over all 18 HRUs: March-April flights bias -17.5% to +11.8% (mean about -5%), HRU-to-HRU r 0.85-0.96. By elevation band the model is +37% at about 2700 m, +1% at about 3000 m, -22% at about 3350 m, -1% at about 3700 m.
  May-June flights vary in sign by year: model 44% below ASO on 2018-05-24, about 30% above on 2023-05-23, 20% above on 2024-05-20 and almost snow-free (10 mm vs 133 mm) on 2025-05-20.
