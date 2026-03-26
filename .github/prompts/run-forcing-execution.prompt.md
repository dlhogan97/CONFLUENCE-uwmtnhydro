---
description: "Generate executable scripts and command steps to build hourly 4 km SUMMA forcing from PRISM, ERA5-Land, Daymet, and MetSim."
name: "Run Forcing Execution"
argument-hint: "Basin, date range, data directories, output NetCDF path, and compute environment"
agent: "Forcing Fusion Engineer"
model: ["GPT-5 (copilot)", "Claude Sonnet 4.5 (copilot)"]
---
Produce an execution-ready implementation for this forcing run:

{{input}}

Do not return high-level strategy only. Return runnable artifacts and commands.

Execution policy:
- Radiation targets: incoming shortwave and incoming longwave.
- Source mapping: PRISM FTP provides tmin, tmax, ppt, and soltotal; ERA5-Land (cdsapi) provides wind inputs; Daymet (pydaymet) provides vapor pressure.
- Temporal handling: use daily source data where applicable and use MetSim to disaggregate to hourly output.
- Interpolation policy: use Ordinary Kriging by default; if unstable or unsupported, use scipy grid interpolation.
- Pressure policy: use MetSim-derived pressure as default.
- Chunking policy: run acquisition, harmonization, and assembly in monthly chunks and write monthly files for the full time series.
- Spatial subset policy: use the catchment shapefile in `domain_East_River_lumped` to identify/select source grid cells before downloading or clipping where APIs support geometry filtering.
- Exclusion policy: do not use GridMET shortwave in this workflow.

Required output sections:
1. Preconditions and dependency install commands.
2. Directory layout and expected inputs.
3. Python scripts/modules to create or edit with exact file paths.
4. Commands to run each stage in order.
5. Validation commands and expected checks.
6. Recovery steps for common failures (missing tiles, kriging failure, time gaps).

Implementation requirements:
- Prefer scripts under ess-project/1_forcing and reusable utilities under utils/custom or utils/data.
- Include concrete command lines and environment assumptions.
- Include expected output files for each stage.
- Include monthly filename conventions and a final monthly-continuity verification step.
