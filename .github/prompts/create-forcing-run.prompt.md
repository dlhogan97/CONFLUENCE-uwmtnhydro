---
description: "Create a complete multi-source forcing workflow plan and implementation steps for SUMMA using PRISM, ERA5-Land, Daymet, MetSim, and kriging to produce hourly 4 km NetCDF outputs."
name: "Create Forcing Run"
argument-hint: "Basin, date range, output path, variables, and any constraints"
agent: "Forcing Fusion Engineer"
model: ["GPT-5 (copilot)", "Claude Sonnet 4.5 (copilot)"]
---
Build a forcing workflow for CONFLUENCE/SUMMA using the following run specification:

{{input}}

If details are missing, make minimal explicit assumptions and list them.

Required workflow policy:
- Radiation targets: incoming shortwave and incoming longwave.
- Source mapping: PRISM FTP provides tmin, tmax, ppt, and soltotal; ERA5-Land (cdsapi) provides wind inputs; Daymet (pydaymet) provides vapor pressure.
- Temporal handling: use daily source data where applicable and use MetSim to disaggregate to hourly output.
- Interpolation policy: use Ordinary Kriging by default; if unstable or unsupported, use scipy grid interpolation.
- Pressure policy: use MetSim-derived pressure as default.
- Chunking policy: process and save data by month across the full period (monthly files for all intermediate and assembled outputs).
- Spatial subset policy: use the catchment shapefile in `domain_East_River_lumped` to identify/select source grid cells prior to download or clipping where supported.
- Exclusion policy: do not use GridMET shortwave in this workflow.

Required output sections:
1. Objective and basin setup.
2. Source-by-source acquisition plan (PRISM FTP, ERA5-Land cdsapi, Daymet pydaymet).
3. Spatial harmonization method and rationale.
4. MetSim configuration and variable derivation strategy.
5. Hourly NetCDF schema for SUMMA (dimensions, variable names, units, encoding, and time handling).
6. Validation checklist and pass/fail criteria.
7. Implementation plan as ordered tasks with file-level changes.
8. Risks, uncertainty hotspots, and recommended improvements.

In all sections, explicitly describe monthly chunk boundaries, monthly output naming conventions, and how catchment shapefile masking/selecting is applied.

For implementation steps, include concrete Python module/file suggestions under ess-project and utils, and call out dependencies to install if missing.
