#!/usr/bin/env bash
set -euo pipefail

# Rebuild Tuolumne lumped forcing using MetSim, then regenerate basin averages.
# Execution environment requirement: ess-project-env.

REPO_ROOT="/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro"
CONFIG_PATH="$REPO_ROOT/ess-project/0_config_files/config_Tuolumne_lumped_noXplict.yaml"
BASIN_ROOT="/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_lumped"
FORCING_ROOT="$BASIN_ROOT/forcing"
MERGED_DIR="$FORCING_ROOT/merged_data"
METSIM_OUT_DIR="$FORCING_ROOT/metsim_outputs"
TMP_ROOT="$FORCING_ROOT/metsim_batch_tmp"
CATCHMENT_SHP="$BASIN_ROOT/shapefiles/catchment/Tuolumne_River_lumped_HRUs_GRUs.shp"
DEM_PATH="$BASIN_ROOT/attributes/elevation/dem/domain_Tuolumne_River_lumped_elv.tif"

# Requested reconstruction window (download/rebuild): back to 1990.
START_MONTH="${START_MONTH:-1990-01}"
END_MONTH="${END_MONTH:-2021-09}"
WORKERS="${WORKERS:-8}"

# Override if your MetSim executable is elsewhere.
METSIM_EXE="${METSIM_EXE:-/home/dlhogan/miniforge3/envs/metsim-run/bin/ms}"

# Simple run log target (works with foreground and nohup runs).
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$FORCING_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/rebuild_tuolumne_metsim_${RUN_TS}.log}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_ROOT"

echo "=== Tuolumne MetSim Rebuild ==="
echo "Log file: $LOG_FILE"
echo "Config: $CONFIG_PATH"
echo "Basin root: $BASIN_ROOT"
echo "Window: $START_MONTH to $END_MONTH"
echo "Workers: $WORKERS"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Missing config: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$MERGED_DIR" ]]; then
  echo "ERROR: Missing merged forcing directory: $MERGED_DIR" >&2
  exit 1
fi
if [[ ! -f "$CATCHMENT_SHP" ]]; then
  echo "ERROR: Missing catchment shapefile: $CATCHMENT_SHP" >&2
  exit 1
fi
if [[ ! -f "$DEM_PATH" ]]; then
  echo "ERROR: Missing DEM: $DEM_PATH" >&2
  exit 1
fi
if [[ ! -x "$METSIM_EXE" ]]; then
  echo "ERROR: MetSim executable not found or not executable: $METSIM_EXE" >&2
  echo "Set METSIM_EXE=/path/to/ms and retry." >&2
  exit 1
fi

# 1) Archive existing basin averages (ERA5-derived) so they remain available.
if [[ -d "$FORCING_ROOT/basin_averaged_data" ]] && find "$FORCING_ROOT/basin_averaged_data" -maxdepth 1 -name '*.nc' | grep -q .; then
  ts="$(date +%Y%m%d_%H%M%S)"
  archive_dir="$FORCING_ROOT/basin_averaged_data_era5_archive_${ts}"
  echo "Archiving existing basin_averaged_data -> $archive_dir"
  mv "$FORCING_ROOT/basin_averaged_data" "$archive_dir"
fi

mkdir -p "$METSIM_OUT_DIR" "$TMP_ROOT"

# 2) Pick a reference merged file for metadata (prefer 2015-12 if present).
REFERENCE_FILE="$MERGED_DIR/ERA5_merged_201512.nc"
if [[ ! -f "$REFERENCE_FILE" ]]; then
  REFERENCE_FILE="$(find "$MERGED_DIR" -maxdepth 1 -type f -name '*.nc' | sort | head -n 1)"
fi
if [[ -z "${REFERENCE_FILE:-}" || ! -f "$REFERENCE_FILE" ]]; then
  echo "ERROR: Could not find a reference merged NetCDF in $MERGED_DIR" >&2
  exit 1
fi

echo "Using reference file: $REFERENCE_FILE"

# 3) Reconstruct MetSim outputs.
conda run -n ess-project-env python "$REPO_ROOT/ess-project/1_forcing/run_metsim_batch.py" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --basin-root "$BASIN_ROOT" \
  --output-dir "$METSIM_OUT_DIR" \
  --tmp-root "$TMP_ROOT" \
  --reference "$REFERENCE_FILE" \
  --dem "$DEM_PATH" \
  --catchment-shp "$CATCHMENT_SHP" \
  --workers "$WORKERS" \
  --metsim-exe "$METSIM_EXE" \
  --build-missing-daily \
  --allow-download

# 4) Regenerate basin averages from METSIM forcing via model-agnostic preprocessing.
conda run -n ess-project-env python - <<'PY'
from pathlib import Path
from CONFLUENCE import CONFLUENCE

config_path = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/0_config_files/config_Tuolumne_lumped_noXplict.yaml")
confluence = CONFLUENCE(config_path)
confluence.managers['project'].setup_project()
confluence.managers['data'].run_model_agnostic_preprocessing()
print("Model-agnostic preprocessing complete.")
PY

# 5) Lightweight validation summary.
metsim_count="$(find "$METSIM_OUT_DIR" -maxdepth 1 -type f -name '*.nc' | wc -l)"
basin_count="$(find "$FORCING_ROOT/basin_averaged_data" -maxdepth 1 -type f -name '*.nc' | wc -l)"

echo "=== Rebuild complete ==="
echo "METSIM files: $metsim_count"
echo "Basin-averaged files: $basin_count"
echo "Primary forcing path now: $METSIM_OUT_DIR"
echo "Fallback forcing path: $MERGED_DIR"
