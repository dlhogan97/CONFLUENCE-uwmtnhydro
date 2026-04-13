#!/usr/bin/env bash
set -euo pipefail

# Re-run Tuolumne lumped model-agnostic preprocessing to regenerate
# EASYMORE basin-averaged forcing from existing forcing files.

REPO_ROOT="/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro"
CONFIG_PATH="$REPO_ROOT/ess-project/0_config_files/config_Tuolumne_lumped_noXplict.yaml"
BASIN_ROOT="/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_lumped"
FORCING_ROOT="$BASIN_ROOT/forcing"
BASIN_AVG_DIR="$FORCING_ROOT/basin_averaged_data"
METSIM_DIR="$FORCING_ROOT/metsim_outputs"
MERGED_DIR="$FORCING_ROOT/merged_data"

ARCHIVE_EXISTING="${ARCHIVE_EXISTING:-1}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$FORCING_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/rerun_tuolumne_lumped_preprocessing_${RUN_TS}.log}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_ROOT"

echo "=== Tuolumne Lumped Preprocessing (EASYMORE rerun) ==="
echo "Log file: $LOG_FILE"
echo "Config: $CONFIG_PATH"
echo "Primary forcing: $METSIM_DIR"
echo "Fallback forcing: $MERGED_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Missing config: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$METSIM_DIR" ]]; then
  echo "ERROR: Missing metsim forcing directory: $METSIM_DIR" >&2
  exit 1
fi

if [[ "$ARCHIVE_EXISTING" == "1" && -d "$BASIN_AVG_DIR" ]] && find "$BASIN_AVG_DIR" -maxdepth 1 -name '*.nc' | grep -q .; then
  archive_dir="$FORCING_ROOT/basin_averaged_data_archive_${RUN_TS}"
  echo "Archiving existing basin_averaged_data -> $archive_dir"
  mv "$BASIN_AVG_DIR" "$archive_dir"
fi

conda run -n ess-project-env python -c "from pathlib import Path; from CONFLUENCE import CONFLUENCE; config_path = Path('/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/0_config_files/config_Tuolumne_lumped_noXplict.yaml'); confluence = CONFLUENCE(config_path); confluence.managers['project'].setup_project(); confluence.managers['data'].run_model_agnostic_preprocessing(); print('Model-agnostic preprocessing complete.')"

count="$(find "$BASIN_AVG_DIR" -maxdepth 1 -type f -name '*.nc' | wc -l || true)"
echo "Basin-averaged files now present: $count"
echo "=== Preprocessing complete ==="
