#!/bin/bash
# Run inside tmux:  tmux new -s emulation
# Then:             bash run_emulation.sh [options]
# Detach:           Ctrl+B, D
# Reconnect:        tmux attach -t emulation
#
# Runs SUMMA parameter emulation (neural-network surrogate calibration) and
# produces a feature-importance report without running the full seasonal
# ensemble pipeline.  Training data is cached and can be reused.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_CONFIG="$SCRIPT_DIR/../0_config_files/config_East_River_lumped_bigBuckt.yaml"

usage() {
    cat <<'EOF'
Usage:
  bash run_emulation.sh [options]

Optional:
  -c, --config PATH          CONFLUENCE config YAML
                             (default: ../0_config_files/config_East_River_lumped_bigBuckt.yaml)

Optional:
  --n-train N                Training samples (default: 500)
  --n-validation N           Validation samples (default: 100)
  --epochs N                 NN training epochs (default: 300)
  --force-retrain            Regenerate training data even if cached
  --emulator-setting MODE    EMULATOR | FD  (default: EMULATOR)
  --report-only              Skip SUMMA runs; generate importance report from
                             existing cached training data
  --mpi-processes N          Override MPI_PROCESSES from config (default: use config value)
  -h, --help                 Show this help text

Examples:
  bash run_emulation.sh \
      --config ../0_config_files/config_East_River_lumped_bigBuckt.yaml

  bash run_emulation.sh \
      --config ../0_config_files/config_East_River_lumped_bigBuckt.yaml \
      --n-train 800 --epochs 500

  bash run_emulation.sh \
      --config ../0_config_files/config_East_River_lumped_bigBuckt.yaml \
      --report-only
EOF
}

CONFIG_FILE="$DEFAULT_CONFIG"
N_TRAIN=500
N_VALIDATION=100
EPOCHS=300
FORCE_RETRAIN=false
EMULATOR_SETTING="EMULATOR"
REPORT_ONLY=false
MPI_PROCESSES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)         CONFIG_FILE="$2";        shift 2 ;;
        --n-train)           N_TRAIN="$2";             shift 2 ;;
        --n-validation)      N_VALIDATION="$2";        shift 2 ;;
        --epochs)            EPOCHS="$2";              shift 2 ;;
        --force-retrain)     FORCE_RETRAIN=true;       shift   ;;
        --emulator-setting)  EMULATOR_SETTING="$2";    shift 2 ;;
        --report-only)       REPORT_ONLY=true;         shift   ;;
        --mpi-processes)     MPI_PROCESSES="$2";       shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        --)                  shift; break ;;
        -*)  echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)
            if [[ -z "$CONFIG_FILE" ]]; then
                CONFIG_FILE="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; usage; exit 1
            fi ;;
    esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config file not found: $CONFIG_FILE" >&2; exit 1
fi

CONFIG_FILE=$(realpath "$CONFIG_FILE")

echo "Config:           $CONFIG_FILE"
echo "Training samples: $N_TRAIN  Validation: $N_VALIDATION"
echo "Epochs:           $EPOCHS"
echo "Emulator setting: $EMULATOR_SETTING"
echo "Force retrain:    $FORCE_RETRAIN"
echo "Report only:      $REPORT_ONLY"

export OMP_NUM_THREADS=1
LOGS="$SCRIPT_DIR/logs"
mkdir -p "$LOGS"
LOG_FILE="$LOGS/emulation_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ess-project-env
cd "$SCRIPT_DIR"

export EMU_CONFIG_FILE="$CONFIG_FILE"
export EMU_LOG_FILE="$LOG_FILE"
export EMU_MPI_PROCESSES="$MPI_PROCESSES"
export EMU_N_TRAIN="$N_TRAIN"
export EMU_N_VALIDATION="$N_VALIDATION"
export EMU_EPOCHS="$EPOCHS"
export EMU_FORCE_RETRAIN="$FORCE_RETRAIN"
export EMU_EMULATOR_SETTING="$EMULATOR_SETTING"
export EMU_REPORT_ONLY="$REPORT_ONLY"

python - <<'PY'
import os
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── locate CONFLUENCE root from the config's CONFLUENCE_CODE_DIR ──────────
with open(os.environ["EMU_CONFIG_FILE"]) as fh:
    cfg_raw = yaml.safe_load(fh)

confluence_root = Path(cfg_raw.get("CONFLUENCE_CODE_DIR", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(confluence_root))

from utils.optimization.optimization_manager import OptimizationManager

# ── build config overlay ───────────────────────────────────────────────────
config = dict(cfg_raw)
config["OPTIMISATION_METHODS"] = ["differentiable_parameter_emulation"]
config["EMULATOR_SETTING"]     = os.environ["EMU_EMULATOR_SETTING"]
config["DPE_TRAINING_SAMPLES"] = int(os.environ["EMU_N_TRAIN"])
config["DPE_VALIDATION_SAMPLES"] = int(os.environ["EMU_N_VALIDATION"])
config["DPE_EPOCHS"]           = int(os.environ["EMU_EPOCHS"])
config["DPE_FORCE_RETRAIN"]    = os.environ["EMU_FORCE_RETRAIN"].lower() == "true"
mpi_override = os.environ.get("EMU_MPI_PROCESSES", "").strip()
if mpi_override:
    config["MPI_PROCESSES"] = int(mpi_override)

report_only = os.environ["EMU_REPORT_ONLY"].lower() == "true"

# ── logger ─────────────────────────────────────────────────────────────────
log_file = Path(os.environ["EMU_LOG_FILE"])
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger("emulation")
logger.info(f"Log file: {log_file}")

# ── paths ──────────────────────────────────────────────────────────────────
data_dir    = Path(config["CONFLUENCE_DATA_DIR"])
domain_name = config["DOMAIN_NAME"]
domain_dir  = data_dir / f"domain_{domain_name}"
cache_path  = domain_dir / "emulation" / "training_data" / f"training_data_{domain_name}.json"

# ── optionally skip SUMMA runs and go straight to the report ───────────────
if not report_only:
    manager = OptimizationManager(config, logger)
    results = manager.run_optimization_workflow()
    logger.info(f"Emulation results: {results}")
else:
    logger.info("--report-only: skipping SUMMA runs, loading cached training data.")

# ── feature importance report ──────────────────────────────────────────────
if not cache_path.exists():
    logger.warning(f"No training data cache found at {cache_path}. Run without --report-only first.")
    sys.exit(0)

logger.info(f"Loading training data from {cache_path}")
with open(cache_path) as fh:
    cached = json.load(fh)

train = cached.get("training", cached)          # handle both cache formats
raw_params = train.get("parameters", [])
raw_objs   = train.get("objectives", [])

if not raw_params or not raw_objs:
    logger.warning("Training data cache is empty.")
    sys.exit(0)

params_array = np.array(raw_params)             # (N, n_params)
objs_array   = np.array(raw_objs)               # (N,) or (N, n_objectives)
if objs_array.ndim > 1:
    kge_scores = objs_array[:, 0]               # first objective (KGE)
else:
    kge_scores = objs_array

# ── recover parameter names from config ───────────────────────────────────
param_names_str = config.get("PARAMS_TO_CALIBRATE", "")
basin_names_str = config.get("BASIN_PARAMS_TO_CALIBRATE", "")
param_names = [p.strip() for p in param_names_str.split(",") if p.strip()]
basin_names = [p.strip() for p in basin_names_str.split(",") if p.strip()]
all_param_names = param_names + basin_names

if len(all_param_names) != params_array.shape[1]:
    # Fall back to generic names if count doesn't match
    all_param_names = [f"param_{i}" for i in range(params_array.shape[1])]
    logger.warning("Parameter name count mismatch — using generic names.")

df = pd.DataFrame(params_array, columns=all_param_names)
df["KGE"] = kge_scores

# ── 1. Pearson / Spearman correlations ────────────────────────────────────
pearson  = df[all_param_names].corrwith(df["KGE"]).rename("pearson_r")
spearman = df[all_param_names].corrwith(df["KGE"], method="spearman").rename("spearman_r")

# ── 2. Random Forest feature importance ───────────────────────────────────
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    mask = np.isfinite(kge_scores)
    X = params_array[mask]
    y = kge_scores[mask]

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    rf_importance = pd.Series(rf.feature_importances_, index=all_param_names, name="rf_importance")
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    perm_importance = pd.Series(perm.importances_mean, index=all_param_names, name="permutation_importance")
    rf_ok = True
except Exception as exc:
    logger.warning(f"sklearn RF importance failed: {exc}")
    rf_ok = False

# ── assemble report ────────────────────────────────────────────────────────
report = pd.concat([pearson.abs().rename("|pearson_r|"),
                    spearman.abs().rename("|spearman_r|")], axis=1)
if rf_ok:
    report = pd.concat([report, rf_importance, perm_importance], axis=1)

report = report.sort_values("|spearman_r|", ascending=False)

report_dir = domain_dir / "emulation" / "feature_importance"
report_dir.mkdir(parents=True, exist_ok=True)
report_csv = report_dir / f"feature_importance_{config['EXPERIMENT_ID']}.csv"
report.to_csv(report_csv)

# ── print to console ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Parameter Sensitivity / Feature Importance Report")
print(f"Domain:  {domain_name}   Experiment: {config['EXPERIMENT_ID']}")
print(f"Samples: {len(df)}  (KGE range: {kge_scores.min():.3f} – {kge_scores.max():.3f})")
print("=" * 60)
print(report.to_string(float_format="{:.4f}".format))
print("=" * 60)
print(f"\nReport saved to: {report_csv}")
print(f"Full training data: {cache_path}")
PY

echo ""
echo "Emulation complete. Feature importance report written to:"
echo "  \$CONFLUENCE_DATA_DIR/domain_<name>/emulation/feature_importance/"
