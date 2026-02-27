#!/bin/bash
#
# CONFLUENCE Optimization Runner
# Complete preprocessing + calibration with flexible algorithm selection
#
# Supported Algorithms: DDS (sequential), PSO, SCE, GA (parallel-friendly)
# Usage: ./optimize.sh --config config_Tuolumne_lumped_v1.yaml [--algorithm PSO]

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/optimize_confluence.py"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  CONFLUENCE Optimization (Complete)                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

CONFIG_FILE=""
RUN_NAME=""
ALGORITHM=""
MPI_PROCESSES=""
USE_PREVIOUS=false
CONFIG_DIR="${SCRIPT_DIR}/../../0_config_files"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            if [[ ! "$ALGORITHM" =~ ^(DDS|PSO|SCE|GA|DE)$ ]]; then
                echo -e "${RED}Error: Invalid algorithm. Choose from: DDS, PSO, SCE, GA, DE${NC}"
                exit 1
            fi
            shift 2
            ;;
        --mpi-processes)
            MPI_PROCESSES="$2"
            if ! [[ "$MPI_PROCESSES" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Error: MPI processes must be a number${NC}"
                exit 1
            fi
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --use-previous)
            USE_PREVIOUS=true
            shift
            ;;
        --help|-h)
            cat << 'HELP'
Usage: ./optimize.sh [OPTIONS]

Options:
  --config FILE            Configuration YAML file (required)
  --algorithm {DDS|PSO|SCE|GA|DE}   Optimization algorithm (overrides config)
                           DDS  = Sequential (recommended with --mpi-processes 1)
                           PSO  = Particle Swarm (parallel-friendly)
                           SCE  = Shuffled Complex (parallel-friendly)
                           GA   = Genetic Algorithm (parallel-friendly)
                           DE   = Differential Evolution (parallel-friendly)
  --mpi-processes N        Number of MPI processes (overrides config)
                           Use 1 with DDS, >1 with other algorithms
  --run-name NAME          Custom experiment name
  --use-previous           Use previous best parameters
  --help                   Show this help

Examples:
  # Standard run
  ./optimize.sh --config config_Tuolumne_lumped_v1.yaml
  
  # DDS with single processor (recommended combo)
  ./optimize.sh --config config_Tuolumne_lumped_v1.yaml --algorithm DDS --mpi-processes 1
  
  # PSO with 4 parallel processes
  ./optimize.sh --config config_Tuolumne_lumped_v1.yaml --algorithm PSO --mpi-processes 4
  
  # DDS with custom experiment name
  ./optimize.sh --config config_Tuolumne_lumped_v1.yaml --algorithm DDS --mpi-processes 1 --run-name "spongy_v2"
HELP
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: --config required${NC}"
    exit 1
fi

CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}Config not found: ${CONFIG_PATH}${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Configuration:${NC}"
echo "  Config: ${CONFIG_FILE}"
[ -n "$ALGORITHM" ] && echo "  Algorithm: ${ALGORITHM}"
[ -n "$MPI_PROCESSES" ] && echo "  MPI Processes: ${MPI_PROCESSES}"
[ -n "$RUN_NAME" ] && echo "  Run name: ${RUN_NAME}"
[ "$USE_PREVIOUS" = true ] && echo "  Using previous best parameters"

echo ""
cd "${SCRIPT_DIR}"

# Build Python command
PYTHON_CMD="python3 \"${PYTHON_SCRIPT}\" --config \"${CONFIG_FILE}\""
[ -n "$ALGORITHM" ] && PYTHON_CMD="${PYTHON_CMD} --algorithm ${ALGORITHM}"
[ -n "$MPI_PROCESSES" ] && PYTHON_CMD="${PYTHON_CMD} --mpi-processes ${MPI_PROCESSES}"
[ -n "$RUN_NAME" ] && PYTHON_CMD="${PYTHON_CMD} --run-name \"${RUN_NAME}\""
[ "$USE_PREVIOUS" = true ] && PYTHON_CMD="${PYTHON_CMD} --use-previous"

eval "${PYTHON_CMD}"

echo -e "\n${GREEN}✅ Complete!${NC}"
