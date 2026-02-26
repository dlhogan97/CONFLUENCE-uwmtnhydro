#!/bin/bash
#
# DDS Optimization Runner
# Complete preprocessing + calibration in one command
#
# Usage: ./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "my_run"

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/optimize_confluence.py"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  CONFLUENCE DDS Optimization (Complete)               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

CONFIG_FILE=""
RUN_NAME=""
USE_PREVIOUS=false
CONFIG_DIR="${SCRIPT_DIR}/../../0_config_files"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
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
  --config FILE       Configuration YAML file (required)
  --run-name NAME     Custom experiment name
  --use-previous      Use previous best parameters
  --help             Show this help

Example:
  ./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "spongy_v2"
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
[ -n "$RUN_NAME" ] && echo "  Run name: ${RUN_NAME}"
[ "$USE_PREVIOUS" = true ] && echo "  Using previous best parameters"

echo ""
cd "${SCRIPT_DIR}"

if [ "$USE_PREVIOUS" = true ]; then
    python3 "${PYTHON_SCRIPT}" --config "${CONFIG_FILE}" --run-name "${RUN_NAME}" --use-previous
else
    python3 "${PYTHON_SCRIPT}" --config "${CONFIG_FILE}" --run-name "${RUN_NAME}"
fi

echo -e "\n${GREEN}✅ Complete!${NC}"
