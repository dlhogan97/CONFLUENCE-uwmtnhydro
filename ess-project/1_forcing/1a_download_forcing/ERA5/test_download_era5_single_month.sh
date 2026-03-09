#!/bin/bash

# Test script: Download 1 month of ERA5 surface data to verify configuration

# Configuration
YEAR=2018
START_MONTH=9
END_MONTH=9
BOUNDING_BOX="39.1/-107.2/38.5/-106.6"
FORCING_PATH="/scratch/dlhogan/ess-project-data/domain_East_River_lumped/forcing/raw_data"

echo "=========================================="
echo "ERA5 Surface-Level Download Test"
echo "=========================================="
echo "Year: $YEAR"
echo "Month range: $(printf '%02d' $START_MONTH) to $(printf '%02d' $END_MONTH)"
echo "Bounding box: $BOUNDING_BOX"
echo "Output path: $FORCING_PATH"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$FORCING_PATH"

# Run the download script
echo "Starting download..."
python download_ERA5_surfaceLevel_annual.py "$YEAR" "$BOUNDING_BOX" "$FORCING_PATH" "$START_MONTH" "$END_MONTH"

echo ""
echo "=========================================="
echo "Download test complete!"
echo "=========================================="
echo ""
echo "Check for downloaded file:"
ls -lh "$FORCING_PATH"/ERA5_surface_${YEAR}$(printf '%02d' $START_MONTH).nc 2>/dev/null || echo "File not found yet (may still be downloading or validating)"
