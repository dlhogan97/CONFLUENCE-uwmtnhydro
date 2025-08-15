#!/bin/bash

# Script to download ERA5 pressure level data
# Reads download path, years to download and spatial extent from 'summaWorkflow_public/0_config_files/control_active.txt'.

# Requires use of the Copernicus Data Store API
# CDS registration: https://cds.climate.copernicus.eu/user/register?destination=%2F%23!%2Fhome
# CDS api setup: https://cds.climate.copernicus.eu/api-how-to

# --- Settings
# get user input for the location of the control file
if [ -z "$1" ]; then
  echo "Please provide the path to the control file (e.g., $control_file_path)"
  exit 1
else
  control_file_path="$1"
fi
echo "Using control file: $control_file_path"

# -- Find where to save data
# Find the line with the forcing path
setting_line=$(grep -m 1 -i "forcing_path" $control_file_path) # -m 1 ensures we only return the top-most result. This is needed because variable names are sometimes used in comments in later lines
echo "Forcing path setting line: $setting_line"
# Extract the path
forcing_path=$(echo ${setting_line##**: }) # remove the part that ends at "|"
forcing_path=$(echo ${forcing_path%%#*}) # remove the part starting at '#'; does nothing if no '#' is present

# Specify the default path if needed
if [ "$forcing_path" = "default" ]; then
  
    # Get the root path
    root_line=$(grep -m 1 -i "confluence_data_dir" $control_file_path)
    root_path=$(echo ${root_line##*:}) 
    root_path=$(echo ${root_path%%#*}) 

    # Get the domain path
    domain_line=$(grep -m 1 -i "domain_name" $control_file_path)
    domain_name=$(echo ${domain_line##*: }) 
    domain_name=$(echo ${domain_name%%#*})  

    forcing_path="${root_path}/domain_${domain_name}/forcing/raw_data/"
fi

# Make the folder if it doesn't exist
mkdir -p $forcing_path
echo "Forcing path: $forcing_path"

# -- Find temporal and spatial domain
# - time
start_setting_line=$(grep -m 1 -i "experiment_time_start" $control_file_path)
start_year=$(echo ${start_setting_line##*|} | cut -d' ' -f2 | cut -d'-' -f1)
end_setting_line=$(grep -m 1 -i "experiment_time_end" $control_file_path)
end_year=$(echo ${end_setting_line##*|} | cut -d' ' -f2 | cut -d'-' -f1)
arrayYears=(${start_year} ${end_year}) # split string into array for later use, based on delimiter ','
echo "Years to download: ${arrayYears[0]} to ${arrayYears[1]}"

# - space
setting_line=$(grep -m 1 -i "bounding_box_coords" $control_file_path)
coordinates=$(echo ${setting_line##*:}) 
coordinates=$(echo ${coordinates%%#*})
echo "Coordinates: $coordinates"

# --- Parallel runs
# Build the target variable (we need this because we can't use brace expansion based on 'arrayYears'
years=""; 
for (( year=$(( arrayYears[0] )); year<=$(( arrayYears[1] )); year++ )); do
 years="$years $year";
done

# Run the ERA5 downloads with reduced parallelism to avoid conflicts
max_jobs=2  # Reduced from 8 to prevent file corruption
count=0
pids=()  # store process IDs here

for y in $years; do
  for c in $coordinates; do
    for f in $forcing_path; do

      # Add some debugging and error checking
      echo "Starting download: Year=$y, Coords=$c, Path=$f"
      
      python download_ERA5_surfaceLevel_annual.py "$y" "$c" "$f" &
      pid=$!
      pids+=($pid)  # save the PID
      echo "Started PID $pid: $y $c $f"

      ((count++))
      if (( count % max_jobs == 0 )); then
        echo "Waiting for batch of $max_jobs jobs to complete..."
        wait  # wait until these finish before starting more
        echo "Batch completed, checking for any failed jobs..."
        
        # Check if any downloads failed
        for pid in "${pids[@]}"; do
          if ! wait $pid; then
            echo "ERROR: Process $pid failed!"
          fi
        done
        pids=()  # reset the array
      fi

    done
  done
done

echo "Waiting for all remaining jobs to complete..."
wait

# Final check for file integrity
echo "Checking downloaded files for readability..."
for y in $years; do
  for month in {01..12}; do
    file="${forcing_path}/ERA5_surface_${y}${month}.nc"
    if [ -f "$file" ]; then
      echo "Checking file: $file"
      
      # Check if file is actually a ZIP archive
      file_type=$(file "$file")
      if [[ "$file_type" == *"Zip archive"* ]]; then
        echo "WARNING: File $file is still a ZIP archive!"
        echo "File type: $file_type"
        echo "You may need to re-run the download for this file."
      elif command -v ncdump >/dev/null 2>&1; then
        # Try to read the file header with ncdump
        if ! ncdump -h "$file" >/dev/null 2>&1; then
          echo "ERROR: File $file is corrupted or unreadable!"
          echo "File type: $file_type"
          ls -la "$file"
        else
          echo "OK: File $file is readable NetCDF"
        fi
      else
        echo "Warning: ncdump not available for file checking"
        echo "File type: $file_type"
      fi
    fi
  done
done

# --- Code provenance
# Generates a basic log file in the domain folder and copies the control file and itself there.

# Make a log directory if it doesn't exist
log_path="${forcing_path}/_workflow_log"
mkdir -p $log_path

# Log filename
today=`date '+%F'`
log_file="${today}_surface_level_log.txt"

# Copy this script
this_file='run_download_ERA5_surfaceLevel.sh'
that_file='download_ERA5_surfaceLevel_annual.py'
cp $this_file $log_path
cp $that_file $log_path

# Create a log file
# echo "Log generated by ${this_file} on `date '+%F %H:%M:%S'`"  > $log_path/$log_file # 1st line, store in new file
echo "Downloaded ERA5 pressure level data for space (lat_max, lon_min, lat_min, lon_max) [${coordinates}] for time Jan-${arrayYears[0]} / Dec-${arrayYears[1]}" >> $log_path/$log_file # 2nd line, append to existing file