# Note: This script was adapted from the original script produced by the CWARHM project team. Original repository can be found at: https://github.com/CH-Earth/CWARHM
# modules
import cdsapi    # copernicus connection
import calendar  # to find days per month
import os        # to check if file already exists
import sys       # to handle command line arguments (sys.argv[0] = name of this file, sys.argv[1] = arg1, ...)
import math
import netCDF4 as nc4  # for file validation
import zipfile   # for extracting ZIP archives
import tempfile  # for temporary file handling
import xarray as xr  # for merging multiple NetCDF files
from pathlib import Path
from shutil import copyfile, move
from datetime import datetime

# CDS registration: https://cds.climate.copernicus.eu/user/register?destination=%2F%23!%2Fhome
# CDS api setup: https://cds.climate.copernicus.eu/api-how-to

''' 
Downloads 1 year of ERA5 data as monthly chunks.
Usage: python download_ERA5_surfaceLevel_annual.py <year> <coordinates> <path/to/save/data> 
'''

# Get the year we're downloading from command line argument
year = int(sys.argv[1]) # arguments are string by default; string to integer

# Get the spatial coordinates as the second command line argument
bounding_box = sys.argv[2] # string
bounding_box = bounding_box.split('/') # split string
bounding_box = [float(value) for value in bounding_box] # string to array

# Get the path as the second command line argument
forcingPath = Path(sys.argv[3]) # string to Path()

# Function to handle ZIP files from CDS API
def extract_if_zip(filepath):
    """
    Check if the downloaded file is a ZIP archive and extract it if needed.
    Returns the path to the final NetCDF file.
    """
    filepath = Path(filepath)
    
    # Check if file is a ZIP archive
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            print(f"File {filepath} is a ZIP archive, extracting...")
            
            # List contents
            zip_contents = zip_ref.namelist()
            print(f"ZIP contents: {zip_contents}")
            
            # Find the NetCDF file in the archive
            nc_files = [f for f in zip_contents if f.endswith('.nc')]
            
            if len(nc_files) == 0:
                raise Exception(f"No NetCDF files found in ZIP archive: {zip_contents}")
            elif len(nc_files) == 1:
                print(f"Single NetCDF file found: {nc_files[0]}")
                nc_file = nc_files[0]
                
                # Extract the single NetCDF file
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    extracted_file = temp_path / nc_file
                    
                    # Extract the file
                    zip_ref.extract(nc_file, temp_path)
                    
                    # Move the extracted file to replace the original ZIP
                    move(str(extracted_file), str(filepath))
                    
            else:
                print(f"Multiple NetCDF files found: {nc_files}")
                print("Merging all NetCDF files into single file...")
                
                # Extract all NetCDF files and merge them
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    extracted_files = []
                    
                    # Extract all NetCDF files
                    for nc_file in nc_files:
                        extracted_file = temp_path / nc_file
                        zip_ref.extract(nc_file, temp_path)
                        extracted_files.append(extracted_file)
                        print(f"  Extracted: {nc_file}")
                    
                    # Load and examine each file
                    datasets = []
                    all_vars = set()
                    
                    for i, extracted_file in enumerate(extracted_files):
                        print(f"  Loading {extracted_file.name}...")
                        ds = xr.open_dataset(extracted_file)
                        datasets.append(ds)
                        
                        # Get variable names (excluding coordinate variables)
                        data_vars = [var for var in ds.data_vars.keys()]
                        all_vars.update(data_vars)
                        print(f"    Variables: {data_vars}")
                    
                    print(f"  All variables found: {sorted(all_vars)}")
                    
                    # Merge datasets
                    print("  Merging datasets...")
                    try:
                        # Try to merge along time dimension if available
                        if 'time' in datasets[0].dims:
                            merged_ds = xr.concat(datasets, dim='time', combine_attrs='override')
                        else:
                            # Merge by combining data variables
                            merged_ds = xr.merge(datasets, combine_attrs='override')
                        
                        # Save merged dataset to original location
                        merged_ds.to_netcdf(filepath)
                        print(f"  Saved merged dataset with {len(merged_ds.data_vars)} variables")
                        
                        # Close datasets
                        for ds in datasets:
                            ds.close()
                        merged_ds.close()
                        
                    except Exception as merge_error:
                        print(f"  Error merging datasets: {merge_error}")
                        print("  Falling back to using first file only...")
                        
                        # Close any open datasets
                        for ds in datasets:
                            ds.close()
                        
                        # Fall back to first file
                        first_file = extracted_files[0]
                        move(str(first_file), str(filepath))
                        
            print(f"Successfully processed {filepath}")
            return filepath
            
    except zipfile.BadZipFile:
        # Not a ZIP file, assume it's already a NetCDF file
        return filepath
    except Exception as e:
        print(f"Error extracting ZIP file {filepath}: {e}")
        raise e

# --- Convert the bounding box to download coordinates
# function to round coordinates of a bounding box to ERA5s 0.25 degree resolution
def round_coords_to_ERA5(coords):
    
    '''Assumes coodinates are an array: [lat_max,lon_min,lat_min,lon_max] (top-left, bottom-right).
    Returns separate lat and lon vectors.'''
    
    # Extract values
    lon = [coords[1],coords[3]]
    lat = [coords[2],coords[0]]
    
    # Round to ERA5 0.25 degree resolution
    rounded_lon = [math.floor(lon[0]*4)/4, math.ceil(lon[1]*4)/4]
    rounded_lat = [math.floor(lat[0]*4)/4, math.ceil(lat[1]*4)/4]
    
    # Find if we are still in the representative area of a different ERA5 grid cell
    if lat[0] > rounded_lat[0]+0.125:
        rounded_lat[0] += 0.25
    if lon[0] > rounded_lon[0]+0.125:
        rounded_lon[0] += 0.25
    if lat[1] < rounded_lat[1]-0.125:
        rounded_lat[1] -= 0.25
    if lon[1] < rounded_lon[1]-0.125:
        rounded_lon[1] -= 0.25
    
    # Make a download string
    dl_string = '{}/{}/{}/{}'.format(rounded_lat[1],rounded_lon[0],rounded_lat[0],rounded_lon[1])
    
    return dl_string, rounded_lat, rounded_lon

# Find the rounded bounding box
coordinates,_,_ = round_coords_to_ERA5(bounding_box)

# --- Start the month loop
for month in range (1,13): # this loops through numbers 1 to 12
       
    # find the number of days in this month
    daysInMonth = calendar.monthrange(year,month) 
        
    # compile the date string in the required format. Append 0's to the month number if needed (zfill(2))
    date = str(year) + '-' + str(month).zfill(2) + '-01/' + \
        str(year) + '-' + str(month).zfill(2) + '-' + str(daysInMonth[1]).zfill(2) 
        
    # compile the file name string
    file = forcingPath / ('ERA5_surface_' + str(year) + str(month).zfill(2) + '.nc')

    # track progress
    print('Trying to download ' + date + ' into ' + str(file))

    # if file doesn't yet exist, download the data
    if not os.path.isfile(file):
            
        # Make sure the connection is re-tried if it fails
        retries_max = 10
        retries_cur = 1
        while retries_cur <= retries_max:
            try:

                # connect to Copernicus (requires .cdsapirc file in $HOME)
                c = cdsapi.Client()

                # specify and retrieve data
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': [
                            'mean_surface_downward_long_wave_radiation_flux',                
                            'mean_surface_downward_short_wave_radiation_flux',
                            'mean_total_precipitation_rate', 
                            'surface_pressure',
                        ],
                        'date': date,
                        'time': '00/to/23/by/1',
                        'area': coordinates,	# North, West, South, East. Default: global
                    	'grid': '0.25/0.25',    # Latitude/longitude grid: east-west (longitude) and north-south
                    },
                    file) # file path and name

                # track progress
                print('Successfully downloaded ' + str(file))
                
                # Handle ZIP extraction if needed
                try:
                    final_file = extract_if_zip(file)
                    print(f'File processing complete: {final_file}')
                except Exception as extract_error:
                    print(f'ERROR: Failed to extract/process downloaded file {file}: {extract_error}')
                    # Remove the problematic file so it can be re-downloaded
                    if os.path.exists(file):
                        os.remove(file)
                        print(f'Removed problematic file: {file}')
                    raise extract_error
                
                # Validate the final NetCDF file
                try:
                    with nc4.Dataset(final_file, 'r') as test_ds:
                        print(f'File validation: {len(test_ds.dimensions)} dimensions, {len(test_ds.variables)} variables')
                        # Check if it has the expected variables
                        expected_vars = ['msdwlwrf', 'msdwswrf', 'mtpr', 'sp']
                        missing_vars = [var for var in expected_vars if var not in test_ds.variables]
                        if missing_vars:
                            print(f'Warning: Missing expected variables: {missing_vars}')
                            print(f'Available variables: {list(test_ds.variables.keys())}')
                        else:
                            print('File validation: All expected variables present')
                except Exception as validation_error:
                    print(f'ERROR: Processed file {final_file} is corrupted: {validation_error}')
                    # Remove the corrupted file so it can be re-downloaded
                    if os.path.exists(final_file):
                        os.remove(final_file)
                        print(f'Removed corrupted file: {final_file}')
                    raise validation_error

            except Exception as e:
                print('Error downloading ' + str(file) + ' on try ' + str(retries_cur))
                print(str(e))
                # If file exists but is corrupted or ZIP, remove it
                if os.path.exists(file):
                    try:
                        # Try to process the file (extract if ZIP, validate if NetCDF)
                        final_file = extract_if_zip(file)
                        with nc4.Dataset(final_file, 'r') as test_ds:
                            pass  # File is readable, keep it
                        print(f'File {file} was successfully processed during error handling')
                        break  # Exit retry loop since file is good
                    except:
                        print(f'Removing corrupted/problematic file: {file}')
                        os.remove(file)
                retries_cur += 1
                continue
            else:
                break