#!/usr/bin/env python3
"""
Script to fix existing ZIP files downloaded from CDS API
Usage: python fix_existing_zip_files.py <path_to_forcing_directory>
"""

import sys
import zipfile
import tempfile
from pathlib import Path
from shutil import move
import netCDF4 as nc4
import xarray as xr

def extract_if_zip(filepath):
    """
    Check if the file is a ZIP archive and extract it if needed.
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
                    
                    # Create backup of original ZIP file before replacing
                    backup_dir = filepath.parent / "_zip_backups"
                    backup_dir.mkdir(exist_ok=True)
                    backup_path = backup_dir / f"{filepath.stem}_backup.zip"
                    
                    # Move original ZIP to backup location
                    move(str(filepath), str(backup_path))
                    print(f"Backed up original ZIP to: {backup_path}")
                    
                    # Move the extracted NetCDF file to the new file
                    savepath = filepath.with_suffix('.nc')
                    move(str(extracted_file), str(savepath))
                    
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
                        
                        # Create backup of original ZIP file
                        backup_dir = filepath.parent / "_zip_backups"
                        backup_dir.mkdir(exist_ok=True)
                        backup_path = backup_dir / f"{filepath.stem}_backup.zip"
                        
                        # Move original ZIP to backup location
                        move(str(filepath), str(backup_path))
                        print(f"  Backed up original ZIP to: {backup_path}")
                        
                        # Save merged dataset to original location
                        savepath = filepath.with_suffix('.nc')
                        merged_ds.to_netcdf(savepath)
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
                        
                        # Create backup and move first file
                        backup_dir = filepath.parent / "_zip_backups"
                        backup_dir.mkdir(exist_ok=True)
                        backup_path = backup_dir / f"{filepath.stem}_backup.zip"
                        
                        move(str(filepath), str(backup_path))
                        print(f"  Backed up original ZIP to: {backup_path}")
                        
                        move(str(first_file), str(filepath))
                
            print(f"Successfully extracted {nc_file} to {savepath}")
            return savepath
            
    except zipfile.BadZipFile:
        # Not a ZIP file, assume it's already a NetCDF file
        print(f"File {filepath} is not a ZIP archive")
        return filepath
    except Exception as e:
        print(f"Error extracting ZIP file {filepath}: {e}")
        raise e

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_existing_zip_files.py <path_to_forcing_directory>")
        sys.exit(1)
    
    forcing_dir = Path(sys.argv[1])
    
    if not forcing_dir.exists():
        print(f"Directory {forcing_dir} does not exist!")
        sys.exit(1)
    
    print(f"Scanning {forcing_dir} for ERA5 surface files...")
    
    # Find all ERA5 surface files (both ZIP and NC)
    zip_pattern = "ERA5_surface_*.zip"
    nc_pattern = "ERA5_surface_*.nc"
    
    zip_files = list(forcing_dir.glob(zip_pattern))
    nc_files = list(forcing_dir.glob(nc_pattern))
    
    print(f"Found {len(zip_files)} ZIP files and {len(nc_files)} NC files")
    
    if not zip_files and not nc_files:
        print(f"No ERA5 surface files found in {forcing_dir}")
        print("Looking for files matching ERA5_surface_*.zip or ERA5_surface_*.nc")
        sys.exit(1)
    
    # Process ZIP files first
    files_to_process = zip_files
    
    if not files_to_process:
        print("No ZIP files to process. Checking existing NC files for integrity...")
        files_to_process = nc_files
    
    print(f"Processing {len(files_to_process)} files")
    
    fixed_count = 0
    error_count = 0
    
    for file_path in sorted(files_to_process):
        print(f"\nProcessing {file_path.name}...")
        
        try:
            # Try to extract if ZIP, or validate if already NC
            final_path = extract_if_zip(file_path)
            
            # Validate the result
            with nc4.Dataset(final_path, 'r') as ds:
                print(f"  ✅ Valid NetCDF: {len(ds.dimensions)} dimensions, {len(ds.variables)} variables")
                
                # Check for expected variables (more flexible check)
                expected_vars = ['msdwlwrf', 'msdwswrf', 'mtpr', 'sp']
                available_vars = list(ds.variables.keys())
                
                # Remove coordinate variables from the check
                data_vars = [var for var in available_vars if var not in ['longitude', 'latitude', 'time']]
                
                missing_vars = [var for var in expected_vars if var not in available_vars]
                
                if missing_vars:
                    print(f"  ⚠️  Missing expected variables: {missing_vars}")
                    print(f"  📋 Available data variables: {data_vars}")
                    print(f"  📋 All variables: {available_vars}")
                else:
                    print(f"  ✅ All expected variables present: {expected_vars}")
            
            fixed_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {e}")
            error_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Files processed: {len(files_to_process)}")
    print(f"Successfully fixed/validated: {fixed_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
