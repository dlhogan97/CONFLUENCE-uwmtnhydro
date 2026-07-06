"""
Forcing Data Processor for CONFLUENCE
======================================
Utilities for processing ERA5 forcing data for SUMMA hydrological modeling.

Pipeline:
1. Merge raw ERA5 (surface + pressure level) into monthly files
2. Basin-average merged data over catchment using EASYMORE
3. Create SUMMA input files with optional adjustments

Usage:
    from utils.custom.forcing_processor import ForcingProcessor
    
    processor = ForcingProcessor(config_dict)
    processor.check_status()
    processor.merge_era5()
    processor.basin_average(confluence)  # requires CONFLUENCE instance
    processor.create_summa_input()
"""

import os
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from utils.custom.calc import empirical_lw_dilley_obrien  # Ensure this function is implemented for longwave recalculation

import numpy as np
import xarray as xr
import netCDF4 as nc4


def _matches_forcing_tag(filename: str, tag: Optional[str]) -> bool:
    """Return True if filename matches the requested forcing product tag.

    Rules:
    - tag is None/empty: always match.
    - tag == 'era5': match 'era5' but exclude names that also include 'metsim'.
      This avoids accidentally selecting hybrid/remapped MetSim files.
    - other tags: simple case-insensitive substring match.
    """
    if tag is None:
        return True

    tag_text = str(tag).strip().lower()
    if tag_text == '':
        return True

    name = filename.lower()
    if tag_text == 'era5':
        return ('era5' in name) and ('metsim' not in name)

    return tag_text in name


class ForcingProcessor:
    """
    Process ERA5 forcing data for SUMMA modeling.
    
    Parameters
    ----------
    config : dict
        CONFLUENCE configuration dictionary containing:
        - CONFLUENCE_DATA_DIR: Base data directory
        - DOMAIN_NAME: Name of the domain
        - EXPERIMENT_TIME_START: Start date (YYYY-MM-DD HH:MM format)
        - EXPERIMENT_TIME_END: End date (YYYY-MM-DD HH:MM format)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.domain_name = config['DOMAIN_NAME']
        
        # Set up paths
        self.project_dir = Path(config['CONFLUENCE_DATA_DIR']) / f"domain_{self.domain_name}"
        self.raw_path = self.project_dir / 'forcing' / 'raw_data'
        self.merged_path = self.project_dir / 'forcing' / 'merged_data'
        self.basin_avg_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.summa_input_path = self.project_dir / 'forcing' / 'SUMMA_input'
        
        # Parse dates
        self.start_year = int(config['EXPERIMENT_TIME_START'][:4])
        self.start_month = int(config['EXPERIMENT_TIME_START'][5:7])
        self.end_year = int(config['EXPERIMENT_TIME_END'][:4])
        self.end_month = int(config['EXPERIMENT_TIME_END'][5:7])
        
        # Track missing files
        self.missing_merged = []
        self.missing_basin_avg = []
        self.missing_summa = []
    
    def _get_required_months(self) -> List[str]:
        """Get list of YYYYMM strings for required period."""
        months = []
        for year in range(self.start_year, self.end_year + 1):
            for month in range(1, 13):
                if year == self.start_year and month < self.start_month:
                    continue
                if year == self.end_year and month > self.end_month:
                    continue
                months.append(f"{year}{month:02d}")
        return months

    def _expected_data_step(self) -> int:
        """Return expected forcing timestep in seconds for SUMMA input files."""
        return int(self.config.get('FORCING_TIME_STEP_SIZE', 3600))

    def validate_summa_input_data_step(self,
                                       fix_missing: bool = True,
                                       fix_mismatch: bool = True,
                                       verbose: bool = True) -> Dict[str, object]:
        """Validate scalar `data_step` across SUMMA input files and optionally repair.

        SUMMA requires every forcing file in the forcing list to expose the same
        timestep metadata. This check prevents mixed files (some with/without
        `data_step`) from causing runtime failures.
        """
        expected = self._expected_data_step()
        files = sorted(self.summa_input_path.glob('*.nc'))
        report = {
            'checked': len(files),
            'expected_data_step': expected,
            'missing': [],
            'mismatch': [],
            'fixed_missing': 0,
            'fixed_mismatch': 0,
        }

        for fp in files:
            try:
                with xr.open_dataset(fp) as ds_src:
                    current = ds_src.get('data_step', None)
                    has_data_step = current is not None
                    current_value = None
                    if has_data_step:
                        current_value = float(np.asarray(current).reshape(-1)[0])

                    needs_missing_fix = (not has_data_step) and fix_missing
                    needs_mismatch_fix = has_data_step and (current_value != float(expected)) and fix_mismatch

                    if needs_missing_fix or needs_mismatch_fix:
                        ds = ds_src.load()
                        ds['data_step'] = xr.DataArray(expected)
                        ds['data_step'].attrs.update({
                            'long_name': 'data step length in seconds',
                            'units': 's'
                        })
                        ds.to_netcdf(fp)
                        if needs_missing_fix:
                            report['fixed_missing'] += 1
                        if needs_mismatch_fix:
                            report['fixed_mismatch'] += 1
                    else:
                        if not has_data_step:
                            report['missing'].append(fp.name)
                        elif current_value != float(expected):
                            report['mismatch'].append({'file': fp.name, 'value': current_value})
            except Exception as exc:
                report['mismatch'].append({'file': fp.name, 'error': str(exc)})

        if verbose:
            print(
                f"data_step validation: checked={report['checked']}, expected={expected}, "
                f"fixed_missing={report['fixed_missing']}, fixed_mismatch={report['fixed_mismatch']}, "
                f"remaining_missing={len(report['missing'])}, remaining_mismatch={len(report['mismatch'])}"
            )

        return report
    
    def check_status(self, verbose: bool = True) -> Dict:
        """
        Check status of forcing data at each processing stage.
        
        Parameters
        ----------
        verbose : bool
            Print status summary
            
        Returns
        -------
        dict
            Status information including counts and missing files
        """
        required_months = self._get_required_months()
        
        # Check each stage
        raw_surface = sorted(self.raw_path.glob('ERA5_surface_*.nc'))
        raw_pressure = sorted(self.raw_path.glob('ERA5_pressureLevel137_*.nc'))
        merged_files = sorted(self.merged_path.glob('ERA5_merged_*.nc'))
        basin_avg_files = sorted(self.basin_avg_path.glob('*.nc'))
        summa_files = sorted(self.summa_input_path.glob('*.nc'))
        
        # Find missing files
        self.missing_merged = []
        self.missing_basin_avg = []
        self.missing_summa = []
        
        for ym in required_months:
            if not (self.merged_path / f"ERA5_merged_{ym}.nc").exists():
                self.missing_merged.append(ym)
            if not list(self.basin_avg_path.glob(f"*_{ym}.nc")):
                self.missing_basin_avg.append(ym)
            if not list(self.summa_input_path.glob(f"*_{ym}.nc")):
                self.missing_summa.append(ym)
        
        status = {
            'required_period': f"{self.start_year}-{self.start_month:02d} to {self.end_year}-{self.end_month:02d}",
            'required_months': len(required_months),
            'raw_surface_count': len(raw_surface),
            'raw_pressure_count': len(raw_pressure),
            'merged_count': len(merged_files),
            'basin_avg_count': len(basin_avg_files),
            'summa_input_count': len(summa_files),
            'missing_merged': self.missing_merged,
            'missing_basin_avg': self.missing_basin_avg,
            'missing_summa': self.missing_summa,
        }
        
        if verbose:
            print("=" * 60)
            print("FORCING DATA STATUS")
            print("=" * 60)
            print(f"\nRequired period: {status['required_period']}")
            print(f"Required months: {status['required_months']}")
            
            print(f"\n📁 Raw ERA5 surface files: {status['raw_surface_count']}")
            print(f"📁 Raw ERA5 pressure files: {status['raw_pressure_count']}")
            
            print(f"\n📁 Merged files: {status['merged_count']}")
            if merged_files:
                first = merged_files[0].stem.split('_')[-1]
                last = merged_files[-1].stem.split('_')[-1]
                print(f"   Range: {first} to {last}")
            
            print(f"\n📁 Basin-averaged files: {status['basin_avg_count']}")
            if basin_avg_files:
                first = basin_avg_files[0].stem.split('_')[-1]
                last = basin_avg_files[-1].stem.split('_')[-1]
                print(f"   Range: {first} to {last}")
            
            print(f"\n📁 SUMMA input files: {status['summa_input_count']}")
            if summa_files:
                first = summa_files[0].stem.split('_')[-1]
                last = summa_files[-1].stem.split('_')[-1]
                print(f"   Range: {first} to {last}")
            
            print(f"\n⚠️  Missing merged: {len(self.missing_merged)}")
            print(f"⚠️  Missing basin-averaged: {len(self.missing_basin_avg)}")
            print(f"⚠️  Missing SUMMA input: {len(self.missing_summa)}")
        
        return status
    
    def _merge_single_month(self, year: int, month: int) -> Tuple[bool, str]:
        """
        Merge surface and pressure level ERA5 files for a single month.
        
        Parameters
        ----------
        year : int
        month : int
        
        Returns
        -------
        tuple
            (success: bool, message: str)
        """
        ym = f"{year}{month:02d}"
        
        pres_file = self.raw_path / f"ERA5_pressureLevel137_{ym}.nc"
        surf_file = self.raw_path / f"ERA5_surface_{ym}.nc"
        dest_file = self.merged_path / f"ERA5_merged_{ym}.nc"
        
        if dest_file.exists():
            return True, f"Already exists: {ym}"
        
        if not pres_file.exists():
            return False, f"Missing pressure file: {pres_file.name}"
        if not surf_file.exists():
            return False, f"Missing surface file: {surf_file.name}"
        
        try:
            with nc4.Dataset(pres_file) as src_pres, nc4.Dataset(surf_file) as src_surf:
                # Get coordinates
                pres_lat = src_pres.variables['latitude'][:]
                pres_lon = src_pres.variables['longitude'][:]
                surf_lat = src_surf.variables['latitude'][:]
                surf_lon = src_surf.variables['longitude'][:]
                surf_time = src_surf.variables['valid_time'][:]
                
                # Fix longitude if needed (convert >180 to negative)
                pres_lon_fixed = pres_lon.copy()
                pres_lon_fixed[pres_lon_fixed > 180] = pres_lon_fixed[pres_lon_fixed > 180] - 360
                
                # Create output file
                with nc4.Dataset(dest_file, 'w', format='NETCDF4') as dst:
                    # Create dimensions
                    dst.createDimension('time', None)
                    dst.createDimension('latitude', len(surf_lat))
                    dst.createDimension('longitude', len(surf_lon))
                    
                    # Create coordinate variables - preserve original ERA5 time format
                    # ERA5 uses seconds since 1970-01-01 with proleptic_gregorian calendar
                    src_time_var = src_surf.variables['valid_time']
                    time_var = dst.createVariable('time', 'i8', ('time',))  # int64 like source
                    time_var.units = src_time_var.units if hasattr(src_time_var, 'units') else 'seconds since 1970-01-01'
                    time_var.calendar = src_time_var.calendar if hasattr(src_time_var, 'calendar') else 'proleptic_gregorian'
                    time_var.long_name = 'time'
                    time_var.standard_name = 'time'
                    time_var[:] = surf_time
                    
                    lat_var = dst.createVariable('latitude', 'f4', ('latitude',))
                    lat_var.units = 'degrees_north'
                    lat_var[:] = surf_lat
                    
                    lon_var = dst.createVariable('longitude', 'f4', ('longitude',))
                    lon_var.units = 'degrees_east'
                    lon_var[:] = surf_lon
                    
                    # Surface variables
                    surf_vars = {
                        'sp': ('airpres', 'Pa', 'air pressure'),
                        'avg_sdlwrf': ('LWRadAtm', 'W m-2', 'downward longwave radiation'),
                        'avg_sdswrf': ('SWRadAtm', 'W m-2', 'downward shortwave radiation'),
                        'avg_tprate': ('pptrate', 'kg m-2 s-1', 'precipitation rate')
                    }
                    
                    for src_name, (dst_name, units, long_name) in surf_vars.items():
                        if src_name in src_surf.variables:
                            src_var = src_surf.variables[src_name]
                            dst_var = dst.createVariable(dst_name, 'f4', 
                                                        ('time', 'latitude', 'longitude'),
                                                        fill_value=-999.0)
                            dst_var[:] = src_var[:]
                            dst_var.units = units
                            dst_var.long_name = long_name
                    
                    # Pressure level variables
                    pres_vars = {
                        't': ('airtemp', 'K', 'air temperature'),
                        'q': ('spechum', 'kg kg-1', 'specific humidity')
                    }
                    
                    for src_name, (dst_name, units, long_name) in pres_vars.items():
                        if src_name in src_pres.variables:
                            src_var = src_pres.variables[src_name]
                            dst_var = dst.createVariable(dst_name, 'f4',
                                                        ('time', 'latitude', 'longitude'),
                                                        fill_value=-999.0)
                            dst_var[:] = src_var[:]
                            dst_var.units = units
                            dst_var.long_name = long_name
                    
                    # Calculate wind speed from u,v components
                    if 'u' in src_pres.variables and 'v' in src_pres.variables:
                        u = src_pres.variables['u'][:]
                        v = src_pres.variables['v'][:]
                        windspd = np.sqrt(u**2 + v**2)
                        wind_var = dst.createVariable('windspd', 'f4',
                                                     ('time', 'latitude', 'longitude'),
                                                     fill_value=-999.0)
                        wind_var[:] = windspd
                        wind_var.units = 'm s-1'
                        wind_var.long_name = 'wind speed'
                    
                    # Global attributes
                    dst.History = f'Created {datetime.now().isoformat()}'
                    dst.Source = 'ERA5 surface and pressure level 137 merged'
                    dst.Conventions = 'CF-1.6'
            
            return True, f"Created: {ym}"
            
        except Exception as e:
            # Clean up partial file
            if dest_file.exists():
                dest_file.unlink()
            return False, f"Error {ym}: {str(e)}"
    
    def merge_era5(self, months: Optional[List[str]] = None, verbose: bool = True) -> int:
        """
        Merge raw ERA5 surface and pressure level files.
        
        Parameters
        ----------
        months : list, optional
            List of YYYYMM strings to process. If None, processes missing months.
        verbose : bool
            Print progress updates
            
        Returns
        -------
        int
            Number of files successfully created
        """
        # Ensure output directory exists
        self.merged_path.mkdir(parents=True, exist_ok=True)
        
        # Use missing months if not specified
        if months is None:
            if not self.missing_merged:
                self.check_status(verbose=False)
            months = self.missing_merged
        
        if not months:
            if verbose:
                print("✅ All merged files already exist")
            return 0
        
        if verbose:
            print(f"\n🔄 Merging {len(months)} ERA5 files...")
        
        success_count = 0
        errors = []
        
        for i, ym in enumerate(months):
            year, month = int(ym[:4]), int(ym[4:])
            success, msg = self._merge_single_month(year, month)
            
            if success:
                success_count += 1
            else:
                errors.append(msg)
            
            if verbose and ((i + 1) % 12 == 0 or i == len(months) - 1):
                print(f"   Progress: {i+1}/{len(months)} - Last: {msg}")
        
        if verbose:
            print(f"✅ ERA5 merging complete: {success_count}/{len(months)} successful")
            if errors:
                print(f"⚠️  {len(errors)} errors occurred")
                for err in errors[:5]:  # Show first 5 errors
                    print(f"   - {err}")
        
        return success_count
    
    def basin_average(self, confluence=None, verbose: bool = True) -> int:
        """
        Basin-average merged forcing data over the catchment.
        
        This uses CONFLUENCE's built-in EASYMORE-based resampler.
        
        Parameters
        ----------
        confluence : CONFLUENCE instance, optional
            If provided, uses CONFLUENCE's preprocessing. Otherwise runs standalone.
        verbose : bool
            Print progress updates
            
        Returns
        -------
        int
            Number of files processed
        """
        # Update missing list
        self.check_status(verbose=False)
        
        if not self.missing_basin_avg:
            if verbose:
                print("✅ All basin-averaged files already exist")
            return 0
        
        if verbose:
            print(f"\n🔄 Basin averaging {len(self.missing_basin_avg)} files...")
        
        if confluence is not None:
            # Use CONFLUENCE's built-in preprocessing
            confluence.managers['data'].run_model_agnostic_preprocessing()
            if verbose:
                print("✅ Basin averaging complete (via CONFLUENCE)")
            return len(self.missing_basin_avg)
        else:
            # Standalone mode - would need EASYMORE setup
            print("⚠️  Standalone basin averaging not implemented.")
            print("   Please provide a CONFLUENCE instance or run:")
            print("   confluence.managers['data'].run_model_agnostic_preprocessing()")
            return 0
    
    def create_summa_input(self, 
                          temp_adjustment: float = 0.0,
                          precip_multiplier: float = 1.0,
                          recalc_longwave: bool = False,
                          months: Optional[List[str]] = None,
                          force_rebuild: bool = False,
                          keep_backup: bool = True,
                          source_tag: Optional[str] = None,
                          verbose: bool = True) -> int:
        """
        Create SUMMA input files from basin-averaged data.
        
        Parameters
        ----------
        temp_adjustment : float
            Temperature offset in Kelvin (added to airtemp)
        precip_multiplier : float
            Precipitation scaling factor (1.0 = no change)
        recalc_longwave : bool
            Recalculate longwave radiation using Dilley & O'Brien (1998) empirical formula
            based on temperature, humidity, and pressure. Useful when adjusting temperature.
        months : list, optional
            List of YYYYMM strings to process. If None, processes missing months.
        force_rebuild : bool
            If True, rebuild SUMMA_input via a staging directory, then atomically
            replace the existing SUMMA_input directory.
        keep_backup : bool
            If force_rebuild=True, keep a timestamped backup of the previous
            SUMMA_input directory before swapping in the rebuilt directory.
        source_tag : str, optional
            If set, only use basin_averaged_data files whose filename contains
            this tag (for example 'metsim' or 'ERA5').
        verbose : bool
            Print progress updates
            
        Returns
        -------
        int
            Number of files created
        """
        # Resolve source selection tag from config unless explicitly provided.
        selected_tag = source_tag
        if selected_tag is None:
            raw_tag = self.config.get('FORCING_PRODUCT_TAG', None)
            if raw_tag is not None and str(raw_tag).strip() != '':
                selected_tag = str(raw_tag).strip()

        # Update missing list
        if months is None:
            if force_rebuild:
                months = self._get_required_months()
            else:
                self.check_status(verbose=False)
                months = self.missing_summa

        if not months:
            if verbose:
                print("✅ All SUMMA input files already exist")
            return 0

        # Determine output target. For forced rebuild we write to a staging dir
        # and atomically swap it in only after successful completion.
        target_output_path = self.summa_input_path
        backup_dir = None
        if force_rebuild:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            target_output_path = self.summa_input_path.parent / f"{self.summa_input_path.name}_staging_{timestamp}"
            if target_output_path.exists():
                shutil.rmtree(target_output_path)
        target_output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"\n🔄 Creating {len(months)} SUMMA input files...")
            if force_rebuild:
                print("   Mode: force rebuild (atomic swap)")
            if selected_tag:
                print(f"   Source tag filter: '{selected_tag}'")
            if temp_adjustment != 0:
                print(f"   Temperature adjustment: {temp_adjustment:+.2f} K")
            if precip_multiplier != 1.0:
                print(f"   Precipitation multiplier: {precip_multiplier:.2f}")
            if recalc_longwave:
                print(f"   Recalculating longwave radiation (Dilley & O'Brien 1998)")

        
        success_count = 0
        
        for i, ym in enumerate(months):
            # Find corresponding basin-averaged file
            ba_files = sorted(self.basin_avg_path.glob(f"*_{ym}.nc"))
            ba_files = [fp for fp in ba_files if _matches_forcing_tag(fp.name, selected_tag)]
            
            if not ba_files:
                if verbose:
                    suffix = f" with tag '{selected_tag}'" if selected_tag else ""
                    print(f"   ⚠️  No basin-averaged file for {ym}{suffix}")
                continue

            if len(ba_files) > 1 and verbose:
                print(f"   ⚠️  Multiple basin-averaged candidates for {ym}; using {ba_files[0].name}")
            
            src_file = ba_files[0]
            dst_file = target_output_path / src_file.name
            
            try:
                # Load, adjust if needed, and save
                with xr.open_dataset(src_file) as ds_src:
                    ds = ds_src.load()

                if temp_adjustment != 0 and 'airtemp' in ds:
                    ds['airtemp'] = ds['airtemp'] + temp_adjustment

                if precip_multiplier != 1.0 and 'pptrate' in ds:
                    ds['pptrate'] = ds['pptrate'] * precip_multiplier

                # Recalculate longwave radiation if requested
                if recalc_longwave and all(v in ds for v in ['airtemp', 'spechum', 'airpres']):
                    ds['LWRadAtm'] = empirical_lw_dilley_obrien(
                        ds['airtemp'], ds['airpres'], ds['spechum']
                    )

                # Enforce SUMMA-required scalar timestep metadata consistently.
                ds['data_step'] = xr.DataArray(self._expected_data_step())
                ds['data_step'].attrs.update({
                    'long_name': 'data step length in seconds',
                    'units': 's'
                })

                ds.to_netcdf(dst_file)
                success_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error processing {ym}: {str(e)}")
                continue
            
            if verbose and ((i + 1) % 12 == 0 or i == len(months) - 1):
                print(f"   Progress: {i+1}/{len(months)}")

        # For force rebuild, only swap if we successfully produced all months.
        if force_rebuild:
            if success_count != len(months):
                shutil.rmtree(target_output_path, ignore_errors=True)
                raise RuntimeError(
                    f"Force rebuild requested but only created {success_count}/{len(months)} files. "
                    "SUMMA_input was not modified."
                )

            if self.summa_input_path.exists():
                if keep_backup:
                    backup_dir = self.summa_input_path.parent / (
                        f"{self.summa_input_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    shutil.move(str(self.summa_input_path), str(backup_dir))
                else:
                    shutil.rmtree(self.summa_input_path)

            shutil.move(str(target_output_path), str(self.summa_input_path))
        
        if verbose:
            print(f"✅ SUMMA input creation complete: {success_count}/{len(months)}")
            if force_rebuild and backup_dir is not None:
                print(f"   Backup saved: {backup_dir}")
        
        return success_count
    
    def process_all(self, 
                   confluence=None,
                   temp_adjustment: float = 0.0,
                   precip_multiplier: float = 1.0,
                   verbose: bool = True) -> Dict:
        """
        Run the complete forcing processing pipeline.
        
        Parameters
        ----------
        confluence : CONFLUENCE instance, optional
            Required for basin averaging step
        temp_adjustment : float
            Temperature offset in Kelvin for SUMMA input
        precip_multiplier : float
            Precipitation scaling factor for SUMMA input
        verbose : bool
            Print progress updates
            
        Returns
        -------
        dict
            Summary of processing results
        """
        results = {
            'merged': 0,
            'basin_averaged': 0,
            'summa_input': 0
        }
        
        # Step 1: Check status
        status = self.check_status(verbose=verbose)
        
        # Step 2: Merge ERA5
        results['merged'] = self.merge_era5(verbose=verbose)
        
        # Step 3: Basin average
        results['basin_averaged'] = self.basin_average(confluence=confluence, verbose=verbose)
        
        # Step 4: Create SUMMA input
        results['summa_input'] = self.create_summa_input(
            temp_adjustment=temp_adjustment,
            precip_multiplier=precip_multiplier,
            verbose=verbose
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE")
            print("=" * 60)
            print(f"  Merged files created: {results['merged']}")
            print(f"  Basin-averaged files: {results['basin_averaged']}")
            print(f"  SUMMA input files: {results['summa_input']}")
        
        return results


def check_forcing_status(config: Dict, verbose: bool = True) -> Dict:
    """
    Quick function to check forcing data status.
    
    Parameters
    ----------
    config : dict
        CONFLUENCE configuration dictionary
    verbose : bool
        Print status summary
        
    Returns
    -------
    dict
        Status information
    """
    processor = ForcingProcessor(config)
    return processor.check_status(verbose=verbose)


def process_forcing(config: Dict, 
                   confluence=None,
                   temp_adjustment: float = 0.0,
                   precip_multiplier: float = 1.0) -> Dict:
    """
    Process forcing data through complete pipeline.
    
    Parameters
    ----------
    config : dict
        CONFLUENCE configuration dictionary
    confluence : CONFLUENCE instance, optional
        Required for basin averaging
    temp_adjustment : float
        Temperature offset in Kelvin
    precip_multiplier : float
        Precipitation scaling factor
        
    Returns
    -------
    dict
        Processing results summary
    """
    processor = ForcingProcessor(config)
    return processor.process_all(
        confluence=confluence,
        temp_adjustment=temp_adjustment,
        precip_multiplier=precip_multiplier
    )
