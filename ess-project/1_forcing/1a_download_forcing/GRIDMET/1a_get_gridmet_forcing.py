import os
import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

def download_gridmet_forcing(domain_name, variable, root_path):
    """
    Download the gridmet forcing data for the domain_name and variable
    
    Parameters
    ----------
    domain_name: str
        The domain_name of the basin
    variable: str
        The variable to download. Must be in the following list: pr (ppt), tmmx (tmax), tmmn (tmin), sph (spechum), srad (sw_down), vs (wind)"""
    
    # set the output directory and check if file exists
    output_fn = root_path / f"domain_{domain_name}/forcing/raw_data/gridmet_{variable}.nc"

    if os.path.exists(output_fn):
        print(f"File already exists. Find it at {output_fn}")
        return
    else:
        print(f"Downloading file: {output_fn}")
        # build URL from the GRIDMET server
        server = "http://thredds.northwestknowledge.net:8080"
        dataset = f"/thredds/dodsC/agg_met_{variable}_1979_CurrentYear_CONUS.nc"
        url = server + dataset

        # gather the url data
        ds = xr.open_dataset(url) 

        # rename day to time and lat to y and lon to x
        ds = ds.rename({'day': 'time', 'lat': 'y', 'lon': 'x'})

        # get the variable names 
        variable_names = list(ds.keys())
                
        # Open the basin shapefile
        basin = gpd.read_file( root_path / f'domain_{domain_name}/shapefiles/catchment/{domain_name}.shp')
        
        # write the crs to the variable name
        ds = ds[variable_names[0]].rio.write_crs(basin.crs)

        # clip to the basin
        ds = ds.rio.clip_box(*basin.total_bounds)

        # save the dataset 
        ds.to_netcdf(output_fn)

        # close the dataset
        ds.close()
        print(f"File saved to {output_fn}")
        return

# using a mixing of 0.6 * tmin + 0.4 * tmax, make a mean temperature dataset
def calculate_average_gridmet_temperature(domain_name, root_path):
    """
    Calculate the average temperature from the max and min temperature
    
    Parameters
    ----------
    tmmx: xarray.DataArray
        The maximum temperature data
    tmnx: xarray.DataArray
        The minimum temperature data
    
    Returns
    -------
    xarray.DataArray
        The average temperature data
    """
    if os.path.exists(root_path / f"domain_{domain_name}/forcing/raw_data/gridmet_tavg.nc"):
        print(f"File already exists. Find it at {root_path / f'domain_{domain_name}/forcing/raw_data/gridmet_tavg.nc'}")
        return
    else:
        print(f"Calculating average temperature and saving to {root_path / f'domain_{domain_name}/forcing/raw_data/gridmet_tavg.nc'}")
        tmmn_ds = xr.open_dataarray(root_path / f"domain_{domain_name}/forcing/raw_data/gridmet_tmmn.nc")
        tmmx_ds = xr.open_dataarray(root_path / f"domain_{domain_name}/forcing/raw_data/gridmet_tmmx.nc")
        # calculate the average temperature
        tavg_ds = 0.6 * tmmn_ds['daily_minimum_temperature'] + 0.4 * tmmx_ds['daily_maximum_temperature']
        # name the new DataArray
        tavg_ds.name = "tavg"
        # add the attributes like units
        tavg_ds.attrs = {"units": "Kelvin",
                        "method": "Calculated from 0.6 * tmin + 0.4 * tmax"}
        # save the dataset
        tavg_ds.to_netcdf(root_path / f"domain_{domain_name}/forcing/raw_data/gridmet_tavg.nc")
        print(f"File saved to {root_path / f'domain_{domain_name}/forcing/raw_data/gridmet_tavg.nc'}")
        return

def separate_to_monthly_files(domain_name, root_path):
    """
    Separate the gridmet data into monthly files
    
    Parameters
    ----------
    domain_name: str
        The domain_name of the basin
    root_path: Path
        The root path to the domain folder
    """
    # list of variables to separate
    variables = ['pr', 'tmmx', 'tmmn', 'sph', 'srad', 'vs', 'tavg']
    
    for variable in variables:
        # open the dataset
        ds = xr.open_dataarray(root_path / f"domain_{domain_name}/forcing/raw_data/gridmet_{variable}.nc")
        # convert time to datetime
        ds['time'] = pd.to_datetime(ds['time'].values)
        # group by month and year and save each month as a separate file
        for (year, month), group in ds.groupby('time.year', 'time.month'):
            output_fn = root_path / f"domain_{domain_name}/forcing/1_raw_data/gridmet_{variable}_{year}{month:02d}.nc"
            group.to_netcdf(output_fn)
            print(f"File saved to {output_fn}")
        # close the dataset
        ds.close()
    return

if __name__ == "__main__":
    gridmet_vars = {
    "tmax": "tmmx",
    "tmin": "tmmn",
    "spechum": "sph",
    "sw_down": "srad",
    "wind": "vs",
    "ppt": "pr"
    }

    # Store the name of the 'active' file in a variable
    domain_name = input("Enter the domain name (e.g., Tuolumne_River): ")  # basin name
    root_path = Path(input("Enter the root path (e.g., /storage/dlhogan/summa_modeling_data/domain_TuolumneRiver): "))
    separate_files = input("Separate to monthly files? (y/n): ").lower() == 'y'
    
    for var in gridmet_vars.values():
        download_gridmet_forcing(domain_name, var, root_path)
    # Calculate average temperature
    calculate_average_gridmet_temperature(domain_name, root_path)
    
    if separate_files:
        separate_to_monthly_files(domain_name, root_path)