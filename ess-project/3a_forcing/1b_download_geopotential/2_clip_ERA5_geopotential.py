# Note: This script was adapted from the original script produced by the CWARHM project team. Original repository can be found at: https://github.com/CH-Earth/CWARHM
# Script to download ERA5 geopotential data.
# Geopotential data can be converted into elevation, which is needed for temperature lapsing.

# Requires use of the Copernicus Data Store API
# CDS registration: https://cds.climate.copernicus.eu/user/register?destination=%2F%23!%2Fhome
# CDS api setup: https://cds.climate.copernicus.eu/api-how-to

# modules
import cdsapi    # copernicus connection
import os        # to check if file already exists
import sys       # to handle command line arguments (sys.argv[0] = name of this file, sys.argv[1] = arg1, ...)
import math
from pathlib import Path
from datetime import datetime

''' 
Downloads ERA5 geopotential invariant data.
Usage: python 2_clip_ERA5_geopotential.py <coordinates> <path/to/save/data> 
'''

# Get the spatial coordinates as the first command line argument
bounding_box = sys.argv[1] # string
bounding_box = bounding_box.split('/') # split string
bounding_box = [float(value) for value in bounding_box] # string to array

# Get the path as the second command line argument
geoPath = Path(sys.argv[2]) # string to Path()

# Make the folder if it doesn't exist
geoPath.mkdir(parents=True, exist_ok=True)

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


# --- Specify date to download

# Geopotential is part of the ERA5 "invariant" data, which are constant through time.
# Therefore, specify an arbitrary date to download
date = '2019-01-01'


# --- Download the data

# Specify a filename
file = geoPath / 'ERA5_geopotential.nc'

# track progress
print('Trying to download geopotential data into ' + str(file))

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
            c.retrieve('reanalysis-era5-complete', {    # do not change this!
                    'stream': 'oper',
                    'levtype': 'sf',
                    'param': '26/228007/27/28/29/30/43/74/129/160/161/162/163/172',
                    'date': date,
                    'time': '00',#/to/23/by/1',
                    'area': coordinates,
                    'grid': '0.25/0.25', # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude).
                    'format'  : 'netcdf',
                }, file)
            
            # track progress
            print('Successfully downloaded ' + str(file))

        except Exception as e:
            print('Error downloading ' + str(file) + ' on try ' + str(retries_cur))
            print(str(e))
            retries_cur += 1
            continue
        else:
            break 