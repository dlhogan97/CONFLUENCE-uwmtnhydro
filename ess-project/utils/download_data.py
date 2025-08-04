import os
from ftplib import FTP
import pandas as pd
import datetime as dt
import py3dep

def download_spires_data(
    year: int,
    months: list[str],
    product: str = "NRT",         # or "HIST"
    tile: str = "h09v05",
    destination_base: str = "~/data/spires"
):
    """
    Download SPIRES data from the CU FTP server for a given year and list of months.
    
    Parameters:
        year (int): The year of data to download (e.g., 2023)
        months (list[str]): List of 2-digit month strings to download (e.g., ["04", "05"])
        product (str): Either "NRT" or "HIST"
        tile (str): Tile identifier, e.g., "h09v05"
        destination_base (str): Base path to store data (e.g., "~/data/spires")
    """
    
    # Validate input
    assert product in ["NRT", "HIST"], "Product must be 'NRT' or 'HIST'"
    for m in months:
        assert len(m) == 2 and m.isdigit(), f"Month {m} must be a 2-digit string"

    # FTP connection info
    host_name = "dtn.rc.colorado.edu"
    ftp_user = "anonymous"
    ftp_pwd = "pwd"
    base_path = "/shares/snow-today/spires"
    sub_path = f"SPIRES_{product}_V01/{tile}"
    remote_path = f"{base_path}/{sub_path}/{year}"

    # Create date range and filter
    start_date = dt.datetime(year, 1, 1)
    end_date = dt.datetime(year, 12, 31)
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    filtered_dates = all_dates[all_dates.strftime("%m").isin(months)]

    # Format filenames
    file_template = f"SPIRES_{product}_h09v05_MOD09GA061_{{date}}_V1.0.nc"
    file_names = [file_template.format(date=d.strftime("%Y%m%d")) for d in filtered_dates]

    # Create local output directory
    output_dir = os.path.expanduser(f"{destination_base}/{year}")
    os.makedirs(output_dir, exist_ok=True)

    # Connect to FTP and download files
    ftp = FTP(host_name)
    ftp.login(user=ftp_user, passwd=ftp_pwd)
    ftp.cwd(remote_path)

    for fname in file_names:
        local_path = os.path.join(output_dir, fname)
        if not os.path.exists(local_path):  # Avoid redownloading
            try:
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {fname}", f.write)
                print(f"Downloaded: {fname}")
            except Exception as e:
                print(f"❌ Could not download {fname}: {e}")
        else:
            print(f"✔️ Already exists: {fname}")

    ftp.quit()
    print(f"All downloads complete. Check {output_dir} for files.")
    return

from pynhd import NLDI
import xarray as xr
import rioxarray as rxr  # Ensure rioxarray is installed for geospatial operations
from dask import delayed, compute
from pathlib import Path

# ---- Clipping Function (wrapped in delayed) ----
@delayed
def clip_file_to_basin(input_path, output_path, gage_id):
    try:
        ds = xr.open_dataset(input_path)
        basin = NLDI().get_basins(gage_id).set_crs("EPSG:4326")

        if not ds.rio.crs:
            ds = ds.rio.write_crs(ds.crs.attrs['crs_wkt'])

        if ds.rio.crs != basin.crs:
            ds = ds.rio.reproject(basin.crs)

        ds_clipped = ds.rio.clip(basin.geometry, basin.crs, drop=True)
        ds_clipped.to_netcdf(output_path)
        # close the dataset to free resources
        ds.close()
        ds_clipped.close()
        return f"✔️ Clipped: {input_path.name}"
    except Exception as e:
        return f"❌ Failed: {input_path.name} — {e}"
    
def get_topo_data(gage_id, map_type="DEM", resolution=500, geo_crs=4326, save_path=None, override=False):
    """
    Get topographic data for a given gage ID.
    Parameters:
    gage_id (str): The gage ID for which to retrieve topographic data.
    map_type (str or list): The type of map to retrieve (e.g., "DEM",
                    "Slope Degrees"). Default is "DEM".
    resolution (int): The resolution of the map in meters. Default is 500.
    geo_crs (int): The EPSG code for the geographic coordinate reference system. Default
                    is 4326 (WGS 84).   
    save_path (str): Optional path to save the retrieved data. If None, data will not be saved.
    Returns:
    xarray.DataArray: The topographic data for the specified gage ID.
    """
    # establish if the file should saved/read as a geoTIFF or netcf
    if type(map_type) is str:
        map_type = [map_type]
        file_type = 'tif'
    elif type(map_type) is list:
        file_type = 'nc'
    else:
        raise ValueError("map_type must be a string or a list of strings.")
    
    if os.path.exists(Path(save_path) / f"{gage_id}_topo.{file_type}") or save_path == None:
        # ignore if file already exists
        print("File already exists.\nNew data will not be saved.\nUse override=True to override old file")
        if file_type == "nc":
            topo = xr.open_dataset(Path(save_path) / f"{gage_id}_topo.{file_type}")
        else:
            topo = rxr.open_rasterio(Path(save_path) / f"{gage_id}_topo.{file_type}")

    elif save_path or override:
        basin = NLDI().get_basins(gage_id).geometry.iloc[0]
        topo = py3dep.get_map(map_type, basin, resolution, geo_crs=4326,)
        # save to the given path
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if file_type == 'nc':
            topo.to_netcdf(save_path / f"{gage_id}_topo.{file_type}")
        else:
            # save as tif
            topo.rio.to_raster(f"{gage_id}_topo.{file_type}")
        print(f"✔️ Topo data saved to {save_path / f'{gage_id}_topo.{file_type}'}")
    return topo