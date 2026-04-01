# utils/data/openet_utils.py
"""
OpenET Data Acquisition Utilities

This module provides utilities for downloading evapotranspiration (ET) data
from the OpenET API. It supports polygon-based timeseries queries using
shapefiles to define the area of interest.

The OpenET API provides access to multiple ET models including:
- DisALEXI
- EEMETRIC 
- geeSEBAL
- PT-JPL
- SIMS
- SSEBop
- Ensemble (ensemble mean of all models)

For more information about OpenET: https://openetdata.org/

API Documentation: https://open-et.github.io/docs/
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import requests
import geopandas as gpd
import pandas as pd
import numpy as np

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


class OpenETClient:
    """
    Client for interacting with the OpenET API to retrieve ET timeseries data.
    
    This client handles authentication, geometry processing, and API requests
    for downloading evapotranspiration data over specified polygon geometries.
    
    Attributes:
        api_key (str): OpenET API key for authentication
        base_url (str): Base URL for the OpenET API
        logger (logging.Logger): Logger instance
        
    Example:
        >>> client = OpenETClient()
        >>> data = client.get_timeseries_from_shapefile(
        ...     shapefile_path='/path/to/shapefile.shp',
        ...     start_date='2018-01-01',
        ...     end_date='2023-12-31',
        ...     variable='et',
        ...     model='ensemble'
        ... )
    """
    
    # API endpoints
    BASE_URL = "https://openet-api.org"
    TIMESERIES_POLYGON_ENDPOINT = "/raster/timeseries/polygon"
    TIMESERIES_MULTIPOLYGON_ENDPOINT = "/raster/timeseries/multipolygon"
    
    # Available models
    AVAILABLE_MODELS = [
        'disalexi', 'eemetric', 'geesebal', 'ptjpl', 
        'sims', 'ssebop', 'ensemble'
    ]
    
    # Available variables
    AVAILABLE_VARIABLES = [
        'et',           # Evapotranspiration (mm)
        'eto',          # Reference ET (mm)
        'etof',         # Fraction of reference ET
        'ndvi',         # Normalized Difference Vegetation Index
        'pr',           # Precipitation (mm)
        'etr',          # Reference ET - ASCE Tall Reference (mm)
        'count'         # Pixel count
    ]
    
    # Default reducer for aggregating pixels within polygon
    DEFAULT_REDUCER = 'mean'
    AVAILABLE_REDUCERS = ['mean', 'median', 'min', 'max', 'sum', 'count']
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        env_file: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the OpenET client.
        
        The API key can be provided directly, loaded from an environment variable
        (OPENET_API_KEY), or loaded from a .env file.
        
        Args:
            api_key: OpenET API key. If None, will try to load from environment.
            env_file: Path to .env file containing OPENET_API_KEY.
            logger: Logger instance. If None, creates a default logger.
            
        Raises:
            ValueError: If no API key is found.
        """
        self.logger = logger or self._setup_logger()
        
        # Load .env file if provided
        if env_file and HAS_DOTENV:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path)
                self.logger.info(f"Loaded environment from {env_file}")
            else:
                self.logger.warning(f"Environment file not found: {env_file}")
        elif env_file and not HAS_DOTENV:
            self.logger.warning("python-dotenv not installed. Cannot load .env file.")
        
        # Get API key
        self.api_key = api_key or os.getenv('OPENET_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenET API key not found. Please provide it directly, "
                "set OPENET_API_KEY environment variable, or specify an env_file."
            )
        
        self.base_url = self.BASE_URL
        self.logger.info("OpenET client initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Create a default logger if none provided."""
        logger = logging.getLogger('OpenETClient')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def _shapefile_to_geojson(
        self,
        shapefile_path: Union[str, Path],
        dissolve: bool = True
    ) -> Dict[str, Any]:
        """
        Convert a shapefile to GeoJSON geometry.
        
        Args:
            shapefile_path: Path to the shapefile.
            dissolve: If True, dissolve all features into a single geometry.
            
        Returns:
            GeoJSON geometry dict.
        """
        shapefile_path = Path(shapefile_path)
        if not shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        
        # Read shapefile
        gdf = gpd.read_file(shapefile_path)
        self.logger.info(f"Loaded shapefile with {len(gdf)} features")
        
        # Ensure WGS84 CRS
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
            self.logger.info("Reprojected shapefile to EPSG:4326")
        
        # Dissolve if requested and multiple features exist
        if dissolve and len(gdf) > 1:
            gdf = gdf.dissolve()
            self.logger.info("Dissolved features into single geometry")
        
        # Convert to GeoJSON
        geometry = gdf.geometry.iloc[0]
        geojson_geometry = json.loads(gpd.GeoSeries([geometry]).to_json())['features'][0]['geometry']
        
        return geojson_geometry
    
    def _geodataframe_to_geojson_list(
        self,
        gdf: gpd.GeoDataFrame
    ) -> List[Dict[str, Any]]:
        """
        Convert a GeoDataFrame to a list of GeoJSON geometries.
        
        Args:
            gdf: GeoDataFrame with polygon geometries.
            
        Returns:
            List of GeoJSON geometry dicts.
        """
        geometries = []
        for idx, row in gdf.iterrows():
            geojson = json.loads(gpd.GeoSeries([row.geometry]).to_json())
            geometries.append(geojson['features'][0]['geometry'])
        return geometries
    
    def get_timeseries(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        variable: str = 'et',
        model: str = 'ensemble',
        reducer: str = 'mean',
        units: str = 'mm',
        interval: str = 'monthly',
        ref_et_source: str = 'gridmet',
        feature_id: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 5.0
    ) -> pd.DataFrame:
        """
        Get ET timeseries data for a single polygon geometry.
        
        Args:
            geometry: GeoJSON geometry dict (Polygon or MultiPolygon).
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            variable: Variable to retrieve (et, eto, etof, ndvi, pr, etr, count).
            model: ET model to use (disalexi, eemetric, geesebal, ptjpl, 
                   sims, ssebop, ensemble).
            reducer: Spatial aggregation method (mean, median, min, max, sum, count).
            units: Units for ET values ('mm' or 'in').
            interval: Temporal interval ('daily', 'monthly', or 'annual').
            ref_et_source: Reference ET source ('gridmet', 'cimis', 'nldas').
            feature_id: Optional identifier for the feature.
            retry_count: Number of retries on failure.
            retry_delay: Delay between retries in seconds.
            
        Returns:
            DataFrame with timeseries data.
            
        Raises:
            ValueError: If invalid parameters are provided.
            requests.RequestException: If API request fails.
        """
        # Validate parameters
        if model.lower() not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model: {model}. Available: {self.AVAILABLE_MODELS}"
            )
        if variable.lower() not in self.AVAILABLE_VARIABLES:
            raise ValueError(
                f"Invalid variable: {variable}. Available: {self.AVAILABLE_VARIABLES}"
            )
        if reducer.lower() not in self.AVAILABLE_REDUCERS:
            raise ValueError(
                f"Invalid reducer: {reducer}. Available: {self.AVAILABLE_REDUCERS}"
            )
        
        # Handle geometry format for OpenET API
        # The API expects coordinates as a flat list of numbers: [lon1, lat1, lon2, lat2, ...]
        geom_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])
        
        if geom_type == 'Polygon':
            # For Polygon, use the outer ring (first element of coordinates)
            coord_pairs = coordinates[0]
        elif geom_type == 'MultiPolygon':
            # For MultiPolygon, merge all polygons into a single list
            self.logger.warning("MultiPolygon detected - merging outer rings")
            coord_pairs = []
            for polygon in coordinates:
                coord_pairs.extend(polygon[0])
        else:
            coord_pairs = coordinates
        
        # Flatten coordinate pairs to [lon1, lat1, lon2, lat2, ...]
        geom_coords = []
        for coord in coord_pairs:
            geom_coords.extend(coord)  # Add lon, then lat
        
        self.logger.info(f"Geometry has {len(coord_pairs)} coordinate pairs ({len(geom_coords)} values)")
        
        # Build request payload
        # OpenET API expects:
        # - geometry: list of [lon, lat] coordinate pairs forming the polygon
        # - date_range: array [start_date, end_date]
        payload = {
            'geometry': geom_coords,
            'date_range': [start_date, end_date],
            'variable': variable.lower(),
            'model': model.lower(),
            'reducer': reducer.lower(),
            'units': units.lower(),
            'interval': interval.lower(),
            'reference_et': ref_et_source.lower(),
            'file_format': 'json'
        }
        
        # Log payload for debugging (without full geometry)
        self.logger.debug(f"Request payload (geometry has {len(geom_coords)} points): "
                         f"date_range={payload['date_range']}, variable={payload['variable']}, "
                         f"model={payload['model']}, interval={payload['interval']}")
        
        if feature_id:
            payload['feature_id'] = feature_id
        
        url = f"{self.base_url}{self.TIMESERIES_POLYGON_ENDPOINT}"
        
        # Make request with retries
        for attempt in range(retry_count):
            try:
                self.logger.info(f"Making API request (attempt {attempt + 1}/{retry_count})")
                response = requests.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=300  # 5 minute timeout for large requests
                )
                
                if response.status_code == 200:
                    data = response.json()
                    df = self._parse_timeseries_response(data, variable, model)
                    self.logger.info(f"Successfully retrieved {len(df)} records")
                    return df
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    self.logger.warning("Rate limited. Waiting before retry...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    if attempt == retry_count - 1:
                        raise requests.RequestException(error_msg)
                    time.sleep(retry_delay)
                    
            except requests.Timeout:
                self.logger.warning(f"Request timed out (attempt {attempt + 1})")
                if attempt == retry_count - 1:
                    raise
                time.sleep(retry_delay)
            except requests.RequestException as e:
                self.logger.error(f"Request error: {e}")
                if attempt == retry_count - 1:
                    raise
                time.sleep(retry_delay)
        
        raise requests.RequestException("Failed after all retries")
    
    def _parse_timeseries_response(
        self,
        data: Union[List, Dict],
        variable: str,
        model: str
    ) -> pd.DataFrame:
        """
        Parse API response into a DataFrame.
        
        Args:
            data: API response data.
            variable: Variable name for column naming.
            model: Model name for column naming.
            
        Returns:
            Parsed DataFrame with datetime index.
        """
        if isinstance(data, list):
            # Response is a list of records
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Response might be nested
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError(f"Unexpected response format: {type(data)}")
        
        # Process date column
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
            df = df.drop(columns=['time'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        if 'date' in df.columns:
            df = df.set_index('date').sort_index()
        
        # Rename value column if present
        if 'value' in df.columns:
            df = df.rename(columns={'value': f'{variable}_{model}'})
        
        return df
    
    def get_timeseries_from_shapefile(
        self,
        shapefile_path: Union[str, Path],
        start_date: str,
        end_date: str,
        variable: str = 'et',
        model: str = 'ensemble',
        reducer: str = 'mean',
        units: str = 'mm',
        interval: str = 'monthly',
        ref_et_source: str = 'gridmet',
        dissolve: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Get ET timeseries for a polygon defined by a shapefile.
        
        Args:
            shapefile_path: Path to the shapefile defining the area of interest.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            variable: Variable to retrieve (et, eto, etof, ndvi, pr, etr, count).
            model: ET model to use.
            reducer: Spatial aggregation method.
            units: Units for ET values ('mm' or 'in').
            interval: Temporal interval ('daily', 'monthly', or 'annual').
            ref_et_source: Reference ET source.
            dissolve: If True, dissolve all shapefile features into one.
            output_path: Optional path to save the data.
            output_format: Output format ('csv' or 'netcdf').
            
        Returns:
            DataFrame with timeseries data.
        """
        self.logger.info(f"Processing shapefile: {shapefile_path}")
        
        # Convert shapefile to GeoJSON
        geometry = self._shapefile_to_geojson(shapefile_path, dissolve=dissolve)
        
        # Get timeseries
        df = self.get_timeseries(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            variable=variable,
            model=model,
            reducer=reducer,
            units=units,
            interval=interval,
            ref_et_source=ref_et_source
        )
        
        # Save if output path provided
        if output_path:
            self.save_timeseries(df, output_path, output_format)
        
        return df
    
    def get_timeseries_per_polygon(
        self,
        shapefile_path: Union[str, Path],
        start_date: str,
        end_date: str,
        variable: str = 'et',
        model: str = 'ensemble',
        reducer: str = 'mean',
        units: str = 'mm',
        interval: str = 'monthly',
        ref_et_source: str = 'gridmet',
        id_column: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        output_format: str = 'csv',
        combine: bool = True,
        delay_between_requests: float = 1.0
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get ET timeseries for each polygon in a shapefile separately.
        
        This method processes each polygon/feature in the shapefile individually,
        making separate API requests for each. Useful when you have multiple
        sub-catchments, HRUs, or GRUs and want data for each.
        
        Args:
            shapefile_path: Path to the shapefile with multiple polygons.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            variable: Variable to retrieve.
            model: ET model to use.
            reducer: Spatial aggregation method.
            units: Units for ET values.
            interval: Temporal interval.
            ref_et_source: Reference ET source.
            id_column: Column name to use as polygon ID. If None, auto-detects
                       (looks for HRU_ID, GRU_ID, ID, or uses index).
            output_dir: Directory to save individual files per polygon.
            output_format: Output format ('csv' or 'netcdf').
            combine: If True, returns single DataFrame with polygon_id column.
                     If False, returns dict of DataFrames keyed by polygon ID.
            delay_between_requests: Seconds to wait between API calls.
            
        Returns:
            If combine=True: DataFrame with all polygons and 'polygon_id' column.
            If combine=False: Dict mapping polygon IDs to individual DataFrames.
        
        Example:
            >>> client = OpenETClient(env_file='.env')
            >>> # Get data for each HRU separately
            >>> df = client.get_timeseries_per_polygon(
            ...     shapefile_path='catchments.shp',
            ...     start_date='2019-01-01',
            ...     end_date='2022-12-31',
            ...     id_column='HRU_ID',
            ...     output_dir='./et_by_hru/'
            ... )
        """
        shapefile_path = Path(shapefile_path)
        if not shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        
        # Read shapefile
        gdf = gpd.read_file(shapefile_path)
        self.logger.info(f"Loaded shapefile with {len(gdf)} features/polygons")
        
        # Ensure WGS84 CRS
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
            self.logger.info("Reprojected shapefile to EPSG:4326")
        
        # Determine ID column
        if id_column and id_column in gdf.columns:
            ids = gdf[id_column].tolist()
        elif 'HRU_ID' in gdf.columns:
            ids = gdf['HRU_ID'].tolist()
            id_column = 'HRU_ID'
        elif 'GRU_ID' in gdf.columns:
            ids = gdf['GRU_ID'].tolist()
            id_column = 'GRU_ID'
        elif 'ID' in gdf.columns:
            ids = gdf['ID'].tolist()
            id_column = 'ID'
        else:
            ids = list(range(len(gdf)))
            id_column = 'index'
        
        self.logger.info(f"Using '{id_column}' as polygon identifier: {ids}")
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each polygon
        results = {}
        for idx, (_, row) in enumerate(gdf.iterrows()):
            polygon_id = str(ids[idx])
            self.logger.info(f"Processing polygon {idx + 1}/{len(gdf)}: {polygon_id}")
            
            # Convert geometry to GeoJSON
            geojson = json.loads(gpd.GeoSeries([row.geometry]).to_json())
            geometry = geojson['features'][0]['geometry']
            
            try:
                df = self.get_timeseries(
                    geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    variable=variable,
                    model=model,
                    reducer=reducer,
                    units=units,
                    interval=interval,
                    ref_et_source=ref_et_source,
                    feature_id=polygon_id
                )
                
                # Add polygon ID to dataframe
                df['polygon_id'] = polygon_id
                results[polygon_id] = df
                
                # Save individual file if output_dir specified
                if output_dir:
                    filename = f"openet_{variable}_{model}_{polygon_id}_{interval}.{output_format}"
                    self.save_timeseries(df.drop(columns=['polygon_id']), 
                                        output_dir / filename, output_format)
                
                self.logger.info(f"Retrieved {len(df)} records for polygon {polygon_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to get data for polygon {polygon_id}: {e}")
                continue
            
            # Delay between requests to avoid rate limiting
            if idx < len(gdf) - 1:
                time.sleep(delay_between_requests)
        
        if not results:
            self.logger.warning("No data retrieved for any polygon")
            return pd.DataFrame() if combine else {}
        
        if combine:
            # Combine all DataFrames into one
            combined_df = pd.concat(results.values(), ignore_index=False)
            combined_df = combined_df.reset_index().rename(columns={'index': 'date'})
            
            # Save combined file if output_dir specified
            if output_dir:
                combined_path = output_dir / f"openet_{variable}_{model}_all_polygons_{interval}.{output_format}"
                self.save_timeseries(combined_df.set_index('date'), combined_path, output_format)
                self.logger.info(f"Saved combined data to {combined_path}")
            
            return combined_df
        else:
            return results
    
    def get_multiple_models(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        models: Optional[List[str]] = None,
        variable: str = 'et',
        reducer: str = 'mean',
        units: str = 'mm',
        interval: str = 'monthly',
        ref_et_source: str = 'gridmet'
    ) -> pd.DataFrame:
        """
        Get a single timeseries table for multiple OpenET models.

        Notes:
            - Each model is requested independently, then joined on datetime index.
            - To prevent column collisions (e.g., repeated 'et'), all non-date/value
              columns are renamed with a model prefix before joining.
              Example: 'et' from model 'ptjpl' becomes 'ptjpl_et'.

        Args:
            geometry: GeoJSON geometry dict.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            models: List of models to query. If None, queries all available.
            variable: Variable to retrieve.
            reducer: Spatial aggregation method.
            units: Units for ET values.
            interval: Temporal interval.
            ref_et_source: Reference ET source.

        Returns:
            DataFrame indexed by date with one column per model/variable.
        """
        models = models or self.AVAILABLE_MODELS

        combined_df = None
        for model in models:
            try:
                self.logger.info(f"Querying model: {model}")
                df = self.get_timeseries(
                    geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    variable=variable,
                    model=model,
                    reducer=reducer,
                    units=units,
                    interval=interval,
                    ref_et_source=ref_et_source
                )

                # Defensive rename in case API returns raw column names like 'et'
                rename_map = {}
                for col in df.columns:
                    if col in {'date', 'time'}:
                        continue
                    if not col.startswith(f"{model}_"):
                        rename_map[col] = f"{model}_{col}"
                if rename_map:
                    df = df.rename(columns=rename_map)

                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = combined_df.join(df, how='outer')

                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Failed to get data for model {model}: {e}")
                continue

        return combined_df if combined_df is not None else pd.DataFrame()

    def get_multiple_models_from_shapefile(
        self,
        shapefile_path: Union[str, Path],
        start_date: str,
        end_date: str,
        models: Optional[List[str]] = None,
        variable: str = 'et',
        reducer: str = 'mean',
        units: str = 'mm',
        interval: str = 'monthly',
        ref_et_source: str = 'gridmet',
        dissolve: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Get ET timeseries from multiple models for a shapefile-defined area.

        This is a wrapper around `get_multiple_models` that:
          1) builds geometry from the shapefile,
          2) queries each model,
          3) merges model outputs into one date-indexed DataFrame,
          4) optionally writes output to disk.

        Args:
            shapefile_path: Path to the shapefile.
            start_date: Start date.
            end_date: End date.
            models: List of models to query.
            variable: Variable to retrieve.
            reducer: Spatial aggregation method.
            units: Units for ET values.
            interval: Temporal interval.
            ref_et_source: Reference ET source.
            dissolve: If True, dissolve shapefile features.
            output_path: Optional path to save data.
            output_format: Output format.

        Returns:
            DataFrame with columns for each model.
        """
        geometry = self._shapefile_to_geojson(shapefile_path, dissolve=dissolve)

        df = self.get_multiple_models(
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            models=models,
            variable=variable,
            reducer=reducer,
            interval=interval,
            units=units,
            ref_et_source=ref_et_source
        )

        if output_path:
            self.save_timeseries(df, output_path, output_format)

        return df
    
    def save_timeseries(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        output_format: str = 'csv'
    ) -> None:
        """
        Save timeseries data to file.
        
        Args:
            df: DataFrame with timeseries data.
            output_path: Path to save the data.
            output_format: Format ('csv' or 'netcdf').
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format.lower() == 'csv':
            df.to_csv(output_path)
            self.logger.info(f"Saved timeseries to {output_path}")
        elif output_format.lower() == 'netcdf':
            # Convert to xarray and save as NetCDF
            import xarray as xr
            ds = df.to_xarray()
            ds.to_netcdf(output_path)
            self.logger.info(f"Saved timeseries to {output_path}")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def download_openet_data(
    shapefile_path: Union[str, Path],
    output_dir: Union[str, Path],
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    env_file: Optional[str] = None,
    variable: str = 'et',
    models: Optional[List[str]] = None,
    interval: str = 'monthly',
    units: str = 'mm',
    output_filename: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to download OpenET data.
    
    This is a simplified interface for common use cases.
    
    Args:
        shapefile_path: Path to shapefile defining the area.
        output_dir: Directory to save the output.
        start_date: Start date ('YYYY-MM-DD').
        end_date: End date ('YYYY-MM-DD').
        api_key: OpenET API key (optional if set in environment).
        env_file: Path to .env file with API key.
        variable: Variable to download.
        models: List of models. If None, uses ensemble only.
        interval: Temporal interval.
        units: Units for values.
        output_filename: Output filename. If None, auto-generated.
        
    Returns:
        DataFrame with downloaded data.
        
    Example:
        >>> df = download_openet_data(
        ...     shapefile_path='/path/to/catchment.shp',
        ...     output_dir='/path/to/output/',
        ...     start_date='2018-01-01',
        ...     end_date='2023-12-31'
        ... )
    """
    # Create client
    client = OpenETClient(api_key=api_key, env_file=env_file)
    
    # Set default models
    if models is None:
        models = ['ensemble']
    
    # Generate output filename
    if output_filename is None:
        shapefile_name = Path(shapefile_path).stem
        output_filename = f"openet_{variable}_{shapefile_name}_{start_date}_{end_date}.csv"
    
    output_path = Path(output_dir) / output_filename
    
    # Download data
    if len(models) == 1:
        df = client.get_timeseries_from_shapefile(
            shapefile_path=shapefile_path,
            start_date=start_date,
            end_date=end_date,
            variable=variable,
            model=models[0],
            interval=interval,
            units=units,
            output_path=output_path
        )
    else:
        df = client.get_multiple_models_from_shapefile(
            shapefile_path=shapefile_path,
            start_date=start_date,
            end_date=end_date,
            models=models,
            variable=variable,
            interval=interval,
            units=units,
            output_path=output_path
        )
    
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download OpenET timeseries data')
    parser.add_argument('--shapefile', required=True, help='Path to shapefile')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--variable', default='et', help='Variable to download')
    parser.add_argument('--interval', default='monthly', help='Temporal interval')
    parser.add_argument('--env-file', help='Path to .env file with API key')

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--model',
        default='ensemble',
        help='Single ET model to use (default: ensemble)'
    )
    model_group.add_argument(
        '--models',
        help='Comma-separated ET models (e.g., disalexi,eemetric,geesebal,ptjpl,sims,ssebop,ensemble)'
    )

    args = parser.parse_args()

    if args.models:
        model_list = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    else:
        model_list = [args.model.strip().lower()]

    invalid = [m for m in model_list if m not in OpenETClient.AVAILABLE_MODELS]
    if invalid:
        raise ValueError(
            f"Invalid model(s): {invalid}. Available: {OpenETClient.AVAILABLE_MODELS}"
        )

    df = download_openet_data(
        shapefile_path=args.shapefile,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        env_file=args.env_file,
        variable=args.variable,
        models=model_list,
        interval=args.interval
    )

    print(f"Downloaded {len(df)} records")
    print(df.head())
