import geopandas as gpd # type: ignore
import numpy as np # type: ignore
from typing import List, Dict, Any, Optional, Tuple
import rasterio # type: ignore
from rasterio.mask import mask # type: ignore
from rasterio.warp import reproject, Resampling # type: ignore
from shapely.geometry import Polygon, MultiPolygon, shape # type: ignore
from shapely.ops import unary_union # type: ignore
import matplotlib.pyplot as plt # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import pvlib # type: ignore
import pandas as pd # type: ignore
from pyproj import CRS # type: ignore
import rasterstats # type: ignore
import time

class DomainDiscretizer:
    """
    A class for discretizing a domain into Hydrologic Response Units (HRUs).

    This class provides methods for various types of domain discretization,
    including elevation-based, soil class-based, land class-based, and
    radiation-based discretization. HRUs are allowed to be MultiPolygons,
    meaning spatially disconnected areas with the same attributes are 
    grouped into single HRUs.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        logger: Logger object for logging information and errors.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        self.dem_path = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        self.catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
        self.catchment_dir.mkdir(parents=True, exist_ok=True)
        delineation_method = self.config.get('DOMAIN_DEFINITION_METHOD')

        if delineation_method == 'delineate':
            self.delineation_suffix = 'delineate'
        elif delineation_method == 'lumped':
            self.delineation_suffix = 'lumped'
        elif delineation_method == 'subset':
            self.delineation_suffix = f"subset_{self.config['GEOFABRIC_TYPE']}"

    def sort_catchment_shape(self):
        """
        Sort the catchment shapefile based on GRU and HRU IDs.

        This method performs the following steps:
        1. Loads the catchment shapefile
        2. Sorts the shapefile based on GRU and HRU IDs
        3. Saves the sorted shapefile back to the original location

        The method uses GRU and HRU ID column names specified in the configuration.

        Raises:
            FileNotFoundError: If the catchment shapefile is not found.
            ValueError: If the required ID columns are not present in the shapefile.
        """
        self.logger.info("Sorting catchment shape")

        self.catchment_path = self.config.get('CATCHMENT_PATH')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            discretization_method = self.config.get('DOMAIN_DISCRETIZATION')
            # Handle comma-separated attributes for output filename
            if ',' in discretization_method:
                method_suffix = discretization_method.replace(',', '_')
            else:
                method_suffix = discretization_method
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{method_suffix}.shp"
        self.gruId = self.config.get('CATCHMENT_SHP_GRUID')
        self.hruId = self.config.get('CATCHMENT_SHP_HRUID')

        if self.catchment_path == 'default':
            self.catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            self.catchment_path = Path(self.catchment_path)

        catchment_file = self.catchment_path / self.catchment_name
        
        try:
            # Open the shape
            shp = gpd.read_file(catchment_file)
            
            # Check if required columns exist
            if self.gruId not in shp.columns or self.hruId not in shp.columns:
                raise ValueError(f"Required columns {self.gruId} and/or {self.hruId} not found in shapefile")
            
            # Sort
            shp = shp.sort_values(by=[self.gruId, self.hruId])
            
            # Save
            shp.to_file(catchment_file)
            
            self.logger.info(f"Catchment shape sorted and saved to {catchment_file}")
        except FileNotFoundError:
            self.logger.error(f"Catchment shapefile not found at {catchment_file}")
            raise
        except ValueError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error sorting catchment shape: {str(e)}")
            raise

    def discretize_domain(self) -> Optional[Path]:
        """
        Discretize the domain based on the method specified in the configuration.
        If CATCHMENT_SHP_NAME is provided and not 'default', it uses the provided shapefile instead.
        Supports both single attributes and comma-separated multiple attributes.
        """
        start_time = time.time()
        
        # Check if a custom catchment shapefile is provided
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if catchment_name != 'default':
            self.logger.info(f"Using provided catchment shapefile: {catchment_name}")
            self.logger.info("Skipping discretization steps")
            
            # Just sort the existing shapefile
            self.logger.info("Sorting provided catchment shape")
            shp = self.sort_catchment_shape()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Catchment processing completed in {elapsed_time:.2f} seconds")
            return shp
        
        # Parse discretization method to check for multiple attributes
        discretization_config = self.config.get('DOMAIN_DISCRETIZATION')
        attributes = [attr.strip() for attr in discretization_config.split(',')]
        
        self.logger.info(f"Starting domain discretization using attributes: {attributes}")

        # Handle single vs multiple attributes
        if len(attributes) == 1:
            # Single attribute - use existing logic
            discretization_method = attributes[0].lower()
            method_map = {
                'grus': self._use_grus_as_hrus,
                'elevation': self._discretize_by_elevation,
                'aspect': self._discretize_by_aspect,
                'soilclass': self._discretize_by_soil_class,
                'landclass': self._discretize_by_land_class,
                'radiation': self._discretize_by_radiation
            }

            if discretization_method not in method_map:
                self.logger.error(f"Invalid discretization method: {discretization_method}")
                raise ValueError(f"Invalid discretization method: {discretization_method}")

            self.logger.info("Step 1/2: Running single attribute discretization method")
            method_map[discretization_method]()
        else:
            # Multiple attributes - use combined discretization
            self.logger.info("Step 1/2: Running combined attributes discretization method")
            self._discretize_combined(attributes)
        
        self.logger.info("Step 2/2: Sorting catchment shape")
        shp = self.sort_catchment_shape()

        elapsed_time = time.time() - start_time
        self.logger.info(f"Domain discretization completed in {elapsed_time:.2f} seconds")
        return shp

    def _discretize_combined(self, attributes: List[str]):
        """
        Discretize the domain based on a combination of geospatial attributes.
        
        Args:
            attributes: List of attribute names to combine (e.g., ['elevation', 'landclass'])
        """
        self.logger.info(f"Starting combined discretization with attributes: {attributes}")
        
        # Get GRU shapefile
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        elif self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               f"{self.domain_name}_riverBasins_with_coastal.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               self.config.get('RIVER_BASINS_NAME'))
        
        # Generate output filename
        method_suffix = '_'.join(attributes)
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", 
                                              f"{self.domain_name}_HRUs_{method_suffix}.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", 
                                         f"{self.domain_name}_HRUs_{method_suffix}.png")
        
        # Get raster paths and thresholds for each attribute
        raster_info = self._get_raster_info_for_attributes(attributes)
        
        # Read GRU data
        gru_gdf = self._read_shapefile(gru_shapefile)
        
        # Create combined HRUs
        hru_gdf = self._create_combined_attribute_hrus(gru_gdf, raster_info, attributes)
        
        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Combined attribute HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            # Create plot with combined attributes
            plot_column = f"combined_{method_suffix}"
            self._plot_hrus(hru_gdf, output_plot, plot_column, f'Combined {method_suffix.replace("_", " + ")} HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _get_raster_info_for_attributes(self, attributes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get raster paths and classification information for each attribute.
        
        Args:
            attributes: List of attribute names
            
        Returns:
            Dictionary containing raster path and classification info for each attribute
        """
        raster_info = {}
        
        for attr in attributes:
            attr_lower = attr.lower()
            
            if attr_lower == 'elevation':
                dem_name = self.config['DEM_NAME']
                if dem_name == "default":
                    dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"
                
                raster_path = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
                band_size = float(self.config.get('ELEVATION_BAND_SIZE'))
                
                raster_info[attr] = {
                    'path': raster_path,
                    'type': 'continuous',
                    'band_size': band_size,
                    'class_name': 'elevClass'
                }
                
            elif attr_lower == 'soilclass':
                raster_path = self._get_file_path("SOIL_CLASS_PATH", "attributes/soilclass/", 
                                                f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif")
                raster_info[attr] = {
                    'path': raster_path,
                    'type': 'discrete',
                    'class_name': 'soilClass'
                }
                
            elif attr_lower == 'landclass':
                raster_path = self._get_file_path("LAND_CLASS_PATH", "attributes/landclass", 
                                                f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif")
                raster_info[attr] = {
                    'path': raster_path,
                    'type': 'discrete',
                    'class_name': 'landClass'
                }
                
            elif attr_lower == 'radiation':
                radiation_raster = self._get_file_path("RADIATION_PATH", "attributes/radiation", 
                                                     "annual_radiation.tif")
                
                # Calculate radiation if it doesn't exist
                if not radiation_raster.exists():
                    self.logger.info("Annual radiation raster not found. Calculating radiation...")
                    dem_name = self.config['DEM_NAME']
                    if dem_name == "default":
                        dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"
                    dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
                    radiation_raster = self._calculate_annual_radiation(dem_raster, radiation_raster)
                    if radiation_raster is None:
                        raise ValueError("Failed to calculate annual radiation")
                
                radiation_class_number = int(self.config.get('RADIATION_CLASS_NUMBER'))
                
                raster_info[attr] = {
                    'path': radiation_raster,
                    'type': 'continuous',
                    'band_size': radiation_class_number,
                    'class_name': 'radiationClass'
                }
            elif attr_lower == 'aspect':
                aspect_raster = self._get_file_path(
                    "ASPECT_PATH", "attributes/elevation/dem",
                    f"domain_{self.config['DOMAIN_NAME']}_aspect.tif"
                )

                # Calculate aspect if it doesn't exist (fallback; prefer compute_aspect_raster step)
                if not aspect_raster.exists():
                    self.logger.info("Aspect raster not found. Calculating aspect...")
                    dem_name = self.config.get('DEM_NAME', 'default')
                    if dem_name == "default":
                        dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"
                    dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
                    aspect_raster = self._calculate_aspect(dem_raster, aspect_raster)
                    if aspect_raster is None:
                        raise ValueError("Failed to calculate aspect")
                
                raster_info[attr] = {
                    'path': aspect_raster,
                    'type': 'discrete',
                    'class_name': 'aspectClass'
                }

            elif attr_lower == 'tpi':
                # Resolve TPI class raster with optional explicit override.
                # Priority:
                # 1) TPI_CLASS_RASTER (full file path)
                # 2) TPI_CLASS_PATH/TPI_PATH + TPI_CLASS_NAME
                # 3) domain default path: <project>/attributes/tpi/tpi1000_class.tif
                tpi_raster_override = self.config.get('TPI_CLASS_RASTER')
                tpi_class_name = self.config.get('TPI_CLASS_NAME', 'tpi1000_class.tif')
                if tpi_class_name == 'default' or not tpi_class_name:
                    tpi_class_name = 'tpi1000_class.tif'

                if tpi_raster_override and tpi_raster_override != 'default':
                    tpi_raster = Path(tpi_raster_override)
                else:
                    tpi_base = self.config.get('TPI_CLASS_PATH', self.config.get('TPI_PATH', 'default'))
                    if tpi_base == 'default' or not tpi_base:
                        tpi_raster = self.project_dir / 'attributes' / 'tpi' / tpi_class_name
                    else:
                        tpi_raster = Path(tpi_base) / tpi_class_name

                if not tpi_raster.exists():
                    raise ValueError(
                        f"TPI class raster not found at {tpi_raster}. "
                        "Set TPI_CLASS_RASTER or place tpi1000_class.tif under attributes/tpi/."
                    )

                raster_info[attr] = {
                    'path': tpi_raster,
                    'type': 'discrete',
                    'class_name': 'tpiClass'
                }

            else:
                raise ValueError(f"Unsupported attribute for discretization: {attr}")
        
        return raster_info

    def _create_combined_attribute_hrus(self, gru_gdf: gpd.GeoDataFrame, 
                                       raster_info: Dict[str, Dict[str, Any]], 
                                       attributes: List[str]) -> gpd.GeoDataFrame:
        """
        Create HRUs based on unique combinations of multiple attributes within each GRU.
        
        Args:
            gru_gdf: GeoDataFrame containing GRU data
            raster_info: Dictionary containing raster information for each attribute
            attributes: List of attribute names
            
        Returns:
            GeoDataFrame containing combined attribute HRUs
        """
        self.logger.info(f"Creating combined attribute HRUs within {len(gru_gdf)} GRUs")
        
        all_hrus = []
        hru_id_counter = 1
        
        # Process each GRU individually
        for gru_idx, gru_row in gru_gdf.iterrows():
            self.logger.info(f"Processing GRU {gru_idx + 1}/{len(gru_gdf)}")
            
            gru_geometry = gru_row.geometry
            gru_id = gru_row.get('GRU_ID', gru_idx + 1)
            
            # Extract all raster data for this GRU
            raster_data = {}
            common_transform = None
            common_shape = None
            common_crs = None
            
            for attr in attributes:
                attr_info = raster_info[attr]
                raster_path = attr_info['path']
                
                try:
                    with rasterio.open(raster_path) as src:
                        # Determine the effective raster CRS.  When it is not embedded (common
                        # for TPI/aspect rasters saved without CRS metadata), infer the UTM zone
                        # from the GRU centroid so that geometry reprojection and alignment work.
                        effective_src_crs = src.crs
                        if effective_src_crs is None and gru_gdf.crs is not None:
                            try:
                                centroid = gru_gdf.geometry.union_all().centroid
                                if gru_gdf.crs.is_geographic:
                                    lon, lat = centroid.x, centroid.y
                                else:
                                    _pt = gpd.GeoSeries([centroid], crs=gru_gdf.crs).to_crs("EPSG:4326").iloc[0]
                                    lon, lat = _pt.x, _pt.y
                                utm_zone = int((lon + 180) / 6) + 1
                                epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
                                from rasterio.crs import CRS as RioCRS
                                effective_src_crs = RioCRS.from_epsg(epsg)
                                self.logger.warning(
                                    f"Raster {raster_path.name} has no embedded CRS. "
                                    f"Inferred EPSG:{epsg} (UTM zone {utm_zone}) from GRU centroid."
                                )
                            except Exception:
                                pass  # leave effective_src_crs as None; masking may still succeed

                        # Reproject GRU geometry to raster CRS when needed before masking.
                        geometry_for_mask = gru_geometry
                        if gru_gdf.crs is not None and effective_src_crs is not None and gru_gdf.crs != effective_src_crs:
                            geometry_for_mask = (
                                gpd.GeoSeries([gru_geometry], crs=gru_gdf.crs)
                                .to_crs(effective_src_crs)
                                .iloc[0]
                            )

                        out_image, out_transform = mask(src, [geometry_for_mask], crop=True,
                                                       all_touched=True, filled=False)
                        out_image = out_image[0]
                        nodata_value = src.nodata

                        if nodata_value is None:
                            # Preserve missing values when source has no explicit nodata.
                            nodata_value = np.nan
                            out_image = out_image.astype(np.float32)

                        # Convert masked arrays to plain ndarrays so downstream numpy ops
                        # (comparisons, broadcasting with other rasters) behave predictably.
                        if isinstance(out_image, np.ma.MaskedArray):
                            _fill = nodata_value if not (isinstance(nodata_value, float) and np.isnan(nodata_value)) else out_image.fill_value
                            out_image = out_image.filled(_fill)

                        # Set reference grid from first raster and align subsequent rasters to it.
                        if common_transform is None:
                            common_transform = out_transform
                            common_shape = out_image.shape
                            common_crs = effective_src_crs if effective_src_crs is not None else gru_gdf.crs
                        else:
                            needs_alignment = (
                                out_image.shape != common_shape
                                or out_transform != common_transform
                                or (common_crs is not None and effective_src_crs is not None and effective_src_crs != common_crs)
                            )

                            if needs_alignment:
                                aligned_image = np.full(common_shape, nodata_value, dtype=out_image.dtype)
                                reproject(
                                    source=out_image,
                                    destination=aligned_image,
                                    src_transform=out_transform,
                                    src_crs=effective_src_crs if effective_src_crs is not None else common_crs,
                                    src_nodata=nodata_value,
                                    dst_transform=common_transform,
                                    dst_crs=common_crs,
                                    dst_nodata=nodata_value,
                                    resampling=Resampling.nearest,
                                )
                                out_image = aligned_image
                        
                        # Store raster data and metadata
                        raster_data[attr] = {
                            'data': out_image,
                            'nodata': nodata_value,
                            'info': attr_info
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Could not extract {attr} raster data for GRU {gru_id}: {str(e)}")
                    continue

            missing_attrs = [attr for attr in attributes if attr not in raster_data]
            if missing_attrs:
                self.logger.warning(
                    f"Skipping GRU {gru_id}: missing raster extracts for attributes {missing_attrs}"
                )
                continue
            
            if not raster_data:
                self.logger.warning(f"No valid raster data found for GRU {gru_id}")
                continue
            
            # Create combined valid mask (pixels that are valid in all rasters)
            combined_valid_mask = np.ones(common_shape, dtype=bool)
            for attr, data in raster_data.items():
                raster_array = data['data']
                nodata_value = data['nodata']
                
                if nodata_value is not None:
                    valid_mask = raster_array != nodata_value
                else:
                    valid_mask = ~np.isnan(raster_array) if raster_array.dtype == np.float64 else np.ones_like(raster_array, dtype=bool)
                
                combined_valid_mask &= valid_mask
            
            if not np.any(combined_valid_mask):
                self.logger.warning(f"No valid pixels found in GRU {gru_id}")
                continue
            
            # Classify each attribute and find unique combinations
            classified_data = {}
            for attr in attributes:
                data_info = raster_data[attr]
                raster_array = data_info['data']
                attr_info = data_info['info']
                
                if attr_info['type'] == 'continuous':
                    # Classify continuous data into bands
                    classified_data[attr] = self._classify_continuous_data(
                        raster_array, combined_valid_mask, attr_info['band_size']
                    )
                else:
                    # Use discrete values directly
                    classified_data[attr] = raster_array
            
            # Find unique combinations of classified values
            unique_combinations = self._find_unique_combinations(
                classified_data,
                combined_valid_mask,
                attributes,
            )
            
            # Create HRUs for each unique combination
            gru_hrus = self._create_hrus_from_combinations(
                unique_combinations, classified_data, combined_valid_mask, 
                common_transform, gru_geometry, gru_row, hru_id_counter, attributes
            )
            
            all_hrus.extend(gru_hrus)
            hru_id_counter += len(gru_hrus)
        
        self.logger.info(f"Created {len(all_hrus)} combined attribute HRUs across all GRUs")
        if not all_hrus:
            self.logger.warning("No HRUs created during combined discretization")
            return gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs=gru_gdf.crs)
        return gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)

    def _classify_continuous_data(self, raster_array: np.ndarray, valid_mask: np.ndarray, 
                                 band_size: float) -> np.ndarray:
        """
        Classify continuous raster data into discrete bands.
        
        Args:
            raster_array: Input raster array
            valid_mask: Boolean mask for valid pixels
            band_size: Size of bands (for elevation) or number of classes (for radiation)
            
        Returns:
            Classified array with discrete class values
        """
        valid_data = raster_array[valid_mask]
        
        if len(valid_data) == 0:
            return raster_array.copy()
        
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        # Create classification based on band_size
        if isinstance(band_size, int) and band_size < 50:  # Assume it's number of classes for radiation
            # Use quantile-based classification
            quantiles = np.linspace(0, 1, band_size + 1)
            thresholds = np.quantile(valid_data, quantiles)
        else:
            # Use fixed band size for elevation
            thresholds = np.arange(data_min, data_max + band_size, band_size)
            if thresholds[-1] < data_max:
                thresholds = np.append(thresholds, thresholds[-1] + band_size)
        
        # Classify the data
        classified = np.zeros_like(raster_array, dtype=int)
        for i in range(len(thresholds) - 1):
            lower, upper = thresholds[i:i+2]
            if i == len(thresholds) - 2:  # Last band
                mask = valid_mask & (raster_array >= lower) & (raster_array <= upper)
            else:
                mask = valid_mask & (raster_array >= lower) & (raster_array < upper)
            classified[mask] = i + 1
        
        return classified

    def _find_unique_combinations(
        self,
        classified_data: Dict[str, np.ndarray],
        valid_mask: np.ndarray,
        attribute_order: Optional[List[str]] = None,
    ) -> List[Tuple]:
        """
        Find unique combinations of classified values across all attributes.
        
        Args:
            classified_data: Dictionary of classified raster arrays for each attribute
            valid_mask: Boolean mask for valid pixels
            attribute_order: Explicit attribute ordering for combination tuples
            
        Returns:
            List of unique value combinations
        """
        # Stack all classified arrays
        stacked_data = []
        ordered_attrs = [
            attr for attr in (attribute_order or list(classified_data.keys()))
            if attr in classified_data
        ]
        for attr in ordered_attrs:
            stacked_data.append(classified_data[attr][valid_mask])
        
        # Find unique combinations
        combined_array = np.column_stack(stacked_data)
        unique_combinations = [tuple(row) for row in np.unique(combined_array, axis=0)]
        
        return unique_combinations

    def _create_hrus_from_combinations(self, unique_combinations: List[Tuple], 
                                      classified_data: Dict[str, np.ndarray],
                                      valid_mask: np.ndarray, transform: Any,
                                      gru_geometry: Any, gru_row: Any, 
                                      start_hru_id: int, attributes: List[str]) -> List[Dict]:
        """
        Create HRUs for each unique combination of attribute values.
        
        Args:
            unique_combinations: List of unique value combinations
            classified_data: Dictionary of classified raster arrays
            valid_mask: Boolean mask for valid pixels
            transform: Raster transform
            gru_geometry: GRU geometry
            gru_row: GRU data row
            start_hru_id: Starting HRU ID
            attributes: List of attribute names
            
        Returns:
            List of HRU dictionaries
        """
        hrus = []
        current_hru_id = start_hru_id
        
        for combination in unique_combinations:
            # Create mask for this combination
            combination_mask = valid_mask.copy()
            
            for i, attr in enumerate(attributes):
                attr_value = combination[i]
                combination_mask &= (classified_data[attr] == attr_value)
            
            if not np.any(combination_mask):
                continue
            
            # Create HRU from this combination
            hru = self._create_hru_from_combination_mask(
                combination_mask, transform, classified_data, gru_geometry,
                gru_row, current_hru_id, attributes, combination
            )
            
            if hru:
                hrus.append(hru)
                current_hru_id += 1
        
        return hrus

    def _create_hru_from_combination_mask(self, combination_mask: np.ndarray, transform: Any,
                                         classified_data: Dict[str, np.ndarray], 
                                         gru_geometry: Any, gru_row: Any, hru_id: int,
                                         attributes: List[str], combination: Tuple) -> Optional[Dict]:
        """
        Create a single HRU from a combination mask.
        
        Args:
            combination_mask: Boolean mask for the combination
            transform: Raster transform
            classified_data: Dictionary of classified raster arrays
            gru_geometry: GRU geometry
            gru_row: GRU data row
            hru_id: HRU ID
            attributes: List of attribute names
            combination: Tuple of attribute values for this combination
            
        Returns:
            Dictionary representing the HRU or None if creation fails
        """
        try:
            # Extract shapes from the mask
            shapes = list(rasterio.features.shapes(
                combination_mask.astype(np.uint8), 
                mask=combination_mask, 
                transform=transform,
                connectivity=4
            ))
            
            if not shapes:
                return None
            
            # Create polygons from shapes
            polygons = []
            for shp, _ in shapes:
                try:
                    geom = shape(shp)
                    if geom.is_valid and not geom.is_empty and geom.area > 0:
                        polygons.append(geom)
                except Exception:
                    continue
            
            if not polygons:
                return None
            
            # Create final geometry
            if len(polygons) == 1:
                final_geometry = polygons[0]
            else:
                final_geometry = MultiPolygon(polygons)
            
            # Clean the geometry
            if not final_geometry.is_valid:
                final_geometry = final_geometry.buffer(0)
            
            if final_geometry.is_empty or not final_geometry.is_valid:
                return None
            
            # Ensure it's within the GRU boundary
            clipped_geometry = final_geometry.intersection(gru_geometry)
            
            if clipped_geometry.is_empty or not clipped_geometry.is_valid:
                return None
            
            # Create HRU data with combination attributes
            hru_data = {
                'geometry': clipped_geometry,
                'GRU_ID': gru_row.get('GRU_ID', gru_row.name),
                'HRU_ID': hru_id,
                'hru_type': f'combined_{"_".join(attributes)}'
            }
            
            # Add individual attribute values
            for i, attr in enumerate(attributes):
                attr_name = attr.lower()
                if attr_name == 'elevation':
                    hru_data['elevClass'] = combination[i]
                elif attr_name == 'aspect':
                    hru_data['aspectClass'] = combination[i]
                elif attr_name == 'tpi':
                    hru_data['tpiClass'] = combination[i]
                elif attr_name == 'soilclass':
                    hru_data['soilClass'] = combination[i]
                elif attr_name == 'landclass':
                    hru_data['landClass'] = combination[i]
                elif attr_name == 'radiation':
                    hru_data['radiationClass'] = combination[i]
            
            # Add combined attribute identifier
            combined_id = '_'.join([str(val) for val in combination])
            combined_name = f"combined_{'_'.join(attributes)}"
            hru_data[combined_name] = combined_id
            
            # Copy relevant GRU attributes (excluding geometry)
            for col in gru_row.index:
                if col not in ['geometry', 'GRU_ID'] and col not in hru_data:
                    hru_data[col] = gru_row[col]
            
            return hru_data
            
        except Exception as e:
            self.logger.warning(f"Error creating HRU for combination {combination}: {str(e)}")
            return None

    def _use_grus_as_hrus(self):
        """
        Use Grouped Response Units (GRUs) as Hydrologic Response Units (HRUs) without further discretization.

        Returns:
            Path: Path to the output HRU shapefile.
        """
        self.logger.info(f"config domain name {self.config.get('DOMAIN_NAME')}")
        if self.config.get('RIVER_BASINS_NAME') == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp")
        
            if self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
                gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_with_coastal.shp")
        
            elif self.config.get('DOMAIN_DEFINITION_METHOD') == "point":
                gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_point.shp")
            
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", self.config.get('RIVER_BASINS_NAME'))
        
        hru_output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_GRUs.shp")

        gru_gdf = self._read_shapefile(gru_shapefile)
        gru_gdf['HRU_ID'] = range(1, len(gru_gdf) + 1)
        gru_gdf['hru_type'] = 'GRU'

        # Calculate mean elevation for each HRU with proper CRS handling
        self.logger.info("Calculating mean elevation for each HRU")
        
        # Get CRS information
        with rasterio.open(self.dem_path) as src:
            dem_crs = src.crs
            self.logger.info(f"DEM CRS: {dem_crs}")
        
        shapefile_crs = gru_gdf.crs
        self.logger.info(f"Shapefile CRS: {shapefile_crs}")
        
        # Check if CRS match
        if dem_crs != shapefile_crs:
            self.logger.info(f"CRS mismatch detected. Reprojecting shapefile from {shapefile_crs} to {dem_crs}")
            gru_gdf_projected = gru_gdf.to_crs(dem_crs)
        else:
            self.logger.info("CRS match - no reprojection needed")
            gru_gdf_projected = gru_gdf.copy()
        
        # Use rasterstats with the raster file path directly (more efficient and handles CRS properly)
        try:
            zs = rasterstats.zonal_stats(
                gru_gdf_projected.geometry, 
                str(self.dem_path),  # Use file path instead of array
                stats=['mean'],
                nodata=-9999  # Explicit nodata value
            )
            gru_gdf['elev_mean'] = [item['mean'] if item['mean'] is not None else -9999 for item in zs]
            self.logger.info(f"Successfully calculated elevation statistics for {len(gru_gdf)} HRUs")
            
        except Exception as e:
            self.logger.error(f"Error calculating zonal statistics: {str(e)}")
            # Fallback: set all elevation means to -9999
            gru_gdf['elev_mean'] = -9999
            self.logger.warning("Setting all elevation means to -9999 due to calculation error")
        
        # Calculate centroids in projected CRS for accuracy
        # Project to UTM for accurate centroid calculation if not already in UTM
        try:
            if gru_gdf.crs.is_geographic:
                utm_crs = gru_gdf.estimate_utm_crs()
                gru_gdf_utm = gru_gdf.to_crs(utm_crs)
            else:
                # Already in projected coordinate system
                gru_gdf_utm = gru_gdf.copy()
                utm_crs = gru_gdf.crs
            
            centroids_utm = gru_gdf_utm.geometry.centroid
            centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
            
            gru_gdf['center_lon'] = centroids_wgs84.x
            gru_gdf['center_lat'] = centroids_wgs84.y
            
            self.logger.info(f"Calculated centroids in WGS84: lat range {centroids_wgs84.y.min():.6f} to {centroids_wgs84.y.max():.6f}, lon range {centroids_wgs84.x.min():.6f} to {centroids_wgs84.x.max():.6f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating centroids: {str(e)}")
            # Fallback: try to use existing center_lat/center_lon if they exist and look reasonable
            if 'center_lat' in gru_gdf.columns and 'center_lon' in gru_gdf.columns:
                # Check if existing values look like actual lat/lon (rough check)
                if (gru_gdf['center_lat'].between(-90, 90).all() and 
                    gru_gdf['center_lon'].between(-180, 180).all()):
                    self.logger.info("Using existing center_lat/center_lon coordinates")
                else:
                    self.logger.warning("Existing center_lat/center_lon appear to be in projected coordinates, setting to default values")
                    gru_gdf['center_lat'] = 0.0
                    gru_gdf['center_lon'] = 0.0
            else:
                gru_gdf['center_lat'] = 0.0
                gru_gdf['center_lon'] = 0.0
        
        if 'COMID' in gru_gdf.columns:
            gru_gdf['GRU_ID'] = gru_gdf['COMID']
        elif 'fid' in gru_gdf.columns:
            gru_gdf['GRU_ID'] = gru_gdf['fid']

        gru_gdf['HRU_area'] = gru_gdf['GRU_area']
        gru_gdf['HRU_ID'] = gru_gdf['GRU_ID']        

        gru_gdf.to_file(hru_output_shapefile)
        self.logger.info(f"GRUs saved as HRUs to {hru_output_shapefile}")

        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_as_GRUs.png")
        self._plot_hrus(gru_gdf, output_plot, 'HRU_ID', 'GRUs = HRUs')
        return hru_output_shapefile

    def _discretize_by_elevation(self):
        """
        Discretize the domain based on elevation within each GRU.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        elif self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins__with_coastal.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", self.config.get('RIVER_BASINS_NAME'))
        
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_elevation.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_elevation.png")

        elevation_band_size = float(self.config.get('ELEVATION_BAND_SIZE'))
        # Read GRUs first, then compute thresholds only over the GRU-covered basin area.
        # This prevents out-of-basin DEM extents from inflating the number of elevation bands.
        gru_gdf, _ = self._read_and_prepare_data(gru_shapefile, dem_raster, elevation_band_size)
        elevation_thresholds = self._compute_elevation_thresholds_within_grus(
            gru_gdf,
            dem_raster,
            elevation_band_size,
        )
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, dem_raster, elevation_thresholds, 'elevClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Elevation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'elevClass', 'Elevation-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _compute_elevation_thresholds_within_grus(
        self,
        gru_gdf: gpd.GeoDataFrame,
        dem_raster: Path,
        elevation_band_size: float,
    ) -> np.ndarray:
        """Compute elevation thresholds using DEM values only inside GRU geometries."""
        with rasterio.open(dem_raster) as src:
            working_grus = gru_gdf
            if gru_gdf.crs is not None and src.crs is not None and gru_gdf.crs != src.crs:
                working_grus = gru_gdf.to_crs(src.crs)

            domain_union = working_grus.union_all()
            out_image, _ = mask(src, [domain_union], crop=True, all_touched=True, filled=False)
            out_image = out_image[0]

            nodata_value = src.nodata
            if nodata_value is not None:
                valid_data = out_image[out_image != nodata_value]
            else:
                valid_data = out_image

            valid_data = valid_data[np.isfinite(valid_data)]

        if len(valid_data) == 0:
            raise ValueError("No valid DEM pixels found within GRU geometries")

        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))
        thresholds = np.arange(min_val, max_val + elevation_band_size, elevation_band_size)

        if thresholds[-1] < max_val:
            thresholds = np.append(thresholds, thresholds[-1] + elevation_band_size)

        self.logger.info(
            f"Elevation thresholds from GRU-covered DEM area: min={min_val:.2f}, max={max_val:.2f}, "
            f"band_size={elevation_band_size}, bands={len(thresholds) - 1}"
        )
        return thresholds

    def _discretize_by_soil_class(self):
        """
        Discretize the domain based on soil classifications using MultiPolygon HRUs.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        soil_raster = self._get_file_path("SOIL_CLASS_PATH", "attributes/soilclass/", f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_soilclass.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_soilclass.png")

        gru_gdf, soil_classes = self._read_and_prepare_data(gru_shapefile, soil_raster)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, soil_raster, soil_classes, 'soilClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Soil-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'soilClass', 'Soil-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _discretize_by_land_class(self):
        """
        Discretize the domain based on land cover classifications using MultiPolygon HRUs.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        elif self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins__with_coastal.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", self.config.get('RIVER_BASINS_NAME'))

        land_raster = self._get_file_path("LAND_CLASS_PATH","attributes/landclass", f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_landclass.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_landclass.png")

        gru_gdf, land_classes = self._read_and_prepare_data(gru_shapefile, land_raster)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, land_raster, land_classes, 'landClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Land-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'landClass', 'Land-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _discretize_by_aspect(self):
        """
        Discretize the domain based on aspect (slope direction) using MultiPolygon HRUs.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        elif self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               f"{self.domain_name}_riverBasins_with_coastal.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               self.config.get('RIVER_BASINS_NAME'))

        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        aspect_raster = self._get_file_path(
            "ASPECT_PATH", "attributes/elevation/dem",
            f"domain_{self.config['DOMAIN_NAME']}_aspect.tif"
        )
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", 
                                              f"{self.domain_name}_HRUs_aspect.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", 
                                         f"{self.domain_name}_HRUs_aspect.png")

        aspect_class_number = int(self.config.get('ASPECT_CLASS_NUMBER', 8))

        if not aspect_raster.exists():
            self.logger.info("Aspect raster not found. Calculating aspect...")
            aspect_raster = self._calculate_aspect(dem_raster, aspect_raster)
            if aspect_raster is None:
                raise ValueError("Failed to calculate aspect")

        gru_gdf, aspect_classes = self._read_and_prepare_data(gru_shapefile, aspect_raster)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, aspect_raster, aspect_classes, 'aspectClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Aspect-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'aspectClass', 'Aspect-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def compute_aspect_raster(self) -> Optional[Path]:
        """
        Public entry point: compute and save the classified aspect raster from the DEM.

        Called explicitly as a preprocessing step before discretize_domain so that
        the aspect TIF is always available on disk before discretization begins.

        Returns:
            Path to the saved aspect raster, or None on failure.
        """
        dem_name = self.config.get('DEM_NAME', 'default')
        if dem_name == 'default':
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"
        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        aspect_raster = self._get_file_path("ASPECT_PATH", "attributes/elevation/dem",
                                            f"domain_{self.config['DOMAIN_NAME']}_aspect.tif")
        return self._calculate_aspect(dem_raster, aspect_raster)

    def _calculate_aspect(self, dem_raster: Path, aspect_raster: Path) -> Optional[Path]:
        """
        Calculate aspect from the DEM using the Horn (1981) kernel with lat/lon cell-size
        correction, classify into cardinal direction classes, and save the result.

        The Horn kernel is identical to what ``gdaldem aspect`` uses internally and is
        more accurate than a simple ``np.gradient`` central-difference, especially for
        geographic (degree) coordinate systems where east–west cell width varies with
        latitude.

        Args:
            dem_raster: Path to the input DEM raster (any CRS; lat/lon correction is
                applied automatically when CRS is geographic).
            aspect_raster: Destination path for the classified aspect raster.

        Returns:
            Path to the saved aspect raster, or None on failure.
        """
        self.logger.info(f"Calculating aspect from DEM: {dem_raster}")

        try:
            with rasterio.open(dem_raster) as src:
                dem = src.read(1).astype(float)
                transform = src.transform
                crs = src.crs
                nodata = src.nodata

            # Mask nodata before gradient computation
            if nodata is not None:
                dem_nodata_mask = (dem == nodata)
                dem[dem_nodata_mask] = np.nan
            else:
                dem_nodata_mask = np.zeros(dem.shape, dtype=bool)

            # ------------------------------------------------------------------
            # Cell sizes in metres.
            # For geographic CRS (degrees) the east–west size shrinks with cos(lat).
            # For projected CRS (metres) the transform already gives metric sizes.
            # ------------------------------------------------------------------
            cell_x_deg = abs(transform.a)  # width of one cell (transform x-step)
            cell_y_deg = abs(transform.e)  # height of one cell (transform y-step)

            if crs and crs.is_geographic:
                # Build per-row latitude array (centre of each row)
                nrows = dem.shape[0]
                lat_top = transform.f
                lats = lat_top - (np.arange(nrows) + 0.5) * cell_y_deg
                dy_m = cell_y_deg * 111320.0
                # dx varies with latitude → shape (nrows, 1) broadcasts across columns
                dx_m = cell_x_deg * 111320.0 * np.cos(np.deg2rad(lats))[:, np.newaxis]
            else:
                dy_m = cell_y_deg
                dx_m = cell_x_deg

            # ------------------------------------------------------------------
            # Horn (1981) weighted 3×3 gradient kernel.
            # Pad with NaN so border pixels stay NaN (same behaviour as gdaldem).
            # ------------------------------------------------------------------
            p = np.pad(dem, 1, mode='constant', constant_values=np.nan)

            # dz/dx: east–west slope (positive = uphill to the east)
            dz_dx = (
                (p[0:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) -
                (p[0:-2, 0:-2] + 2 * p[1:-1, 0:-2] + p[2:, 0:-2])
            ) / (8.0 * dx_m)

            # dz/dy: north–south slope (positive = uphill to the north)
            dz_dy = (
                (p[2:, 0:-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) -
                (p[0:-2, 0:-2] + 2 * p[0:-2, 1:-1] + p[0:-2, 2:])
            ) / (8.0 * dy_m)

            # ------------------------------------------------------------------
            # Aspect: degrees clockwise from North (0–360).
            # The gradient vector points uphill, so add 180° by default to convert
            # to downslope-facing aspect (the common hydrologic convention).
            # ASPECT_AZIMUTH_OFFSET_DEG can be overridden in config if needed.
            # ------------------------------------------------------------------
            upslope_azimuth_deg = (90.0 - np.degrees(np.arctan2(-dz_dy, dz_dx))) % 360.0
            aspect_offset_deg = float(self.config.get('ASPECT_AZIMUTH_OFFSET_DEG', 180.0))
            aspect_deg = (upslope_azimuth_deg + aspect_offset_deg) % 360.0

            # Flat pixels: slope < threshold → class 0 (no aspect-based SW adjustment).
            # Valley bottoms and other low-gradient terrain are still valid HRU pixels;
            # they receive no aspect correction but ARE included in HRU delineation.
            # Threshold is in degrees of slope (not raw gradient magnitude).
            FLAT_SLOPE_DEG = float(self.config.get('ASPECT_FLAT_SLOPE_THRESHOLD', 5.0))
            slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
            flat_mask = slope_deg < FLAT_SLOPE_DEG

            # ------------------------------------------------------------------
            # Classify into cardinal direction classes
            # ------------------------------------------------------------------
            aspect_class_number = int(self.config.get('ASPECT_CLASS_NUMBER', 4))
            classified_aspect = self._classify_aspect_into_classes(
                aspect_deg, flat_mask, aspect_class_number
            )

            # Apply nodata mask (NaN borders from Horn padding + original nodata)
            classified_aspect[dem_nodata_mask] = -9999
            classified_aspect[np.isnan(aspect_deg)] = -9999

            # Save
            aspect_raster.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                aspect_raster, 'w', driver='GTiff',
                height=classified_aspect.shape[0],
                width=classified_aspect.shape[1],
                count=1, dtype='int16',
                crs=crs, transform=transform, nodata=-9999,
            ) as dst:
                dst.write(classified_aspect.astype('int16'), 1)

            class_ids = np.unique(classified_aspect[classified_aspect != -9999])
            self.logger.info(f"Aspect raster saved to: {aspect_raster}")
            self.logger.info(f"Aspect classes present: {class_ids}")
            return aspect_raster

        except Exception as e:
            self.logger.error(f"Error calculating aspect: {str(e)}", exc_info=True)
            return None

    def _classify_aspect_into_classes(self, aspect_deg: np.ndarray, flat_mask: np.ndarray,
                                     num_classes: int) -> np.ndarray:
        """
        Classify aspect degrees (0–360, CW from North) into cardinal direction classes.

        All bins are 90°-wide and **centred** on their cardinal direction so that, for
        example, North covers 315°–45° (wrapping through 0°).  The wrap-around is
        handled by splitting the North bin into two segments: [0°, 45°) and [315°, 360°],
        both mapped to label 1.

        4-class mapping (default):
            1 = NE (  0° –  90°)
            2 = SE ( 90° – 180°)
            3 = SW (180° – 270°)
            4 = NW (270° – 360°)

        8-class mapping:
            1 = N   (337.5° – 22.5°)
            2 = NE  ( 22.5° – 67.5°)
            3 = E   ( 67.5° – 112.5°)
            4 = SE  (112.5° – 157.5°)
            5 = S   (157.5° – 202.5°)
            6 = SW  (202.5° – 247.5°)
            7 = W   (247.5° – 292.5°)
            8 = NW  (292.5° – 337.5°)

        Class 0 = flat (slope < ASPECT_FLAT_SLOPE_THRESHOLD, default 5°).
        Flat pixels are valid terrain (valley bottoms, meadows) that receive no
        aspect-based solar correction (SW multiplier = 1.0). They are included
        in HRU delineation as their own class — NOT treated as nodata.
        True nodata (-9999) is applied by the caller for DEM nodata pixels only.

        Args:
            aspect_deg: Aspect array in degrees [0, 360), clockwise from North.
            flat_mask:  Boolean mask; True where slope < flat threshold.
            num_classes: 4 or 8 (other values fall back to equal-width bins).

        Returns:
            Integer array: 0 = flat, 1–N = cardinal classes, -9999 = nodata (set by caller).
        """
        classified = np.zeros_like(aspect_deg, dtype=int)

        if num_classes == 4:
            # Bins centred on NE/SE/SW/NW at 45/135/225/315°.
            # Boundaries align to 0/90/180/270 so no wrap-around is needed.
            classified[(aspect_deg >= 0)   & (aspect_deg <  90)]  = 1  # NE
            classified[(aspect_deg >= 90)  & (aspect_deg < 180)]  = 2  # SE
            classified[(aspect_deg >= 180) & (aspect_deg < 270)]  = 3  # SW
            classified[(aspect_deg >= 270) & (aspect_deg <= 360)] = 4  # NW

        elif num_classes == 8:
            # Bins centred on N/NE/E/SE/S/SW/W/NW; North wraps through 0°.
            classified[(aspect_deg >= 0)     & (aspect_deg <  22.5)]  = 1  # N lower
            classified[(aspect_deg >= 22.5)  & (aspect_deg <  67.5)]  = 2  # NE
            classified[(aspect_deg >= 67.5)  & (aspect_deg < 112.5)]  = 3  # E
            classified[(aspect_deg >= 112.5) & (aspect_deg < 157.5)]  = 4  # SE
            classified[(aspect_deg >= 157.5) & (aspect_deg < 202.5)]  = 5  # S
            classified[(aspect_deg >= 202.5) & (aspect_deg < 247.5)]  = 6  # SW
            classified[(aspect_deg >= 247.5) & (aspect_deg < 292.5)]  = 7  # W
            classified[(aspect_deg >= 292.5) & (aspect_deg < 337.5)]  = 8  # NW
            classified[(aspect_deg >= 337.5) & (aspect_deg <= 360)]   = 1  # N upper

        else:
            # Fallback: equal-width bins, no wrap-around correction
            class_width = 360.0 / num_classes
            for i in range(num_classes):
                lower = i * class_width
                upper = (i + 1) * class_width
                if i == num_classes - 1:
                    mask = (aspect_deg >= lower) & (aspect_deg <= upper)
                else:
                    mask = (aspect_deg >= lower) & (aspect_deg < upper)
                classified[mask] = i + 1

        # Flat areas (slope < threshold) → class 0.
        # These are real terrain pixels (valley bottoms, meadows) that receive no
        # aspect-based SW correction. Applied last so it overrides any cardinal bin.
        classified[flat_mask] = 0

        return classified

    def _discretize_by_radiation(self):
        """
        Discretize the domain based on radiation properties using MultiPolygon HRUs.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        radiation_raster = self._get_file_path("RADIATION_PATH", "attributes/radiation", "annual_radiation.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_radiation.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_radiation.png")

        radiation_class_number = int(self.config.get('RADIATION_CLASS_NUMBER'))

        if not radiation_raster.exists():
            self.logger.info("Annual radiation raster not found. Calculating radiation...")
            radiation_raster = self._calculate_annual_radiation(dem_raster, radiation_raster)
            if radiation_raster is None:
                raise ValueError("Failed to calculate annual radiation")

        gru_gdf, radiation_thresholds = self._read_and_prepare_data(gru_shapefile, radiation_raster, radiation_class_number)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, radiation_raster, radiation_thresholds, 'radiationClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Radiation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'radiationClass', 'Radiation-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _calculate_annual_radiation(self, dem_raster: Path, radiation_raster: Path) -> Path:
        self.logger.info(f"Calculating annual radiation from DEM: {dem_raster}")
        
        try:
            with rasterio.open(dem_raster) as src:
                dem = src.read(1)
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
            
            center_lat = (bounds.bottom + bounds.top) / 2
            center_lon = (bounds.left + bounds.right) / 2
            
            # Calculate slope and aspect
            dy, dx = np.gradient(dem)
            slope = np.arctan(np.sqrt(dx*dx + dy*dy))
            aspect = np.arctan2(-dx, dy)
            
            # Create a DatetimeIndex for the entire year (daily)
            times = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D')
            
            # Create location object
            location = pvlib.location.Location(latitude=center_lat, longitude=center_lon, altitude=np.mean(dem))
            
            # Calculate solar position
            solar_position = location.get_solarposition(times=times)
            
            # Calculate clear sky radiation
            clearsky = location.get_clearsky(times=times)
            
            # Initialize the radiation array
            radiation = np.zeros_like(dem)
            
            self.logger.info("Calculating radiation for each pixel...")
            for i in range(dem.shape[0]):
                for j in range(dem.shape[1]):
                    surface_tilt = np.degrees(slope[i, j])
                    surface_azimuth = np.degrees(aspect[i, j])
                    
                    total_irrad = pvlib.irradiance.get_total_irradiance(
                        surface_tilt, surface_azimuth,
                        solar_position['apparent_zenith'], solar_position['azimuth'],
                        clearsky['dni'], clearsky['ghi'], clearsky['dhi']
                    )
                    
                    radiation[i, j] = total_irrad['poa_global'].sum()
            
            # Save the radiation raster
            radiation_raster.parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(radiation_raster, 'w', driver='GTiff',
                            height=radiation.shape[0], width=radiation.shape[1],
                            count=1, dtype=radiation.dtype,
                            crs=crs, transform=transform) as dst:
                dst.write(radiation, 1)
            
            self.logger.info(f"Radiation raster saved to: {radiation_raster}")
            return radiation_raster
        
        except Exception as e:
            self.logger.error(f"Error calculating annual radiation: {str(e)}", exc_info=True)
            return None

    def _read_and_prepare_data(self, shapefile_path, raster_path, band_size=None):
        """
        Read and prepare data with chunking for large rasters.
        
        Args:
            shapefile_path: Path to the GRU shapefile
            raster_path: Path to the raster file
            band_size: Optional band size for discretization
            
        Returns:
            tuple: (gru_gdf, thresholds) where:
                - gru_gdf is the GeoDataFrame containing GRU data
                - thresholds are the class boundaries for discretization
        """
        # Read the GRU shapefile
        gru_gdf = self._read_shapefile(shapefile_path)
        
        # Process raster in chunks
        CHUNK_SIZE = 1024  # Adjust based on available memory
        valid_data = []
        
        with rasterio.open(raster_path) as src:
            height = src.height
            width = src.width
            nodata = src.nodata
            
            self.logger.info(f"Raster info: {width}x{height} pixels, nodata={nodata}")
            
            for y in range(0, height, CHUNK_SIZE):
                for x in range(0, width, CHUNK_SIZE):
                    window = rasterio.windows.Window(x, y, 
                        min(CHUNK_SIZE, width - x),
                        min(CHUNK_SIZE, height - y))
                    chunk = src.read(1, window=window)
                    
                    # Filter out nodata values
                    if nodata is not None:
                        valid_chunk = chunk[chunk != nodata]
                    else:
                        valid_chunk = chunk[~np.isnan(chunk)] if chunk.dtype == np.float64 else chunk
                    
                    if len(valid_chunk) > 0:
                        valid_data.extend(valid_chunk.flatten())
        
        if len(valid_data) == 0:
            raise ValueError("No valid data found in raster")
        
        valid_data = np.array(valid_data)
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        self.logger.info(f"Valid data range: {data_min:.2f} to {data_max:.2f}")
        self.logger.info(f"Total valid pixels: {len(valid_data)}")
        
        # Calculate thresholds based on the data
        if band_size is not None:
            # For elevation-based or radiation-based discretization
            # Ensure thresholds cover the full data range
            min_val = data_min
            max_val = data_max
            
            # Create bands that fully cover the data range
            thresholds = np.arange(min_val, max_val + band_size, band_size)
            
            # Ensure the last threshold covers the maximum value
            if thresholds[-1] < max_val:
                thresholds = np.append(thresholds, thresholds[-1] + band_size)
            
            self.logger.info(f"Created {len(thresholds)-1} bands with size {band_size}")
            self.logger.info(f"Threshold range: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
        else:
            # For soil or land class-based discretization
            thresholds = np.unique(valid_data)
            self.logger.info(f"Found {len(thresholds)} unique classes: {thresholds}")
        
        return gru_gdf, thresholds

    def _create_multipolygon_hrus(self, gru_gdf, raster_path, thresholds, attribute_name):
        """
        Create HRUs by discretizing each GRU based on raster values within it.
        Each unique raster value within a GRU becomes an HRU (Polygon or MultiPolygon).
        
        Args:
            gru_gdf: GeoDataFrame containing GRU data
            raster_path: Path to the classification raster
            thresholds: Array of threshold values for classification
            attribute_name: Name of the attribute column
            
        Returns:
            GeoDataFrame containing HRUs
        """
        self.logger.info(f"Creating HRUs within {len(gru_gdf)} GRUs based on {attribute_name}")
        
        all_hrus = []
        hru_id_counter = 1
        
        # Process each GRU individually
        for gru_idx, gru_row in gru_gdf.iterrows():
            self.logger.info(f"Processing GRU {gru_idx + 1}/{len(gru_gdf)}")
            
            gru_geometry = gru_row.geometry
            gru_id = gru_row.get('GRU_ID', gru_idx + 1)
            
            # Extract raster data within this GRU
            with rasterio.open(raster_path) as src:
                try:
                    # Mask the raster to this GRU's geometry
                    out_image, out_transform = mask(src, [gru_geometry], crop=True, all_touched=True, filled=False)
                    out_image = out_image[0]
                    nodata_value = src.nodata
                except Exception as e:
                    self.logger.warning(f"Could not extract raster data for GRU {gru_id}: {str(e)}")
                    continue
            
            # Create mask for valid pixels
            if nodata_value is not None:
                valid_mask = out_image != nodata_value
            else:
                valid_mask = ~np.isnan(out_image) if out_image.dtype == np.float64 else np.ones_like(out_image, dtype=bool)
            
            if not np.any(valid_mask):
                self.logger.warning(f"No valid pixels found in GRU {gru_id}")
                continue
            
            # Find unique values within this GRU
            valid_values = out_image[valid_mask]
            
            if attribute_name in ['elevClass', 'radiationClass']:
                # For continuous data, classify into bands
                gru_hrus = self._create_hrus_from_bands(
                    out_image, valid_mask, out_transform, thresholds, 
                    attribute_name, gru_geometry, gru_row, hru_id_counter
                )
            else:
                # For discrete classes, use unique values
                unique_values = np.unique(valid_values)
                gru_hrus = self._create_hrus_from_classes(
                    out_image, valid_mask, out_transform, unique_values,
                    attribute_name, gru_geometry, gru_row, hru_id_counter
                )
            
            all_hrus.extend(gru_hrus)
            hru_id_counter += len(gru_hrus)
        
        self.logger.info(f"Created {len(all_hrus)} HRUs across all GRUs")
        return gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)

    def _create_hrus_from_bands(self, raster_data, valid_mask, transform, thresholds, 
                               attribute_name, gru_geometry, gru_row, start_hru_id):
        """Create HRUs from elevation/radiation bands within a single GRU."""
        hrus = []
        current_hru_id = start_hru_id
        
        for i in range(len(thresholds) - 1):
            lower, upper = thresholds[i:i+2]
            
            # Make the last band inclusive of the upper bound
            if i == len(thresholds) - 2:  # Last band
                class_mask = valid_mask & (raster_data >= lower) & (raster_data <= upper)
            else:
                class_mask = valid_mask & (raster_data >= lower) & (raster_data < upper)
            
            if np.any(class_mask):
                hru = self._create_hru_from_mask(
                    class_mask, transform, raster_data, gru_geometry,
                    gru_row, current_hru_id, attribute_name, i + 1
                )
                if hru:
                    hrus.append(hru)
                    current_hru_id += 1
        
        return hrus

    def _create_hrus_from_classes(self, raster_data, valid_mask, transform, unique_values,
                                 attribute_name, gru_geometry, gru_row, start_hru_id):
        """Create HRUs from discrete classes within a single GRU."""
        hrus = []
        current_hru_id = start_hru_id
        
        for class_value in unique_values:
            class_mask = valid_mask & (raster_data == class_value)
            
            if np.any(class_mask):
                hru = self._create_hru_from_mask(
                    class_mask, transform, raster_data, gru_geometry,
                    gru_row, current_hru_id, attribute_name, class_value
                )
                if hru:
                    hrus.append(hru)
                    current_hru_id += 1
        
        return hrus

    def _create_hru_from_mask(self, class_mask, transform, raster_data, gru_geometry,
                             gru_row, hru_id, attribute_name, class_value):
        """Create a single HRU from a class mask within a GRU."""
        try:
            # Extract shapes from the mask
            shapes = list(rasterio.features.shapes(
                class_mask.astype(np.uint8), 
                mask=class_mask, 
                transform=transform,
                connectivity=4
            ))
            
            if not shapes:
                return None
            
            # Create polygons from shapes
            polygons = []
            for shp, _ in shapes:
                try:
                    geom = shape(shp)
                    if geom.is_valid and not geom.is_empty and geom.area > 0:
                        polygons.append(geom)
                except Exception:
                    continue
            
            if not polygons:
                return None
            
            # Create final geometry (naturally Polygon or MultiPolygon)
            if len(polygons) == 1:
                final_geometry = polygons[0]
            else:
                # This naturally creates a MultiPolygon if there are disconnected areas
                final_geometry = MultiPolygon(polygons)
            
            # Clean the geometry
            if not final_geometry.is_valid:
                final_geometry = final_geometry.buffer(0)
            
            if final_geometry.is_empty or not final_geometry.is_valid:
                return None
            
            # Ensure it's within the GRU boundary
            clipped_geometry = final_geometry.intersection(gru_geometry)
            
            if clipped_geometry.is_empty or not clipped_geometry.is_valid:
                return None
            
            # Calculate average attribute value
            avg_value = np.mean(raster_data[class_mask]) if np.any(class_mask) else class_value
            
            # Create HRU data
            hru_data = {
                'geometry': clipped_geometry,
                'GRU_ID': gru_row.get('GRU_ID', gru_row.name),
                'HRU_ID': hru_id,
                attribute_name: class_value,
                f'avg_{attribute_name.lower()}': avg_value,
                'hru_type': f'{attribute_name}_within_gru'
            }
            
            # Copy relevant GRU attributes (excluding geometry)
            for col in gru_row.index:
                if col not in ['geometry', 'GRU_ID'] and col not in hru_data:
                    hru_data[col] = gru_row[col]
            
            return hru_data
            
        except Exception as e:
            self.logger.warning(f"Error creating HRU for class {class_value} in GRU: {str(e)}")
            return None

    def _create_single_multipolygon_hru(self, class_mask, out_transform, domain_boundary, 
                                       class_value, out_image, attribute_name, gru_gdf):
        """
        Create a single MultiPolygon HRU from a class mask.
        
        Args:
            class_mask: Boolean mask for the class
            out_transform: Raster transform
            domain_boundary: Boundary of the domain
            class_value: Value of the class
            out_image: Original raster data
            attribute_name: Name of the attribute
            gru_gdf: Original GRU GeoDataFrame
            
        Returns:
            Dictionary representing the HRU
        """
        try:
            # Extract shapes from the mask
            shapes = list(rasterio.features.shapes(
                class_mask.astype(np.uint8), 
                mask=class_mask, 
                transform=out_transform,
                connectivity=8
            ))
            
            if not shapes:
                return None
            
            # Create polygons from shapes
            polygons = []
            for shp, _ in shapes:
                geom = shape(shp)
                if geom.is_valid and not geom.is_empty:
                    # Intersect with domain boundary to ensure it's within the domain
                    intersected = geom.intersection(domain_boundary)
                    if not intersected.is_empty:
                        if isinstance(intersected, (Polygon, MultiPolygon)):
                            polygons.append(intersected)
            
            if not polygons:
                return None
            
            # Create a single MultiPolygon from all polygons
            if len(polygons) == 1:
                multipolygon = polygons[0]
            else:
                multipolygon = MultiPolygon(polygons)
            
            # Clean the geometry
            multipolygon = multipolygon.buffer(0)  # Fix any topology issues
            
            if multipolygon.is_empty or not multipolygon.is_valid:
                return None
            
            # Calculate average attribute value
            avg_value = np.mean(out_image[class_mask])
            
            # Get a representative GRU for metadata (use the first one)
            representative_gru = gru_gdf.iloc[0]
            
            return {
                'geometry': multipolygon,
                'GRU_ID': 1,  # Single domain-wide unit
                attribute_name: class_value,
                f'avg_{attribute_name.lower()}': avg_value,
                'HRU_ID': class_value,  # Use class value as HRU ID
                'hru_type': f'{attribute_name}_multipolygon'
            }
            
        except Exception as e:
            self.logger.warning(f"Error creating MultiPolygon HRU for class {class_value}: {str(e)}")
            return None

    def _clean_and_prepare_hru_gdf(self, hru_gdf):
        """
        Clean and prepare the HRU GeoDataFrame for output.
        """
        # Ensure all geometries are valid
        hru_gdf['geometry'] = hru_gdf['geometry'].apply(self._clean_geometries)
        hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
        
        # Final check: ensure only Polygon or MultiPolygon geometries
        valid_rows = []
        for idx, row in hru_gdf.iterrows():
            geom = row['geometry']
            if isinstance(geom, (Polygon, MultiPolygon)) and geom.is_valid and not geom.is_empty:
                valid_rows.append(row)
            else:
                self.logger.warning(f"Removing HRU {idx} with invalid geometry type: {type(geom)}")
        
        if not valid_rows:
            self.logger.error("No valid HRUs after final geometry validation")
            return gpd.GeoDataFrame(columns=hru_gdf.columns, crs=hru_gdf.crs)
        
        hru_gdf = gpd.GeoDataFrame(valid_rows, crs=hru_gdf.crs)
        
        self.logger.info(f"Retained {len(hru_gdf)} HRUs after geometry validation")
        
        # Calculate areas and centroids
        self.logger.info("Calculating HRU areas and centroids")
        
        # Project to UTM for accurate area calculation
        utm_crs = hru_gdf.estimate_utm_crs()
        hru_gdf_utm = hru_gdf.to_crs(utm_crs)
        hru_gdf_utm['HRU_area'] = hru_gdf_utm.geometry.area
        
        # Calculate centroids (use representative point for MultiPolygons)
        centroids_utm = hru_gdf_utm.geometry.representative_point()
        centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
        
        hru_gdf_utm['center_lon'] = centroids_wgs84.x
        hru_gdf_utm['center_lat'] = centroids_wgs84.y
        
        # Convert back to original CRS
        hru_gdf = hru_gdf_utm.to_crs(hru_gdf.crs)
        
        # Calculate mean elevation for each HRU with proper CRS handling
        self.logger.info("Calculating mean elevation for each HRU")
        try:
            # Get CRS information
            with rasterio.open(self.dem_path) as src:
                dem_crs = src.crs
            
            shapefile_crs = hru_gdf.crs
            
            # Check if CRS match
            if dem_crs != shapefile_crs:
                self.logger.info(f"CRS mismatch detected. Reprojecting HRUs from {shapefile_crs} to {dem_crs}")
                hru_gdf_projected = hru_gdf.to_crs(dem_crs)
            else:
                hru_gdf_projected = hru_gdf.copy()
            
            # Use rasterstats with the raster file path directly (more efficient and handles CRS properly)
            zs = rasterstats.zonal_stats(
                hru_gdf_projected.geometry, 
                str(self.dem_path),  # Use file path instead of array
                stats=['mean'],
                nodata=-9999  # Explicit nodata value
            )
            hru_gdf['elev_mean'] = [item['mean'] if item['mean'] is not None else -9999 for item in zs]
            
        except Exception as e:
            self.logger.error(f"Error calculating mean elevation: {str(e)}")
            hru_gdf['elev_mean'] = -9999

        # Calculate mean slope for each HRU using circular mean (sin/cos decomposition).
        self.logger.info("Calculating mean slope for each HRU")
        try:
            with rasterio.open(self.dem_path) as src:
                dem = src.read(1).astype(np.float64)
                transform = src.transform
                dem_crs = src.crs
                nodata = src.nodata

            dem_nodata_mask = ~np.isfinite(dem)
            if nodata is not None:
                dem_nodata_mask |= (dem == nodata)
            dem[dem_nodata_mask] = np.nan

            cell_size_x = abs(transform[0])
            cell_size_y = abs(transform[4])

            # Convert geographic degrees → meters when DEM is in lat/lon.
            if dem_crs is not None and dem_crs.is_geographic:
                nrows = dem.shape[0]
                lat_top = transform.f
                lats = lat_top - (np.arange(nrows) + 0.5) * cell_size_y
                dy_m = cell_size_y * 111320.0
                dx_m = cell_size_x * 111320.0 * np.cos(np.deg2rad(lats))[:, np.newaxis]
            else:
                dy_m = cell_size_y
                dx_m = cell_size_x

            # Horn (1981) weighted 3x3 gradient kernel.
            p = np.pad(dem, 1, mode='constant', constant_values=np.nan)
            dz_dx = (
                (p[0:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
                - (p[0:-2, 0:-2] + 2 * p[1:-1, 0:-2] + p[2:, 0:-2])
            ) / (8.0 * dx_m)
            dz_dy = (
                (p[2:, 0:-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
                - (p[0:-2, 0:-2] + 2 * p[0:-2, 1:-1] + p[0:-2, 2:])
            ) / (8.0 * dy_m)

            slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
            slope_rad[dem_nodata_mask] = np.nan

            nodata_fill = -9999.0
            sin_slope = np.where(np.isfinite(slope_rad), np.sin(slope_rad), nodata_fill)
            cos_slope = np.where(np.isfinite(slope_rad), np.cos(slope_rad), nodata_fill)

            # Project HRU geometry to DEM CRS for zonal stats.
            geom_for_slope = hru_gdf.to_crs(dem_crs).geometry if dem_crs != hru_gdf.crs else hru_gdf.geometry

            zs_sin = rasterstats.zonal_stats(geom_for_slope, sin_slope, affine=transform, stats=['mean'], nodata=nodata_fill)
            zs_cos = rasterstats.zonal_stats(geom_for_slope, cos_slope, affine=transform, stats=['mean'], nodata=nodata_fill)

            mean_sin = np.array([s['mean'] if s['mean'] is not None else 0.0 for s in zs_sin])
            mean_cos = np.array([s['mean'] if s['mean'] is not None else 1.0 for s in zs_cos])
            hru_gdf['slope_mean_deg'] = np.clip(np.degrees(np.arctan2(mean_sin, mean_cos)), 0, 90)

        except Exception as e:
            self.logger.error(f"Error calculating mean slope: {str(e)}")
            hru_gdf['slope_mean_deg'] = 15.0

        # Merge HRUs below MIN_HRU_SIZE into their nearest-elevation neighbour.
        min_area_km2 = float(self.config.get('MIN_HRU_SIZE', 0))
        min_area_m2 = min_area_km2 * 1e6
        if min_area_m2 > 0:
            small_mask = hru_gdf['HRU_area'] < min_area_m2
            n_small = int(small_mask.sum())
            if n_small > 0:
                self.logger.info(
                    f"Merging {n_small} HRU(s) smaller than {min_area_km2} km² "
                    f"into nearest-elevation neighbour"
                )
                small_idxs = hru_gdf.index[small_mask].tolist()
                large_gdf = hru_gdf[~small_mask].copy()

                for sidx in small_idxs:
                    row = hru_gdf.loc[sidx]
                    small_elev = row['elev_mean'] if row['elev_mean'] != -9999 else large_gdf['elev_mean'].median()
                    small_area = row['HRU_area']

                    nearest = (large_gdf['elev_mean'] - small_elev).abs().idxmin()
                    large_area = large_gdf.at[nearest, 'HRU_area']
                    large_elev = large_gdf.at[nearest, 'elev_mean']
                    total_area = large_area + small_area

                    merged_geometry = large_gdf.at[nearest, 'geometry'].union(row['geometry'])
                    merged_geometry = self._clean_geometries(merged_geometry)
                    if merged_geometry is None:
                        # Keep the small HRU if union produced a non-polygon geometry.
                        self.logger.warning(
                            f"Could not safely merge small HRU {sidx}; retaining as standalone HRU"
                        )
                        large_gdf = pd.concat([large_gdf, hru_gdf.loc[[sidx]]])
                        continue

                    large_gdf.at[nearest, 'geometry'] = merged_geometry
                    large_gdf.at[nearest, 'HRU_area'] = total_area
                    if large_elev != -9999:
                        large_gdf.at[nearest, 'elev_mean'] = (
                            (large_elev * large_area + small_elev * small_area) / total_area
                        )

                hru_gdf = large_gdf.reset_index(drop=True)
                self.logger.info(f"After merging small HRUs: {len(hru_gdf)} HRUs remain")

        # Merges can produce GeometryCollections with non-polygon parts.
        # Enforce polygon-only geometries again before sorting and export.
        hru_gdf['geometry'] = hru_gdf['geometry'].apply(self._clean_geometries)
        dropped_after_merge = int(hru_gdf['geometry'].isna().sum())
        if dropped_after_merge > 0:
            self.logger.warning(
                f"Dropped {dropped_after_merge} HRU(s) after post-merge geometry cleanup"
            )
        hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()].copy()
        if hru_gdf.empty:
            self.logger.error("No valid HRUs remain after post-merge geometry cleanup")
            return hru_gdf

        # Re-assign sequential HRU IDs using configurable ordering.
        # Default behavior is elevation-descending so HRU_ID=1 is the highest HRU.
        if 'HRU_ID' in hru_gdf.columns:
            original_ids = pd.to_numeric(hru_gdf['HRU_ID'], errors='coerce')
        else:
            original_ids = pd.Series(np.arange(1, len(hru_gdf) + 1), index=hru_gdf.index)

        fallback_ids = pd.Series(np.arange(1, len(hru_gdf) + 1), index=hru_gdf.index)
        hru_gdf['_original_hru_id'] = original_ids.fillna(fallback_ids).astype(int)

        order_strategy = str(self.config.get('HRU_ID_ORDER', 'elevation_desc')).strip().lower()
        if order_strategy == 'elevation_desc':
            sort_columns = ['elev_mean']
            ascending = [False]
            if 'GRU_ID' in hru_gdf.columns:
                sort_columns.append('GRU_ID')
                ascending.append(True)
            sort_columns.append('_original_hru_id')
            ascending.append(True)
            hru_gdf = hru_gdf.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
        elif order_strategy in ['elevation_aspect', 'elev_aspect', 'elevation_then_aspect']:
            sort_columns = []
            ascending = []

            # Resolve aspect class column once for optional filtering + ordering.
            aspect_sort_col = None
            if 'aspectClass' in hru_gdf.columns:
                aspect_sort_col = 'aspectClass'
            elif 'aspectClas' in hru_gdf.columns:
                aspect_sort_col = 'aspectClas'

            # Optional: remove flat-aspect HRUs (class 0) so each elevation band has
            # exactly four directional HRUs in the requested sequence.
            exclude_flat_aspect = bool(self.config.get('ASPECT_EXCLUDE_FLAT_HRUS', False))
            if exclude_flat_aspect and aspect_sort_col is not None:
                hru_gdf[aspect_sort_col] = pd.to_numeric(hru_gdf[aspect_sort_col], errors='coerce')
                before_count = len(hru_gdf)
                hru_gdf = hru_gdf[hru_gdf[aspect_sort_col] != 0].copy()
                removed_count = before_count - len(hru_gdf)
                if removed_count > 0:
                    self.logger.info(
                        f"Removed {removed_count} flat-aspect HRUs (class 0) prior to HRU_ID assignment"
                    )

            # Elevation-first ordering for combined HRUs:
            # highest elevation bands first, then aspect class within each band.
            if 'elevClass' in hru_gdf.columns:
                hru_gdf['elevClass'] = pd.to_numeric(hru_gdf['elevClass'], errors='coerce')
                sort_columns.append('elevClass')
                ascending.append(False)
            elif 'elev_mean' in hru_gdf.columns:
                sort_columns.append('elev_mean')
                ascending.append(False)

            if aspect_sort_col is not None:
                hru_gdf[aspect_sort_col] = pd.to_numeric(hru_gdf[aspect_sort_col], errors='coerce')

                # Default 4-class directional order requested for East River workflow:
                # 1=N, 2=E, 4=W, 3=S (class 0 flat is sorted last unless excluded).
                aspect_order_cfg = self.config.get('HRU_ASPECT_CLASS_ORDER', [1, 2, 4, 3, 0])
                if isinstance(aspect_order_cfg, str):
                    parsed_order = [item.strip() for item in aspect_order_cfg.split(',') if item.strip()]
                    try:
                        aspect_order = [int(item) for item in parsed_order]
                    except ValueError:
                        self.logger.warning(
                            f"Invalid HRU_ASPECT_CLASS_ORDER='{aspect_order_cfg}', using default [1,2,4,3,0]"
                        )
                        aspect_order = [1, 2, 4, 3, 0]
                elif isinstance(aspect_order_cfg, (list, tuple)):
                    try:
                        aspect_order = [int(item) for item in aspect_order_cfg]
                    except (TypeError, ValueError):
                        self.logger.warning(
                            f"Invalid HRU_ASPECT_CLASS_ORDER='{aspect_order_cfg}', using default [1,2,4,3,0]"
                        )
                        aspect_order = [1, 2, 4, 3, 0]
                else:
                    aspect_order = [1, 2, 4, 3, 0]

                aspect_rank_map = {cls: rank for rank, cls in enumerate(aspect_order, start=1)}
                hru_gdf['_aspect_rank'] = hru_gdf[aspect_sort_col].map(aspect_rank_map).fillna(999).astype(int)

                sort_columns.append('_aspect_rank')
                ascending.append(True)
                # Keep deterministic numeric ordering as tie-breaker.
                sort_columns.append(aspect_sort_col)
                ascending.append(True)

            if 'GRU_ID' in hru_gdf.columns:
                sort_columns.append('GRU_ID')
                ascending.append(True)

            sort_columns.append('_original_hru_id')
            ascending.append(True)
            hru_gdf = hru_gdf.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
        elif order_strategy in ['elevation_tpi', 'elev_tpi', 'elevation_then_tpi']:
            sort_columns = []
            ascending = []

            # Elevation-first ordering for combined HRUs:
            # highest elevation bands first, then TPI class within each band.
            if 'elevClass' in hru_gdf.columns:
                hru_gdf['elevClass'] = pd.to_numeric(hru_gdf['elevClass'], errors='coerce')
                sort_columns.append('elevClass')
                ascending.append(False)
            elif 'elev_mean' in hru_gdf.columns:
                sort_columns.append('elev_mean')
                ascending.append(False)

            tpi_sort_col = 'tpiClass' if 'tpiClass' in hru_gdf.columns else None
            if tpi_sort_col is not None:
                hru_gdf[tpi_sort_col] = pd.to_numeric(hru_gdf[tpi_sort_col], errors='coerce')

                # Default hydrologic ordering within an elevation band:
                # ridge -> upper_slope -> middle_slope -> flats -> lower_slope -> valley.
                # Valley is expected to drain toward the stream outlet.
                tpi_order_cfg = self.config.get('HRU_TPI_CLASS_ORDER', [1, 2, 3, 4, 5, 6])
                tpi_label_map = {
                    'ridge': 1,
                    'upper_slope': 2,
                    'middle_slope': 3,
                    'flats': 4,
                    'lower_slope': 5,
                    'valley': 6,
                }

                def _parse_tpi_order(order_cfg):
                    if isinstance(order_cfg, str):
                        tokens = [item.strip().lower() for item in order_cfg.split(',') if item.strip()]
                    elif isinstance(order_cfg, (list, tuple)):
                        tokens = [str(item).strip().lower() for item in order_cfg]
                    else:
                        tokens = []

                    parsed = []
                    for token in tokens:
                        if token in tpi_label_map:
                            parsed.append(tpi_label_map[token])
                            continue
                        try:
                            parsed.append(int(token))
                        except ValueError:
                            return [1, 2, 3, 4, 5, 6]

                    # Keep unique values while preserving order.
                    deduped = []
                    for cls in parsed:
                        if cls not in deduped:
                            deduped.append(cls)

                    # Enforce presence of all expected classes.
                    for cls in [1, 2, 3, 4, 5, 6]:
                        if cls not in deduped:
                            deduped.append(cls)
                    return deduped

                tpi_order = _parse_tpi_order(tpi_order_cfg)
                tpi_rank_map = {cls: rank for rank, cls in enumerate(tpi_order, start=1)}
                hru_gdf['_tpi_rank'] = hru_gdf[tpi_sort_col].map(tpi_rank_map).fillna(999).astype(int)

                sort_columns.append('_tpi_rank')
                ascending.append(True)
                # Keep deterministic numeric ordering as tie-breaker.
                sort_columns.append(tpi_sort_col)
                ascending.append(True)

            if 'GRU_ID' in hru_gdf.columns:
                sort_columns.append('GRU_ID')
                ascending.append(True)

            sort_columns.append('_original_hru_id')
            ascending.append(True)
            hru_gdf = hru_gdf.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
        elif order_strategy in ['original', 'input']:
            sort_columns = []
            ascending = []
            if 'GRU_ID' in hru_gdf.columns:
                sort_columns.append('GRU_ID')
                ascending.append(True)
            sort_columns.append('_original_hru_id')
            ascending.append(True)
            hru_gdf = hru_gdf.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
        else:
            self.logger.warning(
                f"Unknown HRU_ID_ORDER='{order_strategy}', using 'elevation_desc'"
            )
            sort_columns = ['elev_mean']
            ascending = [False]
            if 'GRU_ID' in hru_gdf.columns:
                sort_columns.append('GRU_ID')
                ascending.append(True)
            sort_columns.append('_original_hru_id')
            ascending.append(True)
            hru_gdf = hru_gdf.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

        old_ids_sorted = hru_gdf['_original_hru_id'].astype(int).tolist()
        hru_gdf['HRU_ID'] = range(1, len(hru_gdf) + 1)
        mapping_preview = ', '.join(
            [f"{old}->{new}" for new, old in list(enumerate(old_ids_sorted, start=1))[:10]]
        )
        self.logger.info(
            f"Assigned HRU_ID using '{order_strategy}' ordering; old->new preview: {mapping_preview}"
        )
        hru_gdf = hru_gdf.drop(columns=['_original_hru_id', '_aspect_rank', '_tpi_rank'], errors='ignore')
        
        return hru_gdf

    def _clean_geometries(self, geometry):
        """Clean and validate geometries, ensuring only Polygon or MultiPolygon."""
        if geometry is None or geometry.is_empty:
            return None
        
        try:
            # Handle GeometryCollection - extract only Polygons
            from shapely.geometry import GeometryCollection
            if isinstance(geometry, GeometryCollection):
                polygons = []
                for geom in geometry.geoms:
                    if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty:
                        polygons.append(geom)
                    elif isinstance(geom, MultiPolygon):
                        for poly in geom.geoms:
                            if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                                polygons.append(poly)
                
                if not polygons:
                    return None
                elif len(polygons) == 1:
                    geometry = polygons[0]
                else:
                    geometry = MultiPolygon(polygons)
            
            # Ensure we have a valid Polygon or MultiPolygon
            if not isinstance(geometry, (Polygon, MultiPolygon)):
                return None
            
            # Fix invalid geometries
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
                
                # Check again after buffer
                if not isinstance(geometry, (Polygon, MultiPolygon)):
                    return None
            
            return geometry if geometry.is_valid and not geometry.is_empty else None
            
        except Exception as e:
            self.logger.debug(f"Error cleaning geometry: {str(e)}")
            return None

    def _plot_hrus(self, hru_gdf, output_file, class_column, title):
        """Plot HRUs with appropriate coloring."""
        fig, ax = plt.subplots(figsize=(15, 15))

        try:
            if class_column == 'radiationClass':
                # Use the average radiation value for plotting
                if 'avg_radiationclass' in hru_gdf.columns:
                    hru_gdf.plot(column='avg_radiationclass', cmap='viridis', legend=True, ax=ax)
                else:
                    hru_gdf.plot(column=class_column, cmap='viridis', legend=True, ax=ax)
            elif 'combined_' in class_column:
                # For combined attributes, use qualitative colormap
                hru_gdf.plot(column=class_column, cmap='tab20', legend=False, ax=ax)
            else:
                # Use a qualitative colormap for other class types
                hru_gdf.plot(column=class_column, cmap='tab20', legend=True, ax=ax)
                
        except Exception as e:
            self.logger.warning(f"Error plotting with column {class_column}: {str(e)}")
            # Fallback: plot without legend
            hru_gdf.plot(ax=ax, alpha=0.7)
        
        ax.set_title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Create the directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"HRU plot saved to {output_file}")

    def _read_shapefile(self, shapefile_path):
        """
        Read a shapefile and return it as a GeoDataFrame.

        Args:
            shapefile_path (str or Path): Path to the shapefile.

        Returns:
            gpd.GeoDataFrame: The shapefile content as a GeoDataFrame.
        """
        shapefile_path = Path(shapefile_path)

        if not shapefile_path.exists():
            # The configured path doesn't exist — try the project's own river_basins dir.
            # This handles the common case where RIVER_BASINS_PATH points to a base domain
            # that hasn't been delineated yet, but the elevXxx project already has a copy.
            local_rb_dir = self.project_dir / "shapefiles" / "river_basins"
            fallback = None
            if local_rb_dir.exists():
                candidates = sorted(local_rb_dir.glob("*.shp"))
                if candidates:
                    # Prefer files that contain 'lumped' or 'riverBasins' in the name.
                    preferred = [p for p in candidates
                                 if "lumped" in p.name.lower() or "riverbasins" in p.name.lower()]
                    fallback = preferred[0] if preferred else candidates[0]

            if fallback is not None:
                self.logger.warning(
                    f"Configured shapefile not found: {shapefile_path}\n"
                    f"  Falling back to: {fallback}"
                )
                shapefile_path = fallback
            else:
                self.logger.error(f"Error reading shapefile {shapefile_path}: No such file or directory")
                raise FileNotFoundError(
                    f"{shapefile_path}: No such file or directory\n"
                    f"  Also checked: {local_rb_dir}"
                )

        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs is None:
                self.logger.warning(f"CRS is not defined for {shapefile_path}. Setting to EPSG:4326.")
                gdf = gdf.set_crs("EPSG:4326")
            return gdf
        except Exception as e:
            self.logger.error(f"Error reading shapefile {shapefile_path}: {str(e)}")
            raise

    def _get_file_path(self, file_type, file_def_path, file_name):
        """
        Construct file paths based on configuration.

        Args:
            file_type (str): Type of the file (used as a key in config).
            file_def_path (str): Default path relative to project directory.
            file_name (str): Name of the file.

        Returns:
            Path: Constructed file path.
        """
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))