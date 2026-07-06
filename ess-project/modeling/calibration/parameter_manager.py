#!/usr/bin/env python3
"""
parameter_manager.py — Multiplier-based parameter management for staged SUMMA calibration.

In a distributed model each HRU has spatially-derived base parameters (from soil maps,
vegetation, topography). Rather than calibrating per-HRU values (N_HRU × N_param
dimensions), we calibrate a small set of global *multipliers* that scale every HRU
simultaneously, preserving the a-priori spatial structure.

    param_hru_i = base_param_hru_i × multiplier

Usage
-----
    from parameter_manager import ParameterManager
    pm = ParameterManager(base_param_nc="/path/to/trialParams.nc")
    trial_ds = pm.apply_multipliers({"albedoDecayRate": 1.2, "k_soil": 0.8})
    pm.write_trial_params(trial_ds, "/path/to/trial/trialParams.nc")
    pm.freeze_params({"albedoDecayRate": 1.2, ...}, ["albedoDecayRate"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Physical bounds applied *after* multiplication to prevent non-physical values.
# vGn_alpha is stored negative in SUMMA — bounds are negative.
PHYSICAL_BOUNDS: Dict[str, tuple] = {
    "albedoDecayRate":               (1e5,  5e6),
    "k_soil":                        (1e-7, 1e-2),
    "vGn_alpha":                     (-3.0, -1.),
    "vGn_n":                         (1.0, 2.0),
    "qSurfScale":                    (1.0,  10.0),
    "rootingDepth":                  (0.1,  3.0),
    "theta_sat":                     (0.3,  0.6),
    "aquiferScaleFactor":            (0.01, 100.0),
    "aquiferBaseflowExp":            (0.5,  10.0),
    "aquiferBaseflowRate":           (1e-10, 1e-3),
    "zScale_TOPMODEL":               (1.0, 5.0),
    "kAnisotropic":                  (0.01, 5.0),
    "minStomatalResistance":          (1,  20.0),
    # Routing (GRU-level)
    "routingGammaShape":             (1.0,  10.0),
    "routingGammaScale":             (500.0, 172800.0),
    "frozenPrecipMultip":            (0.0,   3.0),    # widened: spatial-weighted by ASO ratios
    "tempCritRain":                 (270.0, 275.15),  # 0-3 °C in Kelvin
}

# Default multiplier search bounds for differential_evolution
MULTIPLIER_BOUNDS: Dict[str, tuple] = {
    "albedoDecayRate":           (0.1,  10.0),
    "k_soil":                    (0.1,  5.0),
    "vGn_alpha":                 (0.5,  2.0),
    "vGn_n":                     (1.0,  2.0),
    "qSurfScale":                (0.5,  3.0),
    "rootingDepth":              (0.5,  2.0),
    "theta_sat":                 (0.5,  1.5),
    "aquiferScaleFactor":        (0.1,  5.0),
    "aquiferBaseflowExp":        (0.5,  3.0),
    "aquiferBaseflowRate":       (0.1,  5.0),
    "routingGammaShape":         (0.5,  3.0),
    "routingGammaScale":         (0.5,  3.0),
    "frozenPrecipMultip":        (0.7,  1.3),
    "tempCritRain":             (0.985, 1.015),
}


class ParameterManager:
    """Read, multiply, clamp, and write SUMMA trialParams.nc files."""

    def __init__(self, base_param_nc: str | Path):
        self.base_path = Path(base_param_nc)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base trialParams.nc not found: {self.base_path}")
        self._base_ds: xr.Dataset = xr.open_dataset(self.base_path)
        logger.info("Loaded base params from %s — HRUs: %d", self.base_path,
                    self._base_ds.dims.get("hru", 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_multipliers(
        self,
        multipliers: Dict[str, float],
        param_names: Optional[List[str]] = None,
        spatial_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> xr.Dataset:
        """Return a copy of the base dataset with multipliers applied and bounds clamped.

        Parameters
        ----------
        multipliers:
            Mapping of param_name → scalar multiplier value.
        param_names:
            Subset of params to modify.  Defaults to all keys in *multipliers*.
        spatial_weights:
            Optional per-HRU weight arrays: {param_name: np.ndarray of shape (n_hru,)}.
            When provided for a parameter, the effective multiplication is::

                param_hru_i = base_hru_i × spatial_weight_i × M

            This lets the YAML config carry directional spatial information (some HRUs
            above 1, some below 1) while M carries only the global magnitude.
            Spatial weights are *not* applied to GRU-dimension parameters
            (e.g. routingGammaShape, basin__aquiferScaleFactor).
        """
        ds = self._base_ds.copy(deep=True)
        names = param_names if param_names is not None else list(multipliers.keys())

        for name in names:
            mult = multipliers.get(name, 1.0)
            if name not in ds:
                logger.warning("Parameter '%s' not found in trialParams.nc — skipping", name)
                continue

            original = ds[name].values.copy()

            # Apply per-HRU spatial weights only for HRU-dimension variables.
            # GRU-level params (routing, basin aquifer) get a plain scalar multiply.
            is_hru_var = "hru" in ds[name].dims
            if spatial_weights and name in spatial_weights and is_hru_var:
                w = spatial_weights[name]  # shape (n_hru,)
                scaled = original * w * float(mult)
                logger.debug("%s: applying spatial weights %s × M=%.4f", name, list(w), mult)
            else:
                scaled = original * float(mult)

            clamped = self._clamp(name, scaled)

            n_clamp = np.sum(clamped != scaled)
            if n_clamp > 0:
                frac = n_clamp / clamped.size
                logger.debug(
                    "%s: multiplier=%.4f clamped %d/%d values (%.0f%%)",
                    name, mult, n_clamp, clamped.size, frac * 100
                )

            ds[name].values[:] = clamped

        return ds

    def write_trial_params(self, ds: xr.Dataset, output_path: str | Path) -> None:
        """Write a modified parameter dataset to *output_path* as NetCDF-4."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out, format="NETCDF4")
        logger.debug("Wrote trial params → %s", out)

    def freeze_params(
        self,
        best_multipliers: Dict[str, float],
        param_names: List[str],
        spatial_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> "ParameterManager":
        """Permanently bake best multipliers (and spatial weights) into the base dataset.

        After freezing, the stored base values reflect ``base × weight × M`` so the
        next stage inherits the correct spatially-differentiated parameter values.

        Returns self so stages can chain:  pm = pm.freeze_params(...)
        """
        new_base = self.apply_multipliers(best_multipliers, param_names,
                                          spatial_weights=spatial_weights)
        self._base_ds = new_base
        logger.info(
            "Froze %d parameters into base: %s",
            len(param_names), param_names
        )
        return self

    def save_base(self, path: str | Path) -> None:
        """Save the current (possibly frozen) base dataset to disk."""
        self.write_trial_params(self._base_ds, path)

    def apply_calib_bounds_values(self, calib_bounds_json: str | Path) -> "ParameterManager":
        """Overwrite base parameters using `value` fields from calib-bounds JSON.

        This is optional and intended for explicit initialization control.
        Expected schema follows distributed_settings_builder output:
        - shared: [{param, value, min, max}, ...]
        - per_hru: [{param, hru_index, value, min, max}, ...]
        """
        path = Path(calib_bounds_json)
        if not path.exists():
            raise FileNotFoundError(f"Calibration bounds JSON not found: {path}")

        with path.open() as fh:
            payload = json.load(fh)

        ds = self._base_ds.copy(deep=True)
        n_updates = 0

        for row in payload.get("shared", []):
            name = row.get("param")
            if not name or name not in ds:
                continue
            value = float(row.get("value"))
            arr = np.asarray(ds[name].values)
            arr[...] = value
            ds[name].values[:] = self._clamp(name, arr)
            n_updates += arr.size

        for row in payload.get("per_hru", []):
            name = row.get("param")
            if not name or name not in ds or "hru" not in ds[name].dims:
                continue
            hru_index = int(row.get("hru_index"))
            if hru_index < 0 or hru_index >= ds.sizes.get("hru", 0):
                continue
            value = float(row.get("value"))
            arr = np.asarray(ds[name].values).copy()
            arr[hru_index] = value
            ds[name].values[:] = self._clamp(name, arr)
            n_updates += 1

        self._base_ds = ds
        logger.info("Applied %d calibration-bound initial values from %s", n_updates, path)
        return self

    @property
    def base_dataset(self) -> xr.Dataset:
        return self._base_ds

    def get_base_values(self, param_names: List[str]) -> Dict[str, np.ndarray]:
        """Return current base values for diagnostics."""
        return {
            name: self._base_ds[name].values.copy()
            for name in param_names
            if name in self._base_ds
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(name: str, values: np.ndarray) -> np.ndarray:
        if name not in PHYSICAL_BOUNDS:
            return values
        lo, hi = PHYSICAL_BOUNDS[name]
        return np.clip(values, lo, hi)

    def __repr__(self) -> str:
        n_hru = self._base_ds.dims.get("hru", "?")
        params = list(self._base_ds.data_vars)
        return f"ParameterManager(hrus={n_hru}, params={params})"
