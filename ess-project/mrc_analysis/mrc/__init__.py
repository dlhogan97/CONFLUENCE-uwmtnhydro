"""Master recession curve (MRC) analysis package.

A config-driven, modular pipeline for identifying baseflow recessions in a long
daily streamflow record, constructing master recession curves via an automated
matching strip, and testing whether the recession constant is stationary.

Modules
-------
config       : typed configuration (dataclasses) + YAML loader
io           : streamflow / forcing loading, gap detection, synthetic data
filter       : Lyne-Hollick recursive digital baseflow filter
recession_id : recession segment detection and screening
mrc          : per-recession tail fit + automated matching-strip master curve
temporal     : sliding-window / per-year MRC series + Mann-Kendall trend test
plotting     : diagnostic figures and optional interactive QC viewer
cli          : end-to-end orchestrator
"""

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
__version__ = "0.1.0"
