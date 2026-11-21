"""DRIFT — Drought Regions Identified From the space–time component Tree.

Detect, characterize and evaluate space–time drought events selected from the component tree of a deficit field.
"""
from . import baselines, confidence, indices, metrics, significance, synthetic
from .core import cell_area_km2, detect, event_table, merge_branches, noise_persistence, select_floor, smooth

__version__ = "0.1.1"
__all__ = ["detect", "smooth", "select_floor", "event_table", "merge_branches", "noise_persistence", "cell_area_km2",
           "baselines", "confidence", "indices", "metrics", "significance", "synthetic", "__version__"]
