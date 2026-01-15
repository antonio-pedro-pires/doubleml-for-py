"""
The :mod:`doubleml.data` module implements data classes for double machine learning.
"""

from .base_data import DoubleMLData
from .did_data import DoubleMLDIDData
from .med_data import DoubleMLMEDData
from .panel_data import DoubleMLPanelData

__all__ = [
    "DoubleMLData",
    "DoubleMLClusterData",
    "DoubleMLPanelData",
    "DoubleMLRDDData",
    "DoubleMLSSMData",
    "DoubleMLMEDData",
]
