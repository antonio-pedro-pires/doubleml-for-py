"""
The :mod:`doubleml.med` module implements double machine learning estimates for causal mediation analysis.
"""

from .med import DoubleMLMED
from .meds import DoubleMLMEDS

__all__ = [
    "DoubleMLMED",
    "DoubleMLMEDS",
]
