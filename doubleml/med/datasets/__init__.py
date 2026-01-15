"""
The :mod:`doubleml.med.datasets` module implements data generating processes for causal mediation analysis models.
"""

from .med_dataset import make_med_data

__all__ = [
    "make_med_data",
]
