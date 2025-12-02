"""
The :mod:`doubleml.med.datasets` module implements data generating processes for causal mediation analysis models.
"""

from .dgp_mediation_data import make_med_data

__all__ = [
    "make_med_data",
]
