import warnings

import numpy as np
import pandas as pd
from sklearn.utils import check_X_y

from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import (
    _check_binary_predictions,
    _check_finite_predictions,
    _check_is_propensity,
    _check_score,
    _check_trimming,
    _check_weights,
)

class DoubleMLMED(LinearScoreMixin, DoubleML):
    """ Double machine learning for causal mediation analysis.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    n_folds : int
        Number of folds.
        Default is ``3``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``5e-2``.

    """

    def __init__(
            self,
            obj_dml_data,
            ml_g,
            ml_m,
            treatment_level,
            n_folds=5,
            n_rep=1,
            score="",
            weights=None,
            normalize_ipw=False,
            trimming_rule="truncate",
            trimming_threshold=1e-2,
            draw_sample_splitting=True,
    ):
        super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting)