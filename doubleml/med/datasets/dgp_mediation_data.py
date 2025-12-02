import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from doubleml.data import DoubleMLMediationData
from doubleml.utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_mediation_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_mediation_data_alias = _get_dml_mediation_data_alias()


def make_med_data(n_obs=1000, dim_x=20, return_type="DoubleMLMediationData", **kwargs):
    """
    Generates data from a mediation analysis model used in  Helmut Farbmacher et al. (2022) for Table 1.
    The data generating process is defined as

    .. math::
        Y = 0.5D + 0.5M + 0.5DM + X'\\beta + U,

        M = 1\\left\\lbrace 0.5D + X'\\beta + V > 0 \\right\\rbrace,

        D = 1\\left\\lbrace X'\\beta + W > 0 \\right\\rbrace,

    with covariates :math:`X \\sim \\mathcal{N}(0, \\Sigma)`, where :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{ij} = 0.5^{|i-j|}`,

    with a coefficients vector :math:`\\beta` of dimension dim_x with the i-th element equal to :math:`\\frac{b}{i^2}`

    with errors :math:`U, V, W \\sim \\mathcal{N}(0, 1)` independent of each other and of X.

    The mediation variable and the treatment variable are both binary and one-dimensional.

    Parameters
    ----------
    n_obs :
        The number of observations to simulate.
    dim_x :
        The number of covariates.
    return_type :
        If ``'DoubleMLData'`` or ``DoubleMLData``, returns a ``DoubleMLMediationData`` object.

        If ``'DataFrame'``, ``'pd.DataFrame'`` or ``pd.DataFrame``, returns a ``pd.DataFrame``.

        If ``'array'``, ``'np.ndarray'``, ``'np.array'`` or ``np.ndarray``, returns ``np.ndarray``'s ``(x, y, d)``.
    **kwargs
        Additional keyword arguments to set non-default values for the parameters
        :math:`b=0.3`
    References
    ----------

    Helmut Farbmacher , Martin Huber , Lukáš Lafférs , Henrika Langen , Martin Spindler
    The Econometrics Journal, Volume 25, Issue 2, May 2022, Pages 277–300, https://doi.org/10.1093/ectj/utac003
    """

    # Draw beta vector
    b = kwargs.get("b", 0.3)
    beta = [b / (i**2) for i in range(1, dim_x + 1)]

    # Draw covariates X.

    # Each entry ij in the covariance matrix equals 0.5^(abs(i-j)). which is a Toeplitz matrix.
    covmat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(mean=np.zeros(dim_x), cov=covmat, size=[n_obs])
    # Draw errors
    u = np.random.standard_normal(
        size=[
            n_obs,
        ]
    )
    v = np.random.standard_normal(
        size=[
            n_obs,
        ]
    )
    w = np.random.standard_normal(
        size=[
            n_obs,
        ]
    )

    # Generate d, m, y vectors
    d = 1.0 * (x.dot(beta) + w > 0)
    m = 1.0 * (0.5 * d + x.dot(beta) + v > 0)
    y = 0.5 * d + 0.5 * m + 0.5 * m * d + u

    if return_type in _array_alias:
        return x, y, d, m
    elif return_type in _data_frame_alias + _dml_mediation_data_alias:
        x_cols = [f"X{i + 1}" for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d, m)), columns=x_cols + ["y", "d", "m"])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLMediationData(data, "y", "d", "m", x_cols)
    else:
        raise ValueError("Invalid return_type.")
