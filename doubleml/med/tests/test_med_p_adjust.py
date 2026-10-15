"""Test p_adjust of DoubleMLMED, which has a single coefficient."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data

N_OBS = 300
N_REP_BOOT = 50


@pytest.fixture(scope="module")
def fitted_med():
    np.random.seed(3141)
    dml_data = make_med_data(n_obs=N_OBS, dim_x=5)
    dml_med_obj = DoubleMLMED(dml_data, 1, "factual", ml_m=LogisticRegression(), ml_g=Lasso(), n_folds=3)
    dml_med_obj.fit()
    dml_med_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_med_obj


@pytest.mark.ci
@pytest.mark.parametrize("method", ["romano-wolf", "bonferroni", "holm"])
def test_p_adjust_single_coefficient(fitted_med, method):
    """Test that with one coefficient the adjusted p-value equals the unadjusted p-value."""
    np.random.seed(3141)
    p_adj = fitted_med.p_adjust(method=method)
    assert isinstance(p_adj, pd.DataFrame)
    assert p_adj.shape == (1, 2)
    np.testing.assert_allclose(p_adj["pval"].to_numpy(), fitted_med.pval, rtol=1e-9, atol=1e-12)
