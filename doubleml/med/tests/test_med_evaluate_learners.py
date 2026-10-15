"""Test evaluate_learners and the nuisance losses of DoubleMLMED."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data

N_FOLDS = 3
N_REP = 2
SPLITTING_CASES = [(True, 2), (True, 1), (False, 2)]
SPLITTING_IDS = ["double_split", "two_subsamples", "single_split"]


def rmse(y_true, y_pred):
    """Root mean squared error that ignores the observations without a target."""
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    available = ~np.isnan(y_true)
    return np.sqrt(np.mean((y_true[available] - y_pred[available]) ** 2))


@pytest.fixture(scope="module")
def small_data():
    np.random.seed(3141)
    return make_med_data(n_obs=300, dim_x=5)


@pytest.fixture(scope="module", params=SPLITTING_CASES, ids=SPLITTING_IDS)
def fitted_med(small_data, request):
    double_sample_splitting, n_folds_inner = request.param
    np.random.seed(3141)
    med_obj = DoubleMLMED(
        obj_dml_data=small_data,
        treatment_level=1,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_G=LinearRegression(),
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=N_FOLDS,
        n_rep=N_REP,
        n_folds_inner=n_folds_inner,
        double_sample_splitting=double_sample_splitting,
    )
    return med_obj.fit()


@pytest.mark.ci
def test_ml_G_evaluated_on_its_treatment_level(small_data, fitted_med):
    """Test that ml_G is compared with the outcome of the observations with D = d only."""
    treated = small_data.d == 1
    score = fitted_med.evaluate_learners(learners=["ml_G"], metric=rmse)["ml_G"]

    for i_rep in range(N_REP):
        expected = rmse(small_data.y[treated], fitted_med.predictions["ml_G"][treated, i_rep, 0])
        np.testing.assert_allclose(score[i_rep, 0], expected)


@pytest.mark.ci
def test_ml_nested_g_evaluated_against_ml_G(small_data, fitted_med):
    """Test that ml_nested_g is compared with the predictions of ml_G for the observations with D = 1-d."""
    untreated = small_data.d == 0
    score = fitted_med.evaluate_learners(learners=["ml_nested_g"], metric=rmse)["ml_nested_g"]

    assert np.all(np.isfinite(score))
    for i_rep in range(N_REP):
        nested = fitted_med.predictions["ml_nested_g"][untreated, i_rep, 0]
        expected = rmse(fitted_med.predictions["ml_G"][untreated, i_rep, 0], nested)
        np.testing.assert_allclose(score[i_rep, 0], expected)


@pytest.mark.ci
def test_ml_G_inner_targets_only_for_own_level(small_data, fitted_med):
    """Test that the inner predictions of ml_G have no target for observations with D = 1-d."""
    untreated = small_data.d == 0
    for name in [name for name in fitted_med.params_names if name.startswith("ml_G_inner_")]:
        assert np.isnan(fitted_med.nuisance_targets[name][untreated]).all(), f"{name} has targets for D = 1-d."


@pytest.mark.ci
@pytest.mark.parametrize("outcome", ["factual", "counterfactual"])
@pytest.mark.parametrize("double_sample_splitting, n_folds_inner", SPLITTING_CASES, ids=SPLITTING_IDS)
def test_fit_does_not_warn_about_the_nuisance_loss(small_data, outcome, double_sample_splitting, n_folds_inner):
    """Test that fit skips learners without targets when computing the loss and raises no 'mean of empty slice' warning."""
    np.random.seed(3141)
    med_obj = DoubleMLMED(
        obj_dml_data=small_data,
        treatment_level=1,
        outcome=outcome,
        ml_m=LogisticRegression(),
        ml_g=LinearRegression(),
        ml_G=LinearRegression(),
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=N_FOLDS,
        n_folds_inner=n_folds_inner,
        double_sample_splitting=double_sample_splitting,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        med_obj.fit()
