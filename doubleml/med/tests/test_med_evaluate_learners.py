import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data

pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")

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
def test_ml_G_is_evaluated_on_the_observations_of_its_treatment_level(small_data, fitted_med):
    """ml_G estimates E[Y|D=d,M,X], so it is compared with the outcome of the observations with D = d only."""
    treated = small_data.d == 1
    score = fitted_med.evaluate_learners(learners=["ml_G"], metric=rmse)["ml_G"]

    for i_rep in range(N_REP):
        expected = rmse(small_data.y[treated], fitted_med.predictions["ml_G"][treated, i_rep, 0])
        np.testing.assert_allclose(score[i_rep, 0], expected)


@pytest.mark.ci
def test_ml_nested_g_is_evaluated_against_the_predictions_of_ml_G(small_data, fitted_med):
    """ml_nested_g has no observed outcome. It is compared with the predictions of ml_G for the observations with D = 1-d."""
    untreated = small_data.d == 0
    score = fitted_med.evaluate_learners(learners=["ml_nested_g"], metric=rmse)["ml_nested_g"]

    assert np.all(np.isfinite(score))
    for i_rep in range(N_REP):
        nested = fitted_med.predictions["ml_nested_g"][untreated, i_rep, 0]
        expected = rmse(fitted_med.predictions["ml_G"][untreated, i_rep, 0], nested)
        np.testing.assert_allclose(score[i_rep, 0], expected)


@pytest.mark.ci
def test_inner_predictions_of_ml_G_have_no_targets_for_the_other_treatment_level(small_data, fitted_med):
    """The outcome of an observation with D = 1-d is not a target of ml_G, also for the inner predictions."""
    untreated = small_data.d == 0
    for name in [name for name in fitted_med.params_names if name.startswith("ml_G_inner_")]:
        assert np.isnan(fitted_med.nuisance_targets[name][untreated]).all(), f"{name} has targets for D = 1-d."


@pytest.mark.ci
@pytest.mark.parametrize("outcome", ["factual", "counterfactual"])
@pytest.mark.parametrize("double_sample_splitting, n_folds_inner", SPLITTING_CASES, ids=SPLITTING_IDS)
def test_fit_does_not_warn_about_the_nuisance_loss(small_data, outcome, double_sample_splitting, n_folds_inner):
    """Learners without targets are skipped when the loss is computed, no 'mean of empty slice' warnings."""
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
