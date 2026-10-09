import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data

pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")

N_FOLDS = 3
N_FOLDS_INNER = 2


@pytest.fixture(scope="module")
def small_data():
    np.random.seed(3141)
    return make_med_data(n_obs=300, dim_x=5)


SPLITTING_CASES = [(True, N_FOLDS_INNER), (True, 1), (False, N_FOLDS_INNER)]
SPLITTING_IDS = ["double_split", "two_subsamples", "single_split"]


def _med_obj(data, outcome, double_sample_splitting, n_folds_inner=N_FOLDS_INNER):
    np.random.seed(3141)
    return DoubleMLMED(
        obj_dml_data=data,
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


def _expected_n_models(learner, double_sample_splitting, n_folds_inner):
    if learner.startswith("ml_G_inner_"):
        # inner models trained within one outer fold, or the model of the outer fold for single sample splitting
        return n_folds_inner if double_sample_splitting else 1
    if learner == "ml_G" and double_sample_splitting:
        return N_FOLDS * n_folds_inner  # all inner models, ordered by outer fold
    return N_FOLDS


@pytest.mark.ci
@pytest.mark.parametrize("outcome", ["factual", "counterfactual"])
@pytest.mark.parametrize("double_sample_splitting, n_folds_inner", SPLITTING_CASES, ids=SPLITTING_IDS)
def test_med_store_models_true(small_data, outcome, double_sample_splitting, n_folds_inner):
    """fit(store_models=True) succeeds and stores the fitted models of every nuisance learner."""
    dml_obj = _med_obj(small_data, outcome, double_sample_splitting, n_folds_inner)
    dml_obj.fit(store_models=True)

    assert dml_obj.models is not None
    d_col = small_data.d_cols[0]
    for learner in dml_obj.params_names:
        context = (
            f"learner '{learner}' (outcome={outcome}, double_sample_splitting={double_sample_splitting},"
            f" n_folds_inner={n_folds_inner})"
        )
        stored = dml_obj.models[learner][d_col][0]
        assert stored is not None, f"No fitted models were stored for {context}."

        expected = _expected_n_models(learner, double_sample_splitting, n_folds_inner)
        assert len(stored) == expected, f"Expected {expected} stored models for {context}, got {len(stored)}."


@pytest.mark.ci
@pytest.mark.parametrize("outcome", ["factual", "counterfactual"])
def test_med_store_models_false(small_data, outcome):
    """fit(store_models=False) keeps no models."""
    dml_obj = _med_obj(small_data, outcome, double_sample_splitting=True)
    dml_obj.fit(store_models=False)

    assert dml_obj.models is None


@pytest.mark.ci
@pytest.mark.parametrize("outcome", ["factual", "counterfactual"])
def test_med_store_predictions_false(small_data, outcome):
    """fit(store_predictions=False) keeps no predictions."""
    dml_obj = _med_obj(small_data, outcome, double_sample_splitting=True)
    dml_obj.fit(store_predictions=False)

    assert dml_obj.predictions is None
