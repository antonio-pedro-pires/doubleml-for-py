"""Test that nuisance parameters reach the learners of DoubleMLMED."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data
from doubleml.tests._utils_tune_optuna import _basic_optuna_settings

N_FOLDS = 3
SPLITTING_CASES = [(True, 2), (True, 1), (False, 2)]
SPLITTING_IDS = ["double_split", "two_subsamples", "single_split"]

# parameters that change the fit of the learner and are valid for the learner type
REGRESSOR_PARAMS = {"fit_intercept": False}
CLASSIFIER_PARAMS = {"C": 0.01}
LEARNER_PARAMS = {
    "ml_g": REGRESSOR_PARAMS,
    "ml_G": REGRESSOR_PARAMS,
    "ml_nested_g": REGRESSOR_PARAMS,
    "ml_m": CLASSIFIER_PARAMS,
    "ml_M": CLASSIFIER_PARAMS,
}


@pytest.fixture(scope="module")
def small_data():
    np.random.seed(3141)
    return make_med_data(n_obs=300, dim_x=5)


def _learners(params_in_constructor=None):
    learners = {
        "ml_g": LinearRegression(),
        "ml_G": LinearRegression(),
        "ml_nested_g": LinearRegression(),
        "ml_m": LogisticRegression(),
        "ml_M": LogisticRegression(),
    }
    for name, params in (params_in_constructor or {}).items():
        learners[name].set_params(**params)
    return learners


def _med_obj(data, outcome, double_sample_splitting, n_folds_inner, params_in_constructor=None):
    np.random.seed(3141)
    return DoubleMLMED(
        obj_dml_data=data,
        treatment_level=1,
        outcome=outcome,
        n_folds=N_FOLDS,
        n_folds_inner=n_folds_inner,
        double_sample_splitting=double_sample_splitting,
        **_learners(params_in_constructor),
    )


@pytest.mark.ci
def test_set_ml_nuisance_params_stores_the_parameters(small_data):
    """Test that set_ml_nuisance_params stores the parameters once for each outer fold."""
    med_obj = _med_obj(small_data, "counterfactual", True, 2)
    assert med_obj.params["ml_G"]["d"] == [None]

    med_obj.set_ml_nuisance_params("ml_G", "d", {"fit_intercept": False})

    assert med_obj.params["ml_G"]["d"] == [[{"fit_intercept": False}] * N_FOLDS]
    assert med_obj.params["ml_nested_g"]["d"] == [None]


@pytest.mark.ci
@pytest.mark.parametrize("double_sample_splitting, n_folds_inner", SPLITTING_CASES, ids=SPLITTING_IDS)
@pytest.mark.parametrize(
    "outcome, learner",
    [
        ("factual", "ml_g"),
        ("factual", "ml_m"),
        ("counterfactual", "ml_G"),
        ("counterfactual", "ml_nested_g"),
        ("counterfactual", "ml_m"),
        ("counterfactual", "ml_M"),
    ],
)
def test_set_ml_nuisance_params_reach_the_learner(small_data, outcome, learner, double_sample_splitting, n_folds_inner):
    """Test that parameters set with set_ml_nuisance_params give the same fit as parameters set on the learner itself."""
    params = LEARNER_PARAMS[learner]

    in_constructor = _med_obj(small_data, outcome, double_sample_splitting, n_folds_inner, {learner: params}).fit()
    via_setter = _med_obj(small_data, outcome, double_sample_splitting, n_folds_inner)
    via_setter.set_ml_nuisance_params(learner, "d", params)
    via_setter.fit()
    default = _med_obj(small_data, outcome, double_sample_splitting, n_folds_inner).fit()

    np.testing.assert_allclose(via_setter.coef, in_constructor.coef)
    np.testing.assert_allclose(via_setter.se, in_constructor.se)
    assert not np.allclose(via_setter.coef, default.coef), f"The parameters of {learner} did not change the estimate."


@pytest.mark.ci
def test_set_ml_nuisance_params_exceptions(small_data):
    """Test that set_ml_nuisance_params raises a ValueError for an unknown learner or treatment variable."""
    med_obj = _med_obj(small_data, "counterfactual", True, 2)

    with pytest.raises(ValueError, match="Invalid nuisance learner ml_g. Valid nuisance learner ml_m or ml_G or ml_M or"):
        med_obj.set_ml_nuisance_params("ml_g", "d", {"fit_intercept": False})
    with pytest.raises(ValueError, match="Invalid treatment variable d1. Valid treatment variable d."):
        med_obj.set_ml_nuisance_params("ml_G", "d1", {"fit_intercept": False})

    factual = _med_obj(small_data, "factual", True, 2)
    with pytest.raises(ValueError, match="Invalid nuisance learner ml_G. Valid nuisance learner ml_g or ml_m."):
        factual.set_ml_nuisance_params("ml_G", "d", {"fit_intercept": False})


@pytest.mark.ci
@pytest.mark.parametrize("n_folds_inner", [1, 2, 3], ids=["two_subsamples", "two_inner_folds", "n_folds_inner=n_folds"])
def test_fold_params_reach_their_fold_models(small_data, n_folds_inner):
    """Test that the parameters of an outer fold are used for all inner models of that fold, whatever n_folds_inner is."""
    med_obj = _med_obj(small_data, "counterfactual", True, n_folds_inner)
    fold_params = [{"fit_intercept": False}, {"fit_intercept": True}, {"fit_intercept": False}]
    med_obj.set_ml_nuisance_params("ml_G", "d", [fold_params])
    med_obj.fit(store_models=True)

    models = med_obj.models["ml_G"]["d"][0]
    assert len(models) == N_FOLDS * n_folds_inner
    for i_fold, params in enumerate(fold_params):
        for model in models[i_fold * n_folds_inner : (i_fold + 1) * n_folds_inner]:
            assert (
                model.fit_intercept == params["fit_intercept"]
            ), f"Wrong parameters for the models of the outer fold {i_fold}."


@pytest.mark.ci
@pytest.mark.parametrize("n_folds_inner", [1, 2, 3], ids=["two_subsamples", "two_inner_folds", "n_folds_inner=n_folds"])
def test_tuned_params_are_used_in_the_fit(small_data, n_folds_inner):
    """Test that fit uses the tuned parameters after tune_ml_models(set_as_params=True), whatever n_folds_inner is."""
    np.random.seed(3141)
    med_obj = DoubleMLMED(
        obj_dml_data=small_data,
        treatment_level=1,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_G=Lasso(),
        ml_M=LogisticRegression(),
        ml_nested_g=Lasso(),
        n_folds=N_FOLDS,
        n_folds_inner=n_folds_inner,
    )
    ml_param_space = {
        "ml_G": lambda trial: {"alpha": trial.suggest_float("alpha", 0.01, 1.0)},
        "ml_nested_g": lambda trial: {"alpha": trial.suggest_float("alpha", 0.01, 1.0)},
    }
    med_obj.tune_ml_models(ml_param_space, set_as_params=True, optuna_settings=_basic_optuna_settings({"n_trials": 3}))
    tuned_alpha = med_obj.params["ml_G"]["d"][0][0]["alpha"]
    assert tuned_alpha != Lasso().alpha, "The tuning did not change alpha, so the test cannot see whether it is used."

    med_obj.fit(store_models=True)

    models = med_obj.models["ml_G"]["d"][0]
    assert len(models) == N_FOLDS * n_folds_inner
    assert all(model.alpha == tuned_alpha for model in models)
    assert all(
        model.alpha == med_obj.params["ml_nested_g"]["d"][0][0]["alpha"] for model in med_obj.models["ml_nested_g"]["d"][0]
    )
