import re

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doubleml import DoubleMLMED
from doubleml.tests._utils_tune_optuna import (
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)

pytestmark = pytest.mark.filterwarnings("ignore:Propensity predictions.*are close to zero or one:UserWarning")


@pytest.fixture(
    scope="module",
)
def untuned_tuned_scores(
    dml_data, binary_outcomes, binary_treats, learner_tree, optuna_params, optuna_settings, ps_processor_config
):

    if binary_outcomes == "factual":
        med_obj = DoubleMLMED(
            dml_data=dml_data,
            outcome=binary_outcomes,
            treatment_level=binary_treats,
            ml_g=learner_tree["ml_g"],
            ml_m=learner_tree["ml_m"],
            ps_processor_config=ps_processor_config,
        )
    else:
        med_obj = DoubleMLMED(
            dml_data=dml_data,
            outcome=binary_outcomes,
            treatment_level=binary_treats,
            ml_m=learner_tree["ml_m"],
            ml_G=learner_tree["ml_G"],
            ml_M=learner_tree["ml_M"],
            ml_nested_g=learner_tree["ml_nested_g"],
            ps_processor_config=ps_processor_config,
        )

    if binary_outcomes == "factual":
        ml_param_space = {
            "ml_g": optuna_params["ml_g"],
            "ml_m": optuna_params["ml_m"],
        }
    else:
        ml_param_space = {
            "ml_m": optuna_params["ml_m"],
            "ml_G": optuna_params["ml_G"],
            "ml_M": optuna_params["ml_M"],
            "ml_nested_g": optuna_params["ml_nested_g"],
        }

    np.random.seed(0)
    med_obj.fit()
    untuned_score = med_obj.evaluate_learners()

    np.random.seed(0)
    tune_res = med_obj.tune_ml_models(
        ml_param_space=ml_param_space,
        optuna_settings=optuna_settings,
        return_tune_res=True,
    )

    np.random.seed(0)
    med_obj.fit()

    np.random.seed(0)
    tuned_score = med_obj.evaluate_learners()
    return {
        "med_obj": med_obj,
        "untuned_score": untuned_score,
        "tuned_score": tuned_score,
        "tune_res": tune_res,
        "ml_param_space": ml_param_space,
    }


@pytest.mark.ci
def test_tune_ml_models(untuned_tuned_scores):
    dml_med_obj = untuned_tuned_scores["med_obj"]
    untuned_score = untuned_tuned_scores["untuned_score"]
    tuned_score = untuned_tuned_scores["tuned_score"]
    tune_res = untuned_tuned_scores["tune_res"]
    optuna_params = untuned_tuned_scores["ml_param_space"]

    params_names = dml_med_obj.params_names

    assert isinstance(tune_res[0], dict)
    assert set(tune_res[0].keys()) == set(optuna_params.keys())
    for key in params_names:
        is_inner_model = re.findall(r"[0-9]", key) != []

        if not (key == "ml_nested_g" or is_inner_model):
            assert hasattr(tune_res[0][key], "best_params")
            _assert_tree_params(tune_res[0][key].best_params)
            msg = (
                f"key: {key}, tuned score is greater or equal to the untuned score: {tuned_score[key]} >= {untuned_score[key]}"
            )
            assert tuned_score[key] < untuned_score[key], msg


class _SpyNestedTree(DecisionTreeRegressor):
    """Decision tree recording the number of features and the target values of every fit call."""

    calls = []

    def fit(self, X, y, sample_weight=None, check_input=True):
        type(self).calls.append((np.asarray(X).shape[1], np.array(y, copy=True)))
        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)


_CONSTANT_G = 42.0


@pytest.fixture(scope="module")
def nested_g_tuning_and_fit_calls(dml_data):
    """Tune only ml_nested_g with a constant ml_G, then fit, and record what the nested learner was trained on."""
    med_obj = DoubleMLMED(
        dml_data=dml_data,
        outcome="counterfactual",
        treatment_level=1,
        ml_m=DecisionTreeClassifier(random_state=123),
        ml_M=DecisionTreeClassifier(random_state=123),
        ml_G=DummyRegressor(strategy="constant", constant=_CONSTANT_G),
        ml_nested_g=_SpyNestedTree(random_state=123),
    )

    _SpyNestedTree.calls.clear()
    med_obj.tune_ml_models(
        ml_param_space={"ml_nested_g": _small_tree_params},
        optuna_settings=_basic_optuna_settings({"n_trials": 3}),
    )
    tuning_calls = list(_SpyNestedTree.calls)

    _SpyNestedTree.calls.clear()
    med_obj.fit()
    fit_calls = list(_SpyNestedTree.calls)
    return tuning_calls, fit_calls


@pytest.mark.ci
def test_tune_nested_g_uses_same_features_as_fit(dml_data, nested_g_tuning_and_fit_calls):
    """ml_nested_g is tuned on the covariates X only, i.e. on the features it is fitted on during estimation."""
    tuning_calls, fit_calls = nested_g_tuning_and_fit_calls
    n_covariates = dml_data.x.shape[1]

    assert len(tuning_calls) > 0 and len(fit_calls) > 0
    assert {n_features for n_features, _ in fit_calls} == {n_covariates}
    assert {n_features for n_features, _ in tuning_calls} == {n_covariates}


@pytest.mark.ci
def test_tune_nested_g_target_is_ml_G_predictions(nested_g_tuning_and_fit_calls):
    """ml_nested_g is tuned on predictions of ml_G (constant here), not on the outcome or on its own predictions."""
    tuning_calls, _ = nested_g_tuning_and_fit_calls

    assert len(tuning_calls) > 0
    for _, target in tuning_calls:
        np.testing.assert_allclose(target, _CONSTANT_G)
