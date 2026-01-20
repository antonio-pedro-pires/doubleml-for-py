import random
import re

import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)

@pytest.fixture(scope="module", params=[{
    "ml_yx": DecisionTreeRegressor(random_state=123),
    "ml_px": DecisionTreeClassifier(random_state=123),
    "ml_ymx": DecisionTreeRegressor(random_state=123),
    "ml_pmx": DecisionTreeClassifier(random_state=123),
    "ml_nested": DecisionTreeRegressor(random_state=123),
}])
def learners(request):
    yield request.param

@pytest.fixture(
    scope="module",
    params=[make_med_data()]
)
def med_data(request):
    yield request.param

@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    yield request.param

@pytest.fixture(scope="module", params=[0, 1])
def treatment_level(request):
    yield request.param

@pytest.fixture(scope="module")
def dml_med_obj(
    med_data,
    target,
    learners):
    if target == "potential":
        yield DoubleMLMED(med_data=med_data,
                          ml_yx=learners["ml_yx"],
                          ml_px=learners["ml_px"],
                          target=target,)
    else:
        yield DoubleMLMED(
            med_data=med_data,
            target=target,
            **learners,
        )


@pytest.fixture(scope="module")
def optuna_params(target):
    if target == "potential":
        return {
            "ml_yx": _small_tree_params,
            "ml_px": _small_tree_params,
        }
    else:
        return {
            "ml_px": _small_tree_params,
            "ml_ymx": _small_tree_params,
            "ml_pmx": _small_tree_params,
            "ml_nested": _small_tree_params,
        }


@pytest.fixture(scope="module", params=_SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def untuned_tuned_scores(request, dml_med_obj, optuna_params):
    sampler_name, optuna_sampler = request.param

    res = {"untuned_score": None, "tuned_score": None, "tune_res": None}
    dml_med_obj.fit()
    untuned_score = dml_med_obj.evaluate_learners()

    random.seed(0)
    tune_res = dml_med_obj.tune_ml_models(
        ml_param_space=optuna_params,
        optuna_settings=_basic_optuna_settings({"sampler": optuna_sampler, "n_trials": 20}),
        return_tune_res=True,
    )

    dml_med_obj.fit()
    tuned_score = dml_med_obj.evaluate_learners()
    return {"untuned_score": untuned_score, "tuned_score": tuned_score, "tune_res": tune_res}


@pytest.mark.ci
def test_tune_ml_models(untuned_tuned_scores, dml_med_obj, optuna_params):
    untuned_score = untuned_tuned_scores["untuned_score"]
    tuned_score = untuned_tuned_scores["tuned_score"]
    tune_res = untuned_tuned_scores["tune_res"]

    params_names = dml_med_obj.params_names

    assert isinstance(tune_res[0], dict)
    assert set(tune_res[0].keys()) == set(optuna_params.keys())
    for key in params_names:
        is_inner_model = re.findall(r"[0-9]", key) != []

        if not (key == "ml_nested" or is_inner_model):
            assert hasattr(tune_res[0][key], "best_params")
            _assert_tree_params(tune_res[0][key].best_params)
            msg = (
                f"key: {key}, tuned score is greater or equal to the untuned score: {tuned_score[key]} >= {untuned_score[key]}"
            )
            assert tuned_score[key] < untuned_score[key], msg
