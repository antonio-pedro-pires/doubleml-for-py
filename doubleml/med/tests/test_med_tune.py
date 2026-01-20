import random
import re

import pytest

from doubleml.tests._utils_tune_optuna import (
    _assert_tree_params,
)


@pytest.fixture(scope="module")
def learners(learner_tree):
    return learner_tree


@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    yield request.param


@pytest.fixture(scope="module")
def dml_med_obj(med_factory, target, treatment_level, learners):
    yield med_factory(target, treatment_level, learners)


@pytest.fixture(
    scope="module",
)
def untuned_tuned_scores(dml_med_obj, optuna_params, optuna_settings, target):
    if target == "potential":
        ml_param_space = {
            "ml_yx": optuna_params["ml_yx"],
            "ml_px": optuna_params["ml_px"],
        }
    else:
        ml_param_space = {
            "ml_px": optuna_params["ml_px"],
            "ml_ymx": optuna_params["ml_ymx"],
            "ml_pmx": optuna_params["ml_pmx"],
            "ml_nested": optuna_params["ml_nested"],
        }

    dml_med_obj.fit()
    untuned_score = dml_med_obj.evaluate_learners()

    random.seed(0)
    tune_res = dml_med_obj.tune_ml_models(
        ml_param_space=ml_param_space,
        optuna_settings=optuna_settings,
        return_tune_res=True,
    )

    dml_med_obj.fit()
    tuned_score = dml_med_obj.evaluate_learners()
    return {"untuned_score": untuned_score, "tuned_score": tuned_score, "tune_res": tune_res, "ml_param_space": ml_param_space}


@pytest.mark.ci
def test_tune_ml_models(untuned_tuned_scores, dml_med_obj):
    untuned_score = untuned_tuned_scores["untuned_score"]
    tuned_score = untuned_tuned_scores["tuned_score"]
    tune_res = untuned_tuned_scores["tune_res"]
    optuna_params = untuned_tuned_scores["ml_param_space"]

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
