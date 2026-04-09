import copy
import random
import re

import pytest

from doubleml import DoubleMLMED
from doubleml.tests._utils_tune_optuna import (
    _assert_tree_params,
)


@pytest.fixture(scope="module")
def learners(learner_tree):
    return learner_tree


@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    yield request.param


@pytest.fixture(
    scope="module",
)
def untuned_tuned_scores(dml_data, binary_targets, binary_treats, learners, optuna_params, optuna_settings):

    if binary_targets == "potential":
        model = DoubleMLMED(
            dml_data=dml_data,
            ml_g=learners["ml_g"],
            ml_m=learners["ml_m"],
            target=binary_targets,
            treatment_level=binary_treats,
        )
    else:
        model = DoubleMLMED(
            dml_data=dml_data,
            target=binary_targets,
            treatment_level=binary_treats,
            ml_g=learners["ml_g"],
            ml_m=learners["ml_m"],
            ml_G=learners["ml_G"],
            ml_M=learners["ml_M"],
            ml_nested_g=learners["ml_nested_g"],
        )

    med_obj = copy.deepcopy(model)
    if binary_targets == "potential":
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

    random.seed(0)
    med_obj.fit()
    untuned_score = med_obj.evaluate_learners()

    random.seed(0)
    tune_res = med_obj.tune_ml_models(
        ml_param_space=ml_param_space,
        optuna_settings=optuna_settings,
        return_tune_res=True,
    )

    random.seed(0)
    med_obj.fit()

    random.seed(0)
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
