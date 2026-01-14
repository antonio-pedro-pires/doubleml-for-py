import random
import re

import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doubleml import DoubleMLMediation
from doubleml.med.datasets import make_med_data
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _assert_tree_params,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    yield request.param


@pytest.fixture(scope="module", params=[DecisionTreeRegressor(random_state=234)])
def ml_yx(request):
    yield request.param


@pytest.fixture(scope="module", params=[DecisionTreeRegressor(random_state=234)])
def ml_ymx(request):
    yield request.param


@pytest.fixture(scope="module", params=[DecisionTreeClassifier(random_state=123)])
def ml_px(request):
    yield request.param


@pytest.fixture(scope="module", params=[DecisionTreeClassifier(random_state=123)])
def ml_pmx(request):
    yield request.param


@pytest.fixture(scope="module", params=[DecisionTreeRegressor(random_state=234)])
def ml_nested(request):
    yield request.param


@pytest.fixture(
    scope="module",
)
def med_data():
    yield make_med_data()


@pytest.fixture(scope="module")
def dml_med_obj(
    med_data,
    target,
    ml_yx,
    ml_px,
    ml_ymx,
    ml_pmx,
    ml_nested,
):
    if target == "potential":
        yield DoubleMLMediation(med_data=med_data, ml_yx=ml_yx, ml_px=ml_px, target=target)
    else:
        yield DoubleMLMediation(
            med_data=med_data, ml_yx=ml_yx, ml_px=ml_px, ml_ymx=ml_ymx, ml_pmx=ml_pmx, ml_nested=ml_nested, target=target
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


# TODO: Refactor this into a utils file?
def _lasso_params(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-2, 1e2),
        "max_iter": trial.suggest_int("max_iter", 500, 1000),
        "tol": trial.suggest_float("tol", 1e-5, 1e-3),
        "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
    }


def _logistic_params(trial):
    return {
        "C": trial.suggest_int("C", 1, 10),
        "max_iter": trial.suggest_int("max_iter", 500, 1000),
        "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        "tol": trial.suggest_float("tol", 1e-5, 1e-3),
        "solver": trial.suggest_categorical(
            "solver",
            ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        ),
    }


def assert_lasso_params(
    lasso_params, alpha_range=(1e-2, 1e2), max_iter_range=(500, 1000), tol_range=(1e-5, 1e-3), selection=("cyclic", "random")
):
    assert isinstance(lasso_params, dict)
    assert set(lasso_params.keys()) == {"alpha", "max_iter", "tol", "selection"}
    assert alpha_range[0] <= lasso_params["alpha"] <= alpha_range[1]
    assert max_iter_range[0] <= lasso_params["max_iter"] <= max_iter_range[1]
    assert tol_range[0] <= lasso_params["tol"] <= tol_range[1]
    assert lasso_params["selection"] in selection


def assert_logistic_params(
    logistic_params,
    C_range=(1, 10),
    max_iter_range=(500, 1000),
    tol_range=(1e-5, 1e-3),
    l1_ratio_range=(0, 1),
    solver=("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"),
):
    assert isinstance(logistic_params, dict)
    assert set(logistic_params.keys()) == {"C", "max_iter", "l1_ratio", "tol", "solver"}
    assert C_range[0] <= logistic_params["C"] <= C_range[1]
    assert max_iter_range[0] <= logistic_params["max_iter"] <= max_iter_range[1]
    assert tol_range[0] <= logistic_params["tol"] <= tol_range[1]
    assert l1_ratio_range[0] <= logistic_params["l1_ratio"] <= l1_ratio_range[1]
    assert logistic_params["solver"] in solver
