import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.fixture(scope="session")
def dml_data():
    return make_med_data()


@pytest.fixture(scope="session")
def learner_linear():
    return {
        "ml_g": LinearRegression(),
        "ml_m": LogisticRegression(solver="saga", l1_ratio=1, max_iter=250, random_state=42),
        "ml_G": LinearRegression(),
        "ml_M": LogisticRegression(solver="saga", l1_ratio=1, max_iter=250, random_state=42),
        "ml_nested_g": LinearRegression(),
    }


@pytest.fixture(scope="session")
def learner_tree():
    return {
        "ml_g": DecisionTreeRegressor(random_state=123),
        "ml_G": DecisionTreeRegressor(random_state=123),
        "ml_m": DecisionTreeClassifier(random_state=123),
        "ml_M": DecisionTreeClassifier(random_state=123),
        "ml_nested_g": DecisionTreeRegressor(random_state=123),
    }


@pytest.fixture(scope="session")
def learner_forest():
    return {
        "ml_g": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
        "ml_G": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
        "ml_m": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        "ml_M": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        "ml_nested_g": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
    }


@pytest.fixture(scope="session", params=[0, 1])
def binary_treats(request):
    return request.param


@pytest.fixture(scope="session", params=["potential", "counterfactual"], ids=["target=potential", "target=counterfactual"])
def binary_targets(request):
    return request.param


@pytest.fixture(scope="session")
def binary_scores(binary_targets, binary_treats):
    return f"{binary_targets}_{binary_treats}"


@pytest.fixture(scope="session")
def optuna_params():
    return {
        "ml_g": _small_tree_params,
        "ml_m": _small_tree_params,
        "ml_G": _small_tree_params,
        "ml_M": _small_tree_params,
        "ml_nested_g": _small_tree_params,
    }


@pytest.fixture(scope="session", params=_SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def optuna_settings(request):
    sampler_name, optuna_sampler = request.param
    return _basic_optuna_settings({"sampler": optuna_sampler, "n_trials": 10})


@pytest.fixture(scope="session")
def med_factory(dml_data):
    def _factory(target, treatment_level, learners, **kwargs):
        if target == "potential":
            active_learners = {k: clone(v) for k, v in learners.items() if k in ["ml_g", "ml_m"]}
        elif target == "counterfactual":
            active_learners = {k: clone(v) for k, v in learners.items() if k in ["ml_m", "ml_G", "ml_M", "ml_nested_g"]}

        return DoubleMLMED(dml_data=dml_data, target=target, treatment_level=treatment_level, **active_learners, **kwargs)

    return _factory
