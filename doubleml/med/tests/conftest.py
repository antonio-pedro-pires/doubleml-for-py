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
def meds_data():
    return make_med_data()


@pytest.fixture(scope="session", params=[0, 1])
def treatment_level(request):
    return request.param


@pytest.fixture(scope="session")
def learner_linear():
    return {
        "ml_yx": LinearRegression(),
        "ml_px": LogisticRegression(solver="saga", l1_ratio=1, max_iter=250, random_state=42),
        "ml_ymx": LinearRegression(),
        "ml_pmx": LogisticRegression(solver="saga", l1_ratio=1, max_iter=250, random_state=42),
        "ml_nested": LinearRegression(),
    }


@pytest.fixture(scope="session")
def learner_tree():
    return {
        "ml_yx": DecisionTreeRegressor(random_state=123),
        "ml_ymx": DecisionTreeRegressor(random_state=123),
        "ml_px": DecisionTreeClassifier(random_state=123),
        "ml_pmx": DecisionTreeClassifier(random_state=123),
        "ml_nested": DecisionTreeRegressor(random_state=123),
    }


@pytest.fixture(scope="session")
def learner_forest():
    return {
        "ml_yx": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
        "ml_ymx": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
        "ml_px": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        "ml_pmx": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        "ml_nested": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
    }


@pytest.fixture(scope="session")
def optuna_params():
    return {
        "ml_yx": _small_tree_params,
        "ml_px": _small_tree_params,
        "ml_ymx": _small_tree_params,
        "ml_pmx": _small_tree_params,
        "ml_nested": _small_tree_params,
    }


@pytest.fixture(scope="session", params=_SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def optuna_settings(request):
    sampler_name, optuna_sampler = request.param
    return _basic_optuna_settings({"sampler": optuna_sampler, "n_trials": 10})


@pytest.fixture(scope="session")
def med_factory(meds_data):
    def _factory(target, treatment_level, learners, **kwargs):
        if target == "potential":
            active_learners = {k: clone(v) for k, v in learners.items() if k in ["ml_yx", "ml_px"]}
        elif target == "counterfactual":
            active_learners = {k: clone(v) for k, v in learners.items() if k in ["ml_px", "ml_ymx", "ml_pmx", "ml_nested"]}

        return DoubleMLMED(med_data=meds_data, target=target, treatment_level=treatment_level, **active_learners, **kwargs)

    return _factory
