"""Shared fixtures for the mediation tests."""

import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doubleml.data import DoubleMLMEDData
from doubleml.med.datasets import make_med_data
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _basic_optuna_settings,
    _small_tree_params,
)
from doubleml.utils import DMLDummyClassifier, DMLDummyRegressor, PSProcessorConfig


# Fixtures that only return plain values (bools, ints) use the "session" scope, as there is no state to reset between modules.
@pytest.fixture(autouse=True)
def _ignore_l1_ratio_warning():
    # scikit-learn < 1.8 warns about l1_ratio without penalty="elasticnet" when LogisticRegression(l1_ratio=1) is fitted.
    # Remove this fixture once the minimum supported scikit-learn version is 1.8, which does not warn.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty")
        yield


@pytest.fixture(scope="session", params=[True, False])
def double_sample_splitting(request):
    return request.param


@pytest.fixture(scope="session", params=[False, True], ids=["normalize_ipw=False", "normalize_ipw=True"])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="session")
def dml_data():
    return make_med_data()


@pytest.fixture(scope="session", params=[1, 2], ids=["n_rep=1", "n_rep=2"])
def n_rep(request):
    return request.param


@pytest.fixture(scope="session")
def n_rep_boot():
    return 499


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
def learner_dummy():
    """Learners that raise an error if they are used, for tests that must not fit or predict."""
    return {
        "ml_g": DMLDummyRegressor(),
        "ml_m": DMLDummyClassifier(),
        "ml_G": DMLDummyRegressor(),
        "ml_M": DMLDummyClassifier(),
        "ml_nested_g": DMLDummyRegressor(),
    }


@pytest.fixture(scope="session", params=[2, 3], ids=["n_folds=2", "n_folds=3"])
def n_folds(request):
    return request.param


@pytest.fixture(scope="session", params=[2, 3], ids=["n_folds_inner=2", "n_folds_inner=3"])
def n_folds_inner(request):
    return request.param


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


@pytest.fixture(scope="session", params=["factual", "counterfactual"], ids=["outcome=factual", "outcome=counterfactual"])
def binary_outcomes(request):
    return request.param


@pytest.fixture(scope="session")
def binary_scores(binary_outcomes, binary_treats):
    return f"{binary_outcomes}_{binary_treats}"


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


@pytest.fixture(scope="session", params=[PSProcessorConfig()])
def ps_processor_config(request):
    return request.param


@pytest.fixture(scope="session")
def med_data_two_treatments():
    """Mediation data with two binary treatment variables."""
    data = make_med_data(n_obs=300, dim_x=5)
    df = data.data.copy()
    df["d2"] = np.random.RandomState(1).binomial(1, 0.4, size=len(df))
    return DoubleMLMEDData(df, y_col=data.y_col, d_cols=[data.d_cols[0], "d2"], m_cols=data.m_cols, x_cols=data.x_cols)
