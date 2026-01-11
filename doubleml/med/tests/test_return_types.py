import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMediation
from doubleml.med.datasets import make_med_data
from doubleml.utils._check_return_types import (
    check_basic_predictions_and_targets,
    check_basic_property_types_and_shapes,
    check_basic_return_types,
    check_sensitivity_return_types,
)

# Test constants
N_OBS = 500
N_TREAT = 1
N_REP = 1
N_FOLDS = 3
N_REP_BOOT = 314

dml_args = {
    "n_rep": N_REP,
    "n_folds": N_FOLDS,
}


# create datasets
np.random.seed(3141)
datasets = {}

datasets["med"] = make_med_data(n_obs=N_OBS)

@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    return request.param

@pytest.fixture(scope="module")
def dml_med_fixture(target):
    dml_args["target"] = target

    dml_med_obj = DoubleMLMediation(datasets["med"], ml_yx=LinearRegression(), ml_px=LogisticRegression(max_iter=1000),
                                    ml_ymx=LinearRegression(), ml_pmx=LogisticRegression(max_iter=1000),
                                    ml_nested=LinearRegression(), **dml_args)

    dml_objs = (dml_med_obj, DoubleMLMediation)
    return dml_objs

@pytest.mark.ci
def test_return_types(dml_med_fixture):
    dml_obj, cls = dml_med_fixture
    check_basic_return_types(dml_obj, cls)

    # further return type tests
    assert isinstance(dml_obj.get_params("ml_px"), dict)


@pytest.fixture(scope="module")
def fitted_dml_obj(request, dml_med_fixture):
    dml_obj, _ = dml_med_fixture
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_property_types_and_shapes(fitted_dml_obj):
    check_basic_property_types_and_shapes(fitted_dml_obj, N_OBS, N_TREAT, N_REP, N_FOLDS, N_REP_BOOT)
    check_basic_predictions_and_targets(fitted_dml_obj, N_OBS, N_TREAT, N_REP)
