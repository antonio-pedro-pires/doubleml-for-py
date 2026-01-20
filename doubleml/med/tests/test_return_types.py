import pytest

from doubleml import DoubleMLMED
from doubleml.utils._check_return_types import (
    check_basic_predictions_and_targets,
    check_basic_property_types_and_shapes,
    check_basic_return_types,
)

# Test constants
N_OBS = 1000
N_TREAT = 1
N_REP = 1
N_FOLDS = 3
N_REP_BOOT = 314

dml_args = {
    "n_rep": N_REP,
    "n_folds": N_FOLDS,
}


@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    return request.param


@pytest.fixture(scope="module")
def dml_med_fixture(target, treatment_level, meds_data, med_factory, learner_linear):
    # dml_args["target"] = target # Removed to avoid duplication

    dml_med_obj = med_factory(target, treatment_level, learner_linear, **dml_args)

    dml_objs = (dml_med_obj, DoubleMLMED)
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
