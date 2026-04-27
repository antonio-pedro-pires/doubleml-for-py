import pytest
from sklearn.base import clone

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

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")


@pytest.fixture(scope="module")
def dml_med_fixture(binary_outcomes, binary_treats, dml_data, learner_linear, ps_processor_config):
    dml_args["ps_processor_config"] = ps_processor_config
    if binary_outcomes == "potential":
        active_learners = {k: clone(v) for k, v in learner_linear.items() if k in ["ml_g", "ml_m"]}
    else:
        active_learners = {k: clone(v) for k, v in learner_linear.items() if k in ["ml_m", "ml_G", "ml_M", "ml_nested_g"]}
    dml_med_obj = DoubleMLMED(
        dml_data=dml_data, outcome=binary_outcomes, treatment_level=binary_treats, **active_learners, **dml_args
    )

    dml_objs = (dml_med_obj, DoubleMLMED)
    return dml_objs


@pytest.mark.ci
def test_return_types(dml_med_fixture):
    dml_obj, cls = dml_med_fixture
    check_basic_return_types(dml_obj, cls)

    # further return type tests
    assert isinstance(dml_obj.get_params("ml_m"), dict)


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
