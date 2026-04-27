import pandas as pd
import pytest

from doubleml import DoubleMLMED, DoubleMLMEDS
from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")


@pytest.fixture(scope="module")
def dml_med_fixture(binary_outcomes, binary_treats, dml_data, learner_linear, ps_processor_config):
    if binary_outcomes == "potential":
        med_obj = DoubleMLMED(
            dml_data=dml_data,
            outcome=binary_outcomes,
            treatment_level=binary_treats,
            ml_g=learner_linear["ml_g"],
            ml_m=learner_linear["ml_m"],
            ps_processor_config=ps_processor_config,
        )
    else:
        med_obj = DoubleMLMED(
            dml_data=dml_data,
            outcome=binary_outcomes,
            treatment_level=binary_treats,
            ml_m=learner_linear["ml_m"],
            ml_G=learner_linear["ml_G"],
            ml_M=learner_linear["ml_M"],
            ml_nested_g=learner_linear["ml_nested_g"],
            ps_processor_config=ps_processor_config,
        )
    return med_obj


@pytest.fixture(scope="module")
def dml_meds_fixture(dml_data, learner_linear, ps_processor_config):
    meds_obj = DoubleMLMEDS(
        dml_data=dml_data,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )
    return meds_obj


@pytest.mark.ci
def test_med_defaults(dml_med_fixture):
    _check_basic_defaults_before_fit(dml_med_fixture)

    _fit_bootstrap(dml_med_fixture)

    _check_basic_defaults_after_fit(dml_med_fixture)


@pytest.mark.ci
def test_meds_defaults(dml_meds_fixture):
    # check defaults before fit
    assert dml_meds_fixture.n_folds == 5
    assert dml_meds_fixture.n_rep == 1
    assert dml_meds_fixture.framework is None
    pd.testing.assert_frame_equal(dml_meds_fixture.summary, pd.DataFrame(columns=["coef", "std err", "t", "P>|t|"]))

    dml_meds_fixture.fit()

    # check defaults after fit
    assert dml_meds_fixture.framework is not None
    assert isinstance(dml_meds_fixture.summary, pd.DataFrame)
