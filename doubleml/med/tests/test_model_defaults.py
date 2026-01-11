import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMediation
from doubleml.med.datasets import make_med_data
from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap



@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    return request.param

@pytest.fixture(scope="module")
def med_data():
    med_data = make_med_data()
    return med_data

@pytest.fixture(scope="module")
def dml_med_fixture(target, med_data):
    if target == "potential":
        med_obj = DoubleMLMediation(med_data, ml_yx=LinearRegression(), ml_px=LogisticRegression(),)
    if target == "counterfactual":
        med_obj = DoubleMLMediation(med_data, ml_yx=LinearRegression(), ml_px=LogisticRegression(),
                                ml_ymx=LinearRegression(), ml_pmx=LogisticRegression(),
                                ml_nested=LinearRegression(), )
    return med_obj


@pytest.mark.ci
def test_med_defaults(dml_med_fixture):
    _check_basic_defaults_before_fit(dml_med_fixture)

    _fit_bootstrap(dml_med_fixture)

    _check_basic_defaults_after_fit(dml_med_fixture)
