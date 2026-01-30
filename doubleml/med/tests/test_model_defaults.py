import pytest

from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap


@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    return request.param


@pytest.fixture(scope="module")
def dml_med_fixture(binary_targets, dml_data, med_factory, learner_linear):
    if binary_targets == "potential":
        med_obj = med_factory(binary_targets, 1, learner_linear)
    if binary_targets == "counterfactual":
        med_obj = med_factory(binary_targets, 1, learner_linear)
    return med_obj


@pytest.mark.ci
def test_med_defaults(dml_med_fixture):
    _check_basic_defaults_before_fit(dml_med_fixture)

    _fit_bootstrap(dml_med_fixture)

    _check_basic_defaults_after_fit(dml_med_fixture)
