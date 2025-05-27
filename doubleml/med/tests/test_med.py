import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.datasets import make_med_data

from ...tests._utils import draw_smpls
from ._utils_med_manual import boot_med, fit_med, fit_sensitivity_elements_med


@pytest.fixture(
    scope="module",
    params=[
        [LinearRegression(), LogisticRegression(solver="lbfgs", max_iter=250, random_state=42)],
        [
            RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
            RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        ],
    ],
)
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module", params=[0.2, 0.15])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 1])
def treatment_level(request):
    return request.param


# TODO: Check what the fixtures do and what the params refer to
@pytest.fixture(scope="module", params=[1])
def order(request):
    """
    Indicates the order of the polynomials used to estimate any conditional probability or conditional mean outcome.
    """
    return request.param


@pytest.fixture(scope="module", params=[0, 1])
def few_splits(request):
    """
    Indicates whether the same training data is used for estimating nested models of nuisance parameters.
    """
    return request.param


@pytest.fixture(scope="module", params=[0, 1])
def score_type(request):
    """
    Indicates the type of the score function.
    """
    return request.param


@pytest.fixture(scope="module")
def dml_med_fixture(learner, normalize_ipw, trimming_threshold, treatment_level):
    boot_methods = ["normal"]
    n_folds = 2
    n_rep_boot = 499

    # Set machine learning methods for m & g & ml_med
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])
    ml_med = clone(learner[2])

    np.random.seed(3141)
    n_obs = 500
    med_data = make_med_data(n_obs)
    y = med_data["y"]
    x = med_data["x"]
    d = med_data["d"]
    m = med_data["m"]

    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    np.random.seed(3141)
    dml_obj = dml.DoubleMLMED(
        med_data,
        ml_g,
        ml_m,
        ml_med,
        treatment_level=treatment_level,
        n_folds=n_folds,
        score="Y(0, M(0))",
        normalize_ipw=normalize_ipw,
        draw_sample_splitting=False,
        trimming_threshold=trimming_threshold,
    )

    # synchronize the sample splitting
    dml_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_obj.fit()

    np.random.seed(3141)
    res_manual = fit_med(
        y,
        x,
        d,
        m,
        clone(learner[0]),
        clone(learner[1]),
        clone(learner[2]),
        treatment_level=treatment_level,
        all_smpls=all_smpls,
        score="Y(0, M(0))",
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )

    np.random.seed(3141)
    # test with external nuisance predictions
    dml_obj_ext = dml.DoubleMLmed(
        med_data,
        ml_g,
        ml_m,
        ml_med,
        treatment_level=treatment_level,
        n_folds=n_folds,
        score="Y(0, M(0))",
        normalize_ipw=normalize_ipw,
        draw_sample_splitting=False,
        trimming_threshold=trimming_threshold,
    )

    # synchronize the sample splitting
    dml_obj_ext.set_sample_splitting(all_smpls=all_smpls)

    prediction_dict = {
        "d": {
            "ml_g_d_lvl0": dml_obj.predictions["ml_g_d_lvl0"].reshape(-1, 1),
            "ml_g_d_lvl1": dml_obj.predictions["ml_g_d_lvl1"].reshape(-1, 1),
            "ml_m": dml_obj.predictions["ml_m"].reshape(-1, 1),
        }
    }
    dml_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = {
        "coef": dml_obj.coef.item(),
        "coef_manual": res_manual["theta"],
        "coef_ext": dml_obj_ext.coef.item(),
        "se": dml_obj.se.item(),
        "se_manual": res_manual["se"],
        "se_ext": dml_obj_ext.se.item(),
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_med(
            y,
            d,
            treatment_level,
            res_manual["thetas"],
            res_manual["ses"],
            res_manual["all_g_hat0"],
            res_manual["all_g_hat1"],
            res_manual["all_m_hat"],
            all_smpls,
            score="med",
            bootstrap=bootstrap,
            n_rep_boot=n_rep_boot,
            normalize_ipw=normalize_ipw,
        )

        np.random.seed(3141)
        dml_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict["boot_t_stat" + bootstrap] = dml_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)
        res_dict["boot_t_stat" + bootstrap + "_ext"] = dml_obj_ext.boot_t_stat

    # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
    dml_obj.sensitivity_analysis(rho=0.0)
    res_dict["sensitivity_ses"] = dml_obj.sensitivity_params["se"]

    # sensitivity tests
    res_dict["sensitivity_elements"] = dml_obj.sensitivity_elements
    res_dict["sensitivity_elements_manual"] = fit_sensitivity_elements_med(
        y,
        d,
        treatment_level,
        all_coef=dml_obj.all_coef,
        predictions=dml_obj.predictions,
        score="Y(0, M(0))",
        n_rep=1,
        normalize_ipw=normalize_ipw,
    )
    return res_dict


@pytest.mark.ci
def test_dml_med_coef(dml_med_fixture):
    assert math.isclose(dml_med_fixture["coef"], dml_med_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_med_fixture["coef"], dml_med_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_se(dml_med_fixture):
    assert math.isclose(dml_med_fixture["se"], dml_med_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_med_fixture["se"], dml_med_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_boot(dml_med_fixture):
    for bootstrap in dml_med_fixture["boot_methods"]:
        assert np.allclose(
            dml_med_fixture["boot_t_stat" + bootstrap],
            dml_med_fixture["boot_t_stat" + bootstrap + "_manual"],
            rtol=1e-9,
            atol=1e-4,
        )
        assert np.allclose(
            dml_med_fixture["boot_t_stat" + bootstrap],
            dml_med_fixture["boot_t_stat" + bootstrap + "_ext"],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.mark.ci
def test_dml_med_sensitivity_rho0(dml_med_fixture):
    assert np.allclose(dml_med_fixture["se"], dml_med_fixture["sensitivity_ses"]["lower"], rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_med_fixture["se"], dml_med_fixture["sensitivity_ses"]["upper"], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_med_sensitivity(dml_med_fixture):
    sensitivity_element_names = ["sigma2", "nu2", "psi_sigma2", "psi_nu2"]
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(
            dml_med_fixture["sensitivity_elements"][sensitivity_element],
            dml_med_fixture["sensitivity_elements_manual"][sensitivity_element],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.fixture(scope="module", params=["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def cov_type(request):
    return request.param
