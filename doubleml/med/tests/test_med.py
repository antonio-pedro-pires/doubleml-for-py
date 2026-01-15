import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.med import DoubleMLMED
from doubleml.med.datasets import make_med_data
from doubleml.tests._utils import draw_smpls


@pytest.fixture(scope="module")
def n_folds():
    return 5


@pytest.fixture(scope="module")
def n_rep_boot():
    return 499


@pytest.fixture(scope="module")
def n_obs():
    return 500


@pytest.fixture(scope="module")
def data_med(n_obs):
    np.random.seed(3141)
    data_med = make_med_data(n_obs=n_obs)
    return data_med


@pytest.fixture(scope="module")
def y(data_med):
    return data_med.y


@pytest.fixture(scope="module")
def d(data_med):
    return data_med.d


@pytest.fixture(scope="module")
def m(data_med):
    return data_med.m


@pytest.fixture(scope="module")
def x(data_med):
    return data_med.x


@pytest.fixture(scope="module")
def data(n_obs, data_med, y, d, m, x):
    df_med = pd.DataFrame(
        np.column_stack((y, d, m, x)), columns=["y", "d", "m"] + ["x" + str(i) for i in range(data_med.x.shape[1])]
    )
    return dml.DoubleMLMEDData(df_med, "y", "d", "m")


@pytest.fixture(scope="module")
def all_smpls(data, n_obs, n_folds, d):
    np.random.seed(3141)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1)
    return all_smpls


# Global parameters for Counterfactual Outcome Tests
learners_counterfactual_params = [
    [
        LinearRegression(),
        LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
        LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
        LinearRegression(),
    ],
    [
        RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
    ],
]

normalize_ipw_params = [False, True]
trimming_threshold_params = [0, 0]  # TODO: Debugging. Put values back to 0.15, 0.2.
treatment_mediation_level_params = [
    [0, 1],
    [1, 0],
]


@pytest.fixture(scope="module", params=learners_counterfactual_params)
def learners_counterfactual(request):
    return request.param


@pytest.fixture(scope="module", params=normalize_ipw_params)
def normalize_ipw_counterfactual(request):
    return request.param


@pytest.fixture(scope="module", params=trimming_threshold_params)
def trimming_threshold_counterfactual(request):
    return request.param


@pytest.fixture(scope="module", params=treatment_mediation_level_params)
def treatment_mediation_level_counterfactual(request):
    return request.param


@pytest.fixture(scope="module")
def dml_med_counterfactual_fixture(
    data,
    y,
    d,
    m,
    x,
    learners_counterfactual,
    all_smpls,
    normalize_ipw_counterfactual,
    trimming_threshold_counterfactual,
    treatment_mediation_level_counterfactual,
    n_folds,
    n_rep_boot,
):
    boot_methods = ["normal"]
    treatment_level, mediation_level = treatment_mediation_level_counterfactual

    # Set machine learning methods
    ml_ymx = clone(learners_counterfactual[0])
    ml_px = clone(learners_counterfactual[1])
    ml_pmx = clone(learners_counterfactual[2])
    ml_nested = clone(learners_counterfactual[3])

    np.random.seed(3141)
    dml_obj = DoubleMLMED(
        med_data=data,
        target="counterfactual",
        treatment_level=treatment_level,
        mediation_level=mediation_level,
        ml_ymx=ml_ymx,
        ml_px=ml_px,
        ml_pmx=ml_pmx,
        ml_nested=ml_nested,
        score="MED",
        score_function="efficient-alt",
        n_folds=n_folds,
        normalize_ipw=normalize_ipw_counterfactual,
        trimming_threshold=trimming_threshold_counterfactual,
    )

    dml_obj.fit()

    # Extract internal splits for manual verification
    smpls = dml_obj.smpls
    smpls_inner = dml_obj._smpls_inner

    dml_obj_ext = DoubleMLMED(
        med_data=data,
        target="counterfactual",
        treatment_level=treatment_level,
        mediation_level=mediation_level,
        ml_ymx=ml_ymx,
        ml_px=ml_px,
        ml_pmx=ml_pmx,
        ml_nested=ml_nested,
        score="MED",
        score_function="efficient-alt",
        n_folds=n_folds,
        normalize_ipw=normalize_ipw_counterfactual,
        trimming_threshold=trimming_threshold_counterfactual,
    )

    # Pass the splits from dml_obj to dml_obj_ext manually to ensure consistency
    dml_obj_ext._smpls = dml_obj.smpls
    dml_obj_ext._smpls_inner = dml_obj._smpls_inner
    dml_obj_ext.fit()

    prediction_dict = {
        "d": {
            "ml_ymx": dml_obj.predictions["ml_ymx"].reshape(-1, 1),
            "ml_px": dml_obj.predictions["ml_px"].reshape(-1, 1),
            "ml_pmx": dml_obj.predictions["ml_pmx"].reshape(-1, 1),
            "ml_nested": dml_obj.predictions["ml_nested"].reshape(-1, 1),
        }
    }

    for i in range(dml_obj_ext.n_folds_inner):
        prediction_dict["d"][f"ml_ymx_inner_{i}"] = dml_obj_ext.predictions[f"ml_ymx_inner_{i}"][:, :, 0]

    dml_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = {
        "coef": dml_obj.coef.item(),
        "coef_ext": dml_obj_ext.coef.item(),
        "se": dml_obj.se.item(),
        "se_ext": dml_obj_ext.se.item(),
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict["boot_t_stat" + bootstrap] = dml_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_ext"] = dml_obj_ext.boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_med_counterfactual_coef(dml_med_counterfactual_fixture):
    assert math.isclose(
        dml_med_counterfactual_fixture["coef"], dml_med_counterfactual_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4
    )


@pytest.mark.ci
def test_dml_med_counterfactual_se(dml_med_counterfactual_fixture):
    assert math.isclose(
        dml_med_counterfactual_fixture["se"], dml_med_counterfactual_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-4
    )


@pytest.mark.ci
def test_dml_med_counterfactual_boot(dml_med_counterfactual_fixture):
    for bootstrap in dml_med_counterfactual_fixture["boot_methods"]:
        assert np.allclose(
            dml_med_counterfactual_fixture["boot_t_stat" + bootstrap],
            dml_med_counterfactual_fixture["boot_t_stat" + bootstrap + "_ext"],
            rtol=1e-9,
            atol=1e-4,
        )


# Global parameters for Potential Outcome Tests
learners_potential_params = [
    [
        LinearRegression(),
        LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
    ],
    [
        RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
    ],
]

normalize_ipw_potential_params = [False, True]
trimming_threshold_potential_params = [0, 0]  # Set to zero for debugging. Reset back to "0.15, 0.2" once done.
treatment_level_potential_params = [0, 1]


@pytest.fixture(scope="module", params=learners_potential_params)
def learners_potential(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=normalize_ipw_potential_params,
    ids=["normalize_ipw=" + str(normalize_ipw_potential_params[0]), "normalize_ipw=" + str(normalize_ipw_potential_params[1])],
)
def normalize_ipw_potential(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=trimming_threshold_potential_params,
    ids=[
        "trimming_threshold=" + str(trimming_threshold_potential_params[0]),
        "trimming_threshold=" + str(trimming_threshold_potential_params[1]),
    ],
)
def trimming_threshold_potential(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=treatment_level_potential_params,
    ids=[
        "treatment_level=" + str(treatment_level_potential_params[0]),
        "treatment_level=" + str(treatment_level_potential_params[1]),
    ],
)
def treatment_level_potential(request):
    return request.param


@pytest.fixture(scope="module")
def dml_med_potential_fixture(
    data,
    y,
    d,
    m,
    x,
    learners_potential,
    all_smpls,
    normalize_ipw_potential,
    trimming_threshold_potential,
    treatment_level_potential,
    n_folds,
    n_rep_boot,
):
    boot_methods = ["normal"]
    treatment_level = treatment_level_potential

    # Set machine learning methods
    ml_yx = clone(learners_potential[0])
    ml_px = clone(learners_potential[1])

    np.random.seed(3141)
    dml_obj = DoubleMLMED(
        med_data=data,
        target="potential",
        treatment_level=treatment_level,
        ml_yx=ml_yx,
        ml_px=ml_px,
        score="MED",
        score_function="efficient-alt",
        n_folds=n_folds,
        normalize_ipw=normalize_ipw_potential,
        trimming_threshold=trimming_threshold_potential,
    )

    # dml_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_obj.fit()

    smpls = dml_obj.smpls
    smpls_inner = getattr(dml_obj, "_smpls_inner", None)

    np.random.seed(3141)
    dml_obj_ext = DoubleMLMED(
        med_data=data,
        target="potential",
        treatment_level=treatment_level,
        ml_yx=ml_yx,
        ml_px=ml_px,
        score="MED",
        score_function="efficient-alt",
        n_folds=n_folds,
        normalize_ipw=normalize_ipw_potential,
        trimming_threshold=trimming_threshold_potential,
    )

    # Pass the splits from dml_obj to dml_obj_ext manually to ensure consistency
    dml_obj_ext._smpls = dml_obj.smpls
    dml_obj_ext._smpls_inner = getattr(dml_obj, "_smpls_inner", None)
    dml_obj_ext.fit()

    prediction_dict = {
        "d": {
            "ml_yx": dml_obj.predictions["ml_yx"].reshape(-1, 1),
            "ml_px": dml_obj.predictions["ml_px"].reshape(-1, 1),
        }
    }
    dml_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = {
        "coef": dml_obj.coef.item(),
        "coef_ext": dml_obj_ext.coef.item(),
        "se": dml_obj.se.item(),
        "se_ext": dml_obj_ext.se.item(),
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict["boot_t_stat" + bootstrap] = dml_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_ext"] = dml_obj_ext.boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_med_potential_coef(dml_med_potential_fixture):
    assert math.isclose(dml_med_potential_fixture["coef"], dml_med_potential_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_potential_se(dml_med_potential_fixture):
    assert math.isclose(dml_med_potential_fixture["se"], dml_med_potential_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_potential_boot(dml_med_potential_fixture):
    for bootstrap in dml_med_potential_fixture["boot_methods"]:
        assert np.allclose(
            dml_med_potential_fixture["boot_t_stat" + bootstrap],
            dml_med_potential_fixture["boot_t_stat" + bootstrap + "_ext"],
            rtol=1e-9,
            atol=1e-4,
        )
