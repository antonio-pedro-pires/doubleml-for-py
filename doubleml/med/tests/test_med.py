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


@pytest.fixture(
    scope="module",
    params=[0, 1],
    ids=[
        "treatment_level=" + str(0),
        "treatment_level=" + str(1),
    ],
)
def treatment_level_potential(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        {
            "ml_yx": LinearRegression(),
            "ml_px": LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
            "ml_ymx": LinearRegression(),
            "ml_pmx": LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
            "ml_nested": LinearRegression(),
        },
        {
            "ml_yx": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
            "ml_ymx": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
            "ml_px": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
            "ml_pmx": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
            "ml_nested": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
        },
    ],
)
def learners(request):
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module", params=[0.15, 0.2])
def trimming_threshold(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        [0, 1],
        [1, 0],
    ],
)
def treatment_mediation_level_counterfactual(request):
    return request.param


@pytest.fixture(scope="module")
def med_objs(
    data,
    learners,
    normalize_ipw,
    trimming_threshold,
    treatment_mediation_level_counterfactual,
    treatment_level_potential,
    n_folds,
    n_rep_boot,
):

    treatment_level, mediation_level = treatment_mediation_level_counterfactual

    # Set machine learning methods
    ml_yx = clone(learners["ml_yx"])
    ml_ymx = clone(learners["ml_ymx"])
    ml_px = clone(learners["ml_px"])
    ml_pmx = clone(learners["ml_pmx"])
    ml_nested = clone(learners["ml_nested"])

    np.random.seed(3141)
    counter_med_obj = DoubleMLMED(
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
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )

    counter_med_obj_ext = DoubleMLMED(
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
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )

    treatment_level = treatment_level_potential
    mediation_level = treatment_level

    np.random.seed(3141)
    pot_med_obj = DoubleMLMED(
        med_data=data,
        target="potential",
        treatment_level=treatment_level,
        mediation_level=mediation_level,
        ml_yx=ml_yx,
        ml_px=ml_px,
        score="MED",
        score_function="efficient-alt",
        n_folds=n_folds,
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )

    np.random.seed(3141)
    pot_med_obj_ext = DoubleMLMED(
        med_data=data,
        target="potential",
        treatment_level=treatment_level,
        mediation_level=mediation_level,
        ml_yx=ml_yx,
        ml_px=ml_px,
        score="MED",
        score_function="efficient-alt",
        n_folds=n_folds,
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
    )
    return counter_med_obj, counter_med_obj_ext, pot_med_obj, pot_med_obj_ext


@pytest.fixture(scope="module")
def dml_med_fixture(
    med_objs,
    n_rep_boot,
):
    boot_methods = ["normal"]

    counter_med_obj, counter_med_obj_ext, pot_med_obj, pot_med_obj_ext = med_objs

    counter_med_obj.fit()
    pot_med_obj.fit()

    counter_smpls_inner = counter_med_obj._smpls_inner

    counter_med_obj_ext._smpls = counter_med_obj.smpls
    counter_med_obj_ext._set_smpls_inner_splitting(counter_smpls_inner)

    pot_med_obj_ext._smpls = pot_med_obj.smpls

    counter_med_obj_ext.fit()
    pot_med_obj_ext.fit()

    counter_prediction_dict = {
        "d": {
            "ml_ymx": counter_med_obj.predictions["ml_ymx"].reshape(-1, 1),
            "ml_px": counter_med_obj.predictions["ml_px"].reshape(-1, 1),
            "ml_pmx": counter_med_obj.predictions["ml_pmx"].reshape(-1, 1),
            "ml_nested": counter_med_obj.predictions["ml_nested"].reshape(-1, 1),
        }
    }

    pot_prediction_dict = {
        "d": {
            "ml_yx": pot_med_obj.predictions["ml_yx"].reshape(-1, 1),
            "ml_px": pot_med_obj.predictions["ml_px"].reshape(-1, 1),
        }
    }

    for i in range(counter_med_obj_ext.n_folds_inner):
        counter_prediction_dict["d"][f"ml_ymx_inner_{i}"] = counter_med_obj_ext.predictions[f"ml_ymx_inner_{i}"][:, :, 0]

    for i in range(pot_med_obj_ext.n_folds_inner):
        pot_med_obj_ext.fit(external_predictions=pot_prediction_dict)

    counter_res_dict = {
        "coef": counter_med_obj.coef.item(),
        "coef_ext": counter_med_obj_ext.coef.item(),
        "se": counter_med_obj.se.item(),
        "se_ext": counter_med_obj_ext.se.item(),
        "boot_methods": boot_methods,
    }

    pot_res_dict = {
        "coef": pot_med_obj.coef.item(),
        "coef_ext": pot_med_obj_ext.coef.item(),
        "se": pot_med_obj.se.item(),
        "se_ext": pot_med_obj_ext.se.item(),
        "boot_methods": boot_methods,
    }
    for bootstrap in boot_methods:
        np.random.seed(3141)
        counter_med_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        counter_med_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        counter_res_dict["boot_t_stat" + bootstrap] = counter_med_obj.boot_t_stat
        counter_res_dict["boot_t_stat" + bootstrap + "_ext"] = counter_med_obj_ext.boot_t_stat

        np.random.seed(3141)
        pot_med_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        pot_med_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        pot_res_dict["boot_t_stat" + bootstrap] = pot_med_obj.boot_t_stat
        pot_res_dict["boot_t_stat" + bootstrap + "_ext"] = pot_med_obj_ext.boot_t_stat

    return counter_res_dict, pot_res_dict


@pytest.mark.ci
def test_dml_med_coef(dml_med_fixture):
    counter_res_dict, pot_res_dict = dml_med_fixture
    assert math.isclose(counter_res_dict["coef"], counter_res_dict["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(pot_res_dict["coef"], pot_res_dict["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_se(dml_med_fixture):
    counter_res_dict, pot_res_dict = dml_med_fixture
    assert math.isclose(counter_res_dict["se"], counter_res_dict["se_ext"], rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(pot_res_dict["se"], pot_res_dict["se_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_counterfactual_boot(dml_med_fixture):
    counter_res_dict, pot_res_dict = dml_med_fixture

    for bootstrap in counter_res_dict["boot_methods"]:
        assert np.allclose(
            counter_res_dict["boot_t_stat" + bootstrap],
            counter_res_dict["boot_t_stat" + bootstrap + "_ext"],
            rtol=1e-9,
            atol=1e-4,
        )

    for bootstrap in pot_res_dict["boot_methods"]:
        assert np.allclose(
            pot_res_dict["boot_t_stat" + bootstrap],
            pot_res_dict["boot_t_stat" + bootstrap + "_ext"],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.mark.ci
def test_set_smpls_inner_splitting(med_objs):
    med_obj, med_obj_ext, *_ = med_objs

    smpls_inner = med_obj.smpls_inner
    smpls = med_obj.smpls

    med_obj_ext._smpls = smpls
    med_obj_ext._set_smpls_inner_splitting(smpls_inner)
    assert med_obj_ext._smpls_inner == med_obj._smpls_inner
