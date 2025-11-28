import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.datasets import make_med_data
from doubleml.med import DoubleMLMEDC
from doubleml.med.tests._utils_med_manual import ManualMedCAlt
from doubleml.tests._utils import draw_smpls


@pytest.fixture(scope="module")
def n_folds():
    return 2


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
    return dml.DoubleMLMediationData(df_med, "y", "d", "m")


@pytest.fixture(scope="module")
def all_smpls(data, n_obs, n_folds, d):
    np.random.seed(3141)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1)
    return all_smpls


class TestCounterfactualAltOutcomes:

    @pytest.fixture(
        scope="class",
    )
    def outcome_scoring(self):
        return [DoubleMLMEDC, ManualMedCAlt]

    @pytest.fixture(
        scope="class",
        params=[
            [
                LinearRegression(),
                LinearRegression(),
                LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
                LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
                LinearRegression(),
            ],
            [
                RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
                RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
                RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
            ],
        ],
    )
    def learners(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params=[
            False,
            True,
        ],
    )
    def normalize_ipw(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params=[
            0.2,
            0.15,
        ],
    )
    def trimming_threshold(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params=[
            [
                0,
                1,
            ],
            [
                1,
                0,
            ],
        ],
    )
    def treatment_mediation_level(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def dml_med_fixture(
        self,
        data,
        y,
        d,
        m,
        x,
        learners,
        all_smpls,
        outcome_scoring,
        normalize_ipw,
        trimming_threshold,
        treatment_mediation_level,
        n_folds,
        n_rep_boot,
    ):
        boot_methods = ["normal"]

        # Set treatment, mediation levels
        treatment_level, mediation_level = treatment_mediation_level

        # Set machine learning methods for m & g
        ml_yx = clone(learners[0])
        ml_ymx = clone(learners[1])
        ml_px = clone(learners[2])
        ml_pmx = clone(learners[3])
        ml_nested = clone(learners[4])

        np.random.seed(3141)
        dml_obj = outcome_scoring[0](
            med_data=data,
            treatment_level=treatment_level,
            mediation_level=mediation_level,
            ml_yx=ml_yx,
            ml_ymx=ml_ymx,
            ml_px=ml_px,
            ml_pmx=ml_pmx,
            ml_nested=ml_nested,
            score="MED",
            score_function="efficient-alt",
            n_folds=n_folds,
            normalize_ipw=normalize_ipw,
            draw_sample_splitting=False,
            trimming_threshold=trimming_threshold,
        )

        # synchronize the sample splitting
        dml_obj.set_sample_splitting(all_smpls=all_smpls)
        dml_obj.fit()

        np.random.seed(3141)
        manual_med = outcome_scoring[1](
            y=y,
            x=x,
            d=d,
            m=m,
            learner_yx=clone(learners[0]),
            learner_ymx=clone(learners[1]),
            learner_px=clone(learners[2]),
            learner_pmx=clone(learners[3]),
            learner_nested=clone(learners[4]),
            treatment_level=treatment_level,
            mediation_level=mediation_level,
            all_smpls=all_smpls,
            n_rep=1,
            normalize_ipw=normalize_ipw,
            trimming_threshold=trimming_threshold,
            smpls_ratio=0.5,
        )
        res_manual = manual_med.fit_med()

        np.random.seed(3141)
        # test with external nuisance predictions
        dml_obj_ext = outcome_scoring[0](
            med_data=data,
            treatment_level=treatment_level,
            mediation_level=mediation_level,
            ml_yx=ml_yx,
            ml_ymx=ml_ymx,
            ml_px=ml_px,
            ml_pmx=ml_pmx,
            ml_nested=ml_nested,
            score="MED",
            score_function="efficient-alt",
            n_folds=n_folds,
            normalize_ipw=normalize_ipw,
            draw_sample_splitting=False,
            trimming_threshold=trimming_threshold,
        )

        # synchronize the sample splitting
        dml_obj_ext.set_sample_splitting(all_smpls=all_smpls)

        prediction_dict = {
            "d": {
                "ml_yx": dml_obj.predictions["ml_yx"].reshape(-1, 1),
                "ml_ymx": dml_obj.predictions["ml_ymx"].reshape(-1, 1),
                "ml_px": dml_obj.predictions["ml_px"].reshape(-1, 1),
                "ml_pmx": dml_obj.predictions["ml_pmx"].reshape(-1, 1),
                "ml_nested": dml_obj.predictions["ml_nested"].reshape(-1, 1),
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
            boot_t_stat = manual_med.boot_med(
                thetas=res_manual["thetas"],
                ses=res_manual["ses"],
                all_yx_hat=res_manual["all_yx_hat"],
                all_ymx_hat=res_manual["all_ymx_hat"],
                all_px_hat=res_manual["all_px_hat"],
                all_pmx_hat=res_manual["all_pmx_hat"],
                all_nested_hat=res_manual["all_nested_hat"],
                bootstrap=bootstrap,
                n_rep_boot=n_rep_boot,
            )

            np.random.seed(3141)
            dml_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
            np.random.seed(3141)
            dml_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
            res_dict["boot_t_stat" + bootstrap] = dml_obj.boot_t_stat
            res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)
            res_dict["boot_t_stat" + bootstrap + "_ext"] = dml_obj_ext.boot_t_stat

        return res_dict

    @pytest.mark.ci
    def test_dml_med_coef(self, dml_med_fixture):
        assert math.isclose(dml_med_fixture["coef"], dml_med_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)
        assert math.isclose(dml_med_fixture["coef"], dml_med_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)

    @pytest.mark.ci
    def test_dml_med_se(self, dml_med_fixture):
        assert math.isclose(dml_med_fixture["se"], dml_med_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)
        assert math.isclose(dml_med_fixture["se"], dml_med_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-4)

    @pytest.mark.ci
    def test_dml_med_boot(self, dml_med_fixture):
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
