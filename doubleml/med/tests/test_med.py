import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.datasets import make_med_data

from ...tests._utils import draw_smpls
from .. import DoubleMLMEDC, DoubleMLMEDP
from ._utils_med_manual import ManualMedC, ManualMedCAlt, ManualMedP


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


class TestPotentialOutcomes:
    # TODO: Think on how to collect the parameters predictions depending on the outcome_scoring function. Is there a better structure than the current one?s
    @pytest.fixture(scope="module", params=[[DoubleMLMEDP, ManualMedP], [DoubleMLMEDP, ManualMedP]])
    def outcome_scoring(self, request):
        return request.param

    @pytest.fixture(
        scope="module",
        params=[
            [LinearRegression(), LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42)],
            [
                RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
            ],
        ],
    )
    def learner(self, request):
        return request.param

    @pytest.fixture(
        scope="module",
        params=[
            False,
            True,
        ],
    )
    def normalize_ipw(self, request):
        return request.param

    @pytest.fixture(
        scope="module",
        params=[
            0.2,
            0.15,
        ],
    )
    def trimming_threshold(self, request):
        return request.param

    @pytest.fixture(
        scope="module",
        params=[
            [
                0,
                0,
            ],
            [
                1,
                1,
            ],
        ],
    )
    def treatment_mediation_level(self, request):
        return request.param

    @pytest.fixture(scope="module")
    def dml_med_fixture(
        self,
        data,
        y,
        d,
        m,
        x,
        all_smpls,
        learner,
        outcome_scoring,
        normalize_ipw,
        trimming_threshold,
        treatment_mediation_level,
        n_rep_boot,
        n_folds,
    ):
        boot_methods = ["normal"]
        treatment_level, mediation_level = treatment_mediation_level

        # Set machine learning methods for m & g
        ml_g = clone(learner[0])
        ml_m = clone(learner[1])

        np.random.seed(3141)
        dml_obj = outcome_scoring[0](
            med_data=data,
            treatment_level=treatment_level,
            mediation_level=mediation_level,
            ml_g=ml_g,
            ml_m=ml_m,
            score="MED",
            score_function="efficient",
            n_folds=2,
            normalize_ipw=normalize_ipw,
            draw_sample_splitting=False,
            trimming_threshold=trimming_threshold,
        )

        # synchronize the sample splitting
        dml_obj.set_sample_splitting(all_smpls=all_smpls)
        dml_obj.fit()

        np.random.seed(3141)
        manual_med = outcome_scoring[1](
            y,
            x,
            d,
            m,
            learner_g=clone(learner[0]),
            learner_m=clone(learner[1]),
            treatment_level=treatment_level,
            all_smpls=all_smpls,
            n_rep=1,
            normalize_ipw=normalize_ipw,
            trimming_threshold=trimming_threshold,
        )
        res_manual = manual_med.fit_med()

        np.random.seed(3141)
        # test with external nuisance predictions
        dml_obj_ext = outcome_scoring[0](
            med_data=data,
            treatment_level=treatment_level,
            mediation_level=mediation_level,
            ml_g=ml_g,
            ml_m=ml_m,
            score="MED",
            score_function="efficient",
            n_folds=2,
            normalize_ipw=normalize_ipw,
            draw_sample_splitting=False,
            trimming_threshold=trimming_threshold,
        )

        # synchronize the sample splitting
        dml_obj_ext.set_sample_splitting(all_smpls=all_smpls)

        prediction_dict = {
            "d": {
                "ml_g_1": dml_obj.predictions["ml_g_1"].reshape(-1, 1),
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
            boot_t_stat = manual_med.boot_med(
                thetas=res_manual["thetas"],
                ses=res_manual["ses"],
                all_g_hat0=res_manual["all_g_hat0"],
                all_g_hat1=res_manual["all_g_hat1"],
                all_m_hat=res_manual["all_m_hat"],
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

        # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
        # dml_obj.sensitivity_analysis(rho=0.0)
        # res_dict["sensitivity_ses"] = dml_obj.sensitivity_params["se"]

        # sensitivity tests
        # res_dict["sensitivity_elements"] = dml_obj.sensitivity_elements
        # res_dict["sensitivity_elements_manual"] = outcome_scoring[1].fit_sensitivity_elements_med(
        #    y,
        #    d,
        #    treatment_level,
        #    all_coef=dml_obj.all_coef,
        #    predictions=dml_obj.predictions,
        #    score="MED",
        #    n_rep=1,
        #    normalize_ipw=normalize_ipw,
        # )
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


class MEDCounterfactualEstimation:
    @pytest.fixture(
        scope="module",
        params=[
            [DoubleMLMEDC, ManualMedC],
            [DoubleMLMEDC, ManualMedC],
            [DoubleMLMEDC, ManualMedCAlt],
            [DoubleMLMEDC, ManualMedCAlt],
        ],
    )
    def outcome_scoring(request):
        return request.param

    @pytest.fixture(
        scope="module",
        params=[
            [
                LinearRegression(),
                ElasticNet(l1_ratio=1, max_iter=250, random_state=42),
                ElasticNet(l1_ratio=1, max_iter=250, random_state=42),
            ],
            [
                LinearRegression(),
                ElasticNet(l1_ratio=1, max_iter=250, random_state=42),
                ElasticNet(l1_ratio=1, max_iter=250, random_state=42),
            ],
            [
                LinearRegression(),
                ElasticNet(l1_ratio=1, max_iter=250, random_state=42),
                ElasticNet(l1_ratio=1, max_iter=250, random_state=42),
            ],
            [
                LinearRegression(),
                ElasticNet(l1_ratio=1, max_iter=250, random_state=42),
                ElasticNet(l1_ratio=1, max_iter=250, random_state=42),
            ],
        ],
    )
    def learner(request):
        return request.param

    @pytest.fixture(scope="module", params=[False, True, False, True])
    def normalize_ipw(request):
        return request.param

    @pytest.fixture(
        scope="module",
        params=[
            0.2,
            0.15,
            0.2,
            0.15,
        ],
    )
    def trimming_threshold(request):
        return request.param

    @pytest.fixture(scope="module", params=[0, 1, 0, 1])
    def treatment_level(request):
        return request.param

    @pytest.fixture(scope="module", params=[1, 0, 1, 0])
    def mediation_level(request):
        return request.param

    @pytest.fixture(scope="module")
    def dml_med_fixture(learner, outcome_scoring, normalize_ipw, trimming_threshold, treatment_level, mediation_level):
        boot_methods = ["normal"]
        n_folds = 2
        n_rep_boot = 499

        # Set machine learning methods for m & g
        ml_g = clone(learner[0])
        ml_m = clone(learner[1])
        ml_med_or_nested = clone(learner[2])

        np.random.seed(3141)
        n_obs = 500
        data_med = make_med_data(n_obs=n_obs)
        y = data_med["y"]
        x = data_med["x"]
        d = data_med["d"]
        m = data_med["m"]
        df_med = pd.DataFrame(
            np.column_stack((y, d, x)), columns=["y", "d", "m"] + ["x" + str(i) for i in range(data_med["x"].shape[1])]
        )

        dml_data = dml.DoubleMLMediationData(df_med, "y", "d", "m")
        all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

        np.random.seed(3141)
        dml_obj = outcome_scoring[0](
            dml_data,
            ml_g,
            ml_m,
            ml_med_or_nested,
            treatment_level=treatment_level,
            n_folds=n_folds,
            score="MED",
            normalize_ipw=normalize_ipw,
            draw_sample_splitting=False,
            trimming_threshold=trimming_threshold,
        )

        # synchronize the sample splitting
        dml_obj.set_sample_splitting(all_smpls=all_smpls)
        dml_obj.fit()

        np.random.seed(3141)
        res_manual = outcome_scoring[1].fit_med(
            y,
            x,
            d,
            m,
            clone(learner[0]),
            clone(learner[1]),
            clone(learner[2]),
            treatment_level=treatment_level,
            all_smpls=all_smpls,
            score="MED",
            normalize_ipw=normalize_ipw,
            trimming_threshold=trimming_threshold,
        )

        np.random.seed(3141)
        # test with external nuisance predictions
        dml_obj_ext = outcome_scoring[0](
            dml_data,
            ml_g,
            ml_m,
            ml_med_or_nested,
            treatment_level=treatment_level,
            n_folds=n_folds,
            score="MED",
            normalize_ipw=normalize_ipw,
            draw_sample_splitting=False,
            trimming_threshold=trimming_threshold,
        )

        # synchronize the sample splitting
        dml_obj_ext.set_sample_splitting(all_smpls=all_smpls)

        prediction_dict = {
            "d": {
                "ml_g_1": dml_obj.predictions["ml_g_1"].reshape(-1, 1),
                "ml_med_or_nested": dml_obj.predictions["ml_med_or_nested"].reshape(-1, 1),
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
            boot_t_stat = outcome_scoring[1].boot_med(
                y,
                d,
                treatment_level,
                res_manual["thetas"],
                res_manual["ses"],
                res_manual["all_g_hat0"],
                res_manual["all_g_hat1"],
                res_manual["all_m_hat"],
                all_smpls,
                score="MED",
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
        res_dict["sensitivity_elements_manual"] = outcome_scoring[1].fit_sensitivity_elements_med(
            y,
            d,
            treatment_level,
            all_coef=dml_obj.all_coef,
            predictions=dml_obj.predictions,
            score="MED",
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

    @pytest.mark.ci
    def test_dml_med_cmed_gmed(treatment_level, cov_type):
        n = 20
        # collect data
        np.random.seed(42)
        obj_dml_data = make_med_data(n_obs=n, dim_x=2)

        # First stage estimation
        ml_g = RandomForestRegressor(n_estimators=10)
        ml_m = RandomForestClassifier(n_estimators=10)

        dml_obj = dml.DoubleMLMediationData(
            obj_dml_data, ml_m=ml_m, ml_g=ml_g, treatment_level=treatment_level, trimming_threshold=0.05, n_folds=5
        )

        dml_obj.fit()
        # create a random basis
        random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 5)))
        cmed = dml_obj.cmed(random_basis, cov_type=cov_type)
        assert isinstance(cmed, dml.utils.blp.DoubleMLBLP)
        assert isinstance(cmed.confint(), pd.DataFrame)
        assert cmed.blp_model.cov_type == cov_type

        groups_1 = pd.DataFrame(
            np.column_stack([obj_dml_data.data["X1"] <= -1.0, obj_dml_data.data["X1"] > 0.2]), columns=["Group 1", "Group 2"]
        )
        msg = "At least one group effect is estimated with less than 6 observations."
        with pytest.warns(UserWarning, match=msg):
            gmed_1 = dml_obj.gmed(groups_1, cov_type=cov_type)
        assert isinstance(gmed_1, dml.utils.blp.DoubleMLBLP)
        assert isinstance(gmed_1.confint(), pd.DataFrame)
        assert all(gmed_1.confint().index == groups_1.columns.to_list())
        assert gmed_1.blp_model.cov_type == cov_type

        np.random.seed(42)
        groups_2 = pd.DataFrame(np.random.choice(["1", "2"], n, p=[0.1, 0.9]))
        msg = "At least one group effect is estimated with less than 6 observations."
        with pytest.warns(UserWarning, match=msg):
            gmed_2 = dml_obj.gmed(groups_2, cov_type=cov_type)
        assert isinstance(gmed_2, dml.utils.blp.DoubleMLBLP)
        assert isinstance(gmed_2.confint(), pd.DataFrame)
        assert all(gmed_2.confint().index == ["Group_1", "Group_2"])
        assert gmed_2.blp_model.cov_type == cov_type
