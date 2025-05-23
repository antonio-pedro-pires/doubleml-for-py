import numpy as np
import pandas as pd
import plotly
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVR

from doubleml import (
    DoubleMLAPO,
    DoubleMLClusterData,
    DoubleMLCVAR,
    DoubleMLData,
    DoubleMLDID,
    DoubleMLDIDCS,
    DoubleMLFramework,
    DoubleMLIIVM,
    DoubleMLIRM,
    DoubleMLLPQ,
    DoubleMLPLIV,
    DoubleMLPLR,
    DoubleMLPolicyTree,
    DoubleMLPQ,
    DoubleMLSSM,
)
from doubleml.datasets import (
    make_iivm_data,
    make_irm_data,
    make_pliv_CHS2015,
    make_pliv_multiway_cluster_CKMS2021,
    make_plr_CCDDHNR2018,
    make_ssm_data,
)
from doubleml.did.datasets import make_did_SZ2020

np.random.seed(3141)
n_obs = 200
dml_data_plr = make_plr_CCDDHNR2018(n_obs=n_obs)
dml_data_pliv = make_pliv_CHS2015(n_obs=n_obs, dim_z=1)
dml_data_irm = make_irm_data(n_obs=n_obs)
dml_data_iivm = make_iivm_data(n_obs=n_obs)
dml_cluster_data_pliv = make_pliv_multiway_cluster_CKMS2021(N=10, M=10)
dml_data_did = make_did_SZ2020(n_obs=n_obs)
dml_data_did_cs = make_did_SZ2020(n_obs=n_obs, cross_sectional_data=True)
(x, y, d, t) = make_did_SZ2020(n_obs=n_obs, cross_sectional_data=True, return_type="array")
binary_outcome = np.random.binomial(n=1, p=0.5, size=n_obs)
dml_data_did_binary_outcome = DoubleMLData.from_arrays(x, binary_outcome, d)
dml_data_did_cs_binary_outcome = DoubleMLData.from_arrays(x, binary_outcome, d, t=t)
dml_data_ssm = make_ssm_data(n_obs=n_obs)

dml_plr = DoubleMLPLR(dml_data_plr, Lasso(), Lasso())
dml_pliv = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso())
dml_irm = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression())
dml_iivm = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression())
dml_pliv_cluster = DoubleMLPLIV(dml_cluster_data_pliv, Lasso(), Lasso(), Lasso())
dml_cvar = DoubleMLCVAR(dml_data_irm, ml_g=RandomForestRegressor(), ml_m=RandomForestClassifier())
dml_pq = DoubleMLPQ(dml_data_irm, ml_g=RandomForestClassifier(), ml_m=RandomForestClassifier())
dml_lpq = DoubleMLLPQ(dml_data_iivm, ml_g=RandomForestClassifier(), ml_m=RandomForestClassifier())
dml_did = DoubleMLDID(dml_data_did, Lasso(), LogisticRegression())
dml_did_binary_outcome = DoubleMLDID(dml_data_did_binary_outcome, LogisticRegression(), LogisticRegression())
dml_did_cs = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression())
dml_did_cs_binary_outcome = DoubleMLDIDCS(dml_data_did_cs_binary_outcome, LogisticRegression(), LogisticRegression())
dml_ssm = DoubleMLSSM(dml_data_ssm, ml_g=Lasso(), ml_m=LogisticRegression(), ml_pi=LogisticRegression())
dml_apo = DoubleMLAPO(dml_data_irm, Lasso(), LogisticRegression(), treatment_level=0)


@pytest.mark.ci
@pytest.mark.parametrize(
    "dml_obj, cls",
    [
        (dml_plr, DoubleMLPLR),
        (dml_pliv, DoubleMLPLIV),
        (dml_irm, DoubleMLIRM),
        (dml_iivm, DoubleMLIIVM),
        (dml_pliv_cluster, DoubleMLPLIV),
        (dml_cvar, DoubleMLCVAR),
        (dml_pq, DoubleMLPQ),
        (dml_lpq, DoubleMLLPQ),
        (dml_did, DoubleMLDID),
        (dml_did_binary_outcome, DoubleMLDID),
        (dml_did_cs, DoubleMLDIDCS),
        (dml_did_cs_binary_outcome, DoubleMLDIDCS),
        (dml_ssm, DoubleMLSSM),
        (dml_apo, DoubleMLAPO),
    ],
)
def test_return_types(dml_obj, cls):
    # ToDo: A second test case with multiple treatment variables would be helpful
    assert isinstance(dml_obj.__str__(), str)
    assert isinstance(dml_obj.summary, pd.DataFrame)
    assert isinstance(dml_obj.draw_sample_splitting(), cls)
    if not dml_obj._is_cluster_data:
        assert isinstance(dml_obj.set_sample_splitting(dml_obj.smpls), cls)
    else:
        assert isinstance(dml_obj._dml_data, DoubleMLClusterData)
    assert isinstance(dml_obj.fit(), cls)
    assert isinstance(dml_obj.__str__(), str)  # called again after fit, now with numbers
    assert isinstance(dml_obj.summary, pd.DataFrame)  # called again after fit, now with numbers
    if not dml_obj._is_cluster_data:
        assert isinstance(dml_obj.bootstrap(), cls)
    else:
        assert isinstance(dml_obj._dml_data, DoubleMLClusterData)
    assert isinstance(dml_obj.confint(), pd.DataFrame)
    if not dml_obj._is_cluster_data:
        assert isinstance(dml_obj.p_adjust(), pd.DataFrame)
    else:
        isinstance(dml_obj.p_adjust("bonferroni"), pd.DataFrame)
    if isinstance(dml_obj, DoubleMLLPQ):
        assert isinstance(dml_obj.get_params("ml_m_z"), dict)
    elif isinstance(dml_obj, DoubleMLSSM):
        assert isinstance(dml_obj.get_params("ml_g_d0"), dict)
    else:
        assert isinstance(dml_obj.get_params("ml_m"), dict)
    assert isinstance(dml_obj._dml_data.__str__(), str)

    # for the following checks we need additional inputs
    # assert isinstance(dml_obj.set_ml_nuisance_params(), cls)
    # assert isinstance(dml_obj.tune(), cls)


n_treat = 1
n_rep = 2
n_folds = 3
n_obs = 200
n_rep_boot = 314

plr_obj = DoubleMLPLR(dml_data_plr, Lasso(), LinearSVR(), n_rep=n_rep, n_folds=n_folds)
plr_obj.fit(store_models=True)
plr_obj.bootstrap(n_rep_boot=n_rep_boot)

pliv_obj = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso(), n_rep=n_rep, n_folds=n_folds)
pliv_obj.fit()
pliv_obj.bootstrap(n_rep_boot=n_rep_boot)

irm_obj = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(), n_rep=n_rep, n_folds=n_folds, trimming_threshold=0.1)
irm_obj.fit()
irm_obj.bootstrap(n_rep_boot=n_rep_boot)

iivm_obj = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(), n_rep=n_rep, n_folds=n_folds)
iivm_obj.fit()
iivm_obj.bootstrap(n_rep_boot=n_rep_boot)

cvar_obj = DoubleMLCVAR(
    dml_data_irm, ml_g=RandomForestRegressor(), ml_m=RandomForestClassifier(), n_rep=n_rep, n_folds=n_folds
)
cvar_obj.fit()
cvar_obj.bootstrap(n_rep_boot=n_rep_boot)

pq_obj = DoubleMLPQ(dml_data_irm, ml_g=RandomForestClassifier(), ml_m=RandomForestClassifier(), n_rep=n_rep, n_folds=n_folds)
pq_obj.fit()
pq_obj.bootstrap(n_rep_boot=n_rep_boot)

lpq_obj = DoubleMLLPQ(
    dml_data_iivm, ml_g=RandomForestClassifier(), ml_m=RandomForestClassifier(), n_rep=n_rep, n_folds=n_folds
)
lpq_obj.fit()
lpq_obj.bootstrap(n_rep_boot=n_rep_boot)

did_obj = DoubleMLDID(dml_data_did, Lasso(), LogisticRegression(), n_rep=n_rep, n_folds=n_folds)
did_obj.fit()
did_obj.bootstrap(n_rep_boot=n_rep_boot)

did_cs_obj = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression(), n_rep=n_rep, n_folds=n_folds)
did_cs_obj.fit()
did_cs_obj.bootstrap(n_rep_boot=n_rep_boot)

ssm_obj = DoubleMLSSM(
    dml_data_ssm, ml_g=Lasso(), ml_m=LogisticRegression(), ml_pi=LogisticRegression(), n_rep=n_rep, n_folds=n_folds
)
ssm_obj.fit()
ssm_obj.bootstrap(n_rep_boot=n_rep_boot)

apo_obj = DoubleMLAPO(dml_data_irm, Lasso(), LogisticRegression(), treatment_level=0, n_rep=n_rep, n_folds=n_folds)
apo_obj.fit()
apo_obj.bootstrap(n_rep_boot=n_rep_boot)


@pytest.mark.ci
@pytest.mark.parametrize(
    "dml_obj", [plr_obj, pliv_obj, irm_obj, iivm_obj, cvar_obj, pq_obj, lpq_obj, did_obj, did_cs_obj, ssm_obj, apo_obj]
)
def test_property_types_and_shapes(dml_obj):
    # not checked: learner, learner_names, params, params_names, score
    # already checked: summary

    # check that the setting is still in line with the hard-coded values
    assert dml_obj._dml_data.n_treat == n_treat
    assert dml_obj.n_rep == n_rep
    assert dml_obj.n_folds == n_folds
    assert dml_obj._dml_data.n_obs == n_obs
    assert dml_obj.n_rep_boot == n_rep_boot

    assert isinstance(dml_obj.all_coef, np.ndarray)
    assert dml_obj.all_coef.shape == (n_treat, n_rep)

    assert isinstance(dml_obj.all_se, np.ndarray)
    assert dml_obj.all_se.shape == (n_treat, n_rep)

    assert isinstance(dml_obj.boot_t_stat, np.ndarray)
    assert dml_obj.boot_t_stat.shape == (n_rep_boot, n_treat, n_rep)

    assert isinstance(dml_obj.coef, np.ndarray)
    assert dml_obj.coef.shape == (n_treat,)

    assert isinstance(dml_obj.psi, np.ndarray)
    assert dml_obj.psi.shape == (
        n_obs,
        n_rep,
        n_treat,
    )

    assert isinstance(dml_obj.framework, DoubleMLFramework)

    is_nonlinear = isinstance(dml_obj, (DoubleMLPQ, DoubleMLLPQ))
    if is_nonlinear:
        for score_element in dml_obj._score_element_names:
            assert isinstance(dml_obj.psi_elements[score_element], np.ndarray)
            assert dml_obj.psi_elements[score_element].shape == (
                n_obs,
                n_rep,
                n_treat,
            )
    else:
        assert isinstance(dml_obj.psi_elements["psi_a"], np.ndarray)
        assert dml_obj.psi_elements["psi_a"].shape == (
            n_obs,
            n_rep,
            n_treat,
        )

        assert isinstance(dml_obj.psi_elements["psi_b"], np.ndarray)
        assert dml_obj.psi_elements["psi_b"].shape == (
            n_obs,
            n_rep,
            n_treat,
        )

    assert isinstance(dml_obj.pval, np.ndarray)
    assert dml_obj.pval.shape == (n_treat,)

    assert isinstance(dml_obj.se, np.ndarray)
    assert dml_obj.se.shape == (n_treat,)

    assert isinstance(dml_obj.t_stat, np.ndarray)
    assert dml_obj.t_stat.shape == (n_treat,)

    assert isinstance(dml_obj._dml_data.binary_treats, pd.Series)
    assert len(dml_obj._dml_data.binary_treats) == n_treat

    assert isinstance(dml_obj.smpls, list)
    assert len(dml_obj.smpls) == n_rep
    all_tuple = all([all([isinstance(tpl, tuple) for tpl in smpl]) for smpl in dml_obj.smpls])
    assert all_tuple
    all_pairs = all([all([len(tpl) == 2 for tpl in smpl]) for smpl in dml_obj.smpls])
    assert all_pairs
    n_folds_each_smpl = np.array([len(smpl) for smpl in dml_obj.smpls])
    assert np.all(n_folds_each_smpl == n_folds_each_smpl[0])
    assert n_folds_each_smpl[0] == n_folds


@pytest.mark.ci
def test_stored_models():
    assert len(plr_obj.models["ml_l"]["d"]) == n_rep
    assert len(plr_obj.models["ml_m"]["d"]) == n_rep

    n_folds_each_model = np.array([len(mdl) for mdl in plr_obj.models["ml_l"]["d"]])
    assert np.all(n_folds_each_model == n_folds_each_model[0])
    assert n_folds_each_model[0] == n_folds

    n_folds_each_model = np.array([len(mdl) for mdl in plr_obj.models["ml_m"]["d"]])
    assert np.all(n_folds_each_model == n_folds_each_model[0])
    assert n_folds_each_model[0] == n_folds

    assert np.all([isinstance(mdl, plr_obj.learner["ml_l"].__class__) for mdl in plr_obj.models["ml_l"]["d"][0]])
    assert np.all([isinstance(mdl, plr_obj.learner["ml_m"].__class__) for mdl in plr_obj.models["ml_m"]["d"][0]])
    # extend these tests to more models


@pytest.mark.ci
def test_stored_predictions():
    assert plr_obj.predictions["ml_l"].shape == (n_obs, n_rep, n_treat)
    assert plr_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert pliv_obj.predictions["ml_l"].shape == (n_obs, n_rep, n_treat)
    assert pliv_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)
    assert pliv_obj.predictions["ml_r"].shape == (n_obs, n_rep, n_treat)

    assert irm_obj.predictions["ml_g0"].shape == (n_obs, n_rep, n_treat)
    assert irm_obj.predictions["ml_g1"].shape == (n_obs, n_rep, n_treat)
    assert irm_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert iivm_obj.predictions["ml_g0"].shape == (n_obs, n_rep, n_treat)
    assert iivm_obj.predictions["ml_g1"].shape == (n_obs, n_rep, n_treat)
    assert iivm_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)
    assert iivm_obj.predictions["ml_r0"].shape == (n_obs, n_rep, n_treat)
    assert iivm_obj.predictions["ml_r1"].shape == (n_obs, n_rep, n_treat)

    assert cvar_obj.predictions["ml_g"].shape == (n_obs, n_rep, n_treat)
    assert cvar_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert pq_obj.predictions["ml_g"].shape == (n_obs, n_rep, n_treat)
    assert pq_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert lpq_obj.predictions["ml_g_du_z0"].shape == (n_obs, n_rep, n_treat)
    assert lpq_obj.predictions["ml_g_du_z1"].shape == (n_obs, n_rep, n_treat)
    assert lpq_obj.predictions["ml_m_z"].shape == (n_obs, n_rep, n_treat)
    assert lpq_obj.predictions["ml_m_d_z0"].shape == (n_obs, n_rep, n_treat)
    assert lpq_obj.predictions["ml_m_d_z1"].shape == (n_obs, n_rep, n_treat)

    assert did_obj.predictions["ml_g0"].shape == (n_obs, n_rep, n_treat)
    assert did_obj.predictions["ml_g1"].shape == (n_obs, n_rep, n_treat)
    assert did_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert did_cs_obj.predictions["ml_g_d0_t0"].shape == (n_obs, n_rep, n_treat)
    assert did_cs_obj.predictions["ml_g_d0_t1"].shape == (n_obs, n_rep, n_treat)
    assert did_cs_obj.predictions["ml_g_d1_t0"].shape == (n_obs, n_rep, n_treat)
    assert did_cs_obj.predictions["ml_g_d1_t1"].shape == (n_obs, n_rep, n_treat)
    assert did_cs_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert ssm_obj.predictions["ml_g_d0"].shape == (n_obs, n_rep, n_treat)
    assert ssm_obj.predictions["ml_g_d1"].shape == (n_obs, n_rep, n_treat)
    assert ssm_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)
    assert ssm_obj.predictions["ml_pi"].shape == (n_obs, n_rep, n_treat)

    assert apo_obj.predictions["ml_g_d_lvl0"].shape == (n_obs, n_rep, n_treat)
    assert apo_obj.predictions["ml_g_d_lvl1"].shape == (n_obs, n_rep, n_treat)
    assert apo_obj.predictions["ml_m"].shape == (n_obs, n_rep, n_treat)


@pytest.mark.ci
def test_stored_nuisance_targets():
    assert plr_obj.nuisance_targets["ml_l"].shape == (n_obs, n_rep, n_treat)
    assert plr_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert pliv_obj.nuisance_targets["ml_l"].shape == (n_obs, n_rep, n_treat)
    assert pliv_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)
    assert pliv_obj.nuisance_targets["ml_r"].shape == (n_obs, n_rep, n_treat)

    assert irm_obj.nuisance_targets["ml_g0"].shape == (n_obs, n_rep, n_treat)
    assert irm_obj.nuisance_targets["ml_g1"].shape == (n_obs, n_rep, n_treat)
    assert irm_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert iivm_obj.nuisance_targets["ml_g0"].shape == (n_obs, n_rep, n_treat)
    assert iivm_obj.nuisance_targets["ml_g1"].shape == (n_obs, n_rep, n_treat)
    assert iivm_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)
    assert iivm_obj.nuisance_targets["ml_r0"].shape == (n_obs, n_rep, n_treat)
    assert iivm_obj.nuisance_targets["ml_r1"].shape == (n_obs, n_rep, n_treat)

    assert cvar_obj.nuisance_targets["ml_g"].shape == (n_obs, n_rep, n_treat)
    assert cvar_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert pq_obj.nuisance_targets["ml_g"].shape == (n_obs, n_rep, n_treat)
    assert pq_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert lpq_obj.nuisance_targets["ml_g_du_z0"].shape == (n_obs, n_rep, n_treat)
    assert lpq_obj.nuisance_targets["ml_g_du_z1"].shape == (n_obs, n_rep, n_treat)
    assert lpq_obj.nuisance_targets["ml_m_z"].shape == (n_obs, n_rep, n_treat)
    assert lpq_obj.nuisance_targets["ml_m_d_z0"].shape == (n_obs, n_rep, n_treat)
    assert lpq_obj.nuisance_targets["ml_m_d_z1"].shape == (n_obs, n_rep, n_treat)

    assert did_obj.nuisance_targets["ml_g0"].shape == (n_obs, n_rep, n_treat)
    assert did_obj.nuisance_targets["ml_g1"].shape == (n_obs, n_rep, n_treat)
    assert did_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert did_cs_obj.nuisance_targets["ml_g_d0_t0"].shape == (n_obs, n_rep, n_treat)
    assert did_cs_obj.nuisance_targets["ml_g_d0_t1"].shape == (n_obs, n_rep, n_treat)
    assert did_cs_obj.nuisance_targets["ml_g_d1_t0"].shape == (n_obs, n_rep, n_treat)
    assert did_cs_obj.nuisance_targets["ml_g_d1_t1"].shape == (n_obs, n_rep, n_treat)
    assert did_cs_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)

    assert ssm_obj.nuisance_targets["ml_g_d0"].shape == (n_obs, n_rep, n_treat)
    assert ssm_obj.nuisance_targets["ml_g_d1"].shape == (n_obs, n_rep, n_treat)
    assert ssm_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)
    assert ssm_obj.nuisance_targets["ml_pi"].shape == (n_obs, n_rep, n_treat)

    assert apo_obj.nuisance_targets["ml_g_d_lvl0"].shape == (n_obs, n_rep, n_treat)
    assert apo_obj.nuisance_targets["ml_g_d_lvl1"].shape == (n_obs, n_rep, n_treat)
    assert apo_obj.nuisance_targets["ml_m"].shape == (n_obs, n_rep, n_treat)


@pytest.mark.ci
def test_nuisance_loss():
    assert plr_obj.nuisance_loss["ml_l"].shape == (n_rep, n_treat)
    assert plr_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)

    assert pliv_obj.nuisance_loss["ml_l"].shape == (n_rep, n_treat)
    assert pliv_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)
    assert pliv_obj.nuisance_loss["ml_r"].shape == (n_rep, n_treat)

    assert irm_obj.nuisance_loss["ml_g0"].shape == (n_rep, n_treat)
    assert irm_obj.nuisance_loss["ml_g1"].shape == (n_rep, n_treat)
    assert irm_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)

    assert iivm_obj.nuisance_loss["ml_g0"].shape == (n_rep, n_treat)
    assert iivm_obj.nuisance_loss["ml_g1"].shape == (n_rep, n_treat)
    assert iivm_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)
    assert iivm_obj.nuisance_loss["ml_r0"].shape == (n_rep, n_treat)
    assert iivm_obj.nuisance_loss["ml_r1"].shape == (n_rep, n_treat)

    assert cvar_obj.nuisance_loss["ml_g"].shape == (n_rep, n_treat)
    assert cvar_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)

    assert pq_obj.nuisance_loss["ml_g"].shape == (n_rep, n_treat)
    assert pq_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)

    assert lpq_obj.nuisance_loss["ml_g_du_z0"].shape == (n_rep, n_treat)
    assert lpq_obj.nuisance_loss["ml_g_du_z1"].shape == (n_rep, n_treat)
    assert lpq_obj.nuisance_loss["ml_m_z"].shape == (n_rep, n_treat)
    assert lpq_obj.nuisance_loss["ml_m_d_z0"].shape == (n_rep, n_treat)
    assert lpq_obj.nuisance_loss["ml_m_d_z1"].shape == (n_rep, n_treat)

    assert did_obj.nuisance_loss["ml_g0"].shape == (n_rep, n_treat)
    assert did_obj.nuisance_loss["ml_g1"].shape == (n_rep, n_treat)
    assert did_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)

    assert did_cs_obj.nuisance_loss["ml_g_d0_t0"].shape == (n_rep, n_treat)
    assert did_cs_obj.nuisance_loss["ml_g_d0_t1"].shape == (n_rep, n_treat)
    assert did_cs_obj.nuisance_loss["ml_g_d1_t0"].shape == (n_rep, n_treat)
    assert did_cs_obj.nuisance_loss["ml_g_d1_t1"].shape == (n_rep, n_treat)
    assert did_cs_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)

    assert ssm_obj.nuisance_loss["ml_g_d0"].shape == (n_rep, n_treat)
    assert ssm_obj.nuisance_loss["ml_g_d1"].shape == (n_rep, n_treat)
    assert ssm_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)
    assert ssm_obj.nuisance_loss["ml_pi"].shape == (n_rep, n_treat)

    assert apo_obj.nuisance_loss["ml_g_d_lvl0"].shape == (n_rep, n_treat)
    assert apo_obj.nuisance_loss["ml_g_d_lvl1"].shape == (n_rep, n_treat)
    assert apo_obj.nuisance_loss["ml_m"].shape == (n_rep, n_treat)


def _test_sensitivity_return_types(dml_obj, n_rep, n_treat, benchmarking_set):
    assert isinstance(dml_obj.sensitivity_elements, dict)
    for key in ["sigma2", "nu2"]:
        assert isinstance(dml_obj.sensitivity_elements[key], np.ndarray)
        assert dml_obj.sensitivity_elements[key].shape == (1, n_rep, n_treat)
    for key in ["psi_sigma2", "psi_nu2", "riesz_rep"]:
        assert isinstance(dml_obj.sensitivity_elements[key], np.ndarray)
        assert dml_obj.sensitivity_elements[key].shape == (n_obs, n_rep, n_treat)

    assert isinstance(dml_obj.sensitivity_summary, str)
    dml_obj.sensitivity_analysis()
    assert isinstance(dml_obj.sensitivity_summary, str)
    assert isinstance(dml_obj.sensitivity_plot(), plotly.graph_objs._figure.Figure)
    benchmarks = {"cf_y": [0.1, 0.2], "cf_d": [0.15, 0.2], "name": ["test1", "test2"]}
    assert isinstance(dml_obj.sensitivity_plot(value="ci", benchmarks=benchmarks), plotly.graph_objs._figure.Figure)

    assert isinstance(dml_obj.framework._calc_sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95), dict)
    assert isinstance(
        dml_obj.framework._calc_robustness_value(null_hypothesis=0.0, level=0.95, rho=1.0, idx_treatment=0), tuple
    )
    benchmark = dml_obj.sensitivity_benchmark(benchmarking_set=benchmarking_set)
    assert isinstance(benchmark, pd.DataFrame)

    return


@pytest.mark.ci
def test_sensitivity():
    # PLR
    _test_sensitivity_return_types(plr_obj, n_rep, n_treat, benchmarking_set=["X1"])

    # IRM
    _test_sensitivity_return_types(irm_obj, n_rep, n_treat, benchmarking_set=["X1"])

    # DID
    _test_sensitivity_return_types(did_obj, n_rep, n_treat, benchmarking_set=["Z1"])

    # DIDCS
    _test_sensitivity_return_types(did_cs_obj, n_rep, n_treat, benchmarking_set=["Z1"])

    # APO
    _test_sensitivity_return_types(apo_obj, n_rep, n_treat, benchmarking_set=["X1"])


@pytest.mark.ci
def test_policytree():
    dml_irm.fit()
    features = dml_data_irm.data[["X1", "X2"]]
    policy_tree = dml_irm.policy_tree(features, depth=2)
    assert isinstance(policy_tree, DoubleMLPolicyTree)
    predict_features = pd.DataFrame(np.random.normal(size=(5, 2)), columns=features.keys())
    assert isinstance(policy_tree.predict(predict_features), pd.DataFrame)
