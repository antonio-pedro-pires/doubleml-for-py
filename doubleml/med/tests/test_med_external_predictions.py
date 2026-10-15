"""Test that external predictions reproduce the estimates of DoubleMLMED."""

import copy
import math
import re

import numpy as np
import pytest

from doubleml import DoubleMLMED


@pytest.fixture(scope="module")
def med_objs(
    dml_data,
    learner_linear,
    binary_outcomes,
    binary_scores,
    normalize_ipw,
    binary_treats,
    n_folds,
    double_sample_splitting,
    n_rep,
    ps_processor_config,
):

    kwargs = {
        "obj_dml_data": dml_data,
        "outcome": binary_outcomes,
        "treatment_level": binary_treats,
        "score": "efficient-alt",
        "normalize_ipw": normalize_ipw,
        "double_sample_splitting": double_sample_splitting,
        "ps_processor_config": ps_processor_config,
        **learner_linear,
    }

    np.random.seed(3141)
    med_obj = DoubleMLMED(**kwargs)
    np.random.seed(3141)
    med_obj_ext = DoubleMLMED(**kwargs)
    return med_obj, med_obj_ext


@pytest.fixture(scope="module")
def dml_med_fixture(
    med_objs,
    n_rep_boot,
):
    boot_methods = ["normal"]

    med_obj, med_obj_ext = med_objs

    smpls_inner = None if not med_obj.double_sample_splitting else med_obj.smpls_inner
    med_obj_ext.set_sample_splitting(smpls=med_obj.smpls, smpls_inner=smpls_inner)

    np.random.seed(3141)
    med_obj.fit()

    prediction_dict = {"d": _get_preds(med_obj, med_obj.learner.keys())}

    if med_obj.outcome == "counterfactual":
        for i in range(med_obj.n_folds):
            prediction_dict["d"][f"ml_G_inner_{i}"] = med_obj.predictions[f"ml_G_inner_{i}"][:, :, 0]

    med_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = _get_res(med_obj, med_obj_ext, boot_methods, n_rep_boot)

    return res_dict


@pytest.mark.ci
def test_dml_med_coef(dml_med_fixture):
    """Test that the coefficient with external predictions matches the reference fit."""
    res_dict = dml_med_fixture
    assert math.isclose(res_dict["coef"], res_dict["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_se(dml_med_fixture):
    """Test that the standard error with external predictions matches the reference fit."""
    res_dict = dml_med_fixture
    assert math.isclose(res_dict["se"], res_dict["se_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_boot(dml_med_fixture):
    """Test that the bootstrapped t-statistics with external predictions match the reference fit."""
    res_dict = dml_med_fixture

    for bootstrap in res_dict["boot_methods"]:
        assert np.allclose(
            res_dict["boot_t_stat" + bootstrap],
            res_dict["boot_t_stat" + bootstrap + "_ext"],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.fixture(scope="module")
def external_predictions_exceptions_fixture(dml_data, learner_linear, ps_processor_config):
    med_obj = DoubleMLMED(
        dml_data,
        outcome="counterfactual",
        treatment_level=1,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )

    med_obj_ext = copy.deepcopy(med_obj)
    med_obj.fit()

    prediction_dict = {"d": _get_preds(med_obj, med_obj.learner.keys())}

    return prediction_dict, med_obj_ext


@pytest.mark.ci
def test_external_predictions_exceptions(external_predictions_exceptions_fixture):
    """Test that external predictions of ml_G without the inner predictions raise a ValueError."""
    prediction_dict, med_obj_ext = external_predictions_exceptions_fixture
    with pytest.raises(
        ValueError,
        match=re.escape(
            "When providing external predictions for ml_G, also inner predictions for all inner folds "
            + f"have to be provided (missing: {', '.join([str(i) for i in [0, 1, 2, 3, 4]])})."
        ),
    ):
        med_obj_ext.fit(external_predictions=prediction_dict)


def _get_preds(obj, keys):
    return {k: np.array([np.ndarray.flatten(subarray) for subarray in obj.predictions[k]]) for k in keys}


def _get_res(obj, obj_ext, boot_methods, n_rep_boot):
    res_dict = {
        "coef": obj.coef.item(),
        "coef_ext": obj_ext.coef.item(),
        "se": obj.se.item(),
        "se_ext": obj_ext.se.item(),
        "boot_methods": boot_methods,
    }
    for bootstrap in boot_methods:
        np.random.seed(3141)
        obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict["boot_t_stat" + bootstrap] = obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_ext"] = obj_ext.boot_t_stat
    return res_dict


@pytest.mark.ci
@pytest.mark.parametrize("n_outer_folds", [2, 3])
@pytest.mark.parametrize("double_sample_splitting", [True, False], ids=["double_split", "single_split"])
def test_med_inner_learner_names(dml_data, learner_linear, n_outer_folds, double_sample_splitting):
    """Test that the counterfactual model has one ml_G_inner entry per outer fold, with single and double sample splitting."""
    med_obj = DoubleMLMED(
        obj_dml_data=dml_data,
        outcome="counterfactual",
        treatment_level=1,
        n_folds=n_outer_folds,
        double_sample_splitting=double_sample_splitting,
        **learner_linear,
    )

    expected_names = ["ml_m", "ml_G", "ml_M", "ml_nested_g"] + [f"ml_G_inner_{i}" for i in range(n_outer_folds)]
    assert med_obj.params_names == expected_names


@pytest.mark.ci
def test_external_ml_G_needs_inner_preds_single_split(dml_data, learner_linear):
    """Test that external predictions of ml_G also require the per-fold inner predictions without double sample splitting."""
    med_obj = DoubleMLMED(
        obj_dml_data=dml_data,
        outcome="counterfactual",
        treatment_level=1,
        n_folds=3,
        double_sample_splitting=False,
        **learner_linear,
    )
    med_obj.fit()
    external_predictions = {"d": {"ml_G": med_obj.predictions["ml_G"][:, :, 0]}}

    msg = re.escape("When providing external predictions for ml_G, also inner predictions for all inner folds ")
    with pytest.raises(ValueError, match=msg):
        med_obj.fit(external_predictions=external_predictions)
