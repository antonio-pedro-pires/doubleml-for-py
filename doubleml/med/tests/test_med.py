import math

import numpy as np
import pytest


def _get_preds(obj, keys):
    return {k: obj.predictions[k].reshape(-1, 1) for k in keys}


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





@pytest.fixture(scope="module")
def n_folds():
    return 5


@pytest.fixture(scope="module")
def n_rep_boot():
    return 499


@pytest.fixture(
    scope="module",
    params=["learner_linear", "learner_forest"],
)
def learners(request):
    return request.getfixturevalue(request.param)


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
    meds_data,
    learners,
    normalize_ipw,
    trimming_threshold,
    treatment_mediation_level_counterfactual,
    treatment_level,
    n_folds,
    n_rep_boot,
    med_factory,
):

    treatment_level, mediation_level = treatment_mediation_level_counterfactual

    counter_kwargs = {
        "target": "counterfactual",
        "treatment_level": treatment_level,
        "learners": learners,
        "mediation_level": mediation_level,
        "score": "MED",
        "score_function": "efficient-alt",
        "n_folds": n_folds,
        "normalize_ipw": normalize_ipw,
        "trimming_threshold": trimming_threshold,
    }

    np.random.seed(3141)
    counter_med_obj = med_factory(**counter_kwargs)

    np.random.seed(3141)
    counter_med_obj_ext = med_factory(**counter_kwargs)

    mediation_level = treatment_level

    pot_kwargs = {
        "target": "potential",
        "treatment_level": treatment_level,
        "learners": learners,
        "mediation_level": mediation_level,
        "score": "MED",
        "score_function": "efficient-alt",
        "n_folds": n_folds,
        "normalize_ipw": normalize_ipw,
        "trimming_threshold": trimming_threshold,
    }

    np.random.seed(3141)
    pot_med_obj = med_factory(**pot_kwargs)

    np.random.seed(3141)
    pot_med_obj_ext = med_factory(**pot_kwargs)

    counter_smpls_inner = counter_med_obj._smpls_inner
    counter_med_obj_ext._smpls = counter_med_obj.smpls
    counter_med_obj_ext._set_sample_inner_splitting(counter_smpls_inner)

    pot_med_obj_ext._smpls = pot_med_obj.smpls
    return counter_med_obj, counter_med_obj_ext, pot_med_obj, pot_med_obj_ext


@pytest.fixture(scope="module")
def dml_med_fixture(
    med_objs,
    n_rep_boot,
):
    boot_methods = ["normal"]

    counter_med_obj, counter_med_obj_ext, pot_med_obj, pot_med_obj_ext = med_objs

    # Fit objects (bootstrapping handled in _get_res)
    for obj in med_objs:
        np.random.seed(3141)
        obj.fit()

    counter_prediction_dict = {"d": _get_preds(counter_med_obj, ["ml_ymx", "ml_px", "ml_pmx", "ml_nested"])}
    pot_prediction_dict = {"d": _get_preds(pot_med_obj, ["ml_yx", "ml_px"])}

    for i in range(counter_med_obj_ext.n_folds_inner):
        counter_prediction_dict["d"][f"ml_ymx_inner_{i}"] = counter_med_obj_ext.predictions[f"ml_ymx_inner_{i}"][:, :, 0]

    # The original code had a loop here, but pot_med_obj_ext.fit was called without 'i'
    # Assuming it should be called once after preparing pot_prediction_dict
    pot_med_obj_ext.fit(external_predictions=pot_prediction_dict)

    counter_res_dict = _get_res(counter_med_obj, counter_med_obj_ext, boot_methods, n_rep_boot)
    pot_res_dict = _get_res(pot_med_obj, pot_med_obj_ext, boot_methods, n_rep_boot)

    return counter_res_dict, pot_res_dict


@pytest.mark.ci
def test_dml_med_coef(dml_med_fixture):
    # The fixture now returns more values, unpack accordingly
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
    med_obj_ext._set_sample_inner_splitting(smpls_inner)
    assert med_obj_ext._smpls_inner == med_obj._smpls_inner
