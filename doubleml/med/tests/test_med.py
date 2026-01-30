import math

import numpy as np
import pytest


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

@pytest.fixture(
    scope="module",
    params=[True, False]
)
def double_sample_splitting(request):
    return request.param

@pytest.fixture(scope="module")
def med_objs(
    learners,
    binary_targets,
    binary_scores,
    normalize_ipw,
    trimming_threshold,
    binary_treats,
    n_folds,
    med_factory,
    double_sample_splitting,
):

    kwargs = {
        "target": binary_targets,
        "treatment_level": binary_treats,
        "learners": learners,
        "score": binary_scores,
        "n_folds": n_folds,
        "normalize_ipw": normalize_ipw,
        "trimming_threshold": trimming_threshold,
        "double_sample_splitting": double_sample_splitting,

    }

    np.random.seed(3141)
    med_obj = med_factory(**kwargs)

    np.random.seed(3141)
    med_obj_ext = med_factory(**kwargs)

    if med_obj.double_sample_splitting:
        smpls_inner = med_obj._smpls_inner
        med_obj_ext._set_sample_inner_splitting(smpls_inner)

    med_obj_ext._smpls = med_obj.smpls

    return med_obj, med_obj_ext


@pytest.fixture(scope="module")
def dml_med_fixture(
    med_objs,
    n_rep_boot,
):
    boot_methods = ["normal"]

    med_obj, med_obj_ext = med_objs

    for obj in med_objs:
        np.random.seed(3141)
        obj.fit()
    
    prediction_dict = {"d": _get_preds(med_obj, med_obj.learner.keys())}

    if med_obj.double_sample_splitting and med_obj.target == "counterfactual":
        for i in range(med_obj.n_folds_inner):
            prediction_dict["d"][f"ml_ymx_inner_{i}"] = med_obj.predictions[f"ml_ymx_inner_{i}"][:, :, 0]

    med_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = _get_res(med_obj, med_obj_ext, boot_methods, n_rep_boot)

    return res_dict


@pytest.mark.ci
def test_dml_med_coef(dml_med_fixture):
    res_dict = dml_med_fixture
    assert math.isclose(res_dict["coef"], res_dict["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_se(dml_med_fixture):
    res_dict = dml_med_fixture
    assert math.isclose(res_dict["se"], res_dict["se_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_boot(dml_med_fixture):
    res_dict = dml_med_fixture

    for bootstrap in res_dict["boot_methods"]:
        assert np.allclose(
            res_dict["boot_t_stat" + bootstrap],
            res_dict["boot_t_stat" + bootstrap + "_ext"],
            rtol=1e-9,
            atol=1e-4,
        )


#TODO: Add way to run this test only when double_sample_splitting fixture is true.
@pytest.mark.ci
def test_set_smpls_inner_splitting(med_objs):
    med_obj, med_obj_ext = med_objs
    if med_obj.double_sample_splitting:
        smpls_inner = med_obj.smpls_inner
        smpls = med_obj.smpls

        med_obj_ext._smpls = smpls
        med_obj_ext._set_sample_inner_splitting(smpls_inner)
        assert med_obj_ext._smpls_inner == med_obj._smpls_inner


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
