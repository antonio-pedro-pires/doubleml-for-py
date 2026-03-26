import copy
import math
import re

import numpy as np
import pytest


@pytest.fixture(scope="module")
def n_folds():
    return 5


@pytest.fixture(scope="module", params=[1, 2])
def n_rep(request):
    return request.param


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


@pytest.fixture(scope="module", params=[True, False])
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
    n_rep,
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
        "n_rep": n_rep,
    }

    np.random.seed(3141)
    med_obj = med_factory(**kwargs)

    np.random.seed(3141)
    med_obj_ext = med_factory(**kwargs)
    return med_obj, med_obj_ext


@pytest.fixture(scope="module")
def dml_med_fixture(
    med_objs,
    n_rep_boot,
):
    boot_methods = ["normal"]

    med_obj, med_obj_ext = med_objs

    smpls_inner = None if not med_obj.double_sample_splitting else med_obj.smpls_inner
    med_obj_ext._set_smpls_sampling(smpls=med_obj.smpls, smpls_inner=smpls_inner)

    np.random.seed(3141)
    med_obj.fit()

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


@pytest.fixture(scope="module")
def external_predictions_exceptions_fixture(med_factory, learner_linear):
    med_obj = med_factory(target="counterfactual", treatment_level=1, learners=learner_linear)

    med_obj_ext = copy.deepcopy(med_obj)
    med_obj.fit()

    prediction_dict = {"d": _get_preds(med_obj, med_obj.learner.keys())}

    return prediction_dict, med_obj_ext


@pytest.mark.ci
def test_external_predictions_exceptions(external_predictions_exceptions_fixture):
    prediction_dict, med_obj_ext = external_predictions_exceptions_fixture
    with pytest.raises(
        ValueError,
        match=re.escape(
            "When providing external predictions for ml_ymx, also inner predictions for all inner folds "
            + f"have to be provided (missing: {', '.join([str(i) for i in [0, 1, 2, 3, 4]])})."
        ),
    ):
        med_obj_ext.fit(external_predictions=prediction_dict)


@pytest.fixture(scope="module")
def set_smpls_sampling_fixture(med_factory, learner_linear, binary_targets, binary_treats, double_sample_splitting):
    # Treatment_level is hardcoded because set_smpls_sampling does not differentiate based on treatment_level
    med_obj = med_factory(
        target=binary_targets,
        treatment_level=binary_treats,
        learners=learner_linear,
        double_sample_splitting=double_sample_splitting,
    )
    med_obj_ext = med_factory(
        target=binary_targets,
        treatment_level=binary_treats,
        learners=learner_linear,
        double_sample_splitting=double_sample_splitting,
        draw_sample_splitting=False,
    )
    return med_obj, med_obj_ext


@pytest.mark.ci
def test_set_smpls_sampling(set_smpls_sampling_fixture):
    med_obj, med_obj_ext = set_smpls_sampling_fixture
    smpls_inner = None if not med_obj.double_sample_splitting else med_obj.smpls_inner
    med_obj_ext._set_smpls_sampling(smpls=med_obj.smpls, smpls_inner=smpls_inner)
    if med_obj.double_sample_splitting:
        np.testing.assert_equal(med_obj.smpls_inner, med_obj_ext.smpls_inner)
    np.testing.assert_equal(med_obj.smpls, med_obj_ext.smpls)


@pytest.mark.ci
def test_set_smpls_sampling_exceptions(set_smpls_sampling_fixture):
    _, med_obj_ext = set_smpls_sampling_fixture
    with pytest.raises(NotImplementedError, match="sample setting with cluster data and inner samples not supported."):
        med_obj_ext._set_smpls_sampling(smpls=[], smpls_inner=[], all_smpls_cluster=[])
    if med_obj_ext.double_sample_splitting:
        with pytest.raises(ValueError, match="smpls_inner is required"):
            med_obj_ext._set_smpls_sampling(smpls=[], smpls_inner=None)


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
