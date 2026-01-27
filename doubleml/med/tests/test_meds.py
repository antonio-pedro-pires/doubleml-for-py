from random import seed

import numpy as np
import pytest

from doubleml.double_ml_framework import concat
from doubleml.med import DoubleMLMEDS


# TODO: Will need to test with data with multiple m columns
@pytest.fixture(scope="module")
def learners(learner_linear):
    return learner_linear


@pytest.fixture(scope="module")
def meds_kwargs(meds_data, learners, double_sample_splitting):
    return {
        "meds_data": meds_data,
        "n_folds": 5,
        "n_rep": 1,
        "n_folds_inner": 5,
        "score": "MED",
        "normalize_ipw": False,
        "trimming_threshold": 1e-2,
        "order": 1,
        "multmed": True,
        "draw_sample_splitting": True,
        "double_sample_splitting": double_sample_splitting,
        **learners,
    }


@pytest.fixture(scope="module")
def meds_obj(meds_kwargs):
    meds_obj = DoubleMLMEDS(**meds_kwargs)
    return meds_obj


@pytest.fixture(scope="module")
def smpls_inner_outer(meds_obj):
    return meds_obj.smpls, meds_obj.smpls_inner


@pytest.fixture(scope="module")
def treatment_mediation(meds_obj):
    return meds_obj.treatment_mediation_levels

@pytest.fixture(scope="module", params=[True, False])
def double_sample_splitting(request):
    return request.param

@pytest.fixture(scope="module")
def individual_med_objs(meds_obj, learners, med_factory, double_sample_splitting):
    kwargs = {
        "score": "MED",
        "n_folds": meds_obj.n_folds,
        "n_rep": meds_obj.n_rep,
        "n_folds_inner": meds_obj.n_folds_inner,
        "normalize_ipw": meds_obj.normalize_ipw,
        "trimming_threshold": meds_obj.trimming_threshold,
        "draw_sample_splitting": False,
        "double_sample_splitting": double_sample_splitting,
    }
    individual_modeldict = {}
    for target, treatment in meds_obj.scores:
        if target == "potential":
            kwargs.update(
                {
                    "learners": learners,
                    "target": target,
                    "treatment_level": treatment,
                }
            )
            model = med_factory(**kwargs)
            model._smpls = meds_obj._smpls
            individual_modeldict[f"{target}_{treatment}"] = model
        elif target == "counterfactual":
            kwargs.update(
                {
                    "learners": learners,
                    "target": target,
                    "treatment_level": treatment,
                }
            )
            model = med_factory(**kwargs)
            model._smpls = meds_obj._smpls
            model._smpls_inner = meds_obj._smpls_inner
            individual_modeldict[f"{target}_{treatment}"] = model

    return individual_modeldict


@pytest.mark.ci
def test_meds_modeldict(meds_obj):
    # Test that meds has the correct amount of models.
    n_models = len(meds_obj.scores)
    assert len(meds_obj.modeldict) == n_models


@pytest.mark.ci
def test_set_smpls(meds_obj, individual_med_objs):
    # Test that smpls and smpls_inner are correctly set for every model.
    # TODO: Add failing tests
    [
        np.testing.assert_equal(meds_obj.smpls, individual_med_objs[f"{target}_{treatment}"].smpls)
        for (target, treatment) in meds_obj.scores
    ]
    if meds_obj._double_sample_splitting:
        [
            (
                np.testing.assert_equal(meds_obj.smpls_inner, individual_med_objs[f"{target}_{treatment}"].smpls_inner)
                if target == "counterfactual"
                else None
            )
            for (target, treatment) in meds_obj.scores
        ]


@pytest.fixture(scope="module")
def fit_objs(meds_obj, individual_med_objs):
    seed(123)
    meds_obj.fit()

    seed(123)
    framework_list = [None] * len(individual_med_objs)
    for idx, model in enumerate(individual_med_objs.values()):
        framework_list[idx] = model.fit().framework  # TODO: make framework dict instead of list
    individual_med_objs_framework = concat(framework_list)
    return meds_obj, individual_med_objs_framework


@pytest.mark.ci
def test_fit(fit_objs):
    # Test that the meds object models are fitted identically those same models when fitted individually using the DoubleMLMED class.
    meds_obj, individual_med_objs_framework = fit_objs
    [
        np.testing.assert_array_equal(
            meds_obj.framework.__getattribute__(elem), individual_med_objs_framework.__getattribute__(elem)
        )
        for elem in [
            "all_pvals",
            "all_ses",
            "all_t_stats",
            "all_thetas",
        ]
    ]


@pytest.mark.ci
def test_effects(fit_objs, individual_med_objs):
    meds_obj, _ = fit_objs
    effects_names = [
        "ATE",
        "DIR_TREAT",
        "DIR_CONTROL",
        "INDIR_TREAT",
        "INDIR_CONTROL",
    ]
    meds_obj.evaluate_effects()

    assert all([names == valid_names for names, valid_names in zip(meds_obj._effects, effects_names)])

    individual_effects = {
        "ATE": individual_med_objs["potential_1"].framework - individual_med_objs["potential_0"].framework,
        "DIR_TREAT": individual_med_objs["potential_1"].framework - individual_med_objs["counterfactual_0"].framework,
        "DIR_CONTROL": individual_med_objs["counterfactual_1"].framework - individual_med_objs["potential_0"].framework,
        "INDIR_TREAT": individual_med_objs["potential_1"].framework - individual_med_objs["counterfactual_1"].framework,
        "INDIR_CONTROL": individual_med_objs["counterfactual_0"].framework - individual_med_objs["potential_0"].framework,
    }

    for effects in effects_names:
        assert meds_obj._effects[effects].all_thetas == individual_effects[effects].all_thetas
        assert meds_obj._effects[effects].all_ses == individual_effects[effects].all_ses
        assert meds_obj._effects[effects].all_pvals == individual_effects[effects].all_pvals
