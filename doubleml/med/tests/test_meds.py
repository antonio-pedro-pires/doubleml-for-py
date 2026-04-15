from random import seed

import numpy as np
import pytest

from doubleml.double_ml_framework import concat
from doubleml.med import DoubleMLMEDS

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")


@pytest.fixture(scope="module")
def meds_kwargs(dml_data, learner_linear, double_sample_splitting, ps_processor_config):
    return {
        "dml_data": dml_data,
        "n_folds": 5,
        "n_rep": 1,
        "n_folds_inner": 5,
        "score": "efficient-alt",
        "normalize_ipw": False,
        "draw_sample_splitting": True,
        "double_sample_splitting": double_sample_splitting,
        "ps_processor_config": ps_processor_config,
        **learner_linear,
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
def individual_med_objs(meds_obj, learner_linear, med_factory, double_sample_splitting):
    kwargs = {
        "score": "efficient-alt",
        "n_folds": meds_obj.n_folds,
        "n_rep": meds_obj.n_rep,
        "n_folds_inner": meds_obj.n_folds_inner,
        "normalize_ipw": meds_obj.normalize_ipw,
        "draw_sample_splitting": False,
        "double_sample_splitting": double_sample_splitting,
        "ps_processor_config": meds_obj.ps_processor_config,
    }
    smpls_inner = None if not meds_obj.double_sample_splitting else meds_obj.smpls_inner
    smpls = meds_obj._smpls

    individual_modeldict = {}
    for model_id, model in meds_obj.modeldict.items():
        if model.outcome == "potential":
            ind_model = med_factory(
                outcome=model.outcome, treatment_level=model.treatment_level, learners=learner_linear, **kwargs
            )
            ind_model._smpls = meds_obj._smpls
        elif model.outcome == "counterfactual":
            ind_model = med_factory(
                outcome=model.outcome, treatment_level=model.treatment_level, learners=learner_linear, **kwargs
            )
        individual_modeldict[model_id] = ind_model
        ind_model.set_sample_splitting(smpls=smpls, smpls_inner=smpls_inner)
    return individual_modeldict


@pytest.mark.ci
def test_meds_modeldict(meds_obj):
    # Test that meds has the correct amount of models.
    n_models = len(meds_obj.models_ids)
    assert len(meds_obj.modeldict) == n_models


@pytest.mark.ci
def test_set_smpls(meds_obj, individual_med_objs):
    # Test that smpls and smpls_inner are correctly set for every model.
    # TODO: Add failing tests
    reference_smpls = meds_obj.smpls
    reference_smpls_inner = meds_obj.smpls_inner

    [np.testing.assert_equal(reference_smpls, model.smpls) for _, model in individual_med_objs.items()]
    if meds_obj._double_sample_splitting:
        [
            (np.testing.assert_equal(reference_smpls_inner, model.smpls_inner) if model.outcome == "counterfactual" else None)
            for _, model in individual_med_objs.items()
        ]


@pytest.fixture(scope="module")
def fit_objs(meds_obj, individual_med_objs):
    seed(123)
    meds_obj.fit()

    seed(123)
    framework_list = [None] * len(individual_med_objs)
    for idx, model in enumerate(individual_med_objs.values()):
        framework_list[idx] = model.fit().framework
    individual_med_objs_framework = concat(framework_list)
    return meds_obj, individual_med_objs_framework


@pytest.mark.ci
def test_fit(fit_objs):
    # Test that the meds object models are fitted identically.
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
def test_effects_binary_treats(fit_objs, individual_med_objs):
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
