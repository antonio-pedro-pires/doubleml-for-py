import itertools
from copy import deepcopy
from random import seed

import pytest

from doubleml.med import DoubleMLMEDS


@pytest.fixture(scope="module")
def learners(learner_tree):
    return learner_tree


@pytest.fixture(scope="module")
def med_objs(meds_data, learners, med_factory):

    meds_obj = DoubleMLMEDS(meds_data, **learners)

    smpls = meds_obj.smpls
    smpls_inner = meds_obj.smpls_inner

    scores = list(itertools.product(["potential", "counterfactual"], [0, 1]))

    individual_med_objs = {}

    for target, treatment in scores:
        model = med_factory(target, treatment, learners)
        model._set_sample_splitting(smpls)
        if target == "counterfactual":
            model._set_sample_inner_splitting(smpls_inner)

        individual_med_objs[f"{target}_{treatment}"] = model

    return meds_obj, individual_med_objs


def _get_param_space_for_target(target, optuna_params):
    if target == "potential":
        return {k: v for k, v in optuna_params.items() if k in ["ml_yx", "ml_px"]}
    elif target == "counterfactual":
        return {k: v for k, v in optuna_params.items() if k in ["ml_px", "ml_ymx", "ml_pmx", "ml_nested"]}


@pytest.fixture(
    scope="module",
)
def tune_res(med_objs, optuna_params, optuna_settings):
    meds_obj, individual_med_objs = med_objs
    seed(123)

    # Deepcopy settings to ensure independent sampler states
    optuna_settings_meds = deepcopy(optuna_settings)
    tune_meds_res = meds_obj.tune_ml_models(
        ml_param_space=optuna_params, optuna_settings=optuna_settings_meds, return_tune_res=True
    )
    seed(123)
    tune_ind_med_res = {}

    # same idea as above
    optuna_settings_ind = deepcopy(optuna_settings)
    for key, model in individual_med_objs.items():
        ml_param_space = _get_param_space_for_target(model.target, optuna_params)
        tune_ind_med_res[key] = model.tune_ml_models(
            ml_param_space=ml_param_space, optuna_settings=optuna_settings_ind, return_tune_res=True
        )
    return tune_meds_res, tune_ind_med_res


@pytest.mark.ci
def test_tune_meds(med_objs, tune_res):
    # Test that the tuned models in the DoubleMLMEDS object are identical to their individually tuned counterpart.
    meds_obj, _ = med_objs
    meds_scores, ind_med_scores = tune_res
    for score, ind_score in zip(meds_scores, ind_med_scores):
        assert score == ind_score
        meds_res = meds_scores[score][0]
        ind_res = ind_med_scores[ind_score][0]
        for learner_meds, learner_ind in zip(meds_res, ind_res):
            assert learner_meds == learner_ind
            assert meds_res[learner_meds].best_params == ind_res[learner_ind].best_params
            assert meds_res[learner_meds].best_score == ind_res[learner_ind].best_score
            assert meds_res[learner_meds].tuned == ind_res[learner_ind].tuned
            assert meds_res[learner_meds].params_name == ind_res[learner_ind].params_name
