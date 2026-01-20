import itertools
from copy import deepcopy
from random import seed

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from doubleml import DoubleMLMED, concat
from doubleml.med import DoubleMLMEDS
from doubleml.med.datasets import make_med_data
from doubleml.tests._utils_tune_optuna import (
    _SAMPLER_CASES,
    _basic_optuna_settings,
    _small_tree_params,
)


@pytest.fixture(scope="module", params=[{
    "ml_yx": DecisionTreeRegressor(random_state=123),
    "ml_ymx": DecisionTreeRegressor(random_state=123),
    "ml_px": DecisionTreeClassifier(random_state=123),
    "ml_pmx": DecisionTreeClassifier(random_state=123),
    "ml_nested": DecisionTreeRegressor(random_state=123),
}])
def learners(request):
    return request.param

@pytest.fixture(scope="module", params=[make_med_data()])
def med_data(request):
    return request.param

@pytest.fixture(scope="module")
def optuna_params():
    return  {"ml_yx": _small_tree_params,
             "ml_px": _small_tree_params,
             "ml_ymx": _small_tree_params,
             "ml_pmx": _small_tree_params,
             "ml_nested": _small_tree_params,
             }


@pytest.fixture(scope="module", params=_SAMPLER_CASES, ids=[case[0] for case in _SAMPLER_CASES])
def optuna_settings(request):
    sampler_name, optuna_sampler = request.param
    return _basic_optuna_settings({"sampler": optuna_sampler, "n_trials": 10
                                   })

@pytest.fixture(scope="module")
def med_objs(med_data, learners):

    meds_obj = DoubleMLMEDS(med_data,
                            **learners)

    smpls = meds_obj.smpls
    smpls_inner = meds_obj.smpls_inner

    scores = list(itertools.product(["potential", "counterfactual"], [0, 1]))

    individual_med_objs = {}
    for target, treatment in scores:
        if target=="potential":
            model = DoubleMLMED(med_data=med_data,
                                ml_yx=learners["ml_yx"],
                                ml_px=learners["ml_px"],)
            model._set_sample_splitting(smpls)
        elif target=="counterfactual":
            model = DoubleMLMED(med_data=med_data,
                                **learners)
            model._set_sample_splitting(smpls)
            model._set_sample_inner_splitting(smpls_inner)


        individual_med_objs[f"{target}_{treatment}"] = model

    return meds_obj, individual_med_objs

@pytest.fixture(scope="module",)
def tune_res(med_objs,  optuna_params, optuna_settings):
    meds_obj, individual_med_objs = med_objs
    seed(123)
    tune_meds_res = meds_obj.tune_ml_models(ml_param_space=optuna_params,
                                            optuna_settings=optuna_settings,
                                            return_tune_res=True)
    seed(123)
    tune_ind_med_res={}
    for key, model in individual_med_objs.items():
        if model.target=="potential":
            tune_ind_med_res[key] = model.tune_ml_models(ml_param_space={"ml_yx": optuna_params["ml_yx"],
                                                                         "ml_px": optuna_params["ml_px"]},
                                                         optuna_settings=optuna_settings,
                                                         return_tune_res=True,
                                                         )
        elif model.target=="counterfactual":
            tune_ind_med_res[key] = model.tune_ml_models(ml_param_space={"ml_px": optuna_params["ml_px"],
                                                                         "ml_ymx": optuna_params["ml_ymx"],
                                                                         "ml_pmx": optuna_params["ml_pmx"],
                                                                         "ml_nested": optuna_params["ml_nested"],},
                                                         optuna_settings=optuna_settings,
                                                         return_tune_res=True)
    return tune_meds_res, tune_ind_med_res

@pytest.mark.ci
def test_tune_meds(med_objs, tune_res):
    meds_obj, _ = med_objs
    tune_meds_res, tune_ind_med_res = tune_res
    for score in meds_obj.scores:
        np.testing.assert_array_equal(tune_res[score], tune_meds_res[score])


