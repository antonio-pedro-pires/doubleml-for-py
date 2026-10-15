"""Test hyperparameter tuning of DoubleMLMEDS."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml.med import DoubleMLMEDS
from doubleml.med.datasets import make_med_data
from doubleml.tests._utils_tune_optuna import _basic_optuna_settings


@pytest.fixture
def dml_meds_obj(dml_data, learner_tree, ps_processor_config):
    return DoubleMLMEDS(dml_data, ps_processor_config=ps_processor_config, double_sample_splitting=True, **learner_tree)


@pytest.fixture
def tuned_models(dml_meds_obj, optuna_params, optuna_settings):
    return dml_meds_obj.tune_ml_models(ml_param_space=optuna_params, optuna_settings=optuna_settings, return_tune_res=True)


@pytest.mark.ci
def test_tune_meds(tuned_models, dml_meds_obj):
    # doubleml/med/tests/test_med_tune.py already checks that a tuned model is better than an untuned one.
    # DoubleMLMEDS tuning works by calling the tune method on each DoubleMLMED model it instantiates.
    # Therefore, since test_med_tune.py is already responsible for checking that tuned models perform better than
    # untuned ones, test_meds_tune.py's sole responsibility is to check that each DoubleMLMED instance tuning method
    # is correctly called. The tuning of a DoubleMLMEDS is correct when for each given "outcome" and "treatment" parameters,
    # the DoubleMLMED instance tunes the correct learners.

    """Test that tuning DoubleMLMEDS tunes the learners of each model."""
    for (model_id, model), (tuned_id, tuned) in zip(dml_meds_obj.modeldict.items(), tuned_models.items()):
        assert model_id == tuned_id
        model_learners = model.learner_names
        tuned_learners = tuned[0].keys()
        assert set(tuned_learners) == set(model_learners)


def _alpha_space(trial):
    return {"alpha": trial.suggest_float("alpha", 0.01, 1.0)}


@pytest.fixture
def small_meds():
    np.random.seed(3141)
    data = make_med_data(n_obs=300, dim_x=5)
    return DoubleMLMEDS(
        data,
        ml_m=LogisticRegression(),
        ml_g=Lasso(),
        ml_G=Lasso(),
        ml_M=LogisticRegression(),
        ml_nested_g=Lasso(),
        n_folds=3,
        n_folds_inner=2,
    )


def _tuned_alpha(model, learner):
    params = model.params[learner]["d"][0]
    return None if params is None else params[0].get("alpha")


@pytest.mark.ci
def test_tune_meds_with_a_subset_of_the_learners(small_meds):
    """Test that only the learners in ml_param_space are tuned, as in DoubleMLMED."""
    small_meds.tune_ml_models(
        {"ml_G": _alpha_space, "ml_nested_g": _alpha_space}, optuna_settings=_basic_optuna_settings({"n_trials": 3})
    )

    for model in small_meds.modeldict.values():
        if model.outcome == "counterfactual":
            assert _tuned_alpha(model, "ml_G") is not None
            assert _tuned_alpha(model, "ml_nested_g") is not None
        else:
            assert _tuned_alpha(model, "ml_g") is None, "ml_g is not in ml_param_space and must stay untuned."


@pytest.mark.ci
def test_tune_meds_factual_learners_only(small_meds):
    """Test that the counterfactual models are skipped if no learner in ml_param_space belongs to them."""
    small_meds.tune_ml_models({"ml_g": _alpha_space}, optuna_settings=_basic_optuna_settings({"n_trials": 3}))

    for model in small_meds.modeldict.values():
        if model.outcome == "factual":
            assert _tuned_alpha(model, "ml_g") is not None
        else:
            assert _tuned_alpha(model, "ml_G") is None


@pytest.mark.ci
def test_tune_meds_none_leaves_a_learner_untuned(small_meds):
    """Test that a learner set to None in ml_param_space keeps the parameters of the learner passed to the constructor."""
    small_meds.tune_ml_models(
        {"ml_G": _alpha_space, "ml_nested_g": None}, optuna_settings=_basic_optuna_settings({"n_trials": 3})
    )

    counterfactual = [model for model in small_meds.modeldict.values() if model.outcome == "counterfactual"]
    assert all(_tuned_alpha(model, "ml_G") != Lasso().alpha for model in counterfactual)
    assert all(_tuned_alpha(model, "ml_nested_g") == Lasso().alpha for model in counterfactual)


@pytest.mark.ci
def test_tune_meds_exceptions(small_meds):
    """Test that tune_ml_models raises a ValueError for an empty parameter space or unknown learners."""
    with pytest.raises(ValueError, match="ml_param_space must be a non-empty dictionary."):
        small_meds.tune_ml_models({})
    with pytest.raises(
        ValueError, match="Invalid ml_param_space keys for DoubleMLMEDS: ml_x. Valid keys are: ml_g, ml_m, ml_G, ml_M"
    ):
        small_meds.tune_ml_models({"ml_G": _alpha_space, "ml_x": _alpha_space})
