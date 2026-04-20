import pytest

from doubleml.med import DoubleMLMEDS


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

    for (model_id, model), (tuned_id, tuned) in zip(dml_meds_obj.modeldict.items(), tuned_models.items()):
        assert model_id == tuned_id
        model_learners = model.learner_names
        tuned_learners = tuned[0].keys()
        assert set(tuned_learners) == set(model_learners)
