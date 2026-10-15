"""Test the constructor signature of DoubleMLMED."""

import inspect

import pytest

from doubleml import DoubleMLMED, DoubleMLMEDS


@pytest.mark.ci
@pytest.mark.parametrize("model_class", [DoubleMLMED, DoubleMLMEDS])
def test_constructor_signature_follows_other_models(model_class):
    """Test that the data comes first as obj_dml_data and score follows n_rep, as in the other models."""
    parameters = list(inspect.signature(model_class).parameters)

    assert parameters[0] == "obj_dml_data"
    assert parameters.index("score") == parameters.index("n_rep") + 1


@pytest.mark.ci
def test_med_constructor_takes_obj_dml_data(dml_data, learner_linear):
    """Test that DoubleMLMED takes the data as obj_dml_data."""
    med_obj = DoubleMLMED(obj_dml_data=dml_data, outcome="factual", treatment_level=1, **learner_linear)
    assert med_obj.score == "efficient-alt"


@pytest.mark.ci
def test_meds_constructor_takes_obj_dml_data(dml_data, learner_linear):
    """Test that DoubleMLMEDS takes the data as obj_dml_data."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, **learner_linear)
    assert meds_obj.score == "efficient-alt"
