import pytest
import re
from doubleml.med.meds import DoubleMLMEDS

@pytest.fixture(scope="module", params=["learner_linear", "learner_tree"])
def learners(request):
    return request.getfixturevalue(request.param)

@pytest.fixture
def meds_obj(dml_data, learners):
    return DoubleMLMEDS(dml_data=dml_data, **learners)

@pytest.mark.ci
def test_evaluate_effects_exception(meds_obj):
    with pytest.raises(ValueError, match=re.escape("Apply fit() before calling evaluate_effects()")):
        meds_obj.evaluate_effects()