import copy
import re

import numpy as np
import pytest

from doubleml.data import DoubleMLData, DoubleMLMEDData
from doubleml.med.datasets import make_med_data
from doubleml.med.meds import DoubleMLMEDS


@pytest.fixture(scope="module")
def learners(learner_linear):
    return learner_linear


@pytest.fixture
def meds_obj(dml_data, learner_linear):
    return DoubleMLMEDS(dml_data=dml_data, **learner_linear)


@pytest.fixture(scope="module")
def bad_data():
    data = make_med_data()
    x, y, d, m = data.x, data.y, data.d, data.m
    z = copy.deepcopy(m)

    not_med_data = DoubleMLData.from_arrays(x=x, y=y, d=d)
    med_data_iv = DoubleMLMEDData.from_arrays(x=x, y=y, d=d, m=m, z=z)
    d_not_bin = np.where(d == 0, 2, d)
    med_data_non_binary = DoubleMLMEDData.from_arrays(x=x, y=y, d=d_not_bin, m=m)
    return not_med_data, med_data_iv, med_data_non_binary


@pytest.mark.ci
def test_evaluate_effects_exception(meds_obj):
    with pytest.raises(ValueError, match=re.escape("Apply fit() before calling evaluate_effects()")):
        meds_obj.evaluate_effects()


@pytest.mark.ci
def test_meds_data_type_exception(learners, bad_data):
    not_med_data, _, _ = bad_data
    msg = f"The data must be of DoubleMLMediationData type. {str(not_med_data)} of type {str(type(not_med_data))} was passed."
    with pytest.raises(TypeError, match=re.escape(msg)):
        DoubleMLMEDS(dml_data=not_med_data, **learners)


@pytest.mark.ci
def test_meds_non_binary_treatment_exception(learners, bad_data):
    _, _, med_data_non_binary = bad_data
    msg = "Treatment variables for mediation analysis must be binary and take values 1 or 0."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DoubleMLMEDS(dml_data=med_data_non_binary, **learners)


@pytest.mark.ci
def test_meds_instrumental_variables_exception(learners, bad_data):
    _, med_data_iv, _ = bad_data
    msg = "instrumental variables for mediation analysis is not yet implemented."
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        DoubleMLMEDS(dml_data=med_data_iv, **learners)


@pytest.mark.ci
def test_meds_smpls_not_set_exception(dml_data, learners):
    msg = "Sample splitting not specified. Draw samples via .draw_sample splitting(). External samples not implemented yet."
    meds_obj_no_smpls = DoubleMLMEDS(dml_data=dml_data, draw_sample_splitting=False, **learners)
    with pytest.raises(ValueError, match=re.escape(msg)):
        _ = meds_obj_no_smpls.smpls


@pytest.mark.ci
def test_meds_confint_before_fit_exception(meds_obj):
    with pytest.raises(ValueError, match=re.escape("Apply fit() before confint().")):
        meds_obj.confint()


@pytest.mark.ci
def test_meds_external_predictions_invalid_keys_exception(dml_data, learner_linear):
    meds_obj = DoubleMLMEDS(dml_data=dml_data, **learner_linear)
    invalid_ext_preds = {"invalid_key": {}}
    expected_keys = set(meds_obj.models_ids)
    passed_keys = set(invalid_ext_preds.keys())
    msg = (
        "external_predictions must be a subset of all scores. "
        + f"Expected keys: {expected_keys}. "
        + f"Passed keys: {passed_keys}."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        meds_obj.fit(external_predictions=invalid_ext_preds)


@pytest.mark.ci
def test_meds_external_predictions_not_dict_exception(dml_data, learner_linear):
    meds_obj = DoubleMLMEDS(dml_data=dml_data, **learner_linear)
    model_id = meds_obj.models_ids[0]
    invalid_ext_preds = {model_id: "not_a_dict"}
    msg = (
        f"external_predictions[{model_id}] must hold a dictionary. "
        + f"Current value in external_predictions[{model_id}] is of type {type('not_a_dict')}"
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        meds_obj.fit(external_predictions=invalid_ext_preds)
