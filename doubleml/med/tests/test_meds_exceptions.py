"""Test the input validation and error messages of DoubleMLMEDS."""

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
    return DoubleMLMEDS(obj_dml_data=dml_data, **learner_linear)


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
    """Test that evaluate_effects raises a ValueError before fit."""
    with pytest.raises(ValueError, match=re.escape("Apply fit() before calling evaluate_effects()")):
        meds_obj.evaluate_effects()


@pytest.mark.ci
def test_meds_data_type_exception(learners, bad_data):
    """Test that DoubleMLMEDS raises a TypeError for data that is not DoubleMLMEDData."""
    not_med_data, _, _ = bad_data
    with pytest.raises(TypeError, match="Mediation analysis requires data"):
        DoubleMLMEDS(obj_dml_data=not_med_data, **learners)


@pytest.mark.ci
def test_meds_non_binary_treatment_exception(learners, bad_data):
    """Test that DoubleMLMEDS raises a ValueError for a non-binary treatment."""
    _, _, med_data_non_binary = bad_data
    msg = "Treatment variables for mediation analysis must be binary and take values 1 or 0."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DoubleMLMEDS(obj_dml_data=med_data_non_binary, **learners)


@pytest.mark.ci
def test_meds_multiple_treatments_exception(learners, med_data_two_treatments):
    """Test that DoubleMLMEDS raises a ValueError for more than one treatment variable."""
    msg = re.escape("Mediation analysis supports exactly one treatment variable, got 2: ['d', 'd2'].")
    with pytest.raises(ValueError, match=msg):
        DoubleMLMEDS(obj_dml_data=med_data_two_treatments, **learners)


@pytest.mark.ci
def test_meds_instrumental_variables_exception(learners, bad_data):
    """Test that DoubleMLMEDS raises a NotImplementedError for instrumental variables."""
    _, med_data_iv, _ = bad_data
    with pytest.raises(NotImplementedError, match="instrumental variables for mediation analysis is not yet implemented."):
        DoubleMLMEDS(obj_dml_data=med_data_iv, **learners)


@pytest.mark.ci
def test_meds_smpls_not_set_exception(dml_data, learners):
    """Test that accessing smpls without a sample splitting raises a ValueError."""
    msg = "Sample splitting not specified. Draw samples via .draw_sample splitting(). External samples not implemented yet."
    meds_obj_no_smpls = DoubleMLMEDS(obj_dml_data=dml_data, draw_sample_splitting=False, **learners)
    with pytest.raises(ValueError, match=re.escape(msg)):
        _ = meds_obj_no_smpls.smpls


@pytest.mark.ci
def test_meds_confint_before_fit_exception(meds_obj):
    """Test that confint raises a ValueError before fit."""
    with pytest.raises(ValueError, match=re.escape("Apply fit() before confint().")):
        meds_obj.confint()


@pytest.mark.ci
def test_meds_external_invalid_model_id_raises(dml_data, learner_linear):
    """Test that external predictions with keys that are no model id raise a ValueError."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, **learner_linear)
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
def test_meds_external_model_not_dict_raises(dml_data, learner_linear):
    """Test that external predictions of a model that are not a dictionary raise a TypeError."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, **learner_linear)
    model_id = meds_obj.models_ids[0]
    invalid_ext_preds = {model_id: "not_a_dict"}
    msg = (
        f"external_predictions[{model_id}] must hold a dictionary. "
        + f"Current value in external_predictions[{model_id}] is of type {type('not_a_dict')}"
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        meds_obj.fit(external_predictions=invalid_ext_preds)


@pytest.mark.ci
def test_meds_external_invalid_learner_raises(dml_data, learner_linear):
    """Test that external predictions with unknown learner names raise a ValueError."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, **learner_linear)
    model_id = meds_obj.models_ids[0]
    d_col = meds_obj._dml_data.d_cols[0]
    invalid_ext_preds = {model_id: {d_col: {"ml_invalid": np.zeros((1, 1))}}}
    msg = f"external_predictions[{d_col}] must hold a dictionnary whose keys are a "
    with pytest.raises(ValueError, match=re.escape(msg)):
        meds_obj.fit(external_predictions=invalid_ext_preds)


@pytest.mark.ci
def test_meds_external_invalid_treatment_raises(dml_data, learner_linear):
    """Test that external predictions keyed by a column that is no treatment variable raise a ValueError."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, **learner_linear)
    model_id = meds_obj.models_ids[0]
    d_cols = meds_obj._dml_data.d_cols
    invalid_ext_preds = {model_id: {"not_a_column": {"ml_m": np.zeros((1, 1))}}}
    msg = (
        f"external_predictions[{model_id}] must be keyed by the treatment variables. "
        + f"Expected keys: {set(d_cols)}. Passed keys: {{'not_a_column'}}."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        meds_obj.fit(external_predictions=invalid_ext_preds)


@pytest.mark.ci
def test_meds_external_treatment_not_dict_raises(dml_data, learner_linear):
    """Test that external predictions of a treatment variable that are not a dictionary raise a TypeError."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, **learner_linear)
    model_id = meds_obj.models_ids[0]
    d_col = meds_obj._dml_data.d_cols[0]
    invalid_ext_preds = {model_id: {d_col: [1, 2, 3]}}
    msg = f"external_predictions[{model_id}][{d_col}] must hold a dictionary. " + f"Current value is of type {type([1, 2, 3])}"
    with pytest.raises(TypeError, match=re.escape(msg)):
        meds_obj.fit(external_predictions=invalid_ext_preds)


@pytest.mark.ci
def test_meds_external_inner_key_out_of_range_raises(dml_data, learner_linear):
    """Test that external inner predictions for a fold that does not exist raise a ValueError."""
    n_folds = 2
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, n_folds=n_folds, n_folds_inner=2, **learner_linear)
    model_id = "counterfactual_1"
    d_col = meds_obj._dml_data.d_cols[0]
    out_of_range_key = f"ml_G_inner_{n_folds}"  # inner entries exist for the folds 0 to n_folds - 1
    invalid_ext_preds = {model_id: {d_col: {out_of_range_key: np.zeros((1, 1))}}}
    msg = f"external_predictions[{d_col}] must hold a dictionnary whose keys are a "
    with pytest.raises(ValueError, match=re.escape(msg)):
        meds_obj.fit(external_predictions=invalid_ext_preds)


@pytest.mark.ci
@pytest.mark.parametrize(
    "method",
    [
        lambda meds_obj: meds_obj.modeldict,
        lambda meds_obj: meds_obj.fit(),
        lambda meds_obj: meds_obj.tune_ml_models({"ml_g": None}),
    ],
    ids=["modeldict", "fit", "tune_ml_models"],
)
def test_methods_need_a_sample_splitting(dml_data, learner_linear, method):
    """Test that methods that need models report a missing sample splitting like a missing smpls."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, draw_sample_splitting=False, **learner_linear)
    msg = "Sample splitting not specified. Draw samples via .draw_sample splitting()"
    with pytest.raises(ValueError, match=re.escape(msg)):
        method(meds_obj)


@pytest.mark.ci
def test_draw_sample_splitting_creates_the_models(dml_data, learner_linear):
    """Test that drawing a sample splitting after construction creates the four models."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, draw_sample_splitting=False, **learner_linear)
    meds_obj.draw_sample_splitting()
    assert len(meds_obj.modeldict) == 4


@pytest.mark.ci
def test_external_predictions_must_be_a_dictionary(meds_obj):
    """Test that fit raises a TypeError if external_predictions is not a dictionary."""
    msg = "external_predictions must be a dictionary. Object of type <class 'list'> passed."
    with pytest.raises(TypeError, match=re.escape(msg)):
        meds_obj.fit(external_predictions=[1])


@pytest.mark.ci
def test_meds_callable_score_exception(dml_data, learners):
    """Test that DoubleMLMEDS rejects a callable score."""
    msg = r"score should be a string"
    with pytest.raises(TypeError, match=msg):
        DoubleMLMEDS(obj_dml_data=dml_data, score=lambda y, d, g_hat, m_hat, smpls: (y, d), **learners)
