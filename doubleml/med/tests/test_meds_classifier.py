import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMEDData
from doubleml.med.meds import DoubleMLMEDS


@pytest.fixture(scope="module")
def binary_y_data(dml_data):
    x = dml_data.x
    m = dml_data.m
    d = dml_data.d
    y = dml_data.y
    # Binarize the continuous outcome to test classifier support
    binary_y = (y > np.median(y)).astype(int)
    return DoubleMLMEDData.from_arrays(y=binary_y, x=x, m=m, d=d)


@pytest.fixture(scope="module", params=["learner_linear", "learner_forest"])
def classifier_learners(request):
    if request.param == "learner_linear":
        return {
            "ml_g": LogisticRegression(solver="saga", l1_ratio=1, max_iter=250, random_state=42),
            "ml_m": LogisticRegression(solver="saga", l1_ratio=1, max_iter=250, random_state=42),
            "ml_G": LogisticRegression(solver="saga", l1_ratio=1, max_iter=250, random_state=42),
            "ml_M": LogisticRegression(solver="saga", l1_ratio=1, max_iter=250, random_state=42),
            "ml_nested_g": LinearRegression(),
        }
    else:
        return {
            "ml_g": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
            "ml_m": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
            "ml_G": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
            "ml_M": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
            "ml_nested_g": RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
        }


@pytest.fixture(scope="module")
def meds_classifier_obj(binary_y_data, classifier_learners, ps_processor_config):
    # Ensemble wrapper for mediation effects
    meds_obj = DoubleMLMEDS(
        binary_y_data,
        n_rep=1,
        n_folds=2,
        double_sample_splitting=True,
        ps_processor_config=ps_processor_config,
        **classifier_learners,
    )
    meds_obj.fit()
    return meds_obj


@pytest.mark.ci
def test_meds_classifier_coef(meds_classifier_obj):
    assert np.all(np.isfinite(meds_classifier_obj.framework.all_thetas))


@pytest.mark.ci
def test_meds_classifier_se(meds_classifier_obj):
    assert np.all(np.isfinite(meds_classifier_obj.framework.all_ses))


@pytest.mark.ci
def test_meds_classifier_evaluate_effects(meds_classifier_obj):
    meds_classifier_obj.evaluate_effects()
    for effect_name, effect_obj in meds_classifier_obj._effects.items():
        assert np.all(np.isfinite(effect_obj.all_thetas)), f"Non-finite theta in effect {effect_name}"
        assert np.all(np.isfinite(effect_obj.all_ses)), f"Non-finite se in effect {effect_name}"
