import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.data import DoubleMLMEDData


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
def med_classifier_obj(binary_y_data, classifier_learners, binary_outcomes, binary_treats, ps_processor_config):
    med_obj = DoubleMLMED(
        binary_y_data,
        outcome=binary_outcomes,
        treatment_level=binary_treats,
        double_sample_splitting=True,
        n_rep=1,
        n_folds=2,
        n_folds_inner=2,
        ps_processor_config=ps_processor_config,
        **classifier_learners,
    )
    med_obj.fit()
    med_obj.bootstrap()
    return med_obj


# The three following tests verify that DoubleMLMED doesn't break when passed
# mostly classifiers (all learners except for ml_nested_g).
# This is why the assertions check if the returned values are finite,
# not if they are equal to the expected values.
@pytest.mark.ci
def test_med_classifier_coef(med_classifier_obj):
    assert np.all(np.isfinite(med_classifier_obj.coef))


@pytest.mark.ci
def test_med_classifier_se(med_classifier_obj):
    assert np.all(np.isfinite(med_classifier_obj.se))


@pytest.mark.ci
def test_med_classifier_boot(med_classifier_obj):
    assert np.all(np.isfinite(med_classifier_obj.boot_t_stat))
