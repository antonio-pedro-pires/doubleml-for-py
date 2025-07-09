import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.datasets import make_med_data
from doubleml.med import DoubleMLMED


@pytest.mark.ci
def test_initialize_ml_nuisance_params():
    ml_m = LogisticRegression()
    ml_g = LinearRegression()
    ml_med = LinearRegression()
    ml_nested = LinearRegression()
    med_data = make_med_data()

    # Potential efficient learners
    med_obj = DoubleMLMED(med_data, 1, 1, ml_g, ml_m, ml_med, ml_nested, score_type="efficient")
    valid_learner = ["ml_m", "ml_g_d"]
    for learner in valid_learner:
        assert learner in set(med_obj.params.keys())
    # Counterfactual efficient learners
    med_obj = DoubleMLMED(med_data, 1, 0, ml_g, ml_m, ml_med, ml_nested, score_type="efficient")
    valid_learner = ["ml_m", "ml_g_d_med_pot", "ml_g_d_med_counter", "ml_med_pot", "ml_med_counter"]
    for learner in valid_learner:
        assert learner in set(med_obj.params.keys())

    # Counterfactual efficient-alt learners
    med_obj = DoubleMLMED(med_data, 1, 0, ml_g, ml_m, ml_med, ml_nested, score_type="efficient-alt")
    valid_learner = ["ml_m", "ml_g_d", "ml_g_nested", "ml_m_med"]
    for learner in valid_learner:
        assert learner in set(med_obj.params.keys())
