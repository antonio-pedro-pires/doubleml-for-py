import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.datasets import make_med_data
from doubleml.med import DoubleMLMED


@pytest.mark.ci
def test_get_training_params():
    pass


@pytest.mark.ci
def test_initialize_preds_dict():

    ml_m = LogisticRegression()
    ml_g = LinearRegression()
    ml_med = LinearRegression()
    ml_nested = LinearRegression()
    med_data = make_med_data()

    # Potential efficient score
    med_obj = DoubleMLMED(med_data, 1, 1, ml_g, ml_m, ml_med, ml_nested, score_type="efficient")
    expected_dict = {
        "predictions": {
            "ml_g_d": None,
            "ml_m": None,
        },
        "targets": {
            "ml_g_d": None,
            "ml_m": None,
        },
        "models": {
            "ml_g_d": None,
            "ml_m": None,
        },
    }
    assert med_obj._initialize_preds_dict() == expected_dict

    # Counterfactual efficient score
    med_obj = DoubleMLMED(med_data, 1, 0, ml_g, ml_m, ml_med, ml_nested, score_type="efficient")
    expected_dict = {
        "predictions": {
            "ml_g_d_med_d": None,
            "ml_g_d_med_1md": None,
            "ml_m": None,
            "ml_med_d": None,
            "ml_med_1md": None,
        },
        "targets": {
            "ml_g_d_med_d": None,
            "ml_g_d_med_1md": None,
            "ml_m": None,
            "ml_med_d": None,
            "ml_med_1md": None,
        },
        "models": {
            "ml_g_d_med_d": None,
            "ml_g_d_med_1md": None,
            "ml_m": None,
            "ml_med_d": None,
            "ml_med_1md": None,
        },
    }
    assert med_obj._initialize_preds_dict() == expected_dict

    # Counterfactual efficient-alt score
    # ["ml_g_d", "ml_g_d_1md", "ml_m", "ml_m_med_x"]
    med_obj = DoubleMLMED(med_data, 1, 0, ml_g, ml_m, ml_med, ml_nested, score_type="efficient-alt")
    expected_dict = {
        "predictions": {
            "ml_g_d": None,
            "ml_g_d_1md": None,
            "ml_m": None,
            "ml_m_med_x": None,
        },
        "targets": {
            "ml_g_d": None,
            "ml_g_d_1md": None,
            "ml_m": None,
            "ml_m_med_x": None,
        },
        "models": {
            "ml_g_d": None,
            "ml_g_d_1md": None,
            "ml_m": None,
            "ml_m_med_x": None,
        },
    }
    assert med_obj._initialize_preds_dict() == expected_dict


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
    assert valid_learner == list(med_obj.params.keys())

    # Counterfactual efficient learners
    med_obj = DoubleMLMED(med_data, 1, 0, ml_g, ml_m, ml_med, ml_nested, score_type="efficient")
    valid_learner = ["ml_m", "ml_g_d_med_d", "ml_g_d_med_1md", "ml_med_d", "ml_med_1md"]
    assert valid_learner == list(med_obj.params.keys())

    # Counterfactual efficient-alt learners
    med_obj = DoubleMLMED(med_data, 1, 0, ml_g, ml_m, ml_med, ml_nested, score_type="efficient-alt")
    valid_learner = ["ml_m", "ml_g_d", "ml_g_d_1md", "ml_m_med_x"]
    assert valid_learner == list(med_obj.params.keys())


@pytest.mark.ci
def test_extract_predictions():
    # Set framework's prediction
    # See that the conditional works correctly.
    # See that an external prediction returns the correct data shape
    # Test the framework's prediction is identical to a prediction made manually.
    pass
