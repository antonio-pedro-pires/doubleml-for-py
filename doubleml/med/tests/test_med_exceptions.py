"""Test the input validation and error messages of DoubleMLMED."""

import copy
import re

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.data import DoubleMLData, DoubleMLMEDData
from doubleml.med.datasets import make_med_data
from doubleml.med.med import DoubleMLMED


@pytest.fixture(scope="module")
def check_med_data_fixture():
    good_data = make_med_data()

    x = good_data.x
    y = good_data.y
    d = good_data.d
    m = good_data.m
    z = copy.deepcopy(m)

    not_med_data = DoubleMLData.from_arrays(x=x, y=y, d=d)
    med_data_instrumental = DoubleMLMEDData.from_arrays(x=x, y=y, d=d, m=m, z=z)

    d_not_bin = [d[d_obs] if d[d_obs] % 10 != 0 else 2 for d_obs in range(len(d))]
    med_data_not_binary_treats = DoubleMLMEDData.from_arrays(x=x, y=y, m=m, d=d_not_bin)
    return good_data, not_med_data, med_data_instrumental, med_data_not_binary_treats


@pytest.fixture(scope="module")
def binary_y_data(dml_data):
    np.random.seed(3141)
    binary_y = np.random.randint(0, 2, dml_data.n_obs)
    data = DoubleMLMEDData.from_arrays(y=binary_y, x=dml_data.x, m=dml_data.m, d=dml_data.d)
    assert data.binary_outcome
    return data


def _ignore_invalid_learner_warnings(func):
    func = pytest.mark.filterwarnings("ignore:Learner provided for ml_m is probably invalid:UserWarning")(func)
    return pytest.mark.filterwarnings("ignore:Learner provided for ml_M is probably invalid:UserWarning")(func)


@pytest.mark.ci
def test_med_accepts_valid_data(check_med_data_fixture, binary_treats, learner_linear, ps_processor_config):
    """Test that DoubleMLMED accepts valid mediation data."""
    good_data, _, _, _ = check_med_data_fixture
    med_obj = DoubleMLMED(
        obj_dml_data=good_data,
        outcome="factual",
        treatment_level=binary_treats,
        ml_m=learner_linear["ml_m"],
        ml_g=learner_linear["ml_g"],
        ps_processor_config=ps_processor_config,
    )
    assert med_obj.treatment_level == binary_treats


@pytest.mark.ci
def test_med_data_type_raises(check_med_data_fixture, binary_treats, learner_linear, ps_processor_config):
    """Test that DoubleMLMED raises a TypeError for data that is not DoubleMLMEDData."""
    _, not_med_data, _, _ = check_med_data_fixture
    msg = (
        "Mediation analysis requires data of type DoubleMLMEDData."
        + f" Data of type {str(type(not_med_data))} was provided instead."
    )
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(
            obj_dml_data=not_med_data,
            outcome="factual",
            treatment_level=binary_treats,
            ml_m=learner_linear["ml_m"],
            ml_g=learner_linear["ml_g"],
            ps_processor_config=ps_processor_config,
        )


@pytest.mark.ci
def test_med_non_binary_treatment_raises(check_med_data_fixture, binary_treats, learner_linear, ps_processor_config):
    """Test that DoubleMLMED raises a ValueError for a treatment that is not binary."""
    _, _, _, med_data_not_binary_treats = check_med_data_fixture
    with pytest.raises(ValueError, match=r"Treatment variables for mediation analysis must be binary and take values 1 or 0."):
        DoubleMLMED(
            obj_dml_data=med_data_not_binary_treats,
            outcome="factual",
            treatment_level=binary_treats,
            ml_m=learner_linear["ml_m"],
            ml_g=learner_linear["ml_g"],
            ps_processor_config=ps_processor_config,
        )


@pytest.mark.ci
def test_med_instrumental_variables_raises(check_med_data_fixture, binary_treats, learner_linear, ps_processor_config):
    """Test that DoubleMLMED raises a NotImplementedError for data with instrumental variables."""
    _, _, med_data_instrumental, _ = check_med_data_fixture
    with pytest.raises(NotImplementedError, match="instrumental variables for mediation analysis is not yet implemented."):
        DoubleMLMED(
            obj_dml_data=med_data_instrumental,
            outcome="factual",
            treatment_level=binary_treats,
            ml_m=learner_linear["ml_m"],
            ml_g=learner_linear["ml_g"],
            ps_processor_config=ps_processor_config,
        )


@pytest.mark.ci
@pytest.mark.parametrize("outcome", ["factual", "counterfactual"])
def test_med_accepts_valid_outcomes(dml_data, binary_treats, learner_linear, outcome):
    """Test that DoubleMLMED accepts the outcomes factual and counterfactual."""
    med_obj = DoubleMLMED(obj_dml_data=dml_data, outcome=outcome, treatment_level=binary_treats, **learner_linear)
    assert med_obj.outcome == outcome


@pytest.mark.ci
def test_med_outcome_type_raises(dml_data, binary_treats, learner_linear):
    """Test that DoubleMLMED raises a TypeError for an outcome that is not a string."""
    learners = {"ml_g": learner_linear["ml_g"], "ml_m": learner_linear["ml_m"]}
    msg = "Outcome must be a string." + f"{str(2)} of type {str(type(2))} provided instead."
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(obj_dml_data=dml_data, outcome=2, treatment_level=binary_treats, **learners)


@pytest.mark.ci
def test_med_outcome_value_raises(dml_data, binary_treats, learner_linear):
    """Test that DoubleMLMED raises a ValueError for an outcome that is neither factual nor counterfactual."""
    learners = {"ml_g": learner_linear["ml_g"], "ml_m": learner_linear["ml_m"]}
    msg = "Invalid outcome other. Valid outcomes factual or counterfactual."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(obj_dml_data=dml_data, outcome="other", treatment_level=binary_treats, **learners)


@pytest.mark.ci
def test_med_accepts_valid_treatment_levels(dml_data, binary_treats, learner_linear, ps_processor_config):
    """Test that DoubleMLMED accepts the treatment levels 0 and 1."""
    med_obj = DoubleMLMED(
        obj_dml_data=dml_data,
        outcome="factual",
        treatment_level=binary_treats,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )
    assert med_obj.treatment_level == binary_treats


@pytest.mark.ci
def test_med_treatment_level_type_raises(dml_data, learner_linear, ps_processor_config):
    """Test that DoubleMLMED raises a TypeError for a treatment level that is not an integer."""
    msg = "Treatment level must be an integer." + f" Treatment level {str(1.5)} of type {str(type(1.5))} provided."
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(
            obj_dml_data=dml_data,
            outcome="factual",
            treatment_level=1.5,
            ps_processor_config=ps_processor_config,
            **learner_linear,
        )


@pytest.mark.ci
def test_med_treatment_level_value_raises(dml_data, learner_linear, ps_processor_config):
    """Test that DoubleMLMED raises a ValueError for a treatment level that is neither 0 nor 1."""
    msg = "Treatment level must be either 0 or 1" + f" Treatment level provided was {str(2)}."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            obj_dml_data=dml_data,
            outcome="factual",
            treatment_level=2,
            ps_processor_config=ps_processor_config,
            **learner_linear,
        )


@pytest.mark.ci
@_ignore_invalid_learner_warnings
def test_med_factual_accepts_good_learners(dml_data, binary_treats, learner_linear, ps_processor_config):
    """Test that DoubleMLMED with a factual outcome accepts a regressor ml_g and a classifier ml_m."""
    med_obj = DoubleMLMED(
        obj_dml_data=dml_data,
        outcome="factual",
        treatment_level=binary_treats,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )
    assert med_obj.outcome == "factual"


@pytest.mark.ci
def test_med_factual_missing_learner_raises(dml_data, binary_treats, learner_linear, ps_processor_config):
    """Test that DoubleMLMED with a factual outcome raises a ValueError if ml_g is missing."""
    learners = {name: learner for name, learner in learner_linear.items() if name != "ml_g"}
    msg = "Learner ml_g is required when the outcome is factual."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            obj_dml_data=dml_data,
            outcome="factual",
            treatment_level=binary_treats,
            ps_processor_config=ps_processor_config,
            **learners,
        )


@pytest.mark.ci
@_ignore_invalid_learner_warnings
@pytest.mark.parametrize("data_fixture, ml_g", [("binary_y_data", LogisticRegression()), ("dml_data", LinearRegression())])
def test_med_factual_ml_m_regressor_raises(request, binary_treats, data_fixture, ml_g):
    """Test that DoubleMLMED with a factual outcome raises a ValueError if ml_m is a regressor."""
    data = request.getfixturevalue(data_fixture)
    with pytest.raises(ValueError, match=r"Learner 'ml_m' must be a classifier"):
        DoubleMLMED(obj_dml_data=data, outcome="factual", treatment_level=binary_treats, ml_m=LinearRegression(), ml_g=ml_g)


@pytest.mark.ci
@_ignore_invalid_learner_warnings
@pytest.mark.parametrize(
    "data_fixture, ml_g, msg",
    [
        ("binary_y_data", LinearRegression(), "Learner 'ml_g' must be a classifier."),
        ("dml_data", LogisticRegression(), "Learner 'ml_g' must be a regressor."),
    ],
    ids=["binary_y", "continuous_y"],
)
def test_med_factual_ml_g_type_raises(request, binary_treats, data_fixture, ml_g, msg):
    """Test that DoubleMLMED raises a ValueError if ml_g is a regressor for a binary y or a classifier for a continuous y."""
    data = request.getfixturevalue(data_fixture)
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(data, binary_treats, "factual", ml_m=LogisticRegression(), ml_g=ml_g)


@pytest.mark.ci
@_ignore_invalid_learner_warnings
def test_med_counterfactual_accepts_good_learners(dml_data, binary_treats, learner_linear, ps_processor_config):
    """Test that DoubleMLMED with a counterfactual outcome accepts ml_G, ml_nested_g, ml_m and ml_M of the right type."""
    med_obj = DoubleMLMED(
        obj_dml_data=dml_data,
        outcome="counterfactual",
        treatment_level=binary_treats,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )
    assert med_obj.outcome == "counterfactual"


@pytest.mark.ci
@pytest.mark.parametrize("missing", ["ml_G", "ml_nested_g", "ml_M"])
def test_med_counterfactual_missing_learner_raises(dml_data, binary_treats, learner_linear, ps_processor_config, missing):
    """Test that DoubleMLMED with a counterfactual outcome raises a ValueError if ml_G, ml_nested_g or ml_M is missing."""
    learners = {name: learner for name, learner in learner_linear.items() if name != missing}
    msg = f"Learner {missing} is required when the outcome is counterfactual."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            obj_dml_data=dml_data,
            outcome="counterfactual",
            treatment_level=binary_treats,
            ps_processor_config=ps_processor_config,
            **learners,
        )


@pytest.mark.ci
@_ignore_invalid_learner_warnings
@pytest.mark.parametrize("data_fixture, ml_G", [("binary_y_data", LogisticRegression()), ("dml_data", LinearRegression())])
@pytest.mark.parametrize(
    "wrong_learner, msg",
    [
        ("ml_nested_g", "Learner 'ml_nested_g' must be a regressor."),
        ("ml_m", "Learner 'ml_m' must be a classifier."),
        ("ml_M", "Learner 'ml_M' must be a classifier."),
    ],
)
def test_med_counterfactual_fixed_learner_type_raises(request, binary_treats, data_fixture, ml_G, wrong_learner, msg):
    """Test that DoubleMLMED raises a ValueError if ml_nested_g is a classifier or ml_m or ml_M is a regressor."""
    data = request.getfixturevalue(data_fixture)
    learners = {"ml_m": LogisticRegression(), "ml_M": LogisticRegression(), "ml_G": ml_G, "ml_nested_g": LinearRegression()}
    learners[wrong_learner] = LogisticRegression() if wrong_learner == "ml_nested_g" else LinearRegression()
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(obj_dml_data=data, treatment_level=binary_treats, outcome="counterfactual", **learners)


@pytest.mark.ci
@_ignore_invalid_learner_warnings
@pytest.mark.parametrize(
    "data_fixture, ml_G, msg",
    [
        ("binary_y_data", LinearRegression(), "Learner 'ml_G' must be a classifier."),
        ("dml_data", LogisticRegression(), "Learner 'ml_G' must be a regressor."),
    ],
    ids=["binary_y", "continuous_y"],
)
def test_med_counterfactual_ml_G_type_raises(request, binary_treats, data_fixture, ml_G, msg):
    """Test that DoubleMLMED raises a ValueError if ml_G is a regressor for a binary y or a classifier for a continuous y."""
    data = request.getfixturevalue(data_fixture)
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            obj_dml_data=data,
            treatment_level=binary_treats,
            outcome="counterfactual",
            ml_m=LogisticRegression(),
            ml_M=LogisticRegression(),
            ml_G=ml_G,
            ml_nested_g=LinearRegression(),
        )


@pytest.mark.ci
def test_med_sensitivity_not_implemented(dml_data, learner_linear):
    """Test that sensitivity analysis raises a NotImplementedError for DoubleMLMED."""
    dml_med_obj = DoubleMLMED(
        obj_dml_data=dml_data,
        outcome="factual",
        treatment_level=1,
        ml_g=learner_linear["ml_g"],
        ml_m=learner_linear["ml_m"],
        n_folds=2,
    )
    dml_med_obj.fit()

    msg = "Sensitivity analysis is not implemented for this model."
    with pytest.raises(NotImplementedError, match=msg):
        dml_med_obj.sensitivity_analysis()


@pytest.mark.ci
def test_med_multiple_treatments_exception(med_data_two_treatments, learner_linear):
    """Test that DoubleMLMED raises a ValueError for data with more than one treatment variable."""
    msg = re.escape("Mediation analysis supports exactly one treatment variable, got 2: ['d', 'd2'].")
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(obj_dml_data=med_data_two_treatments, treatment_level=1, outcome="factual", **learner_linear)


@pytest.mark.ci
@pytest.mark.parametrize("numpy_level", [np.int64(1), np.int32(0)], ids=["int64", "int32"])
def test_med_accepts_numpy_integer_treatment_level(dml_data, learner_linear, numpy_level):
    """Test that treatment levels given as numpy integers are accepted and stored as int."""
    med_obj = DoubleMLMED(obj_dml_data=dml_data, outcome="factual", treatment_level=numpy_level, **learner_linear)

    assert med_obj.treatment_level == int(numpy_level)
    assert type(med_obj.treatment_level) is int


class _NanRegressor(LinearRegression):
    """Regressor whose predictions are not finite."""

    def predict(self, X):
        return np.full(len(X), np.nan)


@pytest.mark.ci
def test_non_finite_inner_predictions_name_learner(dml_data, learner_linear):
    """Test that the error for non-finite inner predictions of ml_G names the nuisance learner."""
    learners = {**learner_linear, "ml_G": _NanRegressor()}
    med_obj = DoubleMLMED(obj_dml_data=dml_data, outcome="counterfactual", treatment_level=1, **learners)

    msg = r"Predictions from learner _NanRegressor\(\) for ml_G are not finite\."
    with pytest.raises(ValueError, match=msg):
        med_obj.fit()
