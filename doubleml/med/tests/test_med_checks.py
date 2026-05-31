import copy

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
def check_med_outcomes_fixture():
    return ["factual", "counterfactual", "other", 2]


@pytest.fixture(scope="module")
def check_med_levels_fixture():
    return [(1, 1), (1.5, 1), (1, "one"), (2, 1)]


@pytest.fixture(scope="module")
def check_med_learners_fixture(learner_linear):
    good_learners = copy.deepcopy(learner_linear)
    missing_g_learner = copy.deepcopy(learner_linear)
    del missing_g_learner["ml_g"]
    missing_G_learner = copy.deepcopy(learner_linear)
    del missing_G_learner["ml_G"]
    return good_learners, missing_g_learner, missing_G_learner


@pytest.mark.ci
def test_med_data_check(check_med_data_fixture, binary_treats, learner_linear, ps_processor_config):
    good_data, not_med_data, med_data_instrumental, med_data_not_binary_treats = check_med_data_fixture
    ml_m = learner_linear["ml_m"]
    ml_g = learner_linear["ml_g"]
    DoubleMLMED(
        dml_data=good_data,
        outcome="factual",
        treatment_level=binary_treats,
        ml_m=ml_m,
        ml_g=ml_g,
        ps_processor_config=ps_processor_config,
    )

    msg = (
        "Mediation analysis requires data of type DoubleMLMediationData."
        + f" Data of type {str(type(not_med_data))} was provided instead."
    )
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(
            dml_data=not_med_data,
            outcome="factual",
            treatment_level=binary_treats,
            ml_m=ml_m,
            ml_g=ml_g,
            ps_processor_config=ps_processor_config,
        )

    with pytest.raises(ValueError, match=r"Treatment variables for mediation analysis must be binary and take values 1 or 0."):
        DoubleMLMED(
            dml_data=med_data_not_binary_treats,
            outcome="factual",
            treatment_level=binary_treats,
            ml_m=ml_m,
            ml_g=ml_g,
            ps_processor_config=ps_processor_config,
        )

    with pytest.raises(NotImplementedError, match="instrumental variables for mediation analysis is not yet implemented."):
        DoubleMLMED(
            dml_data=med_data_instrumental,
            outcome="factual",
            treatment_level=binary_treats,
            ml_m=ml_m,
            ml_g=ml_g,
            ps_processor_config=ps_processor_config,
        )


@pytest.mark.ci
def test_med_outcome_check(dml_data, binary_treats, check_med_outcomes_fixture, learner_linear):
    factual_t, counterfactual_t, value_error_t, type_error_t = check_med_outcomes_fixture

    learners = {"ml_g": learner_linear["ml_g"], "ml_m": learner_linear["ml_m"]}

    msg = "Outcome must be a string." + f"{str(type_error_t)} of type {str(type(type_error_t))} provided instead."
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(dml_data=dml_data, outcome=type_error_t, treatment_level=binary_treats, **learners)

    valid_outcomes = ["factual", "counterfactual"]
    msg = f"Invalid outcome {value_error_t}. " + "Valid outcomes " + " or ".join(valid_outcomes) + "."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(dml_data=dml_data, outcome=value_error_t, treatment_level=binary_treats, **learners)


@pytest.mark.ci
def test_med_levels_check(dml_data, learner_linear, check_med_levels_fixture, ps_processor_config):

    good_levels, treat_not_int_levels, med_not_number_levels, not_01_treat_levels = check_med_levels_fixture
    outcome = "factual"
    DoubleMLMED(
        dml_data=dml_data,
        outcome=outcome,
        treatment_level=good_levels[0],
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )

    msg = (
        "Treatment level must be an integer."
        + f" Treatment level {str(treat_not_int_levels[0])} of type {str(type(treat_not_int_levels[0]))} provided."
    )
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome=outcome,
            treatment_level=treat_not_int_levels[0],
            ps_processor_config=ps_processor_config,
            **learner_linear,
        )

    msg = "Treatment level must be either 0 or 1" + f" Treatment level provided was {str(not_01_treat_levels[0])}."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome=outcome,
            treatment_level=not_01_treat_levels[0],
            ps_processor_config=ps_processor_config,
            **learner_linear,
        )


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:Learner provided for ml_m is probably invalid:UserWarning")
@pytest.mark.filterwarnings("ignore:Learner provided for ml_M is probably invalid:UserWarning")
def test_med_learners_check_factual(dml_data, binary_treats, check_med_learners_fixture, ps_processor_config):
    good_learners, missing_g_learner, missing_G_learner = check_med_learners_fixture
    outcome = "factual"
    DoubleMLMED(
        dml_data=dml_data,
        outcome=outcome,
        treatment_level=binary_treats,
        ps_processor_config=ps_processor_config,
        **good_learners,
    )

    # Prepare data with binary outcome for tests with classifiers.
    x = dml_data.x
    d = dml_data.d
    m = dml_data.m

    binary_y = np.random.randint(0, 2, dml_data.n_obs)
    binary_y_data = DoubleMLMEDData.from_arrays(y=binary_y, x=x, m=m, d=d)
    assert binary_y_data.binary_outcome

    # ---  Test mismatch between provided learners and the type of outcome.
    msg = "Learner ml_g is required when the outcome is factual."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome=outcome,
            treatment_level=binary_treats,
            ps_processor_config=ps_processor_config,
            **missing_g_learner,
        )
    # --- End tests for missing learners ---

    # Test for fixed classifier: ml_m must be a classifier regardless of outcome type
    # With binary y:
    with pytest.raises(ValueError, match=r"Learner 'ml_m' must be a classifier"):
        DoubleMLMED(
            dml_data=binary_y_data,
            treatment_level=binary_treats,
            outcome=outcome,
            ml_m=LinearRegression(),
            ml_g=LogisticRegression(),
        )

    # With continuous y:
    with pytest.raises(ValueError, match=r"Learner 'ml_m' must be a classifier"):
        DoubleMLMED(
            dml_data=dml_data, outcome=outcome, treatment_level=binary_treats, ml_m=LinearRegression(), ml_g=LinearRegression()
        )
    # Test for not-fixed learners: ml_g depends on the type of outcome.
    # ml_g is a regressor if y is continuous, and a classifier if y is binary.
    msg = "Learner 'ml_g' must be a classifier."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(binary_y_data, binary_treats, outcome, ml_m=LogisticRegression(), ml_g=LinearRegression())

    msg = "Learner 'ml_g' must be a regressor."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(dml_data, binary_treats, outcome, ml_m=LogisticRegression(), ml_g=LogisticRegression())


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:Learner provided for ml_m is probably invalid:UserWarning")
@pytest.mark.filterwarnings("ignore:Learner provided for ml_M is probably invalid:UserWarning")
def test_med_learners_check_counterfactual(
    dml_data, binary_treats, check_med_learners_fixture, learner_linear, ps_processor_config
):
    good_learners, missing_g_learner, missing_G_learner = check_med_learners_fixture
    outcome = "counterfactual"
    DoubleMLMED(
        dml_data=dml_data,
        outcome=outcome,
        treatment_level=binary_treats,
        ps_processor_config=ps_processor_config,
        **good_learners,
    )

    # Prepare data with binary outcome for tests with classifiers.
    x = dml_data.x
    d = dml_data.d
    m = dml_data.m
    binary_y = np.random.randint(0, 2, dml_data.n_obs)
    binary_y_data = DoubleMLMEDData.from_arrays(y=binary_y, x=x, m=m, d=d)
    assert binary_y_data.binary_outcome

    # Test for missing learners
    msg = "Learner ml_G is required when the outcome is counterfactual."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome=outcome,
            treatment_level=binary_treats,
            ps_processor_config=ps_processor_config,
            ml_m=LogisticRegression(),
            ml_M=LogisticRegression(),
            ml_nested_g=LinearRegression(),
        )

    msg = "Learner ml_nested_g is required when the outcome is counterfactual."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome=outcome,
            treatment_level=binary_treats,
            ml_m=LogisticRegression(),
            ml_G=LinearRegression(),
            ml_M=LogisticRegression(),
            ps_processor_config=ps_processor_config,
        )

    msg = "Learner ml_M is required when the outcome is counterfactual."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome=outcome,
            treatment_level=binary_treats,
            ml_m=LogisticRegression(),
            ml_G=LinearRegression(),
            ml_nested_g=LinearRegression(),
            ps_processor_config=ps_processor_config,
        )
    # --- End tests for missing learners ---

    # --- Test mismatch between provided learners and required task (classification/regression)
    # Test Fixed Regressor: ml_nested_g must be a regressor regardless of outcome type
    msg = "Learner 'ml_nested_g' must be a regressor."

    # With binary outcomes:
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=binary_y_data,
            treatment_level=binary_treats,
            outcome=outcome,
            ml_m=LogisticRegression(),
            ml_M=LogisticRegression(),
            ml_G=LogisticRegression(),
            ml_nested_g=LogisticRegression(),  # Invalid
        )

    # With continuous outcomes:
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome=outcome,
            treatment_level=binary_treats,
            ml_m=LogisticRegression(),
            ml_M=LogisticRegression(),
            ml_G=LinearRegression(),
            ml_nested_g=LogisticRegression(),
        )

    # Test for fixed classifier: ml_m, ml_M must be classifiers regardless of outcome type
    msg = "Learner 'ml_m' must be a classifier."

    # With binary y:
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=binary_y_data,
            treatment_level=binary_treats,
            outcome=outcome,
            ml_m=LinearRegression(),
            ml_M=LogisticRegression(),
            ml_G=LogisticRegression(),
            ml_nested_g=LinearRegression(),
        )

    # With continuous y:
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome=outcome,
            treatment_level=binary_treats,
            ml_m=LinearRegression(),
            ml_M=LogisticRegression(),
            ml_G=LogisticRegression(),
            ml_nested_g=LinearRegression(),
        )

    # Test for not-fixed learners: ml_G depends on the type of outcome.
    # ml_G is a regressor if y is continuous, and a classifier if y is binary.
    msg = "Learner 'ml_G' must be a classifier."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=binary_y_data,
            treatment_level=binary_treats,
            outcome=outcome,
            ml_m=LogisticRegression(),
            ml_M=LogisticRegression(),
            ml_G=LinearRegression(),
            ml_nested_g=LinearRegression(),
        )

    msg = "Learner 'ml_G' must be a regressor."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data,
            binary_treats,
            outcome=outcome,
            ml_m=LogisticRegression(),
            ml_M=LogisticRegression(),
            ml_G=LogisticRegression(),
            ml_nested_g=LinearRegression(),
        )
