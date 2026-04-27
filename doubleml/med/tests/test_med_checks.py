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
def check_med_outcomes_fixture():
    return ["potential", "counterfactual", "other", 2]


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
        outcome="potential",
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
            outcome="potential",
            treatment_level=binary_treats,
            ml_m=ml_m,
            ml_g=ml_g,
            ps_processor_config=ps_processor_config,
        )

    msg = (
        f"Treatment data {med_data_not_binary_treats.d} must be a binary variable with values either 0 or 1."
        + f" Treatment data contains levels {np.unique(med_data_not_binary_treats.d)}."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DoubleMLMED(
            dml_data=med_data_not_binary_treats,
            outcome="potential",
            treatment_level=binary_treats,
            ml_m=ml_m,
            ml_g=ml_g,
            ps_processor_config=ps_processor_config,
        )

    msg = (
        "The current framework for causal mediation analysis does not perform analysis with instrumental variables."
        + " The results will not take into account the instrumental variables."
    )
    with pytest.warns(UserWarning, match=msg):
        DoubleMLMED(
            dml_data=med_data_instrumental,
            outcome="potential",
            treatment_level=binary_treats,
            ml_m=ml_m,
            ml_g=ml_g,
            ps_processor_config=ps_processor_config,
        )


@pytest.mark.ci
def test_med_outcome_check(dml_data, binary_treats, check_med_outcomes_fixture, learner_linear):
    potential_t, counterfactual_t, value_error_t, type_error_t = check_med_outcomes_fixture

    learners = {"ml_g": learner_linear["ml_g"], "ml_m": learner_linear["ml_m"]}

    msg = "Outcome must be a string." + f"{str(type_error_t)} of type {str(type(type_error_t))} provided instead."
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(dml_data=dml_data, outcome=type_error_t, treatment_level=binary_treats, **learners)

    valid_outcomes = ["potential", "counterfactual"]
    msg = f"Invalid outcome {value_error_t}. " + "Valid outcomes " + " or ".join(valid_outcomes) + "."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(dml_data=dml_data, outcome=value_error_t, treatment_level=binary_treats, **learners)


@pytest.mark.ci
def test_med_levels_check(dml_data, learner_linear, check_med_levels_fixture, ps_processor_config):

    good_levels, treat_not_int_levels, med_not_number_levels, not_01_treat_levels = check_med_levels_fixture
    outcome = "potential"
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
def test_med_learners_check(
    dml_data, binary_treats, binary_outcomes, check_med_learners_fixture, learner_linear, ps_processor_config
):
    good_learners, missing_g_learner, missing_G_learner = check_med_learners_fixture
    DoubleMLMED(
        dml_data=dml_data,
        outcome="potential",
        treatment_level=binary_treats,
        ps_processor_config=ps_processor_config,
        **good_learners,
    )
    DoubleMLMED(
        dml_data=dml_data,
        outcome="counterfactual",
        treatment_level=binary_treats,
        ps_processor_config=ps_processor_config,
        **good_learners,
    )

    msg = "Learner ml_g is required when the outcome is potential."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome="potential",
            treatment_level=binary_treats,
            ps_processor_config=ps_processor_config,
            **missing_g_learner,
        )

    msg = "Learner ml_G is required when the outcome is counterfactual."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome="counterfactual",
            treatment_level=binary_treats,
            ps_processor_config=ps_processor_config,
            **missing_G_learner,
        )

    missmatched_learner = copy.deepcopy(missing_g_learner)
    missmatched_learner["ml_g"] = LogisticRegression()
    msg = "The learner ml_g was identified as a classifier " + "but the outcome variable is not binary with values 0 and 1."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            outcome="potential",
            treatment_level=binary_treats,
            ps_processor_config=ps_processor_config,
            **missmatched_learner,
        )

    x = dml_data.x
    m = dml_data.m
    d = dml_data.d

    binary_y = copy.deepcopy(d)
    binary_y_data = DoubleMLMEDData.from_arrays(y=binary_y, x=x, m=m, d=d)
    assert binary_y_data.binary_outcome

    missmatched_learner["ml_g"] = LinearRegression()
    msg = "The learner ml_g must be a classifier" + "since the outcome variable is binary with values 0 and 1."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=binary_y_data,
            treatment_level=binary_treats,
            outcome="potential",
            ml_m=missmatched_learner["ml_m"],
            ml_g=missmatched_learner["ml_g"],
        )

    # Test ml_nested_g must be a regressor
    missmatched_learner = copy.deepcopy(missing_G_learner)
    missmatched_learner["ml_G"] = LogisticRegression()  # classifier
    missmatched_learner["ml_nested_g"] = LogisticRegression()  # classifier (invalid)

    msg = "The learner ml_nested_g must be a regressor because its target is a continuous probability."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(
            dml_data=binary_y_data,
            treatment_level=binary_treats,
            outcome="counterfactual",
            ml_m=missmatched_learner["ml_m"],
            ml_M=missmatched_learner["ml_m"],
            ml_G=missmatched_learner["ml_G"],
            ml_nested_g=missmatched_learner["ml_nested_g"],
        )

    # Test ml_m must be a classifier
    msg = r"Invalid learner provided for ml_m: .* has no method .predict_proba\(\)."
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            treatment_level=binary_treats,
            outcome="potential",
            ml_m=LinearRegression(),
            ml_g=LinearRegression(),
        )

    # Test ml_M must be a classifier
    msg = r"Invalid learner provided for ml_M: .* has no method .predict_proba\(\)."
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(
            dml_data=dml_data,
            treatment_level=binary_treats,
            outcome="counterfactual",
            ml_m=LogisticRegression(),
            ml_M=LinearRegression(),
            ml_g=LinearRegression(),
            ml_G=LinearRegression(),
            ml_nested_g=LinearRegression(),
        )
