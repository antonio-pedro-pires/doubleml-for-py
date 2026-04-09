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
def check_med_targets_fixture():
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
def test_med_data_check(check_med_data_fixture, learner_linear):
    good_data, not_med_data, med_data_instrumental, med_data_not_binary_treats = check_med_data_fixture
    ml_m = learner_linear["ml_m"]
    ml_g = learner_linear["ml_g"]
    DoubleMLMED(dml_data=good_data, ml_m=ml_m, ml_g=ml_g)

    msg = (
        "Mediation analysis requires data of type DoubleMLMediationData."
        + f" Data of type {str(type(not_med_data))} was provided instead."
    )
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(dml_data=not_med_data, ml_m=ml_m, ml_g=ml_g)

    msg = (
        f"Treatment data {med_data_not_binary_treats.d} must be a binary variable with values either 0 or 1."
        + f" Treatment data contains levels {np.unique(med_data_not_binary_treats.d)}."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DoubleMLMED(dml_data=med_data_not_binary_treats, ml_m=ml_m, ml_g=ml_g)

    msg = (
        "The current framework for causal mediation analysis does not perform analysis with instrumental variables."
        + " The results will not take into account the instrumental variables."
    )
    with pytest.warns(UserWarning, match=msg):
        DoubleMLMED(dml_data=med_data_instrumental, ml_m=ml_m, ml_g=ml_g)


@pytest.mark.ci
def test_med_target_check(dml_data, med_factory, check_med_targets_fixture, learner_linear):
    potential_t, counterfactual_t, value_error_t, type_error_t = check_med_targets_fixture
    treatment_level = 1

    learners = {"ml_g": learner_linear["ml_g"], "ml_m": learner_linear["ml_m"]}

    msg = "Target must be a string." + f"{str(type_error_t)} of type {str(type(type_error_t))} provided instead."
    with pytest.raises(TypeError, match=msg):
        DoubleMLMED(dml_data=dml_data, target=type_error_t, treatment_level=treatment_level, **learners)

    valid_targets = ["potential", "counterfactual"]
    msg = f"Invalid target {value_error_t}. " + "Valid targets " + " or ".join(valid_targets) + "."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(dml_data=dml_data, target=value_error_t, treatment_level=treatment_level, **learners)


@pytest.mark.ci
def test_med_levels_check(dml_data, learner_linear, med_factory, check_med_levels_fixture):

    good_levels, treat_not_int_levels, med_not_number_levels, not_01_treat_levels = check_med_levels_fixture
    target = "potential"
    med_factory(target=target, treatment_level=good_levels[0], mediation_level=good_levels[1], learners=learner_linear)

    msg = (
        "Treatment level must be an integer."
        + f" Treatment level {str(treat_not_int_levels[0])} of type {str(type(treat_not_int_levels[0]))} provided."
    )
    with pytest.raises(TypeError, match=msg):
        med_factory(
            target=target,
            treatment_level=treat_not_int_levels[0],
            mediation_level=treat_not_int_levels[1],
            learners=learner_linear,
        )

    msg = (
        "Mediation level must be a number."
        + f" Mediation level {str(med_not_number_levels[1])} of type {str(type(med_not_number_levels[1]))} provided."
    )
    with pytest.raises(TypeError, match=msg):
        med_factory(
            target=target,
            treatment_level=med_not_number_levels[0],
            mediation_level=med_not_number_levels[1],
            learners=learner_linear,
        )

    msg = "Treatment level must be either 0 or 1" + f" Treatment level provided was {str(not_01_treat_levels[0])}."
    with pytest.raises(ValueError, match=msg):
        med_factory(
            target=target,
            treatment_level=not_01_treat_levels[0],
            mediation_level=not_01_treat_levels[1],
            learners=learner_linear,
        )


@pytest.mark.ci
def test_med_learners_check(dml_data, check_med_learners_fixture, learner_linear, med_factory):
    good_learners, missing_g_learner, missing_G_learner = check_med_learners_fixture
    med_factory(target="potential", treatment_level=1, learners=good_learners)
    med_factory(target="counterfactual", treatment_level=0, learners=good_learners)

    msg = "Learner ml_g is required when the target is potential."
    with pytest.raises(ValueError, match=msg):
        med_factory(target="potential", treatment_level=1, learners=missing_g_learner)

    msg = "Learner ml_G is required when the target is counterfactual."
    with pytest.raises(ValueError, match=msg):
        med_factory(target="counterfactual", treatment_level=1, learners=missing_G_learner)

    missmatched_learner = copy.deepcopy(missing_g_learner)
    missmatched_learner["ml_g"] = LogisticRegression()
    msg = "The learner ml_g was identified as a classifier " + "but the outcome variable is not binary with values 0 and 1."
    with pytest.raises(ValueError, match=msg):
        med_factory(target="potential", treatment_level=1, learners=missmatched_learner)

    x = dml_data.x
    m = dml_data.m
    d = dml_data.d

    binary_y = copy.deepcopy(d)
    binary_y_data = DoubleMLMEDData.from_arrays(y=binary_y, x=x, m=m, d=d)
    assert binary_y_data.binary_outcome

    missmatched_learner["ml_g"] = LinearRegression()
    msg = "The learner ml_g must be a classifier" + "since the outcome variable is binary with values 0 and 1."
    with pytest.raises(ValueError, match=msg):
        DoubleMLMED(dml_data=binary_y_data, ml_m=missmatched_learner["ml_m"], ml_g=missmatched_learner["ml_g"])
