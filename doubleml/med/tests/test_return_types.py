"""Test the return types of the properties and methods of DoubleMLMED and DoubleMLMEDS."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from doubleml import DoubleMLMED, DoubleMLMEDS
from doubleml.utils._check_return_types import (
    check_basic_predictions_and_targets,
    check_basic_property_types_and_shapes,
    check_basic_return_types,
)

# Test constants
N_OBS = 1000
N_TREAT = 1
N_REP = 1
N_FOLDS = 3
N_REP_BOOT = 314

dml_args = {
    "n_rep": N_REP,
    "n_folds": N_FOLDS,
}


@pytest.fixture(scope="module")
def dml_med_fixture(binary_outcomes, binary_treats, dml_data, learner_linear, ps_processor_config):
    dml_args["ps_processor_config"] = ps_processor_config
    if binary_outcomes == "factual":
        active_learners = {k: clone(v) for k, v in learner_linear.items() if k in ["ml_g", "ml_m"]}
    else:
        active_learners = {k: clone(v) for k, v in learner_linear.items() if k in ["ml_m", "ml_G", "ml_M", "ml_nested_g"]}
    dml_med_obj = DoubleMLMED(
        obj_dml_data=dml_data, outcome=binary_outcomes, treatment_level=binary_treats, **active_learners, **dml_args
    )

    dml_objs = (dml_med_obj, DoubleMLMED)
    return dml_objs


@pytest.mark.ci
def test_return_types(dml_med_fixture):
    """Test that DoubleMLMED returns the expected basic types."""
    dml_obj, cls = dml_med_fixture
    check_basic_return_types(dml_obj, cls)

    # further return type tests
    assert isinstance(dml_obj.get_params("ml_m"), dict)


@pytest.fixture(scope="module")
def fitted_dml_obj(request, dml_med_fixture):
    dml_obj, _ = dml_med_fixture
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_property_types_and_shapes(fitted_dml_obj):
    """Test that the properties of a fitted DoubleMLMED have the expected types and shapes."""
    check_basic_property_types_and_shapes(fitted_dml_obj, N_OBS, N_TREAT, N_REP, N_FOLDS, N_REP_BOOT)
    check_basic_predictions_and_targets(fitted_dml_obj, N_OBS, N_TREAT, N_REP)


@pytest.mark.ci
def test_score_elements_psi_a_has_obs_shape(fitted_dml_obj):
    """Test that _score_elements returns psi_a as an array of shape (n_obs,) equal to -1, like psi_b."""
    # The private method is called directly: _set_score_elements broadcasts a scalar psi_a into the stored
    # (n_obs, n_rep, n_coefs) array (double_ml.py:1597), so no public path shows the difference.
    preds = {name: values[:, 0, 0] for name, values in fitted_dml_obj.predictions.items()}
    y = fitted_dml_obj._dml_data.y
    if fitted_dml_obj.outcome == "factual":
        psi_a, psi_b = fitted_dml_obj._score_elements(y, m_hat_preds=preds["ml_m"], g_hat_preds=preds["ml_g"])
    else:
        psi_a, psi_b = fitted_dml_obj._score_elements(
            y,
            m_hat_preds=preds["ml_m"],
            M_hat_preds=preds["ml_M"],
            G_hat_preds=preds["ml_G"],
            nested_g_hat_preds=preds["ml_nested_g"],
        )

    assert isinstance(psi_a, np.ndarray)
    assert psi_a.shape == psi_b.shape == (N_OBS,)
    assert np.all(psi_a == -1.0)


@pytest.fixture(scope="module")
def fitted_meds_obj(dml_data, learner_linear, ps_processor_config):
    meds_obj = DoubleMLMEDS(
        obj_dml_data=dml_data,
        n_folds=2,
        n_rep=1,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )
    meds_obj.fit()
    return meds_obj


@pytest.mark.ci
def test_meds_properties_lifecycle(dml_data, learner_linear):
    """Test that the result properties of DoubleMLMEDS are None before fit and set after fit."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, n_folds=2, n_rep=1, **learner_linear)

    # Before fit: every result-related property is None and the effects summary is empty.
    for prop in ["coef", "all_coef", "se", "all_se", "t_stat", "pval", "n_rep_boot", "boot_method", "boot_t_stat", "effects"]:
        assert getattr(meds_obj, prop) is None
    assert meds_obj.effects_summary.empty

    # After fit: estimates are populated; bootstrap-related properties remain None.
    meds_obj.fit()
    for prop in ["coef", "all_coef", "se", "all_se", "t_stat", "pval"]:
        assert getattr(meds_obj, prop) is not None
    assert meds_obj.n_rep_boot is None
    assert meds_obj.boot_method is None
    assert meds_obj.boot_t_stat is None

    # Accessing effects_summary computes the effects automatically.
    assert not meds_obj.effects_summary.empty
    assert meds_obj.effects is not None

    # After bootstrap: bootstrap-related properties are populated.
    meds_obj.bootstrap()
    assert meds_obj.n_rep_boot is not None
    assert meds_obj.boot_method is not None
    assert meds_obj.boot_t_stat is not None


@pytest.mark.ci
def test_meds_return_types(fitted_meds_obj):

    # Basic return types
    """Test that the properties and methods of a fitted DoubleMLMEDS return the expected types."""
    assert isinstance(fitted_meds_obj.__str__(), str)
    assert isinstance(fitted_meds_obj.summary, pd.DataFrame)
    assert isinstance(fitted_meds_obj.effects_summary, pd.DataFrame)

    # Framework and coefficients
    from doubleml.double_ml_framework import DoubleMLFramework

    assert isinstance(fitted_meds_obj.framework, DoubleMLFramework)
    assert isinstance(fitted_meds_obj.coef, np.ndarray)
    assert isinstance(fitted_meds_obj.se, np.ndarray)

    # DoubleMLMEDS has 4 internal models for binary treatment
    n_models = 4
    assert fitted_meds_obj.coef.shape == (n_models,)
    assert fitted_meds_obj.se.shape == (n_models,)
    assert fitted_meds_obj.all_coef.shape == (n_models, 1)
    assert fitted_meds_obj.all_se.shape == (n_models, 1)

    # Effects
    fitted_meds_obj.evaluate_effects()
    assert isinstance(fitted_meds_obj.effects, dict)
    for effect_name, effect_obj in fitted_meds_obj.effects.items():
        assert isinstance(effect_obj, DoubleMLFramework)

    assert isinstance(fitted_meds_obj.effects_summary, pd.DataFrame)
    assert fitted_meds_obj.effects_summary.shape == (5, 6)


@pytest.mark.ci
def test_med_tune_not_implemented(dml_med_fixture):
    """Test that the grid and randomized search tune method raises a NotImplementedError."""
    dml_obj, _ = dml_med_fixture

    param_grids = {k: {"dummy": [1]} for k in dml_obj.learner_names}
    with pytest.raises(NotImplementedError, match="Nuisance tuning using the 'tune' method is not implemented"):
        with pytest.warns(FutureWarning, match="The 'tune' method using grid search or randomized search"):
            dml_obj.tune(param_grids=param_grids)


@pytest.mark.ci
def test_meds_boot_t_stat(fitted_meds_obj):
    """Test that boot_t_stat of DoubleMLMEDS is None before bootstrap."""
    assert fitted_meds_obj.boot_t_stat is None
