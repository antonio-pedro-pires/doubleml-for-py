"""Test that external predictions reproduce the estimates of DoubleMLMEDS."""

import copy

import numpy as np
import pytest

from doubleml.med import DoubleMLMEDS


@pytest.fixture(scope="module")
def meds_obj(dml_data, double_sample_splitting, ps_processor_config, learner_linear):
    meds_obj = DoubleMLMEDS(
        obj_dml_data=dml_data,
        double_sample_splitting=double_sample_splitting,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )

    meds_obj_ext = copy.deepcopy(meds_obj)
    return meds_obj, meds_obj_ext


@pytest.fixture(scope="module")
def meds_fixture_binary_treat(meds_obj):
    meds_obj, meds_obj_ext = meds_obj

    meds_obj.fit()

    d_col = meds_obj._dml_data.d_cols[0]
    external_predictions_dict = {
        model_id: {d_col: {key: value[:, :, 0] for (key, value) in meds_obj.modeldict[model_id].predictions.items()}}
        for model_id in meds_obj.models_ids
    }
    meds_obj_ext.fit(external_predictions=external_predictions_dict)

    return meds_obj, meds_obj_ext


@pytest.mark.ci
def test_external_predictions_binary_treat(meds_fixture_binary_treat):
    """Test that estimates with external predictions match the reference fit for a binary treatment."""
    values_of_interest = [
        "all_pvals",
        "all_ses",
        "all_t_stats",
        "all_thetas",
    ]
    meds_obj, meds_obj_ext = meds_fixture_binary_treat

    for elem in values_of_interest:
        assert np.array_equal(meds_obj.framework.__getattribute__(elem), meds_obj_ext.framework.__getattribute__(elem))


@pytest.mark.ci
def test_meds_external_for_subset_of_models(dml_data, learner_linear):
    """Test that external predictions for only some of the models are used and the other models are estimated as usual."""
    meds_obj = DoubleMLMEDS(obj_dml_data=dml_data, n_folds=2, n_folds_inner=2, **learner_linear)
    constant_propensity = 0.5
    external_predictions = {"factual_1": {"d": {"ml_m": np.full((dml_data.n_obs, 1), constant_propensity)}}}

    meds_obj.fit(external_predictions=external_predictions)

    np.testing.assert_allclose(meds_obj.modeldict["factual_1"].predictions["ml_m"], constant_propensity)
    assert not np.allclose(meds_obj.modeldict["factual_0"].predictions["ml_m"], constant_propensity)
