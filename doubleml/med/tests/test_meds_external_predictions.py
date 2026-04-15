import copy

import numpy as np
import pytest

from doubleml.med import DoubleMLMEDS

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")


@pytest.fixture(scope="module")
def meds_obj(dml_data, double_sample_splitting, ps_processor_config, learner_linear):
    meds_obj = DoubleMLMEDS(
        dml_data=dml_data,
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
    values_of_interest = [
        "all_pvals",
        "all_ses",
        "all_t_stats",
        "all_thetas",
    ]
    meds_obj, meds_obj_ext = meds_fixture_binary_treat

    for elem in values_of_interest:
        assert np.array_equal(meds_obj.framework.__getattribute__(elem), meds_obj_ext.framework.__getattribute__(elem))
