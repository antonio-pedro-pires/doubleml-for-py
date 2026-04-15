import copy

import numpy as np
import pytest

from doubleml.med import DoubleMLMEDS

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")


@pytest.fixture(scope="module")
def meds_kwargs(dml_data, learner_linear, double_sample_splitting):
    return {
        "dml_data": dml_data,
        "n_folds": 5,
        "n_rep": 3,
        "n_folds_inner": 5,
        "score": "efficient-alt",
        "normalize_ipw": False,
        "draw_sample_splitting": True,
        "double_sample_splitting": double_sample_splitting,
        **learner_linear,
    }


@pytest.fixture(scope="module")
def meds_obj(meds_kwargs):
    meds_obj = DoubleMLMEDS(**meds_kwargs)
    return meds_obj


@pytest.fixture(scope="module")
def smpls_inner_outer(meds_obj):
    return meds_obj.smpls, meds_obj.smpls_inner


@pytest.fixture(scope="module")
def treatment_mediation(meds_obj):
    return meds_obj.treatment_mediation_levels


@pytest.fixture(scope="module", params=[True, False])
def double_sample_splitting(request):
    return request.param


@pytest.fixture(scope="module")
def meds_fixture_binary_treat(meds_obj):
    meds_obj_ext = copy.deepcopy(meds_obj)

    meds_obj.fit()
    # The following line is hardcoded for binary treatments.
    # It should be possible to make it work for multiple treatments,
    # but it would require implementing some logic in meds.py.
    external_predictions_dict = {
        model_id: {"d": {key: value[:, :, 0] for (key, value) in meds_obj.modeldict[model_id].predictions.items()}}
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
