import numpy as np
import pytest

from doubleml import DoubleMLMED


@pytest.fixture(scope="module")
def med_objs(
    dml_data,
    learner_linear,
    binary_outcomes,
    binary_scores,
    normalize_ipw,
    binary_treats,
    n_folds,
    double_sample_splitting,
    n_rep,
    ps_processor_config,
):

    kwargs = {
        "dml_data": dml_data,
        "outcome": binary_outcomes,
        "treatment_level": binary_treats,
        "score": "efficient-alt",
        "normalize_ipw": normalize_ipw,
        "double_sample_splitting": double_sample_splitting,
        "ps_processor_config": ps_processor_config,
        **learner_linear,
    }

    np.random.seed(3141)
    med_obj = DoubleMLMED(**kwargs)
    np.random.seed(3141)
    med_obj_ext = DoubleMLMED(**kwargs)
    return med_obj, med_obj_ext


@pytest.fixture(scope="module")
def set_smpls_sampling_fixture(dml_data, learner_linear, binary_outcomes, double_sample_splitting, ps_processor_config):
    treatment_level = 1  # Fixed treatment_level since the treatment level has no effect on set_smpls_sampling.
    med_obj = DoubleMLMED(
        dml_data=dml_data,
        outcome=binary_outcomes,
        treatment_level=treatment_level,
        double_sample_splitting=double_sample_splitting,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )
    med_obj_ext = DoubleMLMED(
        dml_data=dml_data,
        outcome=binary_outcomes,
        treatment_level=treatment_level,
        double_sample_splitting=double_sample_splitting,
        draw_sample_splitting=False,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )
    return med_obj, med_obj_ext


@pytest.mark.ci
def test_set_samples(set_smpls_sampling_fixture):
    med_obj, med_obj_ext = set_smpls_sampling_fixture
    smpls_inner = None if not med_obj.double_sample_splitting else med_obj.smpls_inner
    med_obj_ext.set_samples(all_smpls=med_obj.smpls, all_smpls_inner=smpls_inner)
    if med_obj.double_sample_splitting:
        np.testing.assert_equal(med_obj.smpls_inner, med_obj_ext.smpls_inner)
    np.testing.assert_equal(med_obj.smpls, med_obj_ext.smpls)
