"""Test set_sample_splitting of DoubleMLMED."""

import numpy as np
import pytest

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data


@pytest.fixture(scope="module")
def set_smpls_sampling_fixture(dml_data, learner_dummy, binary_outcomes, double_sample_splitting, ps_processor_config):
    treatment_level = 1  # Fixed treatment_level since the treatment level has no effect on set_smpls_sampling.
    med_obj = DoubleMLMED(
        obj_dml_data=dml_data,
        outcome=binary_outcomes,
        treatment_level=treatment_level,
        double_sample_splitting=double_sample_splitting,
        ps_processor_config=ps_processor_config,
        **learner_dummy,
    )
    med_obj_ext = DoubleMLMED(
        obj_dml_data=dml_data,
        outcome=binary_outcomes,
        treatment_level=treatment_level,
        double_sample_splitting=double_sample_splitting,
        draw_sample_splitting=False,
        ps_processor_config=ps_processor_config,
        **learner_dummy,
    )
    return med_obj, med_obj_ext


@pytest.mark.ci
def test_set_sample_splitting(set_smpls_sampling_fixture):
    """Test that setting the sample splitting of another object reproduces its outer and inner samples."""
    med_obj, med_obj_ext = set_smpls_sampling_fixture
    smpls_inner = None if not med_obj.double_sample_splitting else med_obj.smpls_inner
    med_obj_ext.set_sample_splitting(smpls=med_obj.smpls, smpls_inner=smpls_inner)
    if med_obj.double_sample_splitting:
        np.testing.assert_equal(med_obj.smpls_inner, med_obj_ext.smpls_inner)
    np.testing.assert_equal(med_obj.smpls, med_obj_ext.smpls)


@pytest.mark.ci
def test_set_sample_splitting_exceptions(set_smpls_sampling_fixture):
    """Test that set_sample_splitting requires smpls_inner with double sample splitting."""
    _, med_obj_ext = set_smpls_sampling_fixture
    if med_obj_ext.double_sample_splitting:
        with pytest.raises(ValueError, match="smpls_inner is required"):
            med_obj_ext.set_sample_splitting(smpls=[], smpls_inner=None)


@pytest.mark.ci
def test_set_sample_splitting_cluster_data_raises(set_smpls_sampling_fixture):
    """Test that cluster sample splitting is rejected with and without double sample splitting."""
    _, med_obj_ext = set_smpls_sampling_fixture
    msg = r"Cluster data is not supported for mediation analysis\. Expected all_smpls_cluster=None and is_cluster_data=False"
    with pytest.raises(NotImplementedError, match=msg + r", got all_smpls_cluster=list and is_cluster_data=False"):
        med_obj_ext.set_sample_splitting(smpls=[], all_smpls_cluster=[], smpls_inner=[])
    with pytest.raises(NotImplementedError, match=msg + r", got all_smpls_cluster=NoneType and is_cluster_data=True"):
        med_obj_ext.set_sample_splitting(smpls=[], is_cluster_data=True)


@pytest.fixture(scope="module", params=["factual", "counterfactual"])
def reset_outcome(request):
    return request.param


@pytest.fixture(scope="module")
def reset_splits_fixture(learner_linear, reset_outcome):
    """A reference object with its own splits, and a second object that gets those splits set externally."""
    np.random.seed(3141)
    data = make_med_data(n_obs=300, dim_x=5)
    kwargs = {"obj_dml_data": data, "outcome": reset_outcome, "treatment_level": 1, **learner_linear}

    # n_folds, n_rep and n_folds_inner follow the externally set splits (as n_folds and n_rep do for all DoubleML models)
    med_ref = DoubleMLMED(n_folds=2, n_rep=2, n_folds_inner=3, **kwargs)
    med_ref.fit()
    med_ext = DoubleMLMED(n_folds=3, n_rep=1, n_folds_inner=2, **kwargs)
    returned = med_ext.set_sample_splitting(smpls=med_ref.smpls, smpls_inner=med_ref.smpls_inner)
    med_ext.fit()
    return med_ref, med_ext, returned


@pytest.mark.ci
def test_set_sample_splitting_returns_self(reset_splits_fixture):
    """Test that set_sample_splitting returns the object so that calls can be chained."""
    _, med_ext, returned = reset_splits_fixture
    assert returned is med_ext


@pytest.mark.ci
def test_set_sample_splitting_updates_dimensions(reset_splits_fixture):
    """Test that the number of repetitions, folds and inner folds follow the externally set splits."""
    med_ref, med_ext, _ = reset_splits_fixture
    assert med_ext.n_rep == med_ref.n_rep == 2
    assert med_ext.n_folds == med_ref.n_folds == 2
    assert med_ext.n_folds_inner == med_ref.n_folds_inner == 3


@pytest.mark.ci
def test_set_splitting_then_fit_matches_reference(reset_splits_fixture):
    """Test that fitting after setting splits with other dimensions reproduces the reference fit."""
    med_ref, med_ext, _ = reset_splits_fixture
    assert med_ext.all_coef.shape == med_ref.all_coef.shape
    np.testing.assert_allclose(med_ext.all_coef, med_ref.all_coef, rtol=1e-9)
    np.testing.assert_allclose(med_ext.all_se, med_ref.all_se, rtol=1e-9)


@pytest.mark.ci
def test_set_splitting_returns_self_single_split(dml_data, learner_dummy):
    """Test that set_sample_splitting also returns the object without double sample splitting."""
    kwargs = {"obj_dml_data": dml_data, "outcome": "factual", "treatment_level": 1, "double_sample_splitting": False}
    med_ref = DoubleMLMED(**kwargs, **learner_dummy)
    med_ext = DoubleMLMED(draw_sample_splitting=False, **kwargs, **learner_dummy)
    assert med_ext.set_sample_splitting(smpls=med_ref.smpls) is med_ext
