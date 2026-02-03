import pytest
import re
import numpy as np

from doubleml.med.utils._med_utils import (
    _check_inner_sample_splitting,
    _check_reps_smpls_inner,
    _check_fold_smpls_inner,
    _check_is_inner_partition
)


@pytest.fixture(
    scope="module",
)
def med_obj(med_factory, learner_linear):
    return med_factory("counterfactual", 1, learner_linear, double_sample_splitting=True, n_rep=3, n_folds=4, n_folds_inner=5)

@pytest.fixture
def valid_smpls_structure(med_obj):
    return med_obj.smpls, med_obj.smpls_inner, med_obj.n_rep, med_obj.n_folds, med_obj.n_folds_inner

@pytest.fixture
def valid_smpls_different_n_inner_folds(med_factory, learner_linear, ):
    alt_med_obj= med_factory("counterfactual", 1, learner_linear, double_sample_splitting=True, n_rep=3, n_folds=3, n_folds_inner=3)
    return alt_med_obj.smpls, alt_med_obj.smpls_inner, alt_med_obj.n_rep, alt_med_obj.n_folds, alt_med_obj.n_folds_inner

@pytest.fixture
def valid_smpls_different_n_folds(med_factory, learner_linear, ):
    alt_med_obj= med_factory("counterfactual", 1, learner_linear, double_sample_splitting=True, n_rep=3, n_folds=3, n_folds_inner=5)
    return alt_med_obj.smpls, alt_med_obj.smpls_inner, alt_med_obj.n_rep, alt_med_obj.n_folds, alt_med_obj.n_folds_inner

@pytest.mark.ci
def test_check_inner_sample_splitting_valid(valid_smpls_structure,):
    smpls, smpls_inner, n_rep, n_folds, n_folds_inner = valid_smpls_structure

    res_inner, res_n_inner_folds = _check_inner_sample_splitting(smpls_inner, smpls)
    assert res_inner == smpls_inner
    assert res_n_inner_folds == n_folds_inner

@pytest.mark.ci
def test_check_inner_sample_splitting_exceptions(valid_smpls_structure, valid_smpls_different_n_inner_folds, valid_smpls_different_n_folds):
     
    smpls, smpls_inner, n_rep, n_folds, n_folds_inner = valid_smpls_structure
    alt_smpls, alt_smpls_inner, alt_n_rep, alt_n_folds, alt_n_folds_inner = valid_smpls_different_n_inner_folds
    alt_smpls2, alt_smpls_inner2, alt_n_rep2, alt_n_folds2, alt_n_folds_inner2 = valid_smpls_different_n_folds
    with pytest.raises(TypeError, match="inner_smpls must be a list."):
        _check_inner_sample_splitting(tuple(), [])

    with pytest.raises(TypeError, match="smpls must be a list."):
        _check_inner_sample_splitting([], tuple())

    with pytest.raises(ValueError, match="smpls_inner must have the same number of repetitions"):
        _check_inner_sample_splitting([[]], [[], []])

    assert n_folds_inner != alt_n_folds_inner
    smpls_inner[0], alt_smpls_inner[0] = alt_smpls_inner[0], smpls_inner[0] # Swap smpls_inner content so that smpls_inner contains different amounts of inner_folds per reps.
    smpls[0], alt_smpls[0] = alt_smpls[0], smpls[0] # Same reasoning.
    assert len(smpls_inner[0][0]) == alt_n_folds_inner
    with pytest.raises(ValueError, match="Some reps in smpls_inner contain different amounts of inner_folds."):
        _check_inner_sample_splitting(smpls_inner, smpls)

    smpls_inner[0], alt_smpls_inner2[0] = alt_smpls_inner2[0], smpls_inner[0] # Swap smpls_inner content so that smpls_inner contains different amounts of inner_folds per reps.
    smpls[0], alt_smpls2[0] = alt_smpls2[0], smpls[0] # Same reasoning.
    with pytest.raises(ValueError, match="Some reps in smpls_inner contain different amounts of folds."):
        _check_inner_sample_splitting(smpls_inner, smpls)

    
@pytest.mark.ci
def test_check_reps_smpls_inner_valid(valid_smpls_structure):
    smpls, smpls_inner, *_, n_inner_fold = valid_smpls_structure
    # Testing a single repetition
    n_inner = _check_reps_smpls_inner(0, smpls_inner[0], smpls[0])
    assert n_inner == n_inner_fold

@pytest.mark.ci
def test_check_reps_smpls_inner_mismatch(valid_smpls_structure, valid_smpls_different_n_inner_folds):
    smpls, smpls_inner, *_ = valid_smpls_structure
    alt_smpls, alt_smpls_inner, *_ = valid_smpls_different_n_inner_folds
    # Provide a rep_inner with only 1 fold instead of 2
    with pytest.raises(ValueError, match="Each smpls_inner rep must have the same number of folds"):
        _check_reps_smpls_inner(0, [smpls_inner[0][0]], smpls[0])
    
    smpls_inner[0][0], alt_smpls_inner[0][0] = alt_smpls_inner[0][0], smpls_inner[0][0] # Swap smpls_inner content so that smpls_inner contains different amounts of inner_folds per reps.
    smpls[0][0], alt_smpls[0][0] = alt_smpls[0][0], smpls[0][0] # Same reasoning.
    with pytest.raises(ValueError, match="Some of the folds contain different numbers of inner_folds."):
        _check_reps_smpls_inner(0,smpls_inner[0], smpls[0])


@pytest.mark.ci
def test_check_fold_smpls_inner_valid(valid_smpls_structure):
    smpls, smpls_inner, *_ = valid_smpls_structure
    _check_fold_smpls_inner(smpls_inner[0][0],)

@pytest.mark.ci
def test_check_fold_smpls_inner_not_tuple():
    bad_fold = [[np.array([1]), np.array([0])]] 
    with pytest.raises(TypeError, match="smpls_inner inner_folds must be a tuple"):
        _check_fold_smpls_inner(bad_fold,) 

@pytest.mark.ci 
def test_check_is_inner_partition_valid(valid_smpls_structure):
    smpls, smpls_inner, *_ = valid_smpls_structure
    for rep_inner, rep in zip(smpls_inner, smpls): 
        for fold_inner, fold in zip(rep_inner, rep):
            _check_is_inner_partition(fold_inner, fold)

@pytest.mark.ci
def test_check_is_inner_partition_error(valid_smpls_structure):
    smpls, smpls_inner, *_ = valid_smpls_structure

    for rep_inner, rep in zip(smpls_inner, smpls): 
        for fold_inner, fold in zip(rep_inner, rep):
            fold_inner.pop() #Makes it so that a partition is impossible.
            with pytest.raises(ValueError, match="Some of the smpls_inner_fold do not partition the training samples"):
                _check_is_inner_partition(fold_inner, fold)
