import pytest
import re

from doubleml.med.utils._med_utils import _check_inner_sample_splitting


@pytest.fixture(
    scope="module",
)
def learners(learner_linear):
    return learner_linear


@pytest.fixture(
    scope="module",
)
def med_obj(med_factory, learners, binary_treats):
    return med_factory("counterfactual", binary_treats, learners)


@pytest.mark.ci
def test_check_inner_sample_splitting(med_obj):

    n_rep = med_obj.n_rep
    smpls = med_obj.smpls
    smpls_inner = med_obj.smpls_inner
    _check_inner_sample_splitting(smpls_inner, smpls, n_rep)

    msg=(r"inner samples must be provided")
    test_smpls_inner=None
    with pytest.raises(ValueError, match=msg):
        _check_inner_sample_splitting(test_smpls_inner, smpls, n_rep)

    msg=(r"samples must be provided")
    test_smpls=None
    with pytest.raises(ValueError, match=msg):
        _check_inner_sample_splitting(smpls_inner, test_smpls, n_rep)

    msg=re.escape(r"n_rep must be provided")
    test_n_rep=None
    with pytest.raises(ValueError, match=msg):
        _check_inner_sample_splitting(smpls_inner, smpls, test_n_rep)

    test_smpls_inner = tuple()
    msg= re.escape(r"all_inner_smpls must be a list type. () of type <class 'tuple'> was passed.")
    with pytest.raises(TypeError, match=msg):
        _check_inner_sample_splitting(test_smpls_inner, smpls, n_rep)
    
    test_smpls_inner = list()
    msg=re.escape(r"Data incompatibility. The parameter all_inner_smpls must contain as many folds as the parameter n_rep. "
         r"number of all_inner_smpls folds:" +str(len(test_smpls_inner))+", n_rep: "+str(n_rep))
    with pytest.raises(ValueError, match=msg):
        _check_inner_sample_splitting(test_smpls_inner, smpls, n_rep)
    
    test_smpls_inner = list([0])
    n_rep=1
    msg=re.escape(r"all_inner_smpls must be a list of lists.")
    with pytest.raises(TypeError, match=msg):
        _check_inner_sample_splitting(test_smpls_inner, smpls, n_rep)

    test_smpls_inner = list([[0]])
    msg=("all_inner_smpls must be a list of lists of lists.")
    with pytest.raises(TypeError, match=msg):
        _check_inner_sample_splitting(test_smpls_inner, smpls, n_rep)


def test_check_inner_sample_reps():
    pass


def test_check_inner_sample_fold():
    pass


def test_check_is_inner_partition():
    pass
