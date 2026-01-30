import pytest

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


def test_check_inner_sample_splitting(med_obj):
    _check_inner_sample_splitting(med_obj.smpls_inner, med_obj.smpls, med_obj.n_rep)
    return


def test_check_inner_sample_reps():
    pass


def test_check_inner_sample_fold():
    pass


def test_check_is_inner_partition():
    pass
