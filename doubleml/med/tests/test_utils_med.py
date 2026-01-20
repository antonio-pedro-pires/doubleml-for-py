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
def med_obj(med_factory, learners, treatment_level):
    return med_factory("potential", treatment_level, learners)


@pytest.fixture(
    scope="module",
)
def smpls(med_obj):
    return med_obj.smpls, med_obj.smpls_inner


def test_check_inner_sample_splitting(smpls):
    smpls, smpls_inner = smpls
    _check_inner_sample_splitting(smpls_inner, smpls)
    return


def test_check_inner_sample_reps():
    pass


def test_check_inner_sample_fold():
    pass


def test_check_is_inner_partition():
    pass
