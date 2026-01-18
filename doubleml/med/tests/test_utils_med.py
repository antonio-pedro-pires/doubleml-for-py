
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.med.datasets import make_med_data
from doubleml.med.med import DoubleMLMED
from doubleml.med.utils._med_utils import _check_inner_sample_splitting


@pytest.fixture(
    scope="module",
)
def data():
    return make_med_data()


@pytest.fixture(
    scope="module",
    params=[[LogisticRegression(), LinearRegression(), LinearRegression(), LogisticRegression(), LinearRegression()]],
)
def learners(request):
    return request.param


@pytest.fixture(
    scope="module",
)
def med_obj(data, learners):
    ml_px, ml_yx, ml_ymx, ml_pmx, ml_nested = learners
    return DoubleMLMED(data, ml_px, ml_yx, ml_ymx, ml_pmx, ml_nested)


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
