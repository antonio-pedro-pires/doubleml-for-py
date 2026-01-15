import numpy as np
import pytest

from doubleml.data.med_data import DoubleMLMEDData
from doubleml.med.datasets import make_med_data
from doubleml.med import DoubleMLMED, DoubleMLMEDS

@pytest.fixture(scope='module', params=[])#Fill parameters with ml_params. yx will be none when counterfactual is something.
def ml_yx_px(request):
    return request.param

@pytest.fixture(scope='module', params=[])#Same but on the other side. When target is potential, counterfactual models are None.
def ml_ymx_pmx_nested(request):
    return request.param

@pytest.fixture(scope='module')
def med_data():
    return make_med_data()

@pytest.fixture(scope='module')
def med_obj(med_data, ml_yx_px, ml_ymx_pmx_nested):
