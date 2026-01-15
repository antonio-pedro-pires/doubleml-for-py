import numpy as np
import pytest

from doubleml.data.med_data import DoubleMLMEDData
from doubleml.med.datasets import make_med_data
from doubleml.med import DoubleMLMED, DoubleMLMEDS

from sklearn.linear_model import LinearRegression, LogisticRegression

#TODO: Will need to test with data with multiple m columns
@pytest.fixture(scope='module', params=[[
                    LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
                    LinearRegression(),
                    LinearRegression(),
                    LogisticRegression(penalty="l1", solver="liblinear", max_iter=250, random_state=42),
                    LinearRegression(),
                    ],
                ])
def learners(request):
    return request.param

@pytest.fixture(scope='module')
def meds_data():
    return make_med_data()

@pytest.fixture(scope='module')
def meds_obj(meds_data, learners):
    ml_px, ml_yx, ml_ymx, ml_pmx, ml_nested = learners
    meds_obj = DoubleMLMEDS(meds_data = meds_data,
                            ml_px = ml_px,
                            ml_yx = ml_yx,
                            ml_ymx = ml_ymx,
                            ml_pmx = ml_pmx,
                            ml_nested = ml_nested,
                            )
    return meds_obj

@pytest.mark.ci
def test_meds_obj(meds_obj):
    modeldict = meds_obj._modeldict

    assert len(modeldict) == 2
    for key in modeldict.keys():
        assert len(modeldict[key])==2