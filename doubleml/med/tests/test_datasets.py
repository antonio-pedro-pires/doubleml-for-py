import numpy as np
import pandas as pd
import pytest

from doubleml import DoubleMLMEDData
from doubleml.med.datasets import (
    make_med_data,
)

msg_inv_return_type = "Invalid return_type."


@pytest.mark.ci
def test_make_med_data_return_types():
    np.random.seed(3141)
    res = make_med_data(n_obs=100, return_type="DoubleMLMEDData")
    assert isinstance(res, DoubleMLMEDData)
    res = make_med_data(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d, m = make_med_data(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(m, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_med_data(n_obs=100, return_type="matrix")


@pytest.mark.ci
def test_make_med_data_default_return_type_and_class_return():
    """The default return type is a DoubleMLMEDData object, which can also be requested by passing the class."""
    np.random.seed(3141)
    assert isinstance(make_med_data(n_obs=100), DoubleMLMEDData)
    assert isinstance(make_med_data(n_obs=100, return_type=DoubleMLMEDData), DoubleMLMEDData)


@pytest.mark.ci
def test_make_med_data_unknown_keyword_argument_raises():
    """Misspelled or unsupported keyword arguments raise instead of being silently ignored."""
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'dim_X'"):
        make_med_data(n_obs=100, dim_X=5)
