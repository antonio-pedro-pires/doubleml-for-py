import numpy as np
import pandas as pd
import pytest

from doubleml import DoubleMLMediationData
from doubleml.med.datasets import (
    make_med_data,
)

msg_inv_return_type = "Invalid return_type."


def test_make_med_data_return_types():
    np.random.seed(3141)
    res = make_med_data(n_obs=100, return_type="DoubleMLMediationData")
    assert isinstance(res, DoubleMLMediationData)
    res = make_med_data(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d, m = make_med_data(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(m, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_med_data(n_obs=100, return_type="matrix")
