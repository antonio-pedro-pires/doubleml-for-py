"""Test the data generating process make_med_data."""

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
    """Test that make_med_data returns the type requested by each return_type alias."""
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


@pytest.mark.ci
def test_make_med_data_invalid_return_type_raises():
    """Test that make_med_data raises a ValueError for an unknown return_type."""
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_med_data(n_obs=100, return_type="matrix")


@pytest.mark.ci
def test_make_med_data_default_and_class_return():
    """Test that make_med_data returns a DoubleMLMEDData by default and when the class is passed as return_type."""
    np.random.seed(3141)
    assert isinstance(make_med_data(n_obs=100), DoubleMLMEDData)
    assert isinstance(make_med_data(n_obs=100, return_type=DoubleMLMEDData), DoubleMLMEDData)


@pytest.mark.ci
def test_make_med_data_unknown_keyword_raises():
    """Test that misspelled or unsupported keyword arguments raise an error instead of being ignored."""
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'dim_X'"):
        make_med_data(n_obs=100, dim_X=5)


@pytest.mark.ci
def test_make_med_data_seeded_values():
    """Test that make_med_data reproduces the pinned draws for a fixed seed (guard against changes of the random stream)."""
    np.random.seed(3141)
    x, y, d, m = make_med_data(n_obs=6, dim_x=3, return_type="array")
    np.testing.assert_allclose(
        y,
        [
            0.7040243784972408,
            2.3443582828637792,
            -0.38598121016730597,
            1.0911192616609977,
            1.0480812292385708,
            2.0233670090658693,
        ],
        rtol=1e-9,
    )
    np.testing.assert_array_equal(d, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(m, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(x[0], [-0.2983095564004948, -0.7583107991551773, 0.6270209162547371], rtol=1e-9)
