import pytest

from doubleml import DoubleMLMediationData
from doubleml.datasets import make_med_data


@pytest.mark.ci
def test_obj_vs_from_arrays():
    # data created from obj
    dml_data = make_med_data()

    # data created from arrays
    data_from_array = DoubleMLMediationData.from_arrays(
        dml_data.data[dml_data.x_cols],
        dml_data.data[dml_data.y_col],
        dml_data.data[dml_data.d_cols],
        dml_data.data[dml_data.m_cols],
    )
    assert dml_data.data.equals(data_from_array.data)


@pytest.mark.ci
def test_m_cols_setter():

    pass


@pytest.mark.ci
def test_data_summary_str():
    pass


@pytest.mark.ci
def test_from_arrays():
    pass


@pytest.mark.ci
def test_get_optional_col_sets():
    pass


@pytest.mark.ci
def test_check_binary_mediators():
    pass


@pytest.mark.ci
def test_check_disjoint_sets():
    pass


@pytest.mark.ci
def test_check_disjoint_sets_m_cols():
    pass


@pytest.mark.ci
def test_set_m():
    pass


@pytest.mark.ci
def test_duplicates():
    pass
