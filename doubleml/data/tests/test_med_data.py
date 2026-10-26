"""Test the data container DoubleMLMEDData."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.multiclass import type_of_target

from doubleml import DoubleMLMEDData
from doubleml.med.datasets import make_med_data


@pytest.mark.ci
def test_obj_vs_from_arrays():
    """Test that a data object built from arrays equals the data object it was built from."""
    # data created from obj
    med_data = make_med_data()

    # data created from arrays
    data_from_array = DoubleMLMEDData.from_arrays(
        med_data.data[med_data.x_cols],
        med_data.data[med_data.y_col],
        med_data.data[med_data.d_cols],
        med_data.data[med_data.m_cols],
    )
    assert med_data.data.equals(data_from_array.data)


@pytest.mark.ci
def test_from_arrays():
    """Test that from_arrays passes the DoubleMLData arguments on to DoubleMLMEDData."""
    # create dataset of type DoubleMLMEDData
    med_data = make_med_data()

    # test custom configuration to check that DoubleMLData's
    # parameters are passed correctly to DoubleMLMEDData.
    data_from_array = DoubleMLMEDData.from_arrays(
        x=med_data.data[med_data.x_cols],
        y=med_data.data[med_data.y_col],
        d=med_data.data[med_data.d_cols],
        m=med_data.data[med_data.m_cols],
        z=med_data.data[med_data.x_cols[0]],  # Use one covariate as instrument for testing
        use_other_treat_as_covariate=True,
        force_all_x_finite=False,
        force_all_d_finite=False,
        force_all_m_finite="allow-nan",
    )

    assert data_from_array.z_cols == ["z"]
    assert data_from_array.use_other_treat_as_covariate is True
    assert data_from_array.force_all_x_finite is False
    assert data_from_array.force_all_d_finite is False
    assert data_from_array.force_all_m_finite == "allow-nan"


@pytest.mark.ci
@pytest.mark.parametrize(
    "force_all_m_finite, exception, msg",
    [
        ("nope", ValueError, r"Invalid force_all_m_finite nope. force_all_m_finite must be True, False or 'allow-nan'."),
        (
            5,
            TypeError,
            r"Invalid force_all_m_finite. force_all_m_finite must be True, False or 'allow-nan'. "
            r"5 of type <class 'int'> was passed.",
        ),
    ],
    ids=["invalid_string", "invalid_type"],
)
def test_from_arrays_force_all_m_finite_raises(force_all_m_finite, exception, msg):
    """Test that from_arrays raises an error for an invalid force_all_m_finite."""
    med_data = make_med_data()
    # The other parameters are tested with the data containers in test_dml_data.py.
    with pytest.raises(exception, match=msg):
        _ = DoubleMLMEDData.from_arrays(
            med_data.data[med_data.x_cols],
            med_data.data[med_data.y_col],
            med_data.data[med_data.d_cols],
            med_data.data[med_data.m_cols],
            force_all_m_finite=force_all_m_finite,
        )


@pytest.mark.ci
@pytest.mark.parametrize(
    "d_cols, m_cols, z_cols, msg",
    [
        (
            "yy",
            ["mm1"],
            "zz",
            r"yy cannot be set as outcome variable ``y_col`` and treatment variable in ``d_cols``.",
        ),
        (
            "dd1",
            ["yy"],
            "zz1",
            r"At least one variable/column is set as outcome variable \(``y_col``\) "
            r"and mediation variable\(s\) \(``m_cols``\).",
        ),
        (
            "dd1",
            ["dd1"],
            "zz1",
            r"At least one variable/column is set as treatment variable \(``d_cols``\) "
            r"and mediation variable\(s\) \(``m_cols``\).",
        ),
        (
            "dd1",
            ["xx1"],
            "zz1",
            r"At least one variable/column is set as covariate \(``x_cols``\) and mediation variable\(s\) \(``m_cols``\).",
        ),
        (
            "dd1",
            ["zz1"],
            "zz1",
            r"At least one variable/column is set as instrumental variable \(``z_cols``\) "
            r"and mediation variable\(s\) \(``m_cols``\).",
        ),
    ],
    ids=["base_y_and_d", "y_and_m", "d_and_m", "x_and_m", "z_and_m"],
)
def test_check_disjoint_sets_raises(d_cols, m_cols, z_cols, msg):
    """Test that overlapping columns raise a ValueError, including the checks of the base class."""
    df = pd.DataFrame(np.tile(np.arange(7), (4, 1)), columns=["yy", "dd1", "xx1", "xx2", "mm1", "zz1", "zz"])
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMEDData(df, y_col="yy", d_cols=d_cols, x_cols=["xx1", "xx2"], m_cols=m_cols, z_cols=z_cols)


@pytest.fixture
def med_data_many_mediators():
    df = pd.DataFrame(np.tile(np.arange(10), (4, 1)), columns=["y", "d", "x1", "x2", "m1", "m2", "m3", "z1", "t1", "s1"])
    return DoubleMLMEDData(df, y_col="y", d_cols="d", x_cols=["x1", "x2"], m_cols=["m1", "m2", "m3"], z_cols="z1")


@pytest.mark.ci
def test_m_cols_setter_updates_m(med_data_many_mediators):
    """Test that setting m_cols updates the array of mediators."""
    med_data = med_data_many_mediators
    m_comp = med_data.data[["m2", "m1"]].values
    med_data.m_cols = ["m2", "m1"]
    assert np.array_equal(med_data.m, m_comp)


@pytest.mark.ci
@pytest.mark.parametrize(
    "value, exception, msg",
    [
        (1, TypeError, r"The mediation variable\(s\) m_cols must be of str or list type. 1 of type <class 'int'> was passed."),
        (["m1", "m1"], ValueError, r"Invalid mediation variable\(s\) m_cols: Contains duplicate values."),
        ("not-m", ValueError, r"Invalid mediation variable\(s\) m_cols. At least one mediation variable is no data column."),
        (
            ["m1", "not-m"],
            ValueError,
            r"Invalid mediation variable\(s\) m_cols. At least one mediation variable is no data column.",
        ),
    ],
    ids=["not_str_or_list", "duplicates", "unknown_column", "unknown_column_in_list"],
)
def test_m_cols_setter_raises(med_data_many_mediators, value, exception, msg):
    """Test that setting m_cols to an invalid value raises an error."""
    with pytest.raises(exception, match=msg):
        med_data_many_mediators.m_cols = value


@pytest.mark.ci
def test_data_summary_str():
    """Test that the string representation lists the variables of the data."""
    np.random.seed(3141)
    med_data = make_med_data()

    # Convert the object to string
    med_str = str(med_data)

    # Check that all important sections are present in the string
    assert f"================== {med_data.__class__.__name__} Object ==================" in med_str
    assert "------------------ Data summary      ------------------" in med_str
    assert "------------------ DataFrame info    ------------------" in med_str

    # Check that specific data attributes are correctly included
    assert "Outcome variable: y" in med_str
    assert "Treatment variable(s): ['d']" in med_str
    assert "Mediation variable(s): ['m']" in med_str
    assert "Instrument variable(s): None" in med_str
    assert "Covariates: " in med_str
    assert "No. Observations:" in med_str

    # Test summary string with instrumental variables
    df = pd.DataFrame(np.random.randn(100, 6), columns=["y", "d", "m", "x1", "z1", "x2"])
    med_data_with_iv = DoubleMLMEDData(df, "y", "d", "m", ["x1", "x2"], "z1")
    assert "Instrument variable(s): ['z1']" in str(med_data_with_iv)


@pytest.mark.ci
def test_get_optional_col_sets():
    """Test that the optional column sets include the mediators."""
    np.random.seed(3141)
    df = pd.DataFrame(np.tile(np.arange(10), (4, 1)), columns=["y", "d", "x1", "x2", "m1", "m2", "m3", "z", "t", "s"])
    med_data = DoubleMLMEDData(
        df,
        y_col="y",
        d_cols="d",
        m_cols=["m1", "m2"],
        x_cols=["x1", "x2"],
        z_cols="z",
    )
    opt_col = med_data._get_optional_col_sets()

    # Check if m1 and m2 are in the optional columns set.
    assert {"m1", "m2"} in opt_col
    # Since x_cols is defined, m3 is not in x_cols
    assert "m3" not in med_data.x_cols

    med_data = DoubleMLMEDData(
        df,
        y_col="y",
        d_cols="d",
        m_cols=["m1", "m2"],
        z_cols="z",
    )
    assert "m3" in med_data.x_cols


@pytest.fixture
def med_data_binary_mediators():
    np.random.seed(3141)
    df = pd.DataFrame(
        np.concat((np.tile(np.arange(7), (4, 1)), np.random.randint(0, 2, size=(4, 3))), axis=1),
        columns=["y", "d", "x1", "x2", "z1", "t1", "s1", "m1", "m2", "m3"],
    )
    return DoubleMLMEDData(df, y_col="y", d_cols="d", x_cols=["x1", "x2"], m_cols=["m1", "m2", "m3"], z_cols="z1")


@pytest.mark.ci
def test_check_binary_mediators_with_values_0_and_1(med_data_binary_mediators):
    """Test that mediators with the values 0 and 1 are identified as binary."""
    med_data = med_data_binary_mediators
    for m in med_data.m_cols:
        assert type_of_target(med_data.data[m]) == "binary"
    assert med_data._check_binary_mediators().all()


@pytest.mark.ci
def test_check_binary_mediators_with_values_0_and_2(med_data_binary_mediators):
    """Test that a mediator with the values 0 and 2 is not identified as binary, as only 0 and 1 are accepted."""
    med_data = med_data_binary_mediators
    med_data.data["m1"] *= 2
    assert type_of_target(med_data.data["m1"]) == "binary"
    assert not med_data._check_binary_mediators()["m1"]


@pytest.mark.ci
def test_check_binary_mediators_without_finite_check(med_data_binary_mediators):
    """Test that no mediator is identified as binary if force_all_m_finite is False."""
    med_data = med_data_binary_mediators
    med_data.force_all_m_finite = False
    assert not med_data._check_binary_mediators().any()


@pytest.mark.ci
def test_check_binary_mediators_allow_nan_without_nan(med_data_binary_mediators):
    """Test that mediators are identified by their observed values if force_all_m_finite is 'allow-nan'."""
    med_data = med_data_binary_mediators
    med_data.force_all_m_finite = "allow-nan"
    assert med_data._check_binary_mediators().all()


@pytest.mark.ci
def test_check_binary_mediators_nan_raises(med_data_binary_mediators):
    """Test that setting force_all_m_finite to True raises a ValueError if a mediator contains NaN."""
    med_data = med_data_binary_mediators
    med_data.data["m1"] = np.nan
    with pytest.raises(ValueError, match=r"Input contains NaN."):
        med_data.force_all_m_finite = True


@pytest.mark.ci
def test_check_binary_mediators_with_values_0_1_and_2(med_data_binary_mediators):
    """Test that a mediator with the values 0, 1 and 2 is not identified as binary."""
    med_data = med_data_binary_mediators
    med_data.data["m1"] = [0, 1, 2, 0]
    assert type_of_target(med_data.data["m1"]) != "binary"
    assert not med_data._check_binary_mediators()["m1"]


@pytest.mark.ci
def test_cluster_cols_not_supported():
    """Test that setting cluster variables raises a NotImplementedError and cluster_cols stays None."""
    np.random.seed(3141)
    med_data = make_med_data()
    med_data.data["cluster"] = np.arange(med_data.n_obs) % 10

    assert med_data.cluster_cols is None
    assert not med_data.is_cluster_data

    msg = r"Cluster variables are not supported for mediation analysis\. Expected cluster_cols=None, got \['cluster'\]\."
    with pytest.raises(NotImplementedError, match=msg):
        med_data.cluster_cols = ["cluster"]
    assert med_data.cluster_cols is None
    assert not med_data.is_cluster_data

    med_data.cluster_cols = None
    assert med_data.cluster_cols is None


@pytest.mark.ci
def test_dml_datatype():
    """Test that data that is not a DataFrame raises a TypeError."""
    data_array = np.zeros((100, 10))
    with pytest.raises(TypeError):
        _ = DoubleMLMEDData(data_array, y_col="y", d_cols=["d"], m_cols=["m"])


@pytest.mark.ci
def test_duplicates():
    """Test that duplicate mediator columns raise a ValueError."""
    np.random.seed(3141)
    dml_med_data = make_med_data()

    msg = r"Invalid mediation variable\(s\) m_cols: Contains duplicate values."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMEDData(dml_med_data.data, y_col="y", d_cols=["d"], m_cols=["X3", "X2", "X3"])
    with pytest.raises(ValueError, match=msg):
        dml_med_data.m_cols = ["X3", "X2", "X3"]

    msg = "Invalid pd.DataFrame: Contains duplicate column names."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMEDData(
            pd.DataFrame(np.zeros((100, 5)), columns=["y", "d", "X3", "X2", "y"]), y_col="y", d_cols=["d"], m_cols=["X2"]
        )


@pytest.fixture
def med_data_binary_mediator():
    n = 10
    np.random.seed(3141)
    df = pd.DataFrame(
        {"y": np.random.randn(n), "d": np.random.randint(0, 2, n), "m1": np.random.randint(0, 2, n), "x1": np.random.randn(n)}
    )
    return DoubleMLMEDData(df, "y", "d", ["m1"], "x1")


@pytest.mark.ci
@pytest.mark.parametrize("value", [False, True, "allow-nan"])
def test_force_all_m_finite_setter(med_data_binary_mediator, value):
    """Test that the force_all_m_finite setter accepts True, False and 'allow-nan'."""
    med_data_binary_mediator.force_all_m_finite = value
    assert med_data_binary_mediator.force_all_m_finite == value


@pytest.mark.ci
@pytest.mark.parametrize(
    "value, exception, msg",
    [(123, TypeError, "Invalid force_all_m_finite."), ("string", ValueError, "Invalid force_all_m_finite:")],
    ids=["invalid_type", "invalid_string"],
)
def test_force_all_m_finite_setter_raises(med_data_binary_mediator, value, exception, msg):
    """Test that the force_all_m_finite setter raises an error for an invalid value."""
    with pytest.raises(exception, match=msg):
        med_data_binary_mediator.force_all_m_finite = value


@pytest.mark.ci
def test_from_arrays_multiple_mediators():
    """Test that from_arrays names the columns of several mediators."""
    np.random.seed(42)
    n = 50
    x = np.random.randn(n, 3)
    y = np.random.randn(n)
    d = np.random.randint(0, 2, n)

    # A single mediator is stored in a column named "m".
    med_single = DoubleMLMEDData.from_arrays(x, y, d, np.random.randn(n))
    assert med_single.m_cols == ["m"]
    assert med_single.n_meds == 1
    assert med_single.m.shape == (n, 1)

    # Multiple mediators are stored in columns named "m1", "m2", ...
    med_multi = DoubleMLMEDData.from_arrays(x, y, d, np.random.randn(n, 3))
    assert med_multi.m_cols == ["m1", "m2", "m3"]
    assert med_multi.n_meds == 3
    assert med_multi.m.shape == (n, 3)


@pytest.mark.ci
def test_binary_meds_property():
    """Test that binary_meds is False for a mediator with values other than 0 and 1."""
    np.random.seed(3141)
    n = 50
    df = pd.DataFrame(
        {
            "y": np.random.randn(n),
            "d": np.random.randint(0, 2, n),
            "x1": np.random.randn(n),
            "m1": np.random.randint(0, 2, n),  # binary mediator
            "m2": np.random.randn(n),  # continuous mediator
        }
    )
    med_data = DoubleMLMEDData(df, y_col="y", d_cols="d", x_cols="x1", m_cols=["m1", "m2"])

    binary_meds = med_data.binary_meds
    assert binary_meds["m1"]
    assert not binary_meds["m2"]


@pytest.mark.ci
def test_binary_meds_allow_nan_checks_observed_values():
    """Test that with force_all_m_finite='allow-nan', a mediator is binary if its observed values are 0 and 1."""
    np.random.seed(3141)
    n = 60
    with_nan = np.random.randint(0, 2, n).astype(float)
    with_nan[:5] = np.nan
    three_values = np.random.randint(0, 3, n).astype(float)
    three_values[:5] = np.nan
    zero_two = 2.0 * np.random.randint(0, 2, n)
    zero_two[:5] = np.nan
    df = pd.DataFrame(
        {
            "y": np.random.randn(n),
            "d": np.random.randint(0, 2, n),
            "x1": np.random.randn(n),
            "m_binary_with_nan": with_nan,
            "m_binary_complete": np.random.randint(0, 2, n).astype(float),
            "m_three_values_with_nan": three_values,
            "m_zero_two_with_nan": zero_two,
            "m_all_nan": np.full(n, np.nan),
        }
    )
    m_cols = [c for c in df.columns if c.startswith("m_")]
    med_data = DoubleMLMEDData(df, y_col="y", d_cols="d", x_cols="x1", m_cols=m_cols, force_all_m_finite="allow-nan")

    assert med_data.binary_meds.to_dict() == {
        "m_binary_with_nan": True,
        "m_binary_complete": True,
        "m_three_values_with_nan": False,
        "m_zero_two_with_nan": False,
        "m_all_nan": False,
    }


@pytest.mark.ci
def test_data_summary_str_exact():
    """Test that the data summary lists every variable group and the number of observations in the expected format."""
    np.random.seed(3141)
    med_data = make_med_data(n_obs=6, dim_x=3)

    expected = (
        "Outcome variable: y\n"
        "Treatment variable(s): ['d']\n"
        "Mediation variable(s): ['m']\n"
        "Covariates: ['X1', 'X2', 'X3']\n"
        "Instrument variable(s): None\n"
        "No. Observations: 6\n"
    )
    assert med_data._data_summary_str() == expected


@pytest.fixture
def binary_meds_frame():
    """
    Hand-written frame for the binary_meds setter tests.

    Function-scoped on purpose: the tests build a new object per test because the setters mutate it.
    """
    return pd.DataFrame(
        {
            "y": np.linspace(0.0, 1.0, 8),
            "d": [0, 1] * 4,
            "x": np.arange(8, dtype=float),
            "mb": [0, 1] * 4,
            "mc": [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9],
            "m_nan": [np.nan, 1, 0, 1, 0, 1, 0, 1],
            "m_zero": [0] * 8,
        }
    )


@pytest.mark.ci
def test_binary_meds_follows_m_cols(binary_meds_frame):
    """Test that binary_meds is recomputed for the new mediator after the m_cols setter."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"])
    assert obj.binary_meds.to_dict() == {"mb": True}
    obj.m_cols = ["mc"]
    assert obj.binary_meds.to_dict() == {"mc": False}


@pytest.mark.ci
def test_binary_meds_index_follows_m_cols_order(binary_meds_frame):
    """Test that the binary_meds index follows the order of m_cols after each setter call."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb", "mc"], x_cols=["x"])
    assert list(obj.binary_meds.index) == ["mb", "mc"]
    obj.m_cols = ["mc", "mb"]
    assert list(obj.binary_meds.index) == ["mc", "mb"]
    assert obj.binary_meds.to_dict() == {"mc": False, "mb": True}
    obj.m_cols = ["mb"]
    assert list(obj.binary_meds.index) == ["mb"]
    assert obj.binary_meds.to_dict() == {"mb": True}


@pytest.mark.ci
def test_binary_meds_follows_force_all_m_finite(binary_meds_frame):
    """Test that binary_meds is recomputed after the force_all_m_finite setter."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"])
    assert obj.binary_meds.to_dict() == {"mb": True}
    obj.force_all_m_finite = False
    assert obj.binary_meds.to_dict() == {"mb": False}
    obj.force_all_m_finite = True
    assert obj.binary_meds.to_dict() == {"mb": True}


@pytest.mark.ci
def test_binary_meds_allow_nan_after_setter(binary_meds_frame):
    """Test that binary_meds checks the observed values once force_all_m_finite is set to 'allow-nan'."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["m_nan"], x_cols=["x"], force_all_m_finite=False)
    assert obj.binary_meds.to_dict() == {"m_nan": False}
    obj.force_all_m_finite = "allow-nan"
    assert obj.binary_meds.to_dict() == {"m_nan": True}


@pytest.mark.ci
def test_binary_meds_same_value_twice(binary_meds_frame):
    """Test that setting the same m_cols twice keeps binary_meds (smoke test, not evidence for the refresh)."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"])
    obj.m_cols = ["mb"]
    assert obj.binary_meds.to_dict() == {"mb": True}
    obj.m_cols = ["mb"]
    assert obj.binary_meds.to_dict() == {"mb": True}


# a constant 0/1 mediator is binary, matching binary_treats; DATA-Q1 closed as not-a-bug
@pytest.mark.ci
def test_binary_meds_constant_mediator_is_binary(binary_meds_frame):
    """Test that a constant all-zero mediator counts as binary after the m_cols setter."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mc"], x_cols=["x"])
    assert obj.binary_meds.to_dict() == {"mc": False}
    obj.m_cols = ["m_zero"]
    assert obj.binary_meds.to_dict() == {"m_zero": True}


def _snapshot(obj):
    """Copy of the four attributes that a failed setter must leave unchanged."""
    return {
        "m_cols": list(obj.m_cols),
        "force_all_m_finite": obj.force_all_m_finite,
        "m": obj.m.copy(),
        "binary_meds": obj.binary_meds.copy(),
    }


def _assert_unchanged(obj, snapshot):
    assert obj.m_cols == snapshot["m_cols"]
    assert obj.force_all_m_finite == snapshot["force_all_m_finite"]
    assert np.array_equal(obj.m, snapshot["m"])
    assert obj.binary_meds.equals(snapshot["binary_meds"])


@pytest.mark.ci
@pytest.mark.parametrize("new_flag", [True, "allow-nan"])
def test_force_all_m_finite_failure_leaves_object_unchanged(binary_meds_frame, new_flag):
    """Test that a failed force_all_m_finite assignment (infinite mediator) leaves the object unchanged."""
    df = binary_meds_frame.copy()
    df["m_inf"] = [np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    obj = DoubleMLMEDData(df, "y", "d", ["m_inf"], x_cols=["x"], force_all_m_finite=False)
    snapshot = _snapshot(obj)
    with pytest.raises(ValueError, match="Input contains infinity"):
        obj.force_all_m_finite = new_flag
    _assert_unchanged(obj, snapshot)


@pytest.mark.ci
@pytest.mark.parametrize("new_flag", [True, "allow-nan"])
def test_force_all_m_finite_type_error_leaves_object_unchanged(binary_meds_frame, new_flag):
    """Test that a failed force_all_m_finite assignment (string mediator, TypeError) leaves the object unchanged."""
    df = binary_meds_frame.copy()
    df["m_str"] = ["a", "b"] * 4
    obj = DoubleMLMEDData(df, "y", "d", ["m_str"], x_cols=["x"], force_all_m_finite=False)
    snapshot = _snapshot(obj)
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for \*\* or pow\(\)"):
        obj.force_all_m_finite = new_flag
    _assert_unchanged(obj, snapshot)


@pytest.mark.ci
def test_m_cols_disjoint_failure_leaves_object_unchanged(binary_meds_frame):
    """Test that a failed m_cols assignment (overlap with x_cols) leaves the object unchanged."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"])
    snapshot = _snapshot(obj)
    with pytest.raises(ValueError, match=r"covariate \(``x_cols``\) and mediation variable"):
        obj.m_cols = ["x"]
    _assert_unchanged(obj, snapshot)


@pytest.mark.ci
def test_m_cols_infinite_failure_leaves_object_unchanged(binary_meds_frame):
    """Test that a failed m_cols assignment (infinite mediator) leaves the object unchanged."""
    df = binary_meds_frame.copy()
    df["m_inf"] = [np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    obj = DoubleMLMEDData(df, "y", "d", ["mb"], x_cols=["x"], force_all_m_finite=True)
    snapshot = _snapshot(obj)
    with pytest.raises(ValueError, match="Input contains infinity"):
        obj.m_cols = ["m_inf"]
    _assert_unchanged(obj, snapshot)


@pytest.mark.ci
def test_m_cols_type_error_leaves_object_unchanged(binary_meds_frame):
    """Test that a failed m_cols assignment (string mediator, TypeError) leaves the object unchanged."""
    df = binary_meds_frame.copy()
    df["m_str"] = ["a", "b"] * 4
    obj = DoubleMLMEDData(df, "y", "d", ["mb"], x_cols=["x"], force_all_m_finite=True)
    snapshot = _snapshot(obj)
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for \*\* or pow\(\)"):
        obj.m_cols = ["m_str"]
    _assert_unchanged(obj, snapshot)


@pytest.mark.ci
def test_setter_works_after_failed_assignment(binary_meds_frame):
    """Test that the m_cols setter still works after a failed assignment."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"])
    with pytest.raises(ValueError, match=r"covariate \(``x_cols``\) and mediation variable"):
        obj.m_cols = ["x"]
    obj.m_cols = ["mc"]
    assert obj.m_cols == ["mc"]
    assert obj.binary_meds.to_dict() == {"mc": False}


@pytest.mark.ci
def test_m_cols_empty_list_failure_leaves_object_unchanged(binary_meds_frame):
    """Test that a failed empty m_cols assignment (ValueError) leaves the object unchanged."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"])
    snapshot = _snapshot(obj)
    with pytest.raises(
        ValueError, match=r"Invalid mediation variable\(s\) m_cols: At least one mediation variable is required\."
    ):
        obj.m_cols = []
    _assert_unchanged(obj, snapshot)


@pytest.mark.ci
def test_constructor_rejects_empty_m_cols(binary_meds_frame):
    """Test that the constructor rejects an empty m_cols with a ValueError."""
    with pytest.raises(
        ValueError, match=r"Invalid mediation variable\(s\) m_cols: At least one mediation variable is required\."
    ):
        DoubleMLMEDData(binary_meds_frame, "y", "d", [], x_cols=["x"])


@pytest.mark.ci
def test_single_mediator_is_accepted(binary_meds_frame):
    """Test that a single mediator is accepted at construction and through the setter."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"])
    obj.m_cols = ["mc"]
    assert obj.n_meds == 1
    assert obj.m_cols == ["mc"]


@pytest.mark.ci
@pytest.mark.parametrize("flag", ["foo", "Allow-NaN"])
def test_constructor_rejects_invalid_force_all_m_finite_string(binary_meds_frame, flag):
    """Test that the constructor rejects a string flag other than 'allow-nan'."""
    with pytest.raises(ValueError, match="Invalid force_all_m_finite"):
        DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"], force_all_m_finite=flag)


@pytest.mark.ci
@pytest.mark.parametrize("flag", [1, None])
def test_constructor_rejects_non_bool_force_all_m_finite(binary_meds_frame, flag):
    """Test that the constructor rejects a flag that is neither a bool nor a string."""
    with pytest.raises(TypeError, match="Invalid force_all_m_finite"):
        DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"], force_all_m_finite=flag)


@pytest.mark.ci
@pytest.mark.parametrize("flag", [True, False, "allow-nan"])
def test_constructor_accepts_valid_force_all_m_finite(binary_meds_frame, flag):
    """Test that the constructor stores a valid flag as passed, with the same type."""
    obj = DoubleMLMEDData(binary_meds_frame, "y", "d", ["mb"], x_cols=["x"], force_all_m_finite=flag)
    assert obj.force_all_m_finite == flag
    assert type(obj.force_all_m_finite) is type(flag)


@pytest.mark.ci
def test_constructor_checks_force_all_m_finite_before_mediators(binary_meds_frame):
    """Test that an invalid flag is reported before the mediators are checked for infinity."""
    df = binary_meds_frame.copy()
    df["m_inf"] = [np.inf] + [1.0] * 7
    with pytest.raises(ValueError, match="Invalid force_all_m_finite"):
        DoubleMLMEDData(df, "y", "d", ["m_inf"], x_cols=["x"], force_all_m_finite="foo")
