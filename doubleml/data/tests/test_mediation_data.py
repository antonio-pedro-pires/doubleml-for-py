import numpy as np
import pandas as pd
import pytest
from sklearn.utils.multiclass import type_of_target

from doubleml import DoubleMLMediationData
from doubleml.datasets import make_med_data


@pytest.mark.ci
def test_obj_vs_from_arrays():
    # data created from obj
    med_data = make_med_data()

    # data created from arrays
    data_from_array = DoubleMLMediationData.from_arrays(
        med_data.data[med_data.x_cols],
        med_data.data[med_data.y_col],
        med_data.data[med_data.d_cols],
        med_data.data[med_data.m_cols],
    )
    assert med_data.data.equals(data_from_array.data)


@pytest.mark.ci
def test_from_arrays():
    # create dataset of type DoubleMLMediationData, with force_all_m_finite=True
    med_data = make_med_data()

    # test force_all_m_finite=False
    _ = DoubleMLMediationData.from_arrays(
        med_data.data[med_data.x_cols],
        med_data.data[med_data.y_col],
        med_data.data[med_data.d_cols],
        med_data.data[med_data.m_cols],
        force_all_m_finite=False,
    )

    # test force_all_m_finite="allow_nan"
    _ = DoubleMLMediationData.from_arrays(
        med_data.data[med_data.x_cols],
        med_data.data[med_data.y_col],
        med_data.data[med_data.d_cols],
        med_data.data[med_data.m_cols],
        force_all_m_finite="allow-nan",
    )

    msg = r"Invalid force_all_m_finite " r"nope" r". " r"force_all_m_finite must be True, False or 'allow-nan'."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData.from_arrays(
            med_data.data[med_data.x_cols],
            med_data.data[med_data.y_col],
            med_data.data[med_data.d_cols],
            med_data.data[med_data.m_cols],
            force_all_m_finite="nope",
        )

    msg = (
        r"Invalid force_all_m_finite. "
        r"force_all_m_finite must be True, False or 'allow-nan'. "
        r"5 of type <class 'int'> was passed."
    )
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLMediationData.from_arrays(
            med_data.data[med_data.x_cols],
            med_data.data[med_data.y_col],
            med_data.data[med_data.d_cols],
            med_data.data[med_data.m_cols],
            force_all_m_finite=5,
        )


@pytest.mark.ci
def test_check_disjoint_sets():
    np.random.seed(3141)
    df = pd.DataFrame(np.tile(np.arange(8), (4, 1)), columns=["yy", "dd1", "xx1", "xx2", "mm1", "zz1", "tt1", "ss1"])

    msg = (
        r"At least one variable/column is set as outcome variable \(``y_col``\) "
        r"and mediation variable\(s\) \(``m_cols``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData(
            df,
            y_col="yy",
            d_cols="dd1",
            x_cols=["xx1", "xx2"],
            m_cols=["yy"],
            z_cols="zz1",
            t_col="tt1",
            s_col="ss1",
        )

    msg = (
        r"At least one variable/column is set as treatment variable \(``d_cols``\) "
        r"and mediation variable\(s\) \(``m_cols``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData(
            df,
            y_col="yy",
            d_cols="dd1",
            x_cols=["xx1", "xx2"],
            m_cols=["dd1"],
            z_cols="zz1",
            t_col="tt1",
            s_col="ss1",
        )

    msg = r"At least one variable/column is set as covariate \(``x_cols``\) " r"and mediation variable\(s\) \(``m_cols``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData(
            df,
            y_col="yy",
            d_cols="dd1",
            x_cols=["xx1", "xx2"],
            m_cols=["xx1"],
            z_cols="zz1",
            t_col="tt1",
            s_col="ss1",
        )

    msg = (
        r"At least one variable/column is set as instrumental variable \(``z_cols``\) "
        r"and mediation variable\(s\) \(``m_cols``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData(
            df,
            y_col="yy",
            d_cols="dd1",
            x_cols=["xx1", "xx2"],
            m_cols=["zz1"],
            z_cols="zz1",
            t_col="tt1",
            s_col="ss1",
        )

    msg = r"At least one variable/column is set as time variable \(``t_col``\) " r"and mediation variable\(s\) \(``m_cols``\)."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData(
            df,
            y_col="yy",
            d_cols="dd1",
            x_cols=["xx1", "xx2"],
            m_cols=["tt1"],
            z_cols="zz1",
            t_col="tt1",
            s_col="ss1",
        )

    msg = (
        r"At least one variable/column is set as score or selection variable \(``s_col``\) "
        r"and mediation variable\(s\) \(``m_cols``\)."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData(
            df,
            y_col="yy",
            d_cols="dd1",
            x_cols=["xx1", "xx2"],
            m_cols=["ss1"],
            z_cols="zz1",
            t_col="tt1",
            s_col="ss1",
        )


@pytest.mark.ci
def test_m_cols_setter():
    np.random.seed(3141)
    df = pd.DataFrame(np.tile(np.arange(10), (4, 1)), columns=["y", "d", "x1", "x2", "m1", "m2", "m3", "z1", "t1", "s1"])

    med_data = DoubleMLMediationData(
        df,
        y_col="y",
        d_cols="d",
        x_cols=["x1", "x2"],
        m_cols=["m1", "m2", "m3"],
        z_cols="z1",
        t_col="t1",
        s_col="s1",
    )

    # check that after changing m_cols, the m array gets updated
    m_comp = med_data.data[["m2", "m1"]].values
    med_data.m_cols = ["m2", "m1"]
    assert np.array_equal(med_data.m, m_comp)

    msg = (
        r"The mediation variable\(s\) m_cols must be of str or list type \(or None\). " r"1 of type <class 'int'> was passed."
    )
    with pytest.raises(TypeError, match=msg):
        med_data.m_cols = 1

    msg = r"Invalid mediation variable\(s\) m_cols: Contains duplicate values."
    with pytest.raises(ValueError, match=msg):
        med_data.m_cols = ["m1", "m1"]

    msg = r"Invalid mediation variable\(s\) m_cols. At least one mediation variable is no data column."
    with pytest.raises(ValueError, match=msg):
        med_data.m_cols = "not-m"

    msg = r"Invalid mediation variable\(s\) m_cols. At least one mediation variable is no data column."
    with pytest.raises(ValueError, match=msg):
        med_data.m_cols = ["m1", "not-m"]
    pass


@pytest.mark.ci
def test_data_summary_str():
    # TODO: Add dataset with instrumental variables and test the summary_str() with it.
    np.random.seed(3141)
    med_data = make_med_data()

    # Convert the object to string
    med_str = str(med_data)

    # Check that all important sections are present in the string
    assert "================== DoubleMLMediationData Object ==================" in med_str
    assert "------------------ Data summary      ------------------" in med_str
    assert "------------------ DataFrame info    ------------------" in med_str

    # Check that specific data attributes are correctly included
    assert "Outcome variable: y" in med_str
    assert "Treatment variable(s): ['d']" in med_str
    assert "Mediation variable(s): ['m']" in med_str
    assert "Instrument variable(s): None" in med_str
    assert "Covariates: " in med_str
    assert "No. Observations:" in med_str

    # Test with additional optional attributes
    df = med_data.data.copy()
    df["time_var"] = 1
    df["score_var"] = 0.5

    med_data_with_optional = DoubleMLMediationData(
        data=df,
        y_col="y",
        d_cols="d",
        m_cols=["m"],
        t_col="time_var",
        s_col="score_var",
    )

    med_str_optional = str(med_data_with_optional)
    assert "Time variable: time_var" in med_str_optional
    assert "Score/Selection variable: score_var" in med_str_optional


@pytest.mark.ci
def test_get_optional_col_sets():
    np.random.seed(3141)
    df = pd.DataFrame(np.tile(np.arange(10), (4, 1)), columns=["y", "d", "x1", "x2", "m1", "m2", "m3", "z", "t", "s"])
    med_data = DoubleMLMediationData(
        df,
        y_col="y",
        d_cols="d",
        m_cols=["m1", "m2"],
        x_cols=["x1", "x2"],
        z_cols="z",
        t_col="t",
        s_col="s",
    )
    opt_col = med_data._get_optional_col_sets()

    # Check if m1 and m2 are in the optional columns set.
    assert {"m1", "m2"} in opt_col
    # Since x_cols is defined, m3 is not in x_cols
    assert "m3" not in med_data.x_cols

    med_data = DoubleMLMediationData(
        df,
        y_col="y",
        d_cols="d",
        m_cols=["m1", "m2"],
        z_cols="z",
        t_col="t",
        s_col="s",
    )
    assert "m3" in med_data.x_cols


@pytest.mark.ci
def test_check_binary_mediators():
    np.random.seed(3141)

    # Prepare the data with binary mediation variables.
    df = pd.DataFrame(
        np.concat((np.tile(np.arange(7), (4, 1)), np.random.randint(0, 2, size=(4, 3))), axis=1),
        columns=["y", "d", "x1", "x2", "z1", "t1", "s1", "m1", "m2", "m3"],
    )
    med_data = DoubleMLMediationData(
        df,
        y_col="y",
        d_cols="d",
        x_cols=["x1", "x2"],
        m_cols=["m1", "m2", "m3"],
        z_cols="z1",
        t_col="t1",
        s_col="s1",
    )

    # The mediation variables are forced to be finite and only contain 0s or 1s. Therefore, binary_meds should return True.
    for m in med_data.m_cols:
        assert type_of_target(med_data.data[m]) == "binary"
    assert med_data._check_binary_mediators().all()

    # The mediation variables contain 0s and 2s. Although technically binary, binary mediators only accept values 0 and 1.
    # Therefore, binary_meds should return False.
    med_data.data["m1"] *= 2
    for m in med_data.m_cols:
        assert type_of_target(med_data.data[m]) == "binary"
    assert not med_data._check_binary_mediators().all()

    med_data.data["m1"] /= 2  # reset m1 data to binary.
    # The mediation variables are not forced to be finite. Therefore, binary_meds should return False.
    med_data.force_all_m_finite = False
    for m in med_data.m_cols:
        assert type_of_target(med_data.data[m]) == "binary"
    assert not med_data._check_binary_mediators().all()

    # The mediation variables allow Nans but has no actual Nan in mediation variable. binary_meds should return True
    med_data.force_all_m_finite = "allow-nan"
    for m in med_data.m_cols:
        assert type_of_target(med_data.data[m]) == "binary"
    assert med_data._check_binary_mediators().all()

    # The mediation variables contains Nans, binary_meds should return False.
    med_data.data["m1"] = np.nan
    msg = r"Input contains NaN."
    with pytest.raises(ValueError, match=msg):
        med_data.force_all_m_finite = True
        assert type_of_target(med_data.data.loc["m1"]) != "binary"
        assert not med_data._check_binary_mediators().all()

    # The mediation variables contain 0s 1s and 2s. The mediators are not binary. Therefore, binary_meds should return False.
    np.random.randint(0, 3, size=(3, 4))
    med_data.data["m1"], med_data.data["m2"], med_data.data["m3"] = np.random.randint(0, 3, size=(3, 4))
    med_data.force_all_m_finite = True

    for m in med_data.m_cols:
        if np.unique(m).size > 2:
            assert type_of_target(med_data.data[m]) != "binary"

    assert not med_data._check_binary_mediators().all()


@pytest.mark.ci
def test_dml_datatype():
    data_array = np.zeros((100, 10))
    with pytest.raises(TypeError):
        _ = DoubleMLMediationData(data_array, y_col="y", d_cols=["d"], m_cols=["m"])


@pytest.mark.ci
def test_duplicates():
    np.random.seed(3141)
    dml_med_data = make_med_data()

    msg = r"Invalid mediation variable\(s\) m_cols: Contains duplicate values."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData(dml_med_data.data, y_col="y", d_cols=["d"], m_cols=["X3", "X2", "X3"])
    with pytest.raises(ValueError, match=msg):
        dml_med_data.m_cols = ["X3", "X2", "X3"]

    msg = "Invalid pd.DataFrame: Contains duplicate column names."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLMediationData(
            pd.DataFrame(np.zeros((100, 5)), columns=["y", "d", "X3", "X2", "y"]), y_col="y", d_cols=["d"], m_cols=["X2"]
        )
