"""Tests for DoubleMLMED/DoubleMLMEDS behavior when X, D or M contain missing values."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from doubleml.data import DoubleMLMEDData
from doubleml.med.datasets import make_med_data
from doubleml.med.med import DoubleMLMED
from doubleml.utils import PSProcessorConfig


def _inject_nan(array, frac=0.05, seed=1111):
    rng = np.random.RandomState(seed)
    array = array.copy()
    ind = rng.choice(np.arange(array.size), replace=False, size=max(1, int(array.size * frac)))
    array[np.unravel_index(ind, array.shape)] = np.nan
    return array


@pytest.fixture(scope="module")
def raw_med_arrays():
    np.random.seed(42)
    return make_med_data(n_obs=300, dim_x=5, return_type="array")


@pytest.fixture(scope="module")
def learner_xgboost():
    return {
        "ml_g": XGBRegressor(n_jobs=1, objective="reg:squarederror", eta=0.1, n_estimators=10),
        "ml_m": XGBClassifier(n_jobs=1, objective="binary:logistic", eval_metric="logloss", eta=0.1, n_estimators=10),
        "ml_G": XGBRegressor(n_jobs=1, objective="reg:squarederror", eta=0.1, n_estimators=10),
        "ml_M": XGBClassifier(n_jobs=1, objective="binary:logistic", eval_metric="logloss", eta=0.1, n_estimators=10),
        "ml_nested_g": XGBRegressor(n_jobs=1, objective="reg:squarederror", eta=0.1, n_estimators=10),
    }


@pytest.fixture(scope="module")
def learner_sklearn():
    return {
        "ml_g": LinearRegression(),
        "ml_m": LogisticRegression(solver="lbfgs", max_iter=250),
        "ml_G": LinearRegression(),
        "ml_M": LogisticRegression(solver="lbfgs", max_iter=250),
        "ml_nested_g": LinearRegression(),
    }


# ---------------------------------------------------------------------------
# X: missing covariates
# ---------------------------------------------------------------------------


@pytest.mark.ci
def test_med_data_x_missing_strict_raises(raw_med_arrays):
    """Test that missing values in X raise a ValueError by default."""
    x, y, d, m = raw_med_arrays
    x_nan = _inject_nan(x)
    with pytest.raises(ValueError, match="(?i)nan"):
        DoubleMLMEDData.from_arrays(x_nan, y, d, m)


@pytest.mark.ci
def test_med_data_x_missing_allow_nan_constructs(raw_med_arrays):
    """Test that missing values in X are allowed with force_all_x_finite='allow-nan'."""
    x, y, d, m = raw_med_arrays
    x_nan = _inject_nan(x)
    dml_data = DoubleMLMEDData.from_arrays(x_nan, y, d, m, force_all_x_finite="allow-nan")
    assert np.isnan(dml_data.x).any()


@pytest.mark.ci
def test_med_fit_missing_x_tolerant(raw_med_arrays, learner_xgboost):
    """Test that a learner that handles missing values fits data with missing X."""
    x, y, d, m = raw_med_arrays
    x_nan = _inject_nan(x)
    dml_data = DoubleMLMEDData.from_arrays(x_nan, y, d, m, force_all_x_finite="allow-nan")

    np.random.seed(42)
    dml_med_obj = DoubleMLMED(
        dml_data,
        treatment_level=1,
        outcome="factual",
        ml_m=learner_xgboost["ml_m"],
        ml_g=learner_xgboost["ml_g"],
        n_folds=2,
        ps_processor_config=PSProcessorConfig(),
    )
    dml_med_obj.fit()
    assert np.isfinite(dml_med_obj.coef).all()
    assert np.isfinite(dml_med_obj.se).all()
    assert (dml_med_obj.se > 0).all()


@pytest.mark.ci
def test_med_fit_missing_x_intolerant_raises(raw_med_arrays, learner_sklearn):
    """Test that a learner that cannot handle missing values raises for data with missing X."""
    x, y, d, m = raw_med_arrays
    x_nan = _inject_nan(x)
    dml_data = DoubleMLMEDData.from_arrays(x_nan, y, d, m, force_all_x_finite="allow-nan")

    np.random.seed(42)
    dml_med_obj = DoubleMLMED(
        dml_data,
        treatment_level=1,
        outcome="factual",
        ml_m=learner_sklearn["ml_m"],
        ml_g=learner_sklearn["ml_g"],
        n_folds=2,
        ps_processor_config=PSProcessorConfig(),
    )
    with pytest.raises(ValueError, match="(?i)nan"):
        dml_med_obj.fit()


# ---------------------------------------------------------------------------
# M: missing mediators (only exercised via the 'counterfactual' outcome path)
# ---------------------------------------------------------------------------


@pytest.mark.ci
def test_med_data_m_missing_strict_raises(raw_med_arrays):
    """Test that missing values in M raise a ValueError by default."""
    x, y, d, m = raw_med_arrays
    m_nan = _inject_nan(m if m.ndim > 1 else m.reshape(-1, 1))
    with pytest.raises(ValueError, match="(?i)nan"):
        DoubleMLMEDData.from_arrays(x, y, d, m_nan)


@pytest.mark.ci
def test_med_data_m_missing_allow_nan_constructs(raw_med_arrays):
    """Test that missing values in M are allowed with force_all_m_finite='allow-nan'."""
    x, y, d, m = raw_med_arrays
    m_nan = _inject_nan(m if m.ndim > 1 else m.reshape(-1, 1))
    dml_data = DoubleMLMEDData.from_arrays(x, y, d, m_nan, force_all_m_finite="allow-nan")
    assert np.isnan(dml_data.m).any()


@pytest.mark.ci
def test_med_fit_missing_m_tolerant(raw_med_arrays, learner_xgboost):
    """Test that a learner that handles missing values fits data with missing M."""
    x, y, d, m = raw_med_arrays
    m_nan = _inject_nan(m if m.ndim > 1 else m.reshape(-1, 1))
    dml_data = DoubleMLMEDData.from_arrays(x, y, d, m_nan, force_all_m_finite="allow-nan")

    np.random.seed(42)
    dml_med_obj = DoubleMLMED(
        dml_data,
        treatment_level=1,
        outcome="counterfactual",
        n_folds=2,
        n_folds_inner=2,
        ps_processor_config=PSProcessorConfig(),
        **learner_xgboost,
    )
    dml_med_obj.fit()
    assert np.isfinite(dml_med_obj.coef).all()
    assert np.isfinite(dml_med_obj.se).all()
    assert (dml_med_obj.se > 0).all()


@pytest.mark.ci
def test_med_fit_missing_m_intolerant_raises(raw_med_arrays, learner_sklearn):
    """Test that a learner that cannot handle missing values raises for data with missing M."""
    x, y, d, m = raw_med_arrays
    m_nan = _inject_nan(m if m.ndim > 1 else m.reshape(-1, 1))
    dml_data = DoubleMLMEDData.from_arrays(x, y, d, m_nan, force_all_m_finite="allow-nan")

    np.random.seed(42)
    dml_med_obj = DoubleMLMED(
        dml_data,
        treatment_level=1,
        outcome="counterfactual",
        n_folds=2,
        n_folds_inner=2,
        ps_processor_config=PSProcessorConfig(),
        **learner_sklearn,
    )
    with pytest.raises(ValueError, match="(?i)nan"):
        dml_med_obj.fit()


# ---------------------------------------------------------------------------
# D: missing treatment assignment. force_all_d_finite is always rejected with
# NaN present -- DoubleMLData.from_arrays validates d against
# force_all_x_finite instead of force_all_d_finite (doubleml/data/base_data.py,
# a pre-existing bug in shared, non-med code, out of scope here), so
# force_all_d_finite="allow-nan" currently has no effect. Documented as the
# actual current behavior rather than the intended one.
# ---------------------------------------------------------------------------


@pytest.mark.ci
def test_med_data_d_missing_strict_raises(raw_med_arrays):
    """Test that missing values in D raise a ValueError by default."""
    x, y, d, m = raw_med_arrays
    d_nan = _inject_nan(d.reshape(-1, 1)).reshape(-1)
    with pytest.raises(ValueError, match="(?i)nan"):
        DoubleMLMEDData.from_arrays(x, y, d_nan, m)


@pytest.mark.ci
def test_med_data_d_missing_allow_nan_still_raises(raw_med_arrays):
    """Test that missing values in D raise a ValueError also with force_all_d_finite='allow-nan'."""
    x, y, d, m = raw_med_arrays
    d_nan = _inject_nan(d.reshape(-1, 1)).reshape(-1)
    with pytest.raises(ValueError, match="(?i)nan"):
        DoubleMLMEDData.from_arrays(x, y, d_nan, m, force_all_d_finite="allow-nan")


# DoubleMLMED's own _nuisance_est*/_nuisance_tuning_optuna methods validate d with
# check_array(..., ensure_all_finite=False) (see med.py), independent of the
# construction-time bug above. Since that bug currently blocks any NaN from ever
# reaching a fitted DoubleMLMEDData object, we inject NaN directly into the
# already-constructed object's private _d to exercise med.py's own handling in
# isolation and guard against a regression once the base_data.py bug is fixed.
#
# Unlike x/m, d's finiteness cannot be relaxed by learner choice: d feeds
# PSProcessor.adjust_ps() -> _validate_treatment(), which unconditionally requires
# a clean binary 0/1 vector (IPW weighting is undefined for a missing treatment
# status). A NaN-tolerant learner (XGBoost) still fits ml_m on d successfully, but
# fitting then fails at that later, learner-independent check instead.


@pytest.mark.ci
@pytest.mark.parametrize("learner_fixture_name", ["learner_xgboost", "learner_sklearn"])
def test_med_fit_with_missing_d_always_raises(raw_med_arrays, learner_fixture_name, request):
    """Test that fitting with missing values in D raises a ValueError for every learner."""
    learners = request.getfixturevalue(learner_fixture_name)
    x, y, d, m = raw_med_arrays
    dml_data = DoubleMLMEDData.from_arrays(x, y, d, m)
    dml_data._d.iloc[0] = np.nan

    np.random.seed(42)
    dml_med_obj = DoubleMLMED(
        dml_data,
        treatment_level=1,
        outcome="factual",
        ml_m=learners["ml_m"],
        ml_g=learners["ml_g"],
        n_folds=2,
        ps_processor_config=PSProcessorConfig(),
    )
    with pytest.raises(ValueError, match="(?i)nan|binary"):
        dml_med_obj.fit()
