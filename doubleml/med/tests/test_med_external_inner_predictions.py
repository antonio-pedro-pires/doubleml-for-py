"""Test external inner predictions of ml_G passed to DoubleMLMED."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data

N_FOLDS = 3
SPLITTING_CASES = [(True, 2), (True, 1), (False, 2)]
SPLITTING_IDS = ["double_split", "two_subsamples", "single_split"]


@pytest.fixture(scope="module")
def small_data():
    np.random.seed(3141)
    return make_med_data(n_obs=300, dim_x=5)


def _med_obj(data, double_sample_splitting, n_folds_inner, **kwargs):
    np.random.seed(3141)
    return DoubleMLMED(
        obj_dml_data=data,
        treatment_level=1,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_G=LinearRegression(),
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=N_FOLDS,
        n_folds_inner=n_folds_inner,
        double_sample_splitting=double_sample_splitting,
        **kwargs,
    )


@pytest.fixture(scope="module", params=SPLITTING_CASES, ids=SPLITTING_IDS)
def reference(small_data, request):
    """A fitted reference object, an object with the same splits and the external predictions of the reference."""
    double_sample_splitting, n_folds_inner = request.param
    med_ref = _med_obj(small_data, double_sample_splitting, n_folds_inner).fit()
    # ml_nested_g is not external, so that it is estimated from the external predictions of ml_G
    external = {name: med_ref.predictions[name][:, :, 0].copy() for name in med_ref.params_names if name != "ml_nested_g"}
    return med_ref, external, (double_sample_splitting, n_folds_inner)


def _fit_external(data, med_ref, external, setting):
    double_sample_splitting, n_folds_inner = setting
    med_ext = _med_obj(data, double_sample_splitting, n_folds_inner, draw_sample_splitting=False)
    smpls_inner = med_ref.smpls_inner if double_sample_splitting else None
    med_ext.set_sample_splitting(smpls=med_ref.smpls, smpls_inner=smpls_inner)
    return med_ext.fit(external_predictions={"d": external})


def _nested_training_units(data, med_ref, setting, i_fold):
    """Observations on which ml_nested_g is fitted in the outer fold: D = 1-d and, with two subsamples, in subsample B."""
    train = med_ref.smpls[0][i_fold][0]
    if setting == (True, 1):
        train = med_ref.smpls_inner[0][i_fold][0][1]
    return train[data.d[train] == 0]


@pytest.mark.ci
def test_inner_predictions_reproduce_the_fit(small_data, reference):
    """Test that passing the stored inner predictions of ml_G reproduces the coefficient of the reference fit."""
    med_ref, external, setting = reference
    med_ext = _fit_external(small_data, med_ref, external, setting)
    np.testing.assert_allclose(med_ext.coef, med_ref.coef)


@pytest.mark.ci
def test_inner_predictions_must_be_finite_in_nested_fit(small_data, reference):
    """Test that non-finite inner predictions on the observations used to fit ml_nested_g raise a ValueError."""
    med_ref, external, setting = reference
    needed = _nested_training_units(small_data, med_ref, setting, i_fold=1)
    broken = {name: values.copy() for name, values in external.items()}
    broken["ml_G_inner_1"][needed[:2], 0] = np.nan

    match = r"ml_G_inner_1.*finite.*ml_nested_g.*outer fold 1.*2 of the %d observations" % len(needed)
    with pytest.raises(ValueError, match=match):
        _fit_external(small_data, med_ref, broken, setting)


@pytest.mark.ci
def test_inner_predictions_may_be_nan_outside_nested_fit(small_data, reference):
    """Test that inner predictions may be nan outside the observations on which ml_nested_g is fitted."""
    med_ref, external, setting = reference
    needed = _nested_training_units(small_data, med_ref, setting, i_fold=1)
    not_needed = np.setdiff1d(np.arange(small_data.n_obs), needed)
    sparse = {name: values.copy() for name, values in external.items()}
    sparse["ml_G_inner_1"][not_needed, 0] = np.nan

    med_ext = _fit_external(small_data, med_ref, sparse, setting)
    np.testing.assert_allclose(med_ext.coef, med_ref.coef)
