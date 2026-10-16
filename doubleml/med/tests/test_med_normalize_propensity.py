"""Test the normalization of the propensity weights of the mediation estimators against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from doubleml import DoubleMLMED, DoubleMLMEDData
from doubleml.med.utils._med_utils import _normalize_propensity_med

# Tiny grid with four observations, two treated. m != 0.5, M != m and M != 1 - m in every observation.
D = np.array([1.0, 0.0, 1.0, 0.0])
M_X = np.array([0.8, 0.3, 0.6, 0.4])  # P(D=1|X)
M_MX = np.array([0.7, 0.2, 0.9, 0.5])  # P(D=1|M,X)
N_OBS = 4

# Weights d/m and (1-d)/(1-m), worked out by hand, e.g. (1-0)/(1-0.3) = 10/7 and 1/(1-0.4) = 5/3. The normalization
# multiplies them by n_obs / sum, e.g. 4 / (5/4 + 5/3) = 48/35, so that the weights add up to n_obs.
RAW_LEVEL_1 = np.array([5 / 4, 0.0, 5 / 3, 0.0])
NORM_LEVEL_1 = np.array([12 / 7, 0.0, 16 / 7, 0.0])
RAW_LEVEL_0 = np.array([0.0, 10 / 7, 0.0, 5 / 3])
NORM_LEVEL_0 = np.array([0.0, 24 / 13, 0.0, 28 / 13])
# Counterfactual first weights: (1-d) M / ((1-M) m) for level 0 and d (1-M) / (M (1-m)) for level 1
RAW_CF_1_LEVEL_0 = np.array([0.0, 5 / 6, 0.0, 5 / 2])
NORM_CF_1_LEVEL_0 = np.array([0.0, 1.0, 0.0, 3.0])
RAW_CF_1_LEVEL_1 = np.array([15 / 7, 0.0, 5 / 18, 0.0])
NORM_CF_1_LEVEL_1 = np.array([216 / 61, 0.0, 28 / 61, 0.0])


@pytest.mark.ci
@pytest.mark.parametrize(
    "normalize_ipw, treatment_level, expected",
    [
        (False, 0, RAW_LEVEL_0),
        (True, 0, NORM_LEVEL_0),
        (False, 1, RAW_LEVEL_1),
        (True, 1, NORM_LEVEL_1),
    ],
)
def test_normalize_propensity_factual(normalize_ipw, treatment_level, expected):
    """Test that the factual weights equal the hand-computed values and that normalized weights add up to n_obs."""
    weights = _normalize_propensity_med(normalize_ipw, "factual", D, treatment_level, M_X)
    np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=0)
    if normalize_ipw:
        assert weights.sum() == pytest.approx(N_OBS, rel=1e-12)


@pytest.mark.ci
@pytest.mark.parametrize(
    "normalize_ipw, treatment_level, expected_first, expected_second",
    [
        (False, 0, RAW_CF_1_LEVEL_0, RAW_LEVEL_1),
        (True, 0, NORM_CF_1_LEVEL_0, NORM_LEVEL_1),
        (False, 1, RAW_CF_1_LEVEL_1, RAW_LEVEL_0),
        (True, 1, NORM_CF_1_LEVEL_1, NORM_LEVEL_0),
    ],
)
def test_normalize_propensity_counterfactual(normalize_ipw, treatment_level, expected_first, expected_second):
    """Test that the counterfactual weights equal the hand-computed values, each with its own normalizing sum."""
    ps1, ps2 = _normalize_propensity_med(normalize_ipw, "counterfactual", D, treatment_level, M_X, M_MX)
    np.testing.assert_allclose(ps1, expected_first, rtol=1e-12, atol=0)
    np.testing.assert_allclose(ps2, expected_second, rtol=1e-12, atol=0)
    if normalize_ipw:
        assert ps1.sum() == pytest.approx(N_OBS, rel=1e-12)
        assert ps2.sum() == pytest.approx(N_OBS, rel=1e-12)


@pytest.fixture(scope="module")
def tiny_data():
    """Eight observations, alternating treatment, outcome 1, ..., 8."""
    df = pd.DataFrame(
        {
            "y": np.arange(1.0, 9.0),
            "d": np.tile([1.0, 0.0], 4),
            "m": np.tile([0.0, 0.0, 1.0, 1.0], 2),
            "x": np.linspace(-1.0, 1.0, 8),
        }
    )
    return DoubleMLMEDData(df, y_col="y", d_cols="d", m_cols="m", x_cols=["x"])


def _external_counterfactual(n_obs, n_folds):
    const = {"ml_m": 0.6, "ml_M": 0.7, "ml_G": 1.0, "ml_nested_g": 0.5}
    external = {name: np.full((n_obs, 1), value) for name, value in const.items()}
    external.update({f"ml_G_inner_{i}": np.full((n_obs, 1), 1.0) for i in range(n_folds)})
    return {"d": external}


@pytest.mark.ci
@pytest.mark.parametrize(
    "normalize_ipw, weight_treated, weight_control, expected_coef",
    [
        # P(D=1|X)=0.6, P(D=1|M,X)=0.7: first weight d (1-0.7)/(0.7 (1-0.6)) = 15/14 d, second weight (1-d)/0.4 = 2.5 (1-d)
        (False, 15 / 14, 2.5, 153 / 56),
        # normalized, 4 treated and 4 control observations: both weights become 2
        (True, 2.0, 2.0, 4.0),
    ],
)
def test_med_counterfactual_score_with_constant_external_predictions(
    tiny_data, learner_dummy, normalize_ipw, weight_treated, weight_control, expected_coef
):
    """Test that coef and psi_b of a counterfactual fit with constant external predictions equal the hand values."""
    n_folds = 2
    med_obj = DoubleMLMED(
        obj_dml_data=tiny_data,
        treatment_level=1,
        outcome="counterfactual",
        n_folds=n_folds,
        normalize_ipw=normalize_ipw,
        double_sample_splitting=False,
        **learner_dummy,
    )
    med_obj.fit(external_predictions=_external_counterfactual(tiny_data.n_obs, n_folds))

    # psi_b = ps1 (y - G) + ps2 (G - nested_g) + nested_g with G = 1 and nested_g = 0.5
    y, d = tiny_data.y, tiny_data.d
    treated = d == 1
    expected_psi_b = np.where(treated, weight_treated * (y - 1.0) + 0.5, weight_control * 0.5 + 0.5)
    np.testing.assert_allclose(med_obj.psi_elements["psi_b"][:, 0, 0], expected_psi_b, rtol=1e-12)
    np.testing.assert_allclose(med_obj.psi_elements["psi_a"][:, 0, 0], -1.0)
    np.testing.assert_allclose(med_obj.coef, [expected_coef], rtol=1e-12)
