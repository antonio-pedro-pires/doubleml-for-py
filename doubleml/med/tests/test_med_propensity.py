import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:ps_processor_config not specified")
def test_factual_propensity_independent_of_treatment_level():
    # ml_m is meant to estimate the propensity score P(D=1|X). Because ml_m was fit on
    # self.treated (d == treatment_level), for treatment_level=0 it instead estimated
    # P(D=0|X), which caused a small but persistent bias in the estimate of E[Y(0,M(0))].
    # The fix fits ml_m on d, so it estimates P(D=1|X) for every treatment_level. This test
    # checks that ml_m estimates the same probability regardless of the treatment level.
    data = make_med_data(n_obs=400)
    common = dict(
        dml_data=data,
        outcome="factual",
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        n_folds=3,
        double_sample_splitting=False,
    )

    med_treat = DoubleMLMED(treatment_level=1, **common)
    med_control = DoubleMLMED(treatment_level=0, **common)
    med_control.set_sample_splitting(smpls=med_treat.smpls, smpls_inner=None)

    med_treat.fit()
    med_control.fit()

    np.testing.assert_allclose(med_control.predictions["ml_m"], med_treat.predictions["ml_m"])


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:ps_processor_config not specified")
def test_counterfactual_propensities_independent_of_treatment_level():
    # ml_M estimates P(D=1|M,X) and ml_m estimates P(D=1|X); both are fit on d and so must
    # not depend on treatment_level. This guards the counterfactual against the mistake fixed
    # for the factual ml_m (fitting on self.treated), by checking that the two treatment
    # levels produce the same ml_M and ml_m predictions.
    data = make_med_data(n_obs=400)
    common = dict(
        dml_data=data,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_M=LogisticRegression(),
        ml_G=LinearRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=3,
        n_folds_inner=3,
        double_sample_splitting=True,
    )

    med_treat = DoubleMLMED(treatment_level=1, **common)
    med_control = DoubleMLMED(treatment_level=0, **common)
    med_control.set_sample_splitting(smpls=med_treat.smpls, smpls_inner=med_treat.smpls_inner)

    med_treat.fit()
    med_control.fit()

    np.testing.assert_allclose(med_control.predictions["ml_M"], med_treat.predictions["ml_M"])
    np.testing.assert_allclose(med_control.predictions["ml_m"], med_treat.predictions["ml_m"])
