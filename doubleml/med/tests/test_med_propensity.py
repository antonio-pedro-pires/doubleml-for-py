import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data
from doubleml.utils import PSProcessorConfig

# TODO: Remove warning filter once sklearn gets to version 1.10
pytestmark = pytest.mark.filterwarnings("ignore: l1_ratio parameter is only used when penalty ")


@pytest.mark.ci
def test_factual_propensity_independent_of_treatment_level():
    # ml_m is meant to estimate the propensity score P(D=1|X). Because ml_m was fit on
    # self.treated (d == treatment_level), for treatment_level=0 it instead estimated
    # P(D=0|X), which caused a small but persistent bias in the estimate of E[Y(0,M(0))].
    # The fix fits ml_m on d, so it estimates P(D=1|X) for every treatment_level. This test
    # checks that ml_m estimates the same probability regardless of the treatment level.
    data = make_med_data(n_obs=400)
    common = dict(
        obj_dml_data=data,
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
def test_counterfactual_propensities_independent_of_treatment_level():
    # ml_M estimates P(D=1|M,X) and ml_m estimates P(D=1|X); both are fit on d and so must
    # not depend on treatment_level. This guards the counterfactual against the mistake fixed
    # for the factual ml_m (fitting on self.treated), by checking that the two treatment
    # levels produce the same ml_M and ml_m predictions.
    data = make_med_data(n_obs=400)
    common = dict(
        obj_dml_data=data,
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


@pytest.mark.ci
@pytest.mark.parametrize("outcome", ["factual", "counterfactual"])
def test_ps_calibration_uses_dml_folds(outcome, learner_linear):
    """Cross-validated isotonic calibration of the propensity scores uses the sample splitting of the model."""
    data = make_med_data(n_obs=300, dim_x=5)
    ps_processor_config = PSProcessorConfig(calibration_method="isotonic", cv_calibration=True)
    med_obj = DoubleMLMED(
        obj_dml_data=data,
        outcome=outcome,
        treatment_level=1,
        n_folds=3,
        double_sample_splitting=False,
        ps_processor_config=ps_processor_config,
        **learner_linear,
    )

    # known raw propensity scores are passed as external predictions, all other nuisances are estimated
    rng = np.random.RandomState(3141)
    propensity_learners = ["ml_m"] if outcome == "factual" else ["ml_m", "ml_M"]
    raw_scores = {learner: rng.uniform(0.2, 0.8, size=(data.n_obs, 1)) for learner in propensity_learners}
    med_obj.fit(external_predictions={"d": raw_scores})

    for learner, raw in raw_scores.items():
        calibrated = cross_val_predict(
            IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0),
            raw,
            data.d,
            cv=med_obj.smpls[0],
            method="predict",
        )
        expected = np.clip(calibrated, ps_processor_config.clipping_threshold, 1 - ps_processor_config.clipping_threshold)
        np.testing.assert_allclose(med_obj.predictions[learner][:, 0, 0], expected, err_msg=f"calibration of {learner}")
