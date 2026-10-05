import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED, DoubleMLMEDData
from doubleml.med.datasets import make_med_data

N_OUTCOME_SHIFT = 25.0


def _med_obj(frame, x_cols, n_folds, double_sample_splitting):
    data = DoubleMLMEDData(frame, y_col="y", d_cols="d", m_cols="m", x_cols=x_cols)
    np.random.seed(7)
    return DoubleMLMED(
        obj_dml_data=data,
        treatment_level=1,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_G=LinearRegression(),
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=n_folds,
        n_folds_inner=3,
        double_sample_splitting=double_sample_splitting,
    )


@pytest.mark.ci
@pytest.mark.parametrize("n_folds", [2, 3])
@pytest.mark.parametrize("double_sample_splitting", [True, False], ids=["double_split", "single_split"])
def test_nested_predictions_do_not_depend_on_the_outcomes_of_their_own_fold(n_folds, double_sample_splitting):
    """Cross-fitting: changing the outcomes of the held-out fold leaves its nested predictions unchanged."""
    np.random.seed(3141)
    frame = make_med_data(n_obs=400, dim_x=5).data
    x_cols = [column for column in frame.columns if column.startswith("X")]

    reference = _med_obj(frame, x_cols, n_folds, double_sample_splitting)
    smpls = reference.smpls
    smpls_inner = reference.smpls_inner if double_sample_splitting else None
    held_out = smpls[0][0][1]
    reference.fit()

    shifted_frame = frame.copy()
    treated_held_out = held_out[frame["d"].to_numpy()[held_out] == 1]
    shifted_frame.loc[shifted_frame.index[treated_held_out], "y"] += N_OUTCOME_SHIFT
    shifted = _med_obj(shifted_frame, x_cols, n_folds, double_sample_splitting)
    if double_sample_splitting:
        shifted.set_sample_splitting(smpls=smpls, smpls_inner=smpls_inner)
    else:
        shifted.set_sample_splitting(smpls)
    shifted.fit()

    np.testing.assert_allclose(
        shifted.predictions["ml_nested_g"][held_out, 0, 0],
        reference.predictions["ml_nested_g"][held_out, 0, 0],
        err_msg="the nested predictions of a fold depend on the outcomes of that fold",
    )
