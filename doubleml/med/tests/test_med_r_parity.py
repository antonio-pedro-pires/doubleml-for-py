from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED, DoubleMLMEDData

# The reference numbers in r_reference/ were computed with r_reference/hdmedalt_nuisance.R, which repeats the nuisance part
# of causalweight::hdmedalt (fewsplits=FALSE, MLmethod="parametric", version 1.1.5) on r_reference/data.csv.
# The test does not call R.
REFERENCE_DIR = Path(__file__).parent / "r_reference"


@pytest.fixture(scope="module")
def reference():
    data = pd.read_csv(REFERENCE_DIR / "data.csv")
    return data, pd.read_csv(REFERENCE_DIR / "r_preds.csv").sort_values("unit"), pd.read_csv(REFERENCE_DIR / "r_folds.csv")


@pytest.mark.ci
@pytest.mark.parametrize("treatment_level", [1, 0])
def test_two_subsample_nuisance_predictions_match_hdmedalt(reference, treatment_level):
    """With the folds and subsamples of hdmedalt, ml_G and ml_nested_g reproduce its predictions on the test folds."""
    frame, r_preds, r_folds = reference
    x_cols = [column for column in frame.columns if column.startswith("X")]
    data = DoubleMLMEDData(frame, y_col="y", d_cols="d", m_cols="m", x_cols=x_cols)

    def units(fold, role):
        return r_folds[(r_folds["fold"] == fold) & (r_folds["role"] == role)]["unit"].to_numpy()

    folds = (1, 2, 3)
    smpls = [[(np.concatenate([units(i, "A"), units(i, "B")]), units(i, "test")) for i in folds]]
    smpls_inner = [[[(units(i, "A"), units(i, "B"))] for i in folds]]

    med_obj = DoubleMLMED(
        obj_dml_data=data,
        treatment_level=treatment_level,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_G=LinearRegression(),
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=3,
        n_folds_inner=1,
        draw_sample_splitting=False,
    )
    med_obj.set_sample_splitting(smpls=smpls, smpls_inner=smpls_inner)
    med_obj.fit()

    np.testing.assert_allclose(
        med_obj.predictions["ml_G"][:, 0, 0], r_preds[f"eymx{treatment_level}te"].to_numpy(), rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        med_obj.predictions["ml_nested_g"][:, 0, 0], r_preds[f"regweymx{treatment_level}te"].to_numpy(), rtol=0, atol=1e-10
    )
