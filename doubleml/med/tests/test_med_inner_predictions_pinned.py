"""Pin the estimates and the inner ml_G entries of the counterfactual estimator for each sample splitting scheme."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMED
from doubleml.med.datasets import make_med_data

N_FOLDS = 3
N_REP = 2

# (double_sample_splitting, n_folds_inner) -> pinned values, captured before the refactoring of the mediation module.
# coef, se, number of stored models per inner key and repetition, and for each inner key the sum of the finite
# predictions, the sum of the finite targets and the number of missing targets.
PINNED = {
    (True, 2): {
        "coef": 0.9405534023909443,
        "se": 0.14597448625299328,
        "n_models": 2,
        "pred_sums": [288.7591447023088, 255.7824873434321, 267.20418869375095],
        "target_sums": [187.3315643000785, 179.39176723904365, 178.07756422839765],
        "n_nan_targets": [257, 260, 247],
    },
    (True, 1): {
        "coef": 0.9666204979262695,
        "se": 0.15587165867681954,
        "n_models": 1,
        "pred_sums": [142.63115909503355, 93.8268173227433, 121.15676632901739],
        "target_sums": [88.59623860310779, 92.299243165786, 85.40762939383393],
        "n_nan_targets": [328, 327, 327],
    },
    (False, 2): {
        "coef": 0.9501599071103206,
        "se": 0.14394127914122434,
        "n_models": 1,
        "pred_sums": [101.71005410754454, 95.41945055538218, 92.89852062463592],
        "target_sums": [0.0, 0.0, 0.0],
        "n_nan_targets": [400, 400, 400],
    },
}


@pytest.fixture(scope="module", params=list(PINNED), ids=["double_split", "two_subsamples", "single_split"])
def fitted(request):
    double_sample_splitting, n_folds_inner = request.param
    np.random.seed(3141)
    data = make_med_data(n_obs=200, dim_x=4)
    np.random.seed(3141)
    med_obj = DoubleMLMED(
        obj_dml_data=data,
        treatment_level=1,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_G=LinearRegression(),
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression(),
        n_folds=N_FOLDS,
        n_rep=N_REP,
        n_folds_inner=n_folds_inner,
        double_sample_splitting=double_sample_splitting,
    )
    med_obj.fit(store_models=True)
    return med_obj, PINNED[request.param]


@pytest.mark.ci
def test_pinned_coef_and_se(fitted):
    """Test that coef and se equal the pinned values."""
    med_obj, pinned = fitted
    np.testing.assert_allclose(med_obj.coef[0], pinned["coef"], rtol=1e-7)
    np.testing.assert_allclose(med_obj.se[0], pinned["se"], rtol=1e-7)


@pytest.mark.ci
def test_pinned_inner_models_per_repetition(fitted):
    """Test that each inner ml_G entry stores the pinned number of models in every repetition."""
    med_obj, pinned = fitted
    for i_fold in range(N_FOLDS):
        stored = med_obj.models[f"ml_G_inner_{i_fold}"]["d"]
        assert len(stored) == N_REP
        assert [len(models) for models in stored] == [pinned["n_models"]] * N_REP


@pytest.mark.ci
def test_pinned_inner_predictions_and_targets(fitted):
    """Test that the inner predictions and targets of ml_G equal the pinned sums and missing counts."""
    med_obj, pinned = fitted
    for i_fold in range(N_FOLDS):
        name = f"ml_G_inner_{i_fold}"
        predictions = med_obj.predictions[name]
        targets = med_obj.nuisance_targets[name]
        np.testing.assert_allclose(np.nansum(predictions), pinned["pred_sums"][i_fold], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(np.nansum(targets), pinned["target_sums"][i_fold], rtol=1e-7, atol=1e-9)
        assert int(np.isnan(targets).sum()) == pinned["n_nan_targets"][i_fold]
