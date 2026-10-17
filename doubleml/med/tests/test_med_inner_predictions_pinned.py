"""Pin the estimates and the inner ml_G entries of the counterfactual estimator for each sample splitting scheme."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

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


# Pins of the complete counterfactual nuisance estimation (score, predictions, targets and stored models), captured
# before the extraction of the ml_G branches of DoubleMLMED._nuisance_est_counterfactual.
N_OBS = 200
N_FOLDS_NEW = 3
N_REP_NEW = 2
SPLITTINGS = {"double_k2": (True, 2), "double_k3": (True, 3), "two_subsamples": (True, 1), "single": (False, 2)}


def _checksums(values):
    """Sum, number of missing entries and index-weighted sum of an array."""
    arr = np.asarray(values, dtype=float)
    weights = np.arange(1, arr.size + 1).reshape(arr.shape) / arr.size
    return [float(np.nansum(arr)), int(np.isnan(arr).sum()), float(np.nansum(arr * weights))]


def _models_structure(models):
    """Number of stored models per repetition and key (None if not stored)."""
    if models is None:
        return None
    structure = {}
    for name, per_treatment in models.items():
        reps = None if per_treatment is None else per_treatment["d"]
        structure[name] = None if reps is None else [None if rep is None else len(rep) for rep in reps]
    return structure


def _summary(med_obj):
    """Everything the counterfactual estimator computes that is observable from the outside."""
    return {
        "coef": float(med_obj.coef[0]),
        "se": float(med_obj.se[0]),
        "psi": _checksums(med_obj.psi),
        "predictions": {name: _checksums(values) for name, values in sorted(med_obj.predictions.items())},
        "targets": {name: _checksums(values) for name, values in sorted(med_obj.nuisance_targets.items())},
        "models": _models_structure(med_obj.models),
    }


def _new_med(data, splitting, ml_G=None, ml_nested_g=None, **kwargs):
    double_sample_splitting, n_folds_inner = splitting
    return DoubleMLMED(
        obj_dml_data=data,
        treatment_level=1,
        outcome="counterfactual",
        ml_m=LogisticRegression(),
        ml_G=LinearRegression() if ml_G is None else ml_G,
        ml_M=LogisticRegression(),
        ml_nested_g=LinearRegression() if ml_nested_g is None else ml_nested_g,
        n_folds=N_FOLDS_NEW,
        n_rep=N_REP_NEW,
        n_folds_inner=n_folds_inner,
        double_sample_splitting=double_sample_splitting,
        **kwargs,
    )


def _fit_case(case):
    kind, splitting_name, store_models = case
    splitting = SPLITTINGS[splitting_name]
    np.random.seed(3141)
    data = make_med_data(n_obs=N_OBS, dim_x=4)
    np.random.seed(3141)
    if kind == "plain":
        return _new_med(data, splitting).fit(store_models=store_models)
    if kind == "forest":
        # no random_state: any change in the order of the fits changes the pinned values
        med_obj = _new_med(data, splitting, ml_G=RandomForestRegressor(n_estimators=5))
        return med_obj.fit(store_models=store_models)
    if kind == "params":
        med_obj = _new_med(data, splitting, ml_G=Ridge(), ml_nested_g=Ridge())
        med_obj.set_ml_nuisance_params("ml_G", "d", {"alpha": 25.0})
        med_obj.set_ml_nuisance_params("ml_nested_g", "d", {"alpha": 7.0})
        return med_obj.fit(store_models=store_models)

    reference = _new_med(data, splitting).fit()
    med_obj = _new_med(data, splitting, draw_sample_splitting=False)
    med_obj.set_sample_splitting(smpls=reference.smpls, smpls_inner=reference.smpls_inner if splitting[0] else None)
    if kind == "external_G":
        names = [name for name in reference.params_names if name.startswith("ml_G")]
    else:
        names = ["ml_nested_g"]
    external = {name: reference.predictions[name][:, :, 0].copy() for name in names}
    return med_obj.fit(external_predictions={"d": external}, store_models=store_models)


CASES = {}
for _name in SPLITTINGS:
    for _store in (True, False):
        CASES[f"plain_{_name}_{'models' if _store else 'nomodels'}"] = ("plain", _name, _store)
for _name in ("double_k2", "single"):
    CASES[f"forest_{_name}"] = ("forest", _name, True)
    CASES[f"params_{_name}"] = ("params", _name, True)
for _name in ("double_k2", "two_subsamples", "single"):
    CASES[f"external_G_{_name}"] = ("external_G", _name, True)
    CASES[f"external_nested_{_name}"] = ("external_nested", _name, True)

PINNED_CASES = {
    "external_G_double_k2": {
        "coef": 0.9405534023909443,
        "models": {
            "ml_G": [None, None],
            "ml_G_inner_0": [None, None],
            "ml_G_inner_1": [None, None],
            "ml_G_inner_2": [None, None],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [404.8780706250595, 0, 208.77896510553938],
            "ml_G_inner_0": [288.7591447023088, 134, 148.6724384258225],
            "ml_G_inner_1": [255.7824873434321, 134, 128.62144045485584],
            "ml_G_inner_2": [267.20418869375095, 132, 142.01724389373743],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [361.2970472398607, 0, 182.46248716738594],
        },
        "psi": [-2.842170943040401e-14, 0, 8.61270108517251],
        "se": 0.14597448625299328,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [0.0, 400, 0.0],
            "ml_G_inner_1": [0.0, 400, 0.0],
            "ml_G_inner_2": [0.0, 400, 0.0],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [136.3002988003558, 218, 72.76741287161552],
        },
    },
    "external_G_single": {
        "coef": 0.9501599071103206,
        "models": {
            "ml_G": [None, None],
            "ml_G_inner_0": [None, None],
            "ml_G_inner_1": [None, None],
            "ml_G_inner_2": [None, None],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [412.4251345711083, 0, 212.7878420344251],
            "ml_G_inner_0": [101.71005410754454, 277, 54.16623977056675],
            "ml_G_inner_1": [95.41945055538218, 274, 47.0189319066049],
            "ml_G_inner_2": [92.89852062463592, 285, 51.50914254436097],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [372.2562793377614, 0, 189.21022799297924],
        },
        "psi": [-1.7763568394002505e-15, 0, 7.804882120514607],
        "se": 0.14394127914122434,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [0.0, 400, 0.0],
            "ml_G_inner_1": [0.0, 400, 0.0],
            "ml_G_inner_2": [0.0, 400, 0.0],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [140.68869382798357, 218, 74.66615317976003],
        },
    },
    "external_G_two_subsamples": {
        "coef": 0.9666204979262695,
        "models": {
            "ml_G": [None, None],
            "ml_G_inner_0": [None, None],
            "ml_G_inner_1": [None, None],
            "ml_G_inner_2": [None, None],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [364.1378505313389, 0, 190.01826247346216],
            "ml_G_inner_0": [142.63115909503355, 266, 71.54083571228573],
            "ml_G_inner_1": [93.8268173227433, 266, 44.37521858559627],
            "ml_G_inner_2": [121.15676632901739, 266, 63.73044867167752],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [332.04543874943533, 0, 170.3353246178571],
        },
        "psi": [-1.9539925233402755e-14, 0, -0.9335257604339819],
        "se": 0.15587165867681954,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [0.0, 400, 0.0],
            "ml_G_inner_1": [0.0, 400, 0.0],
            "ml_G_inner_2": [0.0, 400, 0.0],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [110.53942137371226, 218, 58.66459572272909],
        },
    },
    "external_nested_double_k2": {
        "coef": 0.9405534023909443,
        "models": {
            "ml_G": [6, 6],
            "ml_G_inner_0": [2, 2],
            "ml_G_inner_1": [2, 2],
            "ml_G_inner_2": [2, 2],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [None, None],
        },
        "predictions": {
            "ml_G": [404.8780706250595, 0, 208.77896510553938],
            "ml_G_inner_0": [288.7591447023088, 134, 148.6724384258225],
            "ml_G_inner_1": [255.7824873434321, 134, 128.62144045485584],
            "ml_G_inner_2": [267.20418869375095, 132, 142.01724389373743],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [361.2970472398607, 0, 182.46248716738594],
        },
        "psi": [-2.842170943040401e-14, 0, 8.61270108517251],
        "se": 0.14597448625299328,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [187.3315643000785, 257, 94.16855128066373],
            "ml_G_inner_1": [179.39176723904365, 260, 87.60251842006036],
            "ml_G_inner_2": [178.07756422839765, 247, 94.21691966604936],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [136.3002988003558, 218, 72.76741287161552],
        },
    },
    "external_nested_single": {
        "coef": 0.9501599071103206,
        "models": {
            "ml_G": [3, 3],
            "ml_G_inner_0": [1, 1],
            "ml_G_inner_1": [1, 1],
            "ml_G_inner_2": [1, 1],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [None, None],
        },
        "predictions": {
            "ml_G": [412.4251345711083, 0, 212.7878420344251],
            "ml_G_inner_0": [101.71005410754454, 277, 54.16623977056675],
            "ml_G_inner_1": [95.41945055538218, 274, 47.0189319066049],
            "ml_G_inner_2": [92.89852062463592, 285, 51.50914254436097],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [372.2562793377614, 0, 189.21022799297924],
        },
        "psi": [-1.7763568394002505e-15, 0, 7.804882120514607],
        "se": 0.14394127914122434,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [0.0, 400, 0.0],
            "ml_G_inner_1": [0.0, 400, 0.0],
            "ml_G_inner_2": [0.0, 400, 0.0],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [140.68869382798357, 218, 74.66615317976003],
        },
    },
    "external_nested_two_subsamples": {
        "coef": 0.9666204979262695,
        "models": {
            "ml_G": [3, 3],
            "ml_G_inner_0": [1, 1],
            "ml_G_inner_1": [1, 1],
            "ml_G_inner_2": [1, 1],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [None, None],
        },
        "predictions": {
            "ml_G": [364.1378505313389, 0, 190.01826247346216],
            "ml_G_inner_0": [142.63115909503355, 266, 71.54083571228573],
            "ml_G_inner_1": [93.8268173227433, 266, 44.37521858559627],
            "ml_G_inner_2": [121.15676632901739, 266, 63.73044867167752],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [332.04543874943533, 0, 170.3353246178571],
        },
        "psi": [-1.9539925233402755e-14, 0, -0.9335257604339819],
        "se": 0.15587165867681954,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [88.59623860310779, 328, 44.144192435175555],
            "ml_G_inner_1": [92.299243165786, 327, 44.55434597674206],
            "ml_G_inner_2": [85.40762939383393, 327, 43.942715714195664],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [110.53942137371226, 218, 58.66459572272909],
        },
    },
    "forest_double_k2": {
        "coef": 0.9136942114007448,
        "models": {
            "ml_G": [6, 6],
            "ml_G_inner_0": [2, 2],
            "ml_G_inner_1": [2, 2],
            "ml_G_inner_2": [2, 2],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [448.1621401100606, 0, 229.61361704271036],
            "ml_G_inner_0": [302.6312063910474, 134, 148.57676687265842],
            "ml_G_inner_1": [290.5949804204096, 134, 143.79664942717423],
            "ml_G_inner_2": [301.0204614950165, 132, 155.8093136182785],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [408.91794250086036, 0, 207.28712686161867],
        },
        "psi": [-3.552713678800501e-15, 0, 16.700299084384117],
        "se": 0.15838212362593743,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [187.3315643000785, 257, 94.16855128066373],
            "ml_G_inner_1": [179.39176723904365, 260, 87.60251842006036],
            "ml_G_inner_2": [178.07756422839765, 247, 94.21691966604936],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [173.7479750714287, 218, 93.83679792153778],
        },
    },
    "forest_single": {
        "coef": 1.0469012390333807,
        "models": {
            "ml_G": [3, 3],
            "ml_G_inner_0": [1, 1],
            "ml_G_inner_1": [1, 1],
            "ml_G_inner_2": [1, 1],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [444.4280405705747, 0, 227.23339448124733],
            "ml_G_inner_0": [121.53358715251366, 277, 64.17054527111354],
            "ml_G_inner_1": [111.65784176603623, 274, 56.761320957843495],
            "ml_G_inner_2": [86.65503003637056, 285, 49.03959612909552],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [405.4072165763259, 0, 206.33539253584422],
        },
        "psi": [6.394884621840902e-14, 0, 15.0800736256159],
        "se": 0.18175478340640852,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [0.0, 400, 0.0],
            "ml_G_inner_1": [0.0, 400, 0.0],
            "ml_G_inner_2": [0.0, 400, 0.0],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [173.22852418882835, 218, 93.18916063021165],
        },
    },
    "params_double_k2": {
        "coef": 0.9398687699622265,
        "models": {
            "ml_G": [6, 6],
            "ml_G_inner_0": [2, 2],
            "ml_G_inner_1": [2, 2],
            "ml_G_inner_2": [2, 2],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [463.51436268625355, 0, 234.3229600113495],
            "ml_G_inner_0": [333.00303425567097, 134, 164.12737930731785],
            "ml_G_inner_1": [304.2130279617247, 134, 150.09747971659212],
            "ml_G_inner_2": [296.0972675110832, 132, 154.34397322407904],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [452.5145540469041, 0, 227.60442647400367],
        },
        "psi": [7.105427357601002e-15, 0, 16.981892650496487],
        "se": 0.14515713611894412,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [187.3315643000785, 257, 94.16855128066373],
            "ml_G_inner_1": [179.39176723904365, 260, 87.60251842006036],
            "ml_G_inner_2": [178.07756422839765, 247, 94.21691966604936],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [194.5553168754108, 218, 100.68636306704833],
        },
    },
    "params_single": {
        "coef": 0.9455872913479213,
        "models": {
            "ml_G": [3, 3],
            "ml_G_inner_0": [1, 1],
            "ml_G_inner_1": [1, 1],
            "ml_G_inner_2": [1, 1],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [456.5770506456398, 0, 231.71225922913212],
            "ml_G_inner_0": [134.2736043205384, 277, 68.6335397467677],
            "ml_G_inner_1": [130.48960370461435, 274, 64.35036354807258],
            "ml_G_inner_2": [114.79282135502196, 285, 62.89801664145788],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [439.4166438380438, 0, 221.6663494645772],
        },
        "psi": [1.4210854715202004e-14, 0, 14.722919525292095],
        "se": 0.1416817294831709,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [0.0, 400, 0.0],
            "ml_G_inner_1": [0.0, 400, 0.0],
            "ml_G_inner_2": [0.0, 400, 0.0],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [184.74833133385636, 218, 95.88518223760995],
        },
    },
    "plain_double_k2_models": {
        "coef": 0.9405534023909443,
        "models": {
            "ml_G": [6, 6],
            "ml_G_inner_0": [2, 2],
            "ml_G_inner_1": [2, 2],
            "ml_G_inner_2": [2, 2],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [404.8780706250595, 0, 208.77896510553938],
            "ml_G_inner_0": [288.7591447023088, 134, 148.6724384258225],
            "ml_G_inner_1": [255.7824873434321, 134, 128.62144045485584],
            "ml_G_inner_2": [267.20418869375095, 132, 142.01724389373743],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [361.2970472398607, 0, 182.46248716738594],
        },
        "psi": [-2.842170943040401e-14, 0, 8.61270108517251],
        "se": 0.14597448625299328,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [187.3315643000785, 257, 94.16855128066373],
            "ml_G_inner_1": [179.39176723904365, 260, 87.60251842006036],
            "ml_G_inner_2": [178.07756422839765, 247, 94.21691966604936],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [136.3002988003558, 218, 72.76741287161552],
        },
    },
    "plain_double_k2_nomodels": {
        "coef": 0.9405534023909443,
        "models": None,
        "predictions": {
            "ml_G": [404.8780706250595, 0, 208.77896510553938],
            "ml_G_inner_0": [288.7591447023088, 134, 148.6724384258225],
            "ml_G_inner_1": [255.7824873434321, 134, 128.62144045485584],
            "ml_G_inner_2": [267.20418869375095, 132, 142.01724389373743],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [361.2970472398607, 0, 182.46248716738594],
        },
        "psi": [-2.842170943040401e-14, 0, 8.61270108517251],
        "se": 0.14597448625299328,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [187.3315643000785, 257, 94.16855128066373],
            "ml_G_inner_1": [179.39176723904365, 260, 87.60251842006036],
            "ml_G_inner_2": [178.07756422839765, 247, 94.21691966604936],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [136.3002988003558, 218, 72.76741287161552],
        },
    },
    "plain_double_k3_models": {
        "coef": 0.9391181008128902,
        "models": {
            "ml_G": [9, 9],
            "ml_G_inner_0": [3, 3],
            "ml_G_inner_1": [3, 3],
            "ml_G_inner_2": [3, 3],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [413.1610405185852, 0, 212.6459488262086],
            "ml_G_inner_0": [283.16511334908716, 134, 146.04585966195376],
            "ml_G_inner_1": [271.52547722070426, 134, 134.59962031604857],
            "ml_G_inner_2": [277.72907582922153, 132, 148.3436092357921],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [377.1247635171285, 0, 190.6308694249624],
        },
        "psi": [7.105427357601002e-15, 0, 8.64766027435568],
        "se": 0.14332588996604398,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [187.3315643000785, 257, 94.16855128066373],
            "ml_G_inner_1": [179.39176723904365, 260, 87.60251842006036],
            "ml_G_inner_2": [178.07756422839765, 247, 94.21691966604936],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [141.1740769915219, 218, 74.89070597163952],
        },
    },
    "plain_double_k3_nomodels": {
        "coef": 0.9391181008128902,
        "models": None,
        "predictions": {
            "ml_G": [413.1610405185852, 0, 212.6459488262086],
            "ml_G_inner_0": [283.16511334908716, 134, 146.04585966195376],
            "ml_G_inner_1": [271.52547722070426, 134, 134.59962031604857],
            "ml_G_inner_2": [277.72907582922153, 132, 148.3436092357921],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [377.1247635171285, 0, 190.6308694249624],
        },
        "psi": [7.105427357601002e-15, 0, 8.64766027435568],
        "se": 0.14332588996604398,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [187.3315643000785, 257, 94.16855128066373],
            "ml_G_inner_1": [179.39176723904365, 260, 87.60251842006036],
            "ml_G_inner_2": [178.07756422839765, 247, 94.21691966604936],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [141.1740769915219, 218, 74.89070597163952],
        },
    },
    "plain_single_models": {
        "coef": 0.9501599071103206,
        "models": {
            "ml_G": [3, 3],
            "ml_G_inner_0": [1, 1],
            "ml_G_inner_1": [1, 1],
            "ml_G_inner_2": [1, 1],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [412.4251345711083, 0, 212.7878420344251],
            "ml_G_inner_0": [101.71005410754454, 277, 54.16623977056675],
            "ml_G_inner_1": [95.41945055538218, 274, 47.0189319066049],
            "ml_G_inner_2": [92.89852062463592, 285, 51.50914254436097],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [372.2562793377614, 0, 189.21022799297924],
        },
        "psi": [-1.7763568394002505e-15, 0, 7.804882120514607],
        "se": 0.14394127914122434,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [0.0, 400, 0.0],
            "ml_G_inner_1": [0.0, 400, 0.0],
            "ml_G_inner_2": [0.0, 400, 0.0],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [140.68869382798357, 218, 74.66615317976003],
        },
    },
    "plain_single_nomodels": {
        "coef": 0.9501599071103206,
        "models": None,
        "predictions": {
            "ml_G": [412.4251345711083, 0, 212.7878420344251],
            "ml_G_inner_0": [101.71005410754454, 277, 54.16623977056675],
            "ml_G_inner_1": [95.41945055538218, 274, 47.0189319066049],
            "ml_G_inner_2": [92.89852062463592, 285, 51.50914254436097],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [372.2562793377614, 0, 189.21022799297924],
        },
        "psi": [-1.7763568394002505e-15, 0, 7.804882120514607],
        "se": 0.14394127914122434,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [0.0, 400, 0.0],
            "ml_G_inner_1": [0.0, 400, 0.0],
            "ml_G_inner_2": [0.0, 400, 0.0],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [140.68869382798357, 218, 74.66615317976003],
        },
    },
    "plain_two_subsamples_models": {
        "coef": 0.9666204979262695,
        "models": {
            "ml_G": [3, 3],
            "ml_G_inner_0": [1, 1],
            "ml_G_inner_1": [1, 1],
            "ml_G_inner_2": [1, 1],
            "ml_M": [3, 3],
            "ml_m": [3, 3],
            "ml_nested_g": [3, 3],
        },
        "predictions": {
            "ml_G": [364.1378505313389, 0, 190.01826247346216],
            "ml_G_inner_0": [142.63115909503355, 266, 71.54083571228573],
            "ml_G_inner_1": [93.8268173227433, 266, 44.37521858559627],
            "ml_G_inner_2": [121.15676632901739, 266, 63.73044867167752],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [332.04543874943533, 0, 170.3353246178571],
        },
        "psi": [-1.9539925233402755e-14, 0, -0.9335257604339819],
        "se": 0.15587165867681954,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [88.59623860310779, 328, 44.144192435175555],
            "ml_G_inner_1": [92.299243165786, 327, 44.55434597674206],
            "ml_G_inner_2": [85.40762939383393, 327, 43.942715714195664],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [110.53942137371226, 218, 58.66459572272909],
        },
    },
    "plain_two_subsamples_nomodels": {
        "coef": 0.9666204979262695,
        "models": None,
        "predictions": {
            "ml_G": [364.1378505313389, 0, 190.01826247346216],
            "ml_G_inner_0": [142.63115909503355, 266, 71.54083571228573],
            "ml_G_inner_1": [93.8268173227433, 266, 44.37521858559627],
            "ml_G_inner_2": [121.15676632901739, 266, 63.73044867167752],
            "ml_M": [219.91649205125316, 0, 111.27565728715702],
            "ml_m": [220.5773046029587, 0, 110.94085839502964],
            "ml_nested_g": [332.04543874943533, 0, 170.3353246178571],
        },
        "psi": [-1.9539925233402755e-14, 0, -0.9335257604339819],
        "se": 0.15587165867681954,
        "targets": {
            "ml_G": [272.4004478837599, 182, 137.99399468338675],
            "ml_G_inner_0": [88.59623860310779, 328, 44.144192435175555],
            "ml_G_inner_1": [92.299243165786, 327, 44.55434597674206],
            "ml_G_inner_2": [85.40762939383393, 327, 43.942715714195664],
            "ml_M": [218.0, 0, 107.07749999999999],
            "ml_m": [218.0, 0, 107.07749999999999],
            "ml_nested_g": [110.53942137371226, 218, 58.66459572272909],
        },
    },
}


@pytest.fixture(scope="module", params=list(CASES))
def case_fit(request):
    return request.param, _summary(_fit_case(CASES[request.param]))


def _assert_summary_equal(actual, pinned):
    assert actual.keys() == pinned.keys()
    for key, value in actual.items():
        if key in ("coef", "se"):
            np.testing.assert_allclose(value, pinned[key], rtol=1e-7)
        elif key == "models":
            assert value == pinned[key]
        elif key == "psi":
            np.testing.assert_allclose(value, pinned[key], rtol=1e-7, atol=1e-9)
        else:
            assert value.keys() == pinned[key].keys()
            for name, checksums in value.items():
                np.testing.assert_allclose(checksums, pinned[key][name], rtol=1e-7, atol=1e-9, err_msg=f"{key}[{name}]")


@pytest.mark.ci
def test_counterfactual_nuisance_estimation_is_pinned(case_fit):
    """Test that the score, predictions, targets and stored models of the counterfactual estimator equal the pins."""
    case, summary = case_fit
    _assert_summary_equal(summary, PINNED_CASES[case])
