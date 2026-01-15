import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLMEDData
from doubleml.med.datasets import make_med_data
from doubleml.med.med import DoubleMLMED
from doubleml.utils import DMLDummyClassifier, DMLDummyRegressor

@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param

@pytest.fixture(scope="module", params=["potential", "counterfactual"])
def target(request):
    return request.param

@pytest.fixture(scope="module", params=[True, False])
def set_ml_yx_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_ymx_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_px_ext(request):
    return request.param

@pytest.fixture(scope="module", params=[True, False])
def set_ml_pmx_ext(request):
    return request.param

@pytest.fixture(scope="module", params=[True, False])
def set_ml_nested_ext(request):
    return request.param

@pytest.fixture(scope="module")
def dml_med_fixture(n_rep, target, set_ml_yx_ext, set_ml_ymx_ext, set_ml_px_ext, set_ml_pmx_ext, set_ml_nested_ext):
    x, y, d, m = make_med_data(return_type="array")
    ext_predictions = {"d":{}}

    np.random.seed(3141)
    med_data = DoubleMLMEDData.from_arrays(x, y, d, m)

    kwargs = {"n_rep": n_rep, "target": target, "med_data": med_data}

    #TODO: Cache the estimators so that tests are faster?
    dml_med = DoubleMLMED(ml_yx= LinearRegression(), ml_px=LogisticRegression(), ml_ymx=LinearRegression(), ml_pmx=LogisticRegression(max_iter=1000), ml_nested=LinearRegression(), **kwargs)
    np.random.seed(3141)
    dml_med.fit(store_predictions=True)

    if target == "potential":
        if set_ml_yx_ext:
            ext_predictions["d"]["ml_yx"] = dml_med.predictions["ml_yx"][:, :, 0]
            ml_yx = DMLDummyRegressor()
        else:
            ml_yx = LinearRegression()
        #TODO: Potentially take this out of this conditional branch since both targets need to compute px. Same not true for yx.
        if set_ml_px_ext:
            ext_predictions["d"]["ml_px"] = dml_med.predictions["ml_px"][:, :, 0]
            ml_px = DMLDummyClassifier()
        else:
            ml_px = LogisticRegression(max_iter=1000)

        dml_med_ext = DoubleMLMED(ml_yx=ml_yx, ml_px=ml_px, **kwargs)

        np.random.seed(3141)
        dml_med_ext.fit(external_predictions=ext_predictions)

        res_dict = {
            "coef_normal": dml_med.coef[0],
            "coef_ext": dml_med_ext.coef[0],
            "se_normal": dml_med.se[0],
            "se_ext": dml_med_ext.se[0],
        }
    else:
        if set_ml_yx_ext:
            ext_predictions["d"]["ml_yx"] = dml_med.predictions["ml_yx"][:, :, 0]
            ml_yx = DMLDummyRegressor()
        else:
            ml_yx = LinearRegression()
        if set_ml_px_ext:
            ext_predictions["d"]["ml_px"] = dml_med.predictions["ml_px"][:, :, 0]
            ml_px = DMLDummyClassifier()
        else:
            ml_px = LogisticRegression(max_iter=1000)
        if set_ml_ymx_ext:
            ext_predictions["d"]["ml_ymx"] = dml_med.predictions["ml_ymx"][:, :, 0]

            for i in range(dml_med.n_folds_inner):
                ext_predictions["d"][f"ml_ymx_inner_{i}"] = dml_med.predictions[f"ml_ymx_inner_{i}"][:, :, 0]
            ml_ymx = DMLDummyRegressor()
        else:
            ml_ymx = LinearRegression()
        if set_ml_pmx_ext:
            ext_predictions["d"]["ml_pmx"] = dml_med.predictions["ml_pmx"][:, :, 0]
            ml_pmx = DMLDummyClassifier()
        else:
            ml_pmx = LogisticRegression(max_iter=1000)
        if set_ml_nested_ext:
            ext_predictions["d"]["ml_nested"] = dml_med.predictions["ml_nested"][:, :, 0]
            ml_nested = DMLDummyRegressor()
        else:
            ml_nested = LinearRegression()

        dml_med_ext = DoubleMLMED(ml_yx=ml_yx, ml_px=ml_px, ml_ymx=ml_ymx, ml_pmx=ml_pmx, ml_nested=ml_nested, **kwargs)

        np.random.seed(3141)
        dml_med_ext.fit(external_predictions=ext_predictions)

        res_dict = {
            "coef_normal": dml_med.coef[0],
            "coef_ext": dml_med_ext.coef[0],
            "se_normal": dml_med.se[0],
            "se_ext": dml_med_ext.se[0],
        }
    return res_dict

@pytest.mark.ci
def test_dml_med_coef(dml_med_fixture):
    assert math.isclose(dml_med_fixture["coef_normal"], dml_med_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_se(dml_med_fixture):
    assert math.isclose(dml_med_fixture["se_normal"], dml_med_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_med_exceptions(n_rep, target, set_ml_yx_ext, set_ml_ymx_ext, set_ml_px_ext, set_ml_pmx_ext, set_ml_nested_ext,):
    if target == "counterfactual":
        ext_predictions = {"d": {}}

        x, y, d, m = make_med_data(return_type="array")

        np.random.seed(3141)
        med_data = DoubleMLMEDData.from_arrays(x=x, y=y, d=d, m=m)

        kwargs = {"med_data": med_data, "n_rep": n_rep, "target": target}

        dml_med = DoubleMLMED(ml_yx= LinearRegression(), ml_px=LogisticRegression(max_iter=1000), ml_ymx=LinearRegression(), ml_pmx=LogisticRegression(max_iter=1000), ml_nested=LinearRegression(), **kwargs)
        np.random.seed(3141)
        dml_med.fit(store_predictions=True)

        # prepare external predictions and dummy learners

        ml_ymx = LinearRegression()
        ml_pmx = LogisticRegression(max_iter=1000)
        ml_yx = LinearRegression()
        ml_px = LogisticRegression(max_iter=1000)
        ml_nested = LinearRegression()


        dml_med_ext = DoubleMLMED(ml_yx=ml_yx, ml_px=ml_px, ml_ymx=ml_ymx, ml_pmx=ml_pmx, ml_nested=ml_nested, **kwargs)

        ext_predictions["d"]["ml_ymx"] = dml_med.predictions["ml_ymx"][:, :, 0]

        for i in range(dml_med.n_folds_inner - 1):
            ext_predictions["d"][f"ml_ymx_inner_{i}"] = dml_med.predictions[f"ml_ymx_inner_{i}"][:, :, 0]

        msg = r"When providing external predictions for ml_ymx, also inner predictions for all inner folds"
        with pytest.raises(ValueError, match=msg):
            dml_med_ext.fit(external_predictions=ext_predictions)

        ext_predictions["d"][f"ml_ymx_inner_{dml_med.n_folds_inner-1}"] = (dml_med.predictions)[
            f"ml_ymx_inner_{dml_med.n_folds_inner-1}"
        ][:, :, 0]
