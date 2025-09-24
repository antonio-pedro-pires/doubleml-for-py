import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split, KFold
import pytest
import doubleml as dml

from doubleml.med.utils._med_utils import _trim_probabilities, recombine_samples, divide_samples, extract_sets_from_smpls
from doubleml.tests._utils_dml_cv_predict import _dml_cv_predict_ut_version
from doubleml.utils._estimation import _dml_cv_predict

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression

from .. import DoubleMLMEDC, DoubleMLMEDP
from ._utils_med_manual import ManualMedC, ManualMedCAlt, ManualMedP
from ...datasets import make_med_data


@pytest.fixture(
    scope="module",
    params=[
        ["lower", 0.2, np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])],
        [
            "higher",
            0.8,
            np.array(
                [
                    0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                ]
            ),
        ],
        [
            "both",
            0.2,
            np.array(
                [
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                ]
            ),
        ],
    ],
)
def method_threshold_expected_array(request):
    return request.param


@pytest.mark.ci
def test_trim_probabilities_1d(method_threshold_expected_array):
    full_array = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    method = method_threshold_expected_array[0]
    threshold = method_threshold_expected_array[1]
    expected_array = method_threshold_expected_array[2]

    trimmed_array = _trim_probabilities(full_array, threshold, method)
    assert np.array_equal(trimmed_array, expected_array)


@pytest.mark.ci
def test_trim_probabilities_nd():
    full_array = np.array([np.linspace(0, 1, num=10, endpoint=False), np.linspace(1, 0, num=10, endpoint=False)])
    mask = (full_array[0] <= 0.7) & (full_array[1] >= 0.2)
    expected_array = np.array([full_array[0][mask], full_array[1][mask]])
    trimmed_array = _trim_probabilities(full_array, conditions=mask)

    assert np.array_equal(trimmed_array, expected_array)


@pytest.fixture(
    scope="module",
    params=[
        [[np.linspace(0, 100, 100, endpoint=False, dtype=int)]],
        [[np.linspace(0, 100, 100, endpoint=False, dtype=int), np.linspace(101, 200, 100, endpoint=False, dtype=int)]],
    ],
)
def smpls(request):
    full_smpls = request.param[0]
    train_test_smpls = []
    train_smpls = []
    test_smpls = []
    np.random.seed(10)
    for idx, smpl in enumerate(full_smpls):
        train_idx, test_idx = train_test_split(smpl, test_size=0.7)
        train_smpls.append(train_idx)
        test_smpls.append(test_idx)
        train_test_smpls.append((train_idx, test_idx))

    return [train_smpls, test_smpls, train_test_smpls]


@pytest.fixture(scope="module", params=[0, 1, 2])
def idx(request):
    return request.param


@pytest.mark.ci
def test_separate_samples(smpls, idx):
    # TODO: Create fixture for idx
    idx = idx
    expected_indices = smpls[idx]
    train_test_smpls = smpls[2]

    if idx > 1:
        with pytest.raises(AssertionError):
            extract_sets_from_smpls(train_test_smpls, idx)
    else:
        actual_indices = extract_sets_from_smpls(train_test_smpls, idx)

        for a, e in zip(actual_indices, expected_indices):
            assert np.array_equal(a, e)


# TODO: Modify test_divide_samples to only take 1 set of indices
@pytest.mark.ci
def test_divide_samples(smpls):
    smpls_ratio = 0.5
    smpl_to_divide = smpls[0]
    np.random.seed(10)
    expected_results = []
    expected_subsample1 = []
    expected_subsample2 = []
    for smpl in smpl_to_divide:
        subsample1, subsample2 = train_test_split(smpl, test_size=smpls_ratio)
        expected_subsample1.append(subsample1)
        expected_subsample2.append(subsample2)
    expected_results.append((expected_subsample1, expected_subsample2))

    np.random.seed(10)
    actual_results = divide_samples(
        smpls,
        smpls_ratio,
    )
    for a, e in zip(actual_results, expected_results):
        assert np.array_equal(a, e)


@pytest.fixture(scope="module", params=[2, 4, 8])
def results_s1_s2(request):
    subsamples1 = []
    subsamples2 = []
    results = []
    nb_subsamples_arrays = request.param
    for i in np.arange(1, nb_subsamples_arrays):
        s1 = np.linspace(0, i * 100, 10, endpoint=False)
        s2 = np.linspace(0, i * (-100), 10, endpoint=False)
        subsamples1.append(s1)
        subsamples2.append(s2)
        results.append((s1, s2))
    return [results, subsamples1, subsamples2]


def test_recombine_samples(results_s1_s2):
    expected_results = results_s1_s2[0]
    s1 = results_s1_s2[1]
    s2 = results_s1_s2[2]

    actual_results = recombine_samples(s1, s2)
    for a, e in zip(actual_results, expected_results):
        assert np.array_equal(a, e)


@pytest.fixture(scope="module")
def data():
    n_folds = 2
    n_rep_boot = 499
    n_obs = 500
    np.random.seed(10)
    data_med = make_med_data(n_obs=n_obs)

    y, d, m, x = data_med.y, data_med.d, data_med.m, data_med.x

    df_med = pd.DataFrame(
        np.column_stack((y, d, m, x)), columns=["y", "d", "m"] + ["x" + str(i) for i in range(data_med.x.shape[1])]
    )

    return dml.DoubleMLMediationData(df_med, "y", "d", "m")


@pytest.fixture(
    scope="module", params=[LinearRegression(), RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]
)
def learners(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 1])
def treatment_level(request):
    return request.param


@pytest.fixture(
    scope="module",
)
def nested_cv_fit_predict(data, learners, treatment_level):
    n_jobs = None
    est_params = None

    is_classifier = DoubleMLMEDC._check_learner(learner=learners, learner_name="ml_g", regressor=True, classifier=True)
    if is_classifier:
        _predict_method = {"learner": "predict_proba"}
    else:
        _predict_method = {"learner": "predict"}

    return_models = False
    treated = data.d == treatment_level
