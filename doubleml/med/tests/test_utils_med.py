import random

import numpy as np
from sklearn.model_selection import train_test_split
import pytest

from doubleml.med.utils._med_utils import _trim_probabilities, separate_samples_for_nested_estimator


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
    np.random.seed(10)
    for idx, smpl in enumerate(full_smpls):
        train_idx, test_idx = train_test_split(smpl, test_size=0.7)
        train_test_smpls.append((train_idx, test_idx))
    return train_test_smpls


def test_separate_samples_for_nested_estimator(smpls):
    smpls_ratio = 0.5

    np.random.seed(10)
    expected_smpls = []
    for idx, (train_idx, test_idx) in enumerate(smpls):
        expected_delta, expected_mu = train_test_split(train_idx, test_size=smpls_ratio)
        expected_smpls.append((expected_delta, expected_mu, train_idx, test_idx))

    np.random.seed(10)
    actual_smpls = separate_samples_for_nested_estimator(
        smpls,
        smpls_ratio,
    )
    for a, e in zip(actual_smpls, expected_smpls):
        np.array_equal(a, e)
