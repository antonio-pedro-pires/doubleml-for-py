import numpy as np
import pytest

from doubleml.med.utils._med_utils import _trim_probabilities


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
