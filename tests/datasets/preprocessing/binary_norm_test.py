import pytest
import numpy as np
from unittest.mock import patch
from datasets.preprocessing import BinaryNorm


@pytest.fixture
def binary_norm():
    return BinaryNorm()

values = [
    (np.array([0, 1, 0, 0]), np.array([0, 1, 0, 0])),
    (np.array([0, 2, 0, 0]), np.array([0, 1, 0, 0])),
    (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])),
    (np.array([2, 2, 2, 2]), np.array([1, 1, 1, 1])),
]

@pytest.mark.parametrize("labels, expected", values)
def test_apply_numpy(binary_norm, labels, expected):
    images = np.zeros_like(labels)
    result = binary_norm._apply(images, labels, None, None)[1]
    assert np.array_equal(result, expected)

@pytest.mark.parametrize("labels, expected", values)
def test_apply_list(binary_norm, labels, expected):
    images = np.zeros_like(labels)
    result = binary_norm._apply(images, [labels], None, None)[1]
    assert len(result) == 1
    assert np.array_equal(result[0], expected)

def test_apply_long_list(binary_norm):
    labels = [np.array([0, 0, 0, 0]), np.array([0, 0, 255, 0]), np.array([0, 0, 0, 0])]
    result = binary_norm._apply(labels, labels, None, None)[1]
    assert np.array_equal(result, [np.array([0, 0, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 0])])


def test_fail_non_binary_array(binary_norm):
    with pytest.raises(AssertionError):
        labels = np.array([0, 0, 1, 2])  # unsupported label type
        binary_norm._apply(labels, labels, None, None)

def test_fail_non_binary_list(binary_norm):
    with pytest.raises(AssertionError):
        labels = [np.array([0, 0, 1, 0]), np.array([0, 0, 3, 2])]  # unsupported label type
        binary_norm._apply(labels, labels, None, None)

def test_apply_unsupported(binary_norm):
    with pytest.raises(NotImplementedError):
        images = np.zeros((2, 2))
        labels = 123  # unsupported label type
        binary_norm._apply(images, labels, None, None)


def test_apply_skip_unsupported(binary_norm):
    binary_norm.skip_unsupported = True
    with patch('warnings.warn') as mock_warn:
        images = np.zeros((2, 2))
        labels = 123  # unsupported label type
        binary_norm._apply(images, labels, None, None)
        mock_warn.assert_called_once()
