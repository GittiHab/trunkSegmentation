import numpy as np
import pytest
from datasets.preprocessing import Caching, Preprocessing
from unittest.mock import MagicMock
import os


# Create a fixture for the Caching class
@pytest.fixture
def caching(tmp_path):
    preprocessing_mock = MagicMock(spec=Preprocessing)
    return Caching(preprocessing_mock, path=str(tmp_path))


# Test the _apply method of the Caching class when cache exists
def test_apply_cache_exists(caching, tmp_path):
    # Create sample input data
    images = np.array([1, 2, 3])
    labels = np.array([4, 5, 6])
    masks = None
    axis = 0

    # Create cache files
    np.save(os.path.join(tmp_path, "images.npy"), images + 1)
    np.save(os.path.join(tmp_path, "labels.npy"), labels + 1)

    # Call the _apply method of the Caching class
    result_images, result_labels = caching(images, labels, masks, axis)

    # Assert that the returned images and labels are the cached values
    assert np.array_equal(result_images, images + 1)
    assert np.array_equal(result_labels, labels + 1)


# Test the _apply method of the Caching class when cache doesn't exist
def test_apply_cache_not_exists(caching, tmp_path):
    # Create sample input data
    images = np.array([1, 2, 3])
    labels = np.array([4, 5, 6])
    masks = None
    axis = 0

    # Mock preprocessing step to return processed values
    preprocessing_mock = caching.preprocessing
    preprocessing_mock.return_value = (images * 2, labels * 2)

    # Call the _apply method of the Caching class
    result_images, result_labels = caching(images, labels, masks, axis)

    # Assert that the preprocessing step is called
    assert preprocessing_mock.call_args_list == [((images, labels, masks, axis),)]

    # Assert that the returned images and labels are the processed values
    assert np.array_equal(result_images, images * 2)
    assert np.array_equal(result_labels, labels * 2)
