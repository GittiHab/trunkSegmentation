import numpy as np
import pytest
from datasets.preprocessing import Identity

# Create a fixture for the Identity class
@pytest.fixture
def identity():
    return Identity.singleton()

# Test the _apply method of the Identity class
def test_apply_returns_input_images_and_labels(identity):
    # Create sample input data
    images = np.array([1, 2, 3])
    labels = np.array([4, 5, 6])
    masks = None
    axis = 0

    # Call the _apply method of the Identity class
    result_images, result_labels = identity._apply(images, labels, masks, axis)

    # Assert that the returned images and labels are the same as the input
    assert np.array_equal(result_images, images)
    assert np.array_equal(result_labels, labels)
