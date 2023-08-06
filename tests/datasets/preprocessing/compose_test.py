import numpy as np
import pytest
from datasets.preprocessing import Compose, Preprocessing
from unittest.mock import MagicMock


# Create a fixture for the Compose class
@pytest.fixture
def compose():
    steps = [
        MagicMock(spec=Preprocessing),
        MagicMock(spec=Preprocessing),
        MagicMock(spec=Preprocessing)
    ]
    return Compose(steps)


# Test the _apply method of the Compose class
def test_apply(compose):
    # Create sample input data
    images = 2
    labels = 7
    masks = None
    axis = 0

    # Set up mock objects
    step_mocks = compose.steps
    step_mocks[0].return_value = (images + 1, labels + 1)
    step_mocks[1].return_value = (images - 1, labels - 1)
    step_mocks[2].return_value = (images * 2, labels * 2)

    # Call the _apply method of the Compose class
    result_images, result_labels = compose(images, labels, masks, axis)

    # Assert that each step's _apply method is called in order
    assert step_mocks[0].call_args_list == [((images, labels, masks, axis),)]
    assert step_mocks[1].call_args_list == [((images + 1, labels + 1, masks, axis),)]
    assert step_mocks[2].call_args_list == [((images - 1, labels - 1, masks, axis),)]

    # Assert that the returned images and labels are the result of the composition of steps
    assert np.array_equal(result_images, images * 2)
    assert np.array_equal(result_labels, labels * 2)
