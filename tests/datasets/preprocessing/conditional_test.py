import numpy as np
import pytest
from datasets.preprocessing import Conditional, Preprocessing
from unittest.mock import MagicMock


# Create a fixture for the Conditional class
@pytest.fixture
def conditional():
    step = MagicMock(spec=Preprocessing)
    otherwise = MagicMock(spec=Preprocessing)
    return Conditional(step, condition=True, otherwise=otherwise)


# Test the _apply method of the Conditional class when the condition is True
def test_apply_condition_true(conditional):
    # Create sample input data
    images = 2
    labels = 7
    masks = None
    axis = 0

    # Set up mock objects
    step_mock = conditional.step
    otherwise_mock = conditional.otherwise
    step_mock.return_value = (images + 1, labels + 1)
    otherwise_mock.return_value = (images - 1, labels - 1)

    # Call the _apply method of the Conditional class
    result_images, result_labels = conditional(images, labels, masks, axis)

    # Assert that the step's _apply method is called
    step_mock.assert_called_once_with(images, labels, masks, axis)

    # Assert that the returned images and labels are the result of the step's _apply method
    assert np.array_equal(result_images, images + 1)
    assert np.array_equal(result_labels, labels + 1)


# Test the _apply method of the Conditional class when the condition is False
def test_apply_condition_false(conditional):
    # Create sample input data
    images = 2
    labels = 7
    masks = None
    axis = 0

    # Set up mock objects
    step_mock = conditional.step
    otherwise_mock = conditional.otherwise
    step_mock.return_value = (images + 1, labels + 1)
    otherwise_mock.return_value = (images - 1, labels - 1)

    # Update the condition to False
    conditional.condition = False

    # Call the _apply method of the Conditional class
    result_images, result_labels = conditional(images, labels, masks, axis)

    # Assert that the otherwise's _apply method is called
    otherwise_mock.assert_called_once_with(images, labels, masks, axis)

    # Assert that the returned images and labels are the result of the otherwise's _apply method
    assert np.array_equal(result_images, images - 1)
    assert np.array_equal(result_labels, labels - 1)
