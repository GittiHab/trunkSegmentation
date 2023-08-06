import numpy as np
import pytest

from datasets.preprocessing import FilterMasked


@pytest.fixture
def mask():
    mask_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    mask_arr[0, 10:15, 4:9] = 1
    mask_arr[5, 14:19, 10:15] = 1
    mask_arr[3, 10:12, 4:9] = 1
    mask_arr[10:15, 1, 4:9] = 1
    mask_arr[10:15, 4:9, 3] = 1
    mask_arr[10:15, 4:5, 4] = 1
    return mask_arr


@pytest.fixture
def images():
    return np.random.random((20, 20, 20))


@pytest.fixture
def filter_masked():
    return FilterMasked(5)


def test_extract_all_slices(mask, filter_masked: FilterMasked):
    assert np.array_equal(filter_masked._find_annotated_slices(mask, 0), [0, 5])
    assert np.array_equal(filter_masked._find_annotated_slices(mask, 1), [1])
    assert np.array_equal(filter_masked._find_annotated_slices(mask, 2), [3])


def test_return_slices_1_axis(mask, images, filter_masked: FilterMasked):
    images_input = images.copy()
    filtered, _ = filter_masked(images_input, images_input, mask, 0)

    assert np.array_equal(filtered, np.stack((images[0], images[5])))


def test_return_slices_all_axes(mask, images, filter_masked: FilterMasked):
    images_input = images.copy()
    filtered, _ = filter_masked(images_input, images_input, mask, None)

    assert len(filtered) == 4
    assert np.array_equal(filtered[2], images[:, 1, :])

