from numbers import Number

import numpy as np
import pytest
from utils.dataset_utils import extract_patches, expand_seed, extract_rectangles

IMAGE_DIMS = 100


@pytest.fixture
def test_image():
    return np.random.randint(0, 255, (IMAGE_DIMS, IMAGE_DIMS), dtype=np.uint8)


@pytest.fixture
def test_mask():
    mask = np.zeros((IMAGE_DIMS, IMAGE_DIMS), dtype=np.bool_)
    mask[30:70, 30:70] = 1
    return mask


def test_extract_patches_no_mask_drop(test_image):
    # Test the function when no mask is provided
    patches = extract_patches(test_image.copy(), patch_size=32, stride=16, drop_nonmasked=True)
    # Check that the number of patches is correct
    assert patches.shape[0] == 25
    # Check that the shape of each patch is correct
    assert patches.shape[1:] == (32, 32)
    assert np.array_equal(patches[0], test_image[:32, :32])


def test_extract_patches_no_mask(test_image):
    patches = extract_patches(test_image.copy(), patch_size=32, stride=16, drop_nonmasked=False)
    assert patches.shape[0] == 36
    assert np.array_equal(patches[0], test_image[:32, :32])


def test_extract_patches_no_mask_reflect(test_image):
    patches = extract_patches(test_image.copy(), patch_size=32, stride=16, drop_nonmasked=False, pad_mode='reflect')
    assert patches.shape[0] == 36
    assert np.array_equal(patches[-1], np.pad(test_image[-32 + 12:, -32 + 12:], ((0, 12), (0, 12)), mode='reflect'))


def test_extract_patches_no_mask_no_padding(test_image):
    # Test the function when no mask is provided
    patches = extract_patches(test_image, patch_size=10, stride=5)
    # Check that the number of patches is correct
    assert patches.shape[0] == 361
    # Check that the shape of each patch is correct
    assert patches.shape[1:] == (10, 10)


def test_extract_patches_with_mask_drop(test_image, test_mask):
    # Test the function when a mask is provided
    patches = extract_patches(test_image, patch_size=32, stride=16, mask=test_mask, drop_nonmasked=True)
    # Check that the number of patches is correct
    assert patches.shape[0] == 1
    # Check that the shape of each patch is correct
    assert patches.shape[1:] == (32, 32)


def test_extract_patches_with_mask(test_image, test_mask):
    patches = extract_patches(test_image, patch_size=32, stride=16, mask=test_mask, drop_nonmasked=False)
    assert patches.shape[0] == 4


def test_extract_patches_with_mask_reflect(test_image, test_mask):
    patches = extract_patches(test_image, patch_size=32, stride=16, mask=test_mask, drop_nonmasked=False,
                              pad_mode='reflect')
    assert np.array_equal(patches[-1],
                          np.pad(test_image[70 - 32 + 8:70, 70 - 32 + 8:70], ((0, 8), (0, 8)), mode='reflect'))


def test_extract_patches_with_mask_between_windows(test_image):
    # Create a test image and mask
    mask = np.zeros_like(test_image)
    mask[40:72, 40:72] = 1

    # Extract patches from the image using the mask
    patches = extract_patches(test_image, patch_size=32, stride=16, mask=mask)

    # Check that the number of patches is correct
    assert patches.shape[0] == 1

    # Check that the shape of each patch is correct
    assert patches.shape[1:] == (32, 32)


def test_extract_patches_invalid_patch_size(test_image):
    # Test the function with an invalid patch size
    with pytest.raises(AssertionError):
        extract_patches(test_image, patch_size=101, stride=16)


def test_extract_patches_invalid_stride(test_image):
    # Test the function with an invalid stride
    with pytest.raises(AssertionError):
        extract_patches(test_image, patch_size=32, stride=101)


def test_expand_seed_all_different():
    seed = "my_seed"
    seed_1, seed_2, seed_3 = expand_seed(seed)
    assert seed_1 != seed_2
    assert seed_2 != seed_3
    assert seed_3 != seed_1


def test_expand_seed_not_initial_seed():
    seed = "my_seed"
    seed_1, seed_2, seed_3 = expand_seed(seed)

    assert seed != seed_1
    assert seed != seed_2
    assert seed != seed_3


def test_expand_seed_different_across_seeds():
    seed = "my_seed"
    seed_b = "another_seed"
    seed_1, seed_2, seed_3 = expand_seed(seed)
    seed_b_1, seed_b_2, seed_b_3 = expand_seed(seed_b)

    assert seed_1 != seed_b_1
    assert seed_2 != seed_b_2
    assert seed_3 != seed_b_3


def test_expand_seed_works_with_numbers():
    seed_1, seed_2, seed_3 = expand_seed(42)
    assert isinstance(seed_1, Number)
    assert isinstance(seed_2, Number)
    assert isinstance(seed_3, Number)


@pytest.fixture
def input_array():
    return np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
    ])


def test_extract_rectangles(input_array):
    expected_output = [
        np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ]),
        np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]
        ]),
        np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
        ])
    ]
    assert np.all(np.stack(extract_rectangles(input_array, shapes=False)) == np.stack(expected_output)).astype(np.bool_)


def test_extract_rectangles_min_shape(input_array):
    expected_output = [
        np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ]),
        np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]
        ])
    ]
    assert np.all(
        np.stack(extract_rectangles(input_array, min_width=2, min_height=2, shapes=False)) == np.stack(expected_output))
