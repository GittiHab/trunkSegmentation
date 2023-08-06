import hashlib
import logging
import warnings
from pathlib import Path
from types import NoneType
from typing import Union

import numpy as np
import skimage
import torch
from tqdm import tqdm


def get_slice(slice_idx, axis, image_data):
    if axis == 0:
        return image_data[slice_idx, :, :]
    elif axis == 1:
        return image_data[:, slice_idx, :]
    elif axis == 2:
        return image_data[:, :, slice_idx]
    else:
        raise ValueError("Invalid axis value, must be 0, 1, or 2.")


def compute_masked_slices(masks):
    slice_indices = []
    for axis in range(3):
        axis_len = masks.shape[axis]
        for i in range(axis_len):
            mask_plane = get_slice(i, axis, masks)

            if mask_plane.shape[0] > 128 and mask_plane.shape[1] > 128 and np.count_nonzero(mask_plane) > 128 * 128:
                slice_indices.append((axis, i))
    return slice_indices


def extract_patches(image, patch_size, stride, mask=None, axis=None, drop_nonmasked=False, pad_mode='constant',
                    fill_value=0):
    """
    Extracts all possible patches from the masked areas of an image.

    CAUTION: This function may modify the input arrays! It is possible to implement this function without this
    happening, however, it has not been a requirement until now.

    Parameters
    ----------
    image : numpy.ndarray, shape (height, width[, depth])
        The input image in grayscale without any channels.
    patch_size : int
        The size of the patches to extract from the masked areas.
    stride : int
        The step size between consecutive patches.
    mask : numpy.ndarray, shape (height, width)
        A binary mask indicating the masked areas of the image.
        Non-zero values indicate masked areas, while zero values
        indicate non-masked areas.
        If the image is 3-dimensional, masks are ignored!
    axis : int
        Axis to be fixed, patches will be extracted along the other axes.
        This is mandatory if the image is 3-dimensional, otherwise it is ignored.

    Returns
    -------
    numpy.ndarray, shape (num_patches, patch_size, patch_size)
        An array containing all patches extracted from the masked areas
        of the input image. The number of patches will depend on the size
        of the input image and mask, the size of the patches, and the
        value of the stride parameter.

    Function and docstring authored by ChatGPT (GPT 3.5). Modified.
    """
    assert patch_size <= image.shape[0] and patch_size <= image.shape[1], 'Patch size cannot be larger than image size.'
    assert stride <= image.shape[0] and stride <= image.shape[1], 'Stride cannot be larger than image size.'
    assert mask is None or image.shape == mask.shape, 'Mask needs to have same shape as image.'
    assert len(image.shape) == 2 or len(image.shape) == 3 and axis is not None, 'For 3D images axis needs to be set.'
    if len(image.shape) == 3 and mask is not None:
        warnings.warn('Mask is ignored for 3D images.')
    if len(image.shape) == 2:
        axis = None

    if mask is None:  # TODO: add options to use reflect instead of pad 0
        mask = np.ones_like(image, dtype=np.bool_)

    # Crop first so that we really get some tiles if the mask has the right dimensions.
    ## Find the first and last "true" value in each dimension of the mask
    nonzero_rows, nonzero_cols = np.where(mask)
    first_row, last_row = nonzero_rows.min(), nonzero_rows.max()
    first_col, last_col = nonzero_cols.min(), nonzero_cols.max()

    ## Crop the image and mask to the size of the masked area
    image = image[first_row:last_row + 1, first_col:last_col + 1]
    mask = mask[first_row:last_row + 1, first_col:last_col + 1]

    ## If non-masked regions are kept, they need to be replaced with the fill value
    image_shape = image.shape
    if not drop_nonmasked:
        if pad_mode == 'constant':
            image[np.logical_not(mask)] = fill_value
        else:
            logging.warning(
                'When using another pad mode than constant, the mask is expected to be a single rectangle or empty.')
            rows, cols = np.where(mask == 1)
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            image = image[min_row:max_row + 1, min_col:max_col + 1]

    # Pad the image and mask to ensure that all patches can be extracted, pads with 0
    pad_width = [(0, stride - image_shape[0] % stride if image_shape[0] % stride else 0),
                 (0, stride - image_shape[1] % stride if image_shape[1] % stride else 0)]
    if axis is not None:
        pad_width.insert(axis, (0, 0))

    pad_kwargs = {'constant_values': fill_value} if pad_mode == 'constant' else {}
    image_padded = np.pad(image, pad_width=pad_width, mode=pad_mode, **pad_kwargs)
    mask_padded = np.pad(mask, pad_width=pad_width, mode=pad_mode, **pad_kwargs)

    # Create a view of the padded image that extracts patches of size (patch_size, patch_size) with stride `stride`
    patch_dimensions = [patch_size, patch_size]
    if axis is not None:
        patch_dimensions.insert(axis, image_shape[axis])
    patches = skimage.util.view_as_windows(image_padded, patch_dimensions, step=stride)
    # Create a view of the padded mask that extracts patches of size (patch_size, patch_size) with stride `stride`
    mask_patches = skimage.util.view_as_windows(mask_padded, patch_dimensions, step=stride)
    # Use the mask to filter out patches that are not fully within the masked areas
    selection_op = np.all if drop_nonmasked else np.any
    valid_mask = selection_op(mask_patches == 1, axis=(2, 3) if len(image_shape) == 2 else (3, 4, 5))
    patches = patches[valid_mask]
    return np.reshape(patches, (-1, patch_size, patch_size))


def expand_seed(seed):
    # Expand the seed value into multiple seeds using SHA-256
    seed_bytes = str(seed).encode('utf-8')
    seed_hash = hashlib.sha256(seed_bytes).digest()
    seed_1 = int.from_bytes(seed_hash[:8], byteorder='big')
    seed_2 = int.from_bytes(seed_hash[8:16], byteorder='big')
    seed_3 = int.from_bytes(seed_hash[16:24], byteorder='big')
    return seed_1, seed_2, seed_3


def check_no_rectangle(arr, min_width: int, min_height: Union[NoneType, int] = None):
    if min_height is None:
        min_height = min_width

    if not np.any(arr):
        return True
    if np.sum(arr, axis=0).max() < min_height or np.sum(arr, axis=1).max() < min_width or np.sum(
            arr) < min_height * min_width:
        return True
    return False


def extract_rectangles(arr, min_width=1, min_height=1, shapes=True):
    # Skip trivial cases
    if check_no_rectangle(arr, min_width, min_height):
        return []
    if np.all(arr):
        return [(arr.astype(np.bool_), arr.shape) if shapes else arr.astype(np.bool_)]
    rectangles = []
    arr_copy = arr.copy()

    # Iterate over each cell in the input array
    while np.any(arr_copy > 0):
        indices = np.nonzero(arr_copy)
        i, j = (indices[0][0], indices[1][0])
        rect, shape = find_max_rectangle(arr_copy, i, j)
        arr_copy[rect] = 0

        # Check constraints
        if shape[0] >= min_height and shape[1] >= min_width:
            # Add rectangle to list
            rectangles.append((rect, shape) if shapes else rect)

    return rectangles


def find_max_rectangle(arr, i, j):
    # Initialize queue for BFS
    max_area = (arr.shape[0] - i) * (arr.shape[1] - j)
    start_coordinates = np.array([i, j])
    end_coordinates = np.array([i, j])
    expand_y = True
    expand_x = True

    for increment in range(max_area):
        if increment % 2 and expand_x:
            if end_coordinates[1] + 1 < arr.shape[1] and np.all(arr[i:end_coordinates[0], end_coordinates[1] + 1]):
                end_coordinates[1] += 1
            else:
                expand_x = False
        else:
            if end_coordinates[0] + 1 < arr.shape[0] and np.all(arr[end_coordinates[0] + 1, j:end_coordinates[1]]):
                end_coordinates[0] += 1
            else:
                expand_y = False

        if not expand_x and not expand_y:
            return rectangle_in(arr.shape, start_coordinates, end_coordinates)
    return rectangle_in(arr.shape, start_coordinates, end_coordinates)


def rectangle_in(shape, start_coordinates, end_coordinates):
    arr = np.zeros(shape, dtype=np.bool_)
    arr[start_coordinates[0]:end_coordinates[0] + 1, start_coordinates[1]:end_coordinates[1] + 1] = 1

    height = end_coordinates[0] - start_coordinates[0] + 1
    width = end_coordinates[1] - start_coordinates[1] + 1

    return arr, (height, width)


def extract_masked_regions(masks, images, labels, patch_size, axis=None):
    masked_slices = []
    masked_labels = []
    if axis is None:
        for axis in range(3):
            _extract_masked_regions_axis(masks, images, labels, patch_size, axis, masked_slices, masked_labels)
        return masked_slices, masked_labels
    _extract_masked_regions_axis(masks, images, labels, patch_size, axis, masked_slices, masked_labels)
    return masked_slices, masked_labels


def _extract_masked_regions_axis(masks, images, labels, patch_size, axis, masked_slices, masked_labels):
    axis_len = masks.shape[axis]
    for i in tqdm(range(axis_len), desc="Finding masks in axis " + str(axis)):
        mask_plane = get_slice(i, axis, masks)

        rectangles = extract_rectangles(mask_plane, min_width=patch_size, min_height=patch_size)
        for r, shape in rectangles:
            # TODO: if need be, we could also store the axis and index here,
            #  together with the mask itself, if we wanted to stitch it back together.
            masked_slices.append(get_slice(i, axis, images)[r].reshape(shape))
            masked_labels.append(get_slice(i, axis, labels)[r].reshape(shape))


def as_one_hot(labels):
    return torch.nn.functional.one_hot(labels, num_classes=2).transpose(0, -1).double()


CACHE_LABEL_FILENAME = 'labels'
CACHE_IMAGE_FILENAME = 'images'


def cache_exists(path):
    cache_path = Path(path)
    return cache_path.joinpath(CACHE_IMAGE_FILENAME + '.npz').is_file() and cache_path.joinpath(
        CACHE_LABEL_FILENAME + '.npz').is_file() or \
        cache_path.joinpath(CACHE_IMAGE_FILENAME + '.npy').is_file() and cache_path.joinpath(
            CACHE_LABEL_FILENAME + '.npy').is_file()


def load_npz_file(path):
    data = []
    with np.load(path) as npz_file:
        for name in npz_file.files:
            data.append(npz_file[name])
    return data


def load_cached(path):
    cache_file = Path(path)
    image_path = cache_file.joinpath(CACHE_IMAGE_FILENAME)
    label_path = cache_file.joinpath(CACHE_LABEL_FILENAME)
    if image_path.with_suffix('.npy').is_file():
        images = np.load(str(image_path.with_suffix('.npy')))
        labels = np.load(str(label_path.with_suffix('.npy')))
        return images, labels
    else:
        images = load_npz_file(str(image_path.with_suffix('.npz')))
        labels = load_npz_file(str(label_path.with_suffix('.npz')))
        return images, labels


def save_cache(path, images, labels):
    cache_file = Path(path)
    cache_file.mkdir(parents=True, exist_ok=True)
    image_path = cache_file.joinpath(CACHE_IMAGE_FILENAME)
    label_path = cache_file.joinpath(CACHE_LABEL_FILENAME)

    def save_data(path, data):
        if type(data) is np.ndarray:
            np.save(path, data)
        else:
            np.savez_compressed(path, *data)

    save_data(image_path, images)
    save_data(label_path, labels)


def _find_all_slices_without_label(slices, label):
    indices = []
    for i, s in enumerate(slices):
        if label not in s:
            indices.append(i)
    return np.array(indices)


def drop_equalize(data, labels, keep_extra=0.1):
    if type(data) is list:
        warnings.warn("Drop equalize does not support data in lists, only numpy arrays.")
        return data, labels
    if keep_extra == 1:
        return data, labels

    classes = np.unique(labels)
    if len(classes) < 2:
        return data, labels
    fractions = []
    for c in classes:
        count = 0
        for i in range(len(labels)):
            count += np.any(labels[i] == c)
        fractions.append(count / len(labels))
    fractions = np.array(fractions)
    weakest_label_idx = fractions.argmin()
    select_num = int(len(labels) * fractions[weakest_label_idx] * keep_extra)

    selected_indices = _find_all_slices_without_label(labels, classes[weakest_label_idx])
    if len(selected_indices) < 1:
        return data, labels
    if keep_extra == 0:
        final_indices = np.setdiff1d(np.arange(0, len(labels)), selected_indices)
    else:
        keep_indices = np.random.choice(selected_indices, min(select_num, len(selected_indices)), replace=False)
        final_indices = np.union1d(np.setdiff1d(np.arange(0, len(labels)), selected_indices), keep_indices)
    return data[final_indices], labels[final_indices]


def crop(image, mask=None, padding=10):
    meansY, meansX = np.mean(image, axis=1), np.mean(image, axis=0)
    selected_indices = np.asarray(meansX > np.round(np.min(meansX))).nonzero()[0]
    minX = selected_indices[0] - padding
    maxX = selected_indices[-1] + padding
    selected_indices = np.asarray(meansY > np.round(np.min(meansY))).nonzero()[0]
    minY = selected_indices[0] - padding
    maxY = selected_indices[-1] + padding
    if mask is not None:
        return image[minY:maxY, minX:maxX], mask[minY:maxY, minX:maxX]
    return image[minY:maxY, minX:maxX]
