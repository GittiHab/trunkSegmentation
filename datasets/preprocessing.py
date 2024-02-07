import warnings
from typing import Union, Tuple, Iterable, Optional, Collection

import numpy as np
from tqdm import tqdm

from utils.dataset_utils import cache_exists, load_cached, save_cache, extract_masked_regions, drop_equalize, get_slice, \
    check_no_rectangle
from datasets.masked_slices import MaskedSlices


class Preprocessing:
    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self._apply(*args, **kwargs)


class Identity(Preprocessing):
    _singleton = None

    @classmethod
    def singleton(cls):
        if cls._singleton is None:
            cls._singleton = Identity()
        return cls._singleton

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        return images, labels


class Compose(Preprocessing):
    def __init__(self, preprocessing_steps: Iterable[Preprocessing]):
        self.steps = preprocessing_steps

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        prev_image, prev_label = images, labels
        for step in self.steps:
            prev_image, prev_label = step(prev_image, prev_label, masks, axis)
        return prev_image, prev_label


class Conditional(Preprocessing):
    def __init__(self, step: Preprocessing,
                 condition: bool,
                 otherwise: Optional[Preprocessing] = None):
        self.step = step
        self.otherwise = otherwise if otherwise else Identity.singleton()
        self.condition = condition

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        if self.condition:
            return self.step(images, labels, masks, axis)
        return self.otherwise(images, labels, masks, axis)


class Caching(Preprocessing):
    def __init__(self, preprocessing: Preprocessing = None, path: str = './cache/'):
        self.path = path
        self.preprocessing = preprocessing

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        if cache_exists(self.path):
            return load_cached(self.path)
        else:
            processed_images, processed_labels = images, labels
            if self.preprocessing is not None:
                processed_images, processed_labels = self.preprocessing(images, labels, masks, axis)
            save_cache(self.path, processed_images, processed_labels)
            return processed_images, processed_labels

    def needs_source_data(self):
        return not cache_exists(self.path)


class ConditionalCache(Conditional):
    def __init__(self, preprocessing: Preprocessing, condition: bool, path: str = './cache/'):
        super().__init__(Caching(preprocessing, path), condition, preprocessing)

    def needs_source_data(self):
        return self.step.needs_source_data()


class Masking(Preprocessing):
    def __init__(self, min_region_size: int):
        self.min_region_size = min_region_size

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        if masks is None:
            warnings.warn("No mask given, skipping Masking.")
            return images, labels
        return extract_masked_regions(masks, images, labels, self.min_region_size, axis=axis)


class Padding(Preprocessing):
    def __init__(self, width, mode):
        self.width = width
        self.mode = mode

    def _get_width(self, type_):
        if isinstance(self.width, Collection):
            return self.width
        if type_ == np.ndarray:
            return (0, 0), (self.width, self.width), (self.width, self.width)
        return (self.width, self.width), (self.width, self.width)

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        assert type(images) == type(labels), 'Padding preprocessing assumes identical datatypes for images and labels.'
        assert type(images) in [np.ndarray, MaskedSlices, list]

        if type(images) is MaskedSlices:
            raise NotImplementedError('Not implemented yet.')
        width = self._get_width(type(images))
        if type(images) is np.ndarray:
            return np.pad(images, width, mode=self.mode), np.pad(labels, width, mode=self.mode)
        images_pad, labels_pad = [], []
        for image, label in zip(images, labels):
            images_pad.append(np.pad(image, width, mode=self.mode))
            labels_pad.append(np.pad(label, width, mode=self.mode))
        return images_pad, labels_pad


class AbstractFilterContained(Preprocessing):
    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        raise NotImplementedError('Please use a subclass of AbstractFilterContained.')

    def _filter(self, images, labels, unfiltered, axis):
        if axis is None:
            selected_image_slices = []
            selected_label_slices = []
            for axis in range(3):
                selected_slice_indices = self._find_annotated_slices(unfiltered, axis)
                selected_image_slices.append(np.moveaxis(get_slice(selected_slice_indices, axis, images), axis, 0))
                selected_label_slices.append(np.moveaxis(get_slice(selected_slice_indices, axis, labels), axis, 0))
            return MaskedSlices(*selected_image_slices), MaskedSlices(*selected_label_slices)
        selected_slice_indices = self._find_annotated_slices(unfiltered, axis)
        return get_slice(selected_slice_indices, axis, images), get_slice(selected_slice_indices, axis, labels)

    def _find_annotated_slices(self, unfiltered, axis):
        axis_len = unfiltered.shape[axis]
        indices = []
        for i in tqdm(range(axis_len), desc="Finding slices with masks in axis " + str(axis)):
            mask_plane = get_slice(i, axis, unfiltered)
            if self._skip_condition(mask_plane):
                continue
            indices.append(i)
        return np.array(indices, dtype=np.uint)

    def _skip_condition(self, mask_plane):
        raise NotImplementedError('Skip condition needs to be implemented by subclass.')


class FilterMasked(AbstractFilterContained, Masking):
    """
    Only returns slices that contain a mask.
    """

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        if masks is None:
            warnings.warn("No mask given, skipping FilterMasked.")
            return images, labels
        return self._filter(images, labels, masks, axis)

    def _skip_condition(self, mask_plane):
        return check_no_rectangle(mask_plane, self.min_region_size)


class FilterLabeled(AbstractFilterContained):
    """
    Only returns slices that contain a label.
    """

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        if masks is not None:
            warnings.warn("Masks should be applied before this step. Ignore if done so.")
        return self._filter(images, labels, labels, axis)

    @staticmethod
    def contains_only_single_stripes(marked_rows):
        return np.all(np.convolve(marked_rows, np.ones(3, dtype=int), 'valid') == 1)

    def _skip_condition(self, mask_plane):
        if mask_plane.sum() < 2:
            return True
        x_verify = mask_plane.sum(0) > 0
        if self.contains_only_single_stripes(x_verify):
            return True
        y_verify = mask_plane.sum(1) > 0
        if self.contains_only_single_stripes(y_verify):
            return True
        return False


class CheckDimensions(Preprocessing):
    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        # If the images and labels are not the same size, raise an error
        if images.shape != labels.shape or masks is not None and images.shape != masks.shape:
            raise ValueError("Images, labels and masks, if given, must have the same dimensions.")
        return images, labels


class DropUntilEqual(Preprocessing):
    def __init__(self, extra_classes: float = 0.1):
        self.extra_classes = extra_classes

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        return drop_equalize(images, labels, self.extra_classes)

class BinaryNorm(Preprocessing):
    def __init__(self, skip_unsupported: bool = False):
        self.skip_unsupported = skip_unsupported

    def _apply(self, images, labels, masks, axis) -> Tuple[Union[np.ndarray, Iterable], Union[np.ndarray, Iterable]]:
        if isinstance(labels, np.ndarray):
            unique_labels = np.unique(labels)
            if len(unique_labels) > 2:
                raise AssertionError('More than two unique label values not supported by *binary* norm.')
            max_label = unique_labels.max()
            if max_label > 1:
                labels = (labels == max_label).astype(labels.dtype)
            return images, labels
        if isinstance(labels, list):
            unique_labels = np.unique(np.concatenate([np.unique(l) for l in labels]))
            if len(unique_labels) > 2:
                raise AssertionError('More than two unique label values not supported by *binary* norm.')
            max_label = unique_labels.max()
            if max_label > 1:
                labels = [(l == max_label).astype(l.dtype) for l in labels]
            return images, labels
        if not self.skip_unsupported:
            raise NotImplementedError('BinaryNorm only supports list and numpy array type labels.')
        warnings.warn('SKIPPING BinaryNorm! BinaryNorm only supports list and numpy array type labels.')

class CLAHE(Identity):
    pass  # TODO: implement
