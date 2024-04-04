from typing import Optional

import numpy as np
import torch
import torchvision.transforms.v2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.preprocessing import Preprocessing
from utils.dataset_utils import get_slice, extract_patches, as_one_hot


class BaseDataset(Dataset):
    def __init__(self,
                 image_data: np.ndarray,
                 label_data: np.ndarray,
                 mask_data: Optional[np.ndarray] = None,
                 axis: Optional[int] = None,
                 transform: Optional[torchvision.transforms.v2.Transform] = None,
                 preprocessing: Optional[Preprocessing] = None,
                 one_hot_labels: Optional[bool] = False,
                 as_image: bool = False,
                 preprocess_fn: Optional = None,
                 name=''):
        """
        Constructs a multi image dataset object for medical image segmentation.

        :param image_data: A 3D numpy array of images
        :param label_data: A 3D numpy array of labels
        :param mask_data: A 3D numpy array of masks (optional)
        :param mask_path: The path to a mask file (optional)
        :param axis: The axis to use for tiling (optional)
        :param transform: The transformation to apply to the images (optional)
        :param preprocessing: Preprocessing steps applied to the data when it is loaded.
                              Only applied once not for every retrieved item. (optional)
        :param one_hot_labels: Return labels one-hot-encoded instead of as integers (optional)
        :param as_image: Return PIL Image instead of tensor (optional)
        :param preprocess_fn: Applied to retrieved item right in the very end.
                              Might be applicable when using a pretrained model that requires own preprocessing.
                              (optional)
        :param name: Give the dataset a name to, for example, distinguish it during debugging.
        """
        assert axis is not None or mask_data is not None, \
            'Either mask or axis must be given.'

        self.images = image_data
        self.labels = label_data
        self.masks = mask_data
        self.name = name

        self.axis = axis
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.preprocessing = preprocessing
        self.as_image = as_image
        self.one_hot_labels = one_hot_labels

        self.processed_images, self.processed_labels = self.preprocessing(self.images, self.labels,
                                                                          self.masks, self.axis)

    def __len__(self):
        if type(self.processed_images) is not np.ndarray:  # this happens when masks were extracted
            return len(self.processed_images)
        return self.processed_images.shape[self.axis]

    def __getitem__(self, idx):
        if type(self.processed_images) is not np.ndarray:
            image = self.processed_images[idx]
            label = self.processed_labels[idx]
        else:
            image = get_slice(idx, self.axis, self.processed_images)
            label = get_slice(idx, self.axis, self.processed_labels)
        return self._get_item(image, label)

    def _get_item(self, image, label):
        # Apply data augmentation if specified
        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        if self.as_image:
            image = Image.fromarray(image)
            label = label
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
            label = torch.from_numpy(label).long()
            if self.one_hot_labels:
                label = as_one_hot(label)

        if self.preprocess_fn:
            image_size = image.shape
            image = self.preprocess_fn(image.view(*image_size[1:], 1)).view(-1, *image_size[1:])

        return image, label


class GridTiledDataset(BaseDataset):
    def __init__(self, *args, patch_size: int = 128, stride: int = 64, drop_nonmasked: bool = False,
                 pad_mode: str = 'constant', **kwargs):
        super().__init__(*args, **kwargs)

        if type(self.processed_images) is list:
            tiles = []
            tiled_labels = []
            for image, label in tqdm(zip(self.processed_images, self.processed_labels), desc="Extracting patches"):
                tiles.extend(extract_patches(image, patch_size, stride,
                                             drop_nonmasked=drop_nonmasked, pad_mode=pad_mode))
                tiled_labels.extend(extract_patches(label, patch_size, stride,
                                                    drop_nonmasked=drop_nonmasked, pad_mode=pad_mode))
            self.processed_images = np.stack(tiles).astype(np.single)
            self.processed_labels = np.stack(tiled_labels)
        else:
            self.processed_images = extract_patches(self.processed_images, patch_size, stride, self.masks,
                                                    drop_nonmasked=drop_nonmasked, pad_mode=pad_mode).astype(np.single)
            self.processed_labels = extract_patches(self.processed_labels, patch_size, stride, self.masks,
                                                    drop_nonmasked=drop_nonmasked, pad_mode=pad_mode)

    def __len__(self):
        return self.processed_images.shape[0]

    def __getitem__(self, idx):
        image = self.processed_images[idx]
        label = self.processed_labels[idx]
        return self._get_item(image, label)


class RandomPatchDataset(BaseDataset):
    def __init__(self):
        raise NotImplementedError()
