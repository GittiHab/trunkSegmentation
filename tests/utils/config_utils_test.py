import cv2
import pytest
import numpy as np
import albumentations as A

from utils.config_utils import read_transforms
from utils.transforms import Clip


class TestReadTransforms:
    @pytest.fixture
    def transform_config(self):
        return [
            {"Clip": {"min_val": 90, "max_val": 190}},
            {"Blur": {"blur_limit": 5}},
            {"HorizontalFlip": None},
            {"Rotate": {"limit": 45, "border_mode": "BORDER_REFLECT"}}
        ]

    def test_read_transforms_single_transform(self, transform_config):
        transforms = read_transforms(transform_config[1:2])
        assert isinstance(transforms, A.Blur)

    def test_read_transforms_clip(self, transform_config):
        transforms = read_transforms(transform_config[:1])
        assert isinstance(transforms, Clip)

    def test_read_transforms_compose_transforms(self, transform_config):
        transforms = read_transforms(transform_config)
        assert isinstance(transforms, A.Compose)

    def test_read_transforms_no_transforms(self):
        transforms = read_transforms([])
        assert transforms is None

    def test_read_transforms_rotate_params(self, transform_config):
        transform = read_transforms(transform_config)[-1]
        assert isinstance(transform, A.Rotate)
        assert transform.border_mode == cv2.BORDER_REFLECT
