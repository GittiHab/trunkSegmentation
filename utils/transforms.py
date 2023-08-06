import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class Clip(ImageOnlyTransform):
    """Clip and normalize the pixel values of the input image.

    This transform clips the pixel values of the image to the range [min_val, max_val],
    and then normalizes the image so that its minimum value is 0 and its maximum value is 1.

    Args:
        min_val (float): The minimum pixel value to be allowed in the image. All pixel values less
            than this will be set to this value.
        max_val (float): The maximum pixel value to be allowed in the image. All pixel values greater
            than this will be set to this value.
        p (float): The probability of applying this transform. Default is 1.0.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, min_val, max_val, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, img: np.ndarray, **params):
        # Clip the pixel values
        img = np.clip(img, self.min_val, self.max_val)

        # Normalize the pixel values
        img = (img - self.min_val) / (self.max_val - self.min_val)

        return img

    def get_transform_init_args_names(self):
        return ("min_val", "max_val",)
