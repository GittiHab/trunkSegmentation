import numpy as np
import skimage
import cv2


def adjust_exposure(img_slice):
    # TODO: parameterize
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    img_slice = skimage.exposure.rescale_intensity(img_slice, in_range=(90, 150))
    img_slice = clahe.apply(img_slice)
    # p1, p2 = np.percentile(img_slice, (2, 98))
    # img_slice = skimage.exposure.rescale_intensity(img_slice, in_range=(p1,p2))
    return img_slice


def crop_black(image, padding=10, threshold_x=None, threshold_y=None):
    def determine_bounds(means, threshold, padding):
        threshold = np.round(np.min(means)) if threshold is None else threshold
        selected_indices = np.asarray(means > threshold).nonzero()[0]
        lower = selected_indices[0] - padding
        upper = selected_indices[-1] + padding
        return lower, upper

    means_y, means_x = np.mean(image, axis=1), np.mean(image, axis=0)
    min_x, max_x = determine_bounds(means_x, threshold_x, padding)
    min_y, max_y = determine_bounds(means_y, threshold_y, padding)

    return image[min_y:max_y, min_x:max_x]
