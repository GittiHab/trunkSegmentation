import os

import skimage
import torch
import torchio as tio

from training.segmentation_module import BinarySegmentation, MultiClassSegmentation, SegmentationModule


def get_training_module(name) -> SegmentationModule:
    return {'binary': BinarySegmentation, 'multiclass': MultiClassSegmentation}[name]


def is_single_image(idx, axis):
    return axis is not None and idx is not None

def load_single_slice(path, index, axis, as_numpy=False):
    assert axis in [0, 1, 2], 'Axis needs to be integer in interval [0, 2]'
    # Planes: XY : XZ : YZ -> Axes: z : y : x
    multi_img = skimage.io.MultiImage(path)
    if axis == 0:
        image_data = multi_img[0][index, :, :]
    elif axis == 1:
        image_data = multi_img[0][:, index, :]
    elif axis == 2:
        image_data = multi_img[0][:, :, index]
    if as_numpy:
        return image_data
    image_tensor = torch.from_numpy(image_data)
    return tio.ScalarImage(tensor=image_tensor.unsqueeze(0).unsqueeze(-1))


def load_image(path):
    multi_img = skimage.io.MultiImage(path)[0]
    image_tensor = torch.from_numpy(multi_img)
    return tio.ScalarImage(tensor=image_tensor.unsqueeze(0))


def get_patch_dimensions(plane, patch_size, stride):
    plane_mapping = {
        0: ((patch_size, patch_size, 1), (stride, stride, 0)),  # xy plane
        1: ((patch_size, 1, patch_size), (stride, 0, stride)),  # xz plane
        2: ((1, patch_size, patch_size), (0, stride, stride)),  # yz plane
    }
    if plane in plane_mapping:
        return plane_mapping[plane]
    else:
        raise ValueError("Invalid plane. Please choose 0, 1, or 2.")


def output_name(input_name):
    if input_name == '':
        return 'out.tiff'
    input_name = str(input_name)

    # Split the file path into directory, file name, and extension components
    directory, file_name_with_ext = os.path.split(input_name)
    file_name, file_ext = os.path.splitext(file_name_with_ext)

    # Concatenate the modified file name with the directory and extension components
    new_file_name = file_name + '.out' + file_ext
    return os.path.join(directory, new_file_name)
