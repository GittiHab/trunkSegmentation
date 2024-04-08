import albumentations as A
import cv2

from utils.transforms import Clip


def extract_encoder_params(config):
    if 'encoder' not in config:
        return 'efficientnet-b1', 'imagenet'
    if 'weights' not in config['encoder']:
        return config['encoder']['name'], None
    return config['encoder']['name'], config['encoder']['weights']


def read_transforms(transform_config) -> list:
    if len(transform_config) < 1:
        return []
    transformations = []
    for transform in transform_config:
        name = list(transform.keys())[0]
        params = transform[name]

        if name == 'Clip':
            method = Clip
        else:
            method = getattr(A, name)
        if name == 'Rotate' and 'border_mode' in params:
            params['border_mode'] = getattr(cv2, params['border_mode'])
        transformations.append(method(**params) if params is not None else method())
    return transformations


def extract_all_transforms(transforms_config, general_transforms=[], train_transforms=[], val_transforms=[]):
    general_transforms, train_transforms, val_transforms = general_transforms.copy(), train_transforms.copy(), val_transforms.copy()
    if 'general' in transforms_config:
        general_transforms.extend(read_transforms(transforms_config['general']))
    if 'train' in transforms_config:
        train_transforms.extend(read_transforms(transforms_config['train']))
    if 'infer' in transforms_config:
        val_transforms.extend(read_transforms(transforms_config['infer']))
    return general_transforms, train_transforms, val_transforms
