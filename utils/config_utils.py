import albumentations as A
import cv2

from utils.transforms import Clip


def extract_encoder_params(config):
    if 'encoder' not in config:
        return 'efficientnet-b1', 'imagenet'
    if 'weights' not in config['encoder']:
        return config['encoder']['name'], None
    return config['encoder']['name'], config['encoder']['weights']


def read_transforms(transform_config):
    if len(transform_config) < 1:
        return None
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
    if len(transformations) > 1:
        return A.Compose(transformations)
    return transformations[0]
