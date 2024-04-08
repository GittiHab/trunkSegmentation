from copy import copy

import albumentations as A
import pytorch_lightning as pl
import skimage
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset

import datasets.preprocessing as P
from datasets.datasets import GridTiledDataset
from utils.config_utils import extract_encoder_params, read_transforms, extract_all_transforms
from utils.dataset_utils import expand_seed


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.source = SingleSource(config) if not isinstance(config['datasets'], list) else MultiSource(config)

    def prepare_data(self):
        self.source.prepare_data()

    def train_dataloader(self):
        # TODO: consider using weighted sampling:
        #  https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler or
        #  https://github.com/ufoym/imbalanced-dataset-sampler
        return DataLoader(self.source.get_dataset_train(), batch_size=self.config["batch_size"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.source.get_dataset_val(), batch_size=self.config["batch_size"], shuffle=False)

    def setup(self, stage=None):
        general_transforms, train_transforms, val_transforms = [], [], []
        if stage == 'fit':
            if 'transforms' in self.config:
                general_transforms, train_transforms, val_transforms = extract_all_transforms(
                    self.config['transforms'],
                    general_transforms,
                    train_transforms,
                    val_transforms)
        self.source.setup(stage, general_transforms, train_transforms, val_transforms)


class DataSource:
    def __init__(self, config):
        self.config = config

    def prepare_data(self):
        raise NotImplementedError()

    def setup(self, stage, general_transforms, train_transforms, val_transforms):
        raise NotImplementedError()

    def get_dataset_train(self):
        raise NotImplementedError()

    def get_dataset_val(self):
        raise NotImplementedError()


class SingleSource(DataSource):
    def __init__(self, config, dataset_config=None):
        super().__init__(config)
        self.dataset_config = dataset_config if dataset_config is not None else config['datasets']
        self.cache_path = self.dataset_config['train']['cache_path'] if 'cache_path' in self.dataset_config[
            'train'] else './cache/train'
        self.cache_path_val = self.dataset_config['val']['cache_path'] if 'cache_path' in self.dataset_config[
            'val'] else './cache/val'

    def prepare_data(self):
        if not P.Caching(path=self.cache_path).needs_source_data() and not P.Caching(
                path=self.cache_path_val).needs_source_data():
            self.image_data, self.label_data, self.mask_data, self.mask_data_val = [], [], [], []
            return
        multi_img = skimage.io.MultiImage(self.dataset_config['image_path'])
        self.image_data = multi_img[0]
        multi_img = skimage.io.MultiImage(self.dataset_config['label_path'])
        self.label_data = multi_img[0]
        self.mask_data = None
        if 'mask_path' in self.dataset_config['train']:
            multi_img = skimage.io.MultiImage(self.dataset_config['train']['mask_path'])
            self.mask_data = multi_img[0]
        self.mask_data_val = None
        if 'mask_path' in self.dataset_config['val']:
            multi_img = skimage.io.MultiImage(self.dataset_config['val']['mask_path'])
            self.mask_data_val = multi_img[0]

    def own_validation_set(self):
        return self.config['train_val_split_ration'] == 1

    def build_preprocessing_pipeline(self, cache_path):
        return P.Compose([
            P.ConditionalCache(P.Compose([P.CheckDimensions(),
                                          # P.CLAHE(),
                                          P.Masking(self.config['patch_size'])]),
                               condition=self.config['cache'],
                               path=cache_path
                               ),
            P.Conditional(P.BinaryNorm(), self.config.get('binary_norm', False)),
            P.Conditional(P.DropUntilEqual(self.config.get('keep_extra', 0.1)),
                          self.config['drop_until_equal']),
            P.Conditional(P.Padding(self.config.get('pad_masked', self.config['stride']),
                                    self.config.get('pad_mode', 'constant')),
                          'pad_masked' in self.config)
        ])

    def setup(self, stage, general_transforms, train_transforms, val_transforms):
        seed_fit, seed_test, seed_predict = expand_seed(self.config['seeds']['data'])

        if stage == 'fit':
            if 'transforms' in self.dataset_config:
                general_transforms, train_transforms, val_transforms = extract_all_transforms(
                    self.dataset_config['transforms'],
                    general_transforms,
                    train_transforms,
                    val_transforms)

            preprocess_fn = None
            encoder_param = extract_encoder_params(self.config)
            if encoder_param[1] is not None:
                preprocess_fn = get_preprocessing_fn(encoder_param[0],
                                                     pretrained=encoder_param[1])

            generator = torch.Generator().manual_seed(seed_fit)
            self.dataset_main = GridTiledDataset(self.image_data,
                                                 self.label_data,
                                                 mask_data=self.mask_data,
                                                 as_image=False,
                                                 axis=self.config.get('axis', None),
                                                 transform=A.Compose(general_transforms + train_transforms),
                                                 preprocessing=self.build_preprocessing_pipeline(self.cache_path),
                                                 preprocess_fn=preprocess_fn,
                                                 patch_size=self.config['patch_size'],
                                                 stride=self.config['stride'],
                                                 pad_mode=self.config.get('pad_mode', 'constant'),
                                                 name='Main')
            if not self.own_validation_set():
                split = random_split(self.dataset_main,
                                     (self.config['train_val_split_ration'], 1 - self.config['train_val_split_ration']),
                                     generator=generator)
                self.dataset_train = split[0]
                # Validation dataset should not have any transformations:
                self.dataset_val = copy(self.dataset_main)
                self.dataset_val.transform = A.Compose(general_transforms + val_transforms)
                self.dataset_val.name = 'Validation Split'
                self.dataset_val = Subset(self.dataset_val, indices=split[1].indices)
            else:
                self.dataset_train = self.dataset_main
                self.dataset_val = GridTiledDataset(self.image_data,
                                                    self.label_data,
                                                    mask_data=self.mask_data_val,
                                                    as_image=False,
                                                    axis=self.config.get('axis', None),
                                                    transform=A.Compose(general_transforms + train_transforms),
                                                    preprocessing=self.build_preprocessing_pipeline(
                                                        self.cache_path_val),
                                                    preprocess_fn=preprocess_fn,
                                                    patch_size=self.config['patch_size'],
                                                    stride=self.config['stride'],
                                                    name='Validation')

    def get_dataset_train(self):
        return self.dataset_train

    def get_dataset_val(self):
        return self.dataset_val


class MultiSource(DataSource):
    def __init__(self, config):
        super().__init__(config)
        self.sources = [SingleSource(config, dataset_config) for dataset_config in config['datasets']]

    def prepare_data(self):
        for source in self.sources:
            source.prepare_data()

    def setup(self, stage, general_transforms, train_transforms, val_transforms):
        for source in self.sources:
            source.setup(stage, general_transforms, train_transforms, val_transforms)

    def get_dataset_train(self):
        return ConcatDataset([source.get_dataset_train() for source in self.sources])

    def get_dataset_val(self):
        return ConcatDataset([source.get_dataset_val() for source in self.sources])
