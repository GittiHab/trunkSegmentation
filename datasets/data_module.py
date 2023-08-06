import cv2
import pytorch_lightning as pl
import skimage
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import DataLoader, random_split, Subset
from copy import copy
import albumentations as A
import datasets.preprocessing as P

from datasets.datasets import GridTiledDataset
from utils.dataset_utils import expand_seed
from utils.config_utils import extract_encoder_params, read_transforms


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cache_path = self.config['datasets']['train']['cache_path'] if 'cache_path' in self.config['datasets'][
            'train'] else './cache/train'
        self.cache_path_val = self.config['datasets']['val']['cache_path'] if 'cache_path' in self.config['datasets'][
            'val'] else './cache/val'

    def prepare_data(self):
        if not P.Caching(path=self.cache_path).needs_source_data() and not P.Caching(
                path=self.cache_path_val).needs_source_data():
            self.image_data, self.label_data, self.mask_data, self.mask_data_val = [], [], [], []
            return
        multi_img = skimage.io.MultiImage(self.config['datasets']['image_path'])
        self.image_data = multi_img[0]
        multi_img = skimage.io.MultiImage(self.config['datasets']['label_path'])
        self.label_data = multi_img[0]
        self.mask_data = None
        if 'mask_path' in self.config['datasets']['train']:
            multi_img = skimage.io.MultiImage(self.config['datasets']['train']['mask_path'])
            self.mask_data = multi_img[0]
        self.mask_data_val = None
        if 'mask_path' in self.config['datasets']['val']:
            multi_img = skimage.io.MultiImage(self.config['datasets']['val']['mask_path'])
            self.mask_data_val = multi_img[0]

    def train_dataloader(self):
        # TODO: consider using weighted sampling:
        #  https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler or
        #  https://github.com/ufoym/imbalanced-dataset-sampler
        return DataLoader(self.dataset_train, batch_size=self.config["batch_size"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.config["batch_size"], shuffle=False)

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
            P.Conditional(P.DropUntilEqual(self.config.get('keep_extra', 0.1)),
                          self.config['drop_until_equal']),
            P.Conditional(P.Padding(self.config.get('pad_masked', self.config['stride']),
                                    self.config.get('pad_mode', 'constant')),
                          'pad_masked' in self.config)
        ])

    def setup(self, stage=None):
        seed_fit, seed_test, seed_predict = expand_seed(self.config['seeds']['data'])

        if stage == 'fit':
            # TODO: Put this somewhere else/allow this to be specified in config
            general_transforms = read_transforms(self.config['transforms']['general'])
            train_transforms = read_transforms(self.config['transforms']['train'])

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
                                                 transform=A.Compose([general_transforms, train_transforms]),
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
                self.dataset_val.transform = general_transforms
                self.dataset_val.name = 'Validation Split'
                self.dataset_val = Subset(self.dataset_val, indices=split[1].indices)
            else:
                self.dataset_train = self.dataset_main
                self.dataset_val = GridTiledDataset(self.image_data,
                                                    self.label_data,
                                                    mask_data=self.mask_data_val,
                                                    as_image=False,
                                                    axis=self.config.get('axis', None),
                                                    transform=A.Compose([general_transforms, train_transforms]),
                                                    preprocessing=self.build_preprocessing_pipeline(
                                                        self.cache_path_val),
                                                    preprocess_fn=preprocess_fn,
                                                    patch_size=self.config['patch_size'],
                                                    stride=self.config['stride'],
                                                    name='Validation')
