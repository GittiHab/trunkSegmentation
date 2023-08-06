import kornia
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config_utils import extract_encoder_params
from training.losses import FocalLossCustom, BinaryWeightedDiceBCELoss


class SegmentationModule(pl.LightningModule):
    _num_classes = 2

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Load model
        encoder_name, encoder_weights = extract_encoder_params(config)
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,  # or resnet50, efficientnet-b1 when running locally
            encoder_weights=encoder_weights,  # or `imagenet`
            in_channels=1,  # 1 for gray-scale images, 3 for RGB
            classes=self._num_classes,
        )
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        # TODO: make this optional as it may slow down training:
        self.log_values(y_hat, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat,
                         y)  # TODO: when setting another loss function don't forget to verify the parameters of early stopping
        self.log("val_loss", loss)
        self.log_values(y_hat, y, 'val')

    def log_values(self, y_hat, y, stage):
        pass

    def loss(self, y_hat, y):
        weight = None
        if 'loss_weights' in self.config:
            weight = torch.tensor(self.config['loss_weights'], device=self.device)
        return F.cross_entropy(y_hat, y, weight=weight)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])

        # Create a learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.config["lr_factor"],
                                      patience=self.config["lr_patience"], verbose=True)

        # Return the optimizer and scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


class MultiClassSegmentation(SegmentationModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.get('loss', 'ce').lower() == 'focal':
            self._loss = kornia.losses.FocalLoss(alpha=self.config.get('focal_alpha', 0.5), reduction='mean')
        self.dice_loss = torchmetrics.Dice(num_classes=2, ignore_index=0)
        self.jaccard_loss = torchmetrics.classification.MulticlassJaccardIndex(num_classes=2, ignore_index=0)

    def log_values(self, y_hat, y, stage):
        self.log("{}_dice".format(stage), self.dice_loss(y_hat, y))
        self.log("{}_jaccard".format(stage), self.jaccard_loss(y_hat, y))

    def loss(self, y_hat, y):
        if self.config.get('loss', 'ce').lower() != 'ce':
            return self._loss(y_hat, y)
        return super().loss(y_hat, y)

    @property
    def _num_classes(self):
        return self.config.get('num_classes', 2)


class BinarySegmentation(SegmentationModule):
    _num_classes = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dice_loss = torchmetrics.functional.dice
        self.jaccard_loss = torchmetrics.classification.BinaryJaccardIndex()

        loss_config = self.config.get('loss', 'ce').lower()
        self._targets_as_float = False
        self._apply_sigmoid = True
        if loss_config == 'focal':
            self._loss_fn = FocalLossCustom(alpha=self.config.get('focal_alpha', 0.5), reduction='mean')
            self._targets_as_float = True
            self._apply_sigmoid = False
        elif loss_config == 'combined':
            self._loss_fn = BinaryWeightedDiceBCELoss(weight=self.config.get('dice_ce_weight', 0.25))
        else:
            self._loss_fn = nn.BCELoss()
            self._targets_as_float = True

    def forward(self, x):
        prediction = self.model(x)
        if self._apply_sigmoid:
            return prediction.sigmoid().squeeze(1)
        return prediction.squeeze(1)

    def log_values(self, y_hat, y, stage):
        self.log("{}_dice".format(stage), self.dice_loss(y_hat, y))
        self.log("{}_jaccard".format(stage), self.jaccard_loss(y_hat, y))

    def loss(self, y_hat, y):
        return self._loss_fn(y_hat, y.float() if self._targets_as_float else y)
