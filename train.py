import argparse
import warnings

import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datasets.data_module import SegmentationDataModule
from utils.train_utils import EarlyStoppingMinEpochs, get_training_module, get_project_name


def main(config):
    # Create the segmentation module
    seg_module = get_training_module(config)

    # Setup Data
    data_module = SegmentationDataModule(config)

    # Setup W&B logging
    wandb_logger = None
    if config.get('logging', True):
        try:
            wandb_logger = WandbLogger(project=get_project_name(config['logging']),
                                       name=config['name'],
                                       tags=config.get('tags', []))
            wandb_logger.experiment.config.update(config)
            print('W&B logging connected.')
        except Exception:
            wandb_logger = None
            warnings.warn('Skipping wandb logging')
    else:
        print('Logging disabled.')

    # Create the PyTorch Lightning trainer
    save_intermediate_checkpoints = config.get('save_checkpoints_every', 0)
    trainer = pl.Trainer(max_epochs=config['epochs'],
                         callbacks=[
                             EarlyStoppingMinEpochs(config.get('min_epochs', 0),
                                                    monitor=config['early_stopping'].get('metric', 'val_loss'),
                                                    mode=config['early_stopping'].get('mode', 'min'),
                                                    patience=config['early_stopping']['patience'],
                                                    min_delta=config['early_stopping']['min_delta']),
                             ModelCheckpoint(save_last=True,
                                             save_top_k=1 if save_intermediate_checkpoints > 0 else 0,
                                             monitor=config['early_stopping'].get('metric', 'val_loss'),
                                             mode="min",
                                             every_n_epochs=save_intermediate_checkpoints)],
                         logger=wandb_logger)

    # Train the model
    trainer.fit(seg_module, data_module)

    # Finish W&B logging
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-C', '-c', type=str, default='config.yaml',
                        help='Path to the config file. Default: config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
