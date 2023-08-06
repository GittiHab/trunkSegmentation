from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from training.segmentation_module import BinarySegmentation, MultiClassSegmentation


def get_training_module(config, get_instance=True):
    """
    Get the appropriate SegmentationModule for training based on the provided configuration.

    Args:
        config (dict): Configuration parameters for the training. If the key 'load_model' a previous model is loaded.
            In this case, the values concerning the network architecture in the config should be identical to the loaded
            model. No verification or merge is done.
        get_instance (bool, optional): Whether to return an instance of the module or just the class. Defaults to True.

    Returns:
        Union[Type[pl.LightningModule], pl.LightningModule]: The PyTorch Lightning module class or instance.
    """
    cls = BinarySegmentation  # 'binary'
    if config.get('trainer', 'multi') == 'multi' and not config['name'].startswith('Binary Loss'):
        cls = MultiClassSegmentation
    if get_instance:
        if config.get('load_model', False):
            return cls.load_from_checkpoint(config['load_model'], config=config)
        return cls(config)
    return cls


def get_project_name(logging_config):
    if not isinstance(logging_config, dict) or 'project' not in logging_config:
        return 'trunk-segmentation'
    return logging_config['project']


class EarlyStoppingMinEpochs(EarlyStopping):
    def __init__(self, min_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epochs = min_epochs

    def _run_early_stopping_check(self, trainer: "Trainer"):
        if trainer.current_epoch < self.min_epochs:
            return False, "Min epochs not reached"
        super()._run_early_stopping_check(trainer)
