import logging

import torch
import torch.utils.data
import torchio as tio
from tqdm import tqdm

from training.segmentation_module import BinarySegmentation
from utils.config_utils import read_transforms
from utils.inference_utils import get_training_module, get_patch_dimensions


class ImagePredictor:
    def __init__(self, model_path, module_name, axis=0, stride=None, raw=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.module_name = module_name
        self.axis = axis if axis is not None else 0
        self._stride = stride
        self.raw = raw

        self.model = None
        self.config = None
        self.general_transforms = None
        self.load_model(self.module_name, self.model_path)

    @property
    def stride(self):
        if self._stride is None:
            return self.config['stride']
        if self._stride < 1:
            return self.config['patch_size']
        return self._stride

    def load_model(self, module_name, model_path):
        # Load and initialize the model
        self.model = get_training_module(module_name).load_from_checkpoint(model_path).to(self.device)
        self.config = self.model.config
        self.general_transforms = read_transforms(self.config['transforms']['general'])
        self.model.eval()
        logging.info('Loaded model on %s', self.device)

    def create_patch_sampler(self, input_image):
        patch_size, stride = get_patch_dimensions(self.axis, self.config['patch_size'],
                                                  self.config['patch_size'] - self.stride)
        return tio.inference.GridSampler(
            tio.Subject(one_image=input_image),
            patch_size,
            stride,
        )

    def create_data_loader(self, patch_sampler):
        batch_size = self.config['batch_size']
        return torch.utils.data.DataLoader(patch_sampler, batch_size=batch_size)

    def create_aggregator(self, patch_sampler):
        return tio.inference.GridAggregator(patch_sampler, overlap_mode='average')

    @property
    def _binary(self):
        return type(self.model) == BinarySegmentation

    def _process_prediction(self, logits):
        if self._binary:
            return logits.unsqueeze(1).unsqueeze(-1 - self.axis)
        if self.raw:
            raise NotImplementedError('MultiClass raw stitching does not work currently.')
        return logits.unsqueeze(-1 - self.axis).argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True).float()

    def predict_patches(self, data_loader, aggregator):
        with torch.no_grad():
            for patches_batch in tqdm(data_loader, desc='Predicting'):
                input_tensor = patches_batch['one_image'][tio.DATA]
                input_transformed = self.general_transforms(image=input_tensor.cpu().numpy())
                input_transformed = torch.from_numpy(input_transformed['image']).type(torch.FloatTensor).to(self.device)
                locations = patches_batch[tio.LOCATION]
                logits = self.model(input_transformed.squeeze(-1 - self.axis))
                aggregator.add_batch(self._process_prediction(logits), locations)

    def stitch_image(self, aggregator, affine, as_tensor, invert):
        logging.info('Stitching final image')
        # TODO: *argmaxing* after stitching does not work with tio
        #  because it returns the original shape of the image (losing the classes dimensions)
        #  -> one option could be to subclass GridAggregator and change _initialize_output_tensor
        output_tensor = aggregator.get_output_tensor()
        if not self.raw and self._binary:
                output_tensor = output_tensor.round()
        if as_tensor:
            return output_tensor
        if invert:
            output_tensor = output_tensor.permute([0, 3, 2, 1])
        return tio.ScalarImage(tensor=output_tensor, affine=affine)

    def predict_image(self, input_image, as_tensor=False, invert=True):
        patch_sampler = self.create_patch_sampler(input_image)
        data_loader = self.create_data_loader(patch_sampler)
        aggregator = self.create_aggregator(patch_sampler)

        self.predict_patches(data_loader, aggregator)

        final_image = self.stitch_image(aggregator, input_image.affine, as_tensor, invert)
        logging.info('Done predicting')
        return final_image
