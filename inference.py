import argparse
import logging
from pathlib import Path

from inference.predictor import ImagePredictor
from utils.inference_utils import output_name, load_single_slice, load_image, is_single_image


def main(model_path, data_path, output_path, module_name, idx=None, axis=None, stride=None, raw=False, data=None):
    print('Started')
    axis_pred = axis if not is_single_image(idx, axis) else 0
    predictor = ImagePredictor(model_path, module_name, axis=axis_pred, stride=stride, raw=raw, dataset=data)
    if is_single_image(idx, axis):
        input_image = load_single_slice(data_path, idx, axis)
    else:
        input_image = load_image(data_path)
    output_image = predictor.predict_image(input_image)
    logging.info('Saving...')
    output_path = output_path if output_path else output_name(data_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # TODO: add option to save compressed np array instead of tiff s.t. we can access float values and not have a 25GB file.
    output_image.save(output_path)
    logging.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--data_path', '-i', type=str, required=True, help='Path to the input data')
    parser.add_argument('--out_path', '-o', type=str, required=False,
                        help='(Optional) Path where output should be stored', default='')
    parser.add_argument('--module', '-M', type=str, default='binary', help='Which type of training module was used.',
                        choices=['binary', 'multiclass'])
    parser.add_argument('--index', '-idx', type=int, required=False, help='(Optional) Index of slice to predict.')
    parser.add_argument('--axis', '-a', type=int, required=True, choices=[0, 1, 2],
                        help='Axis (i.e. plane) for the index. 0: XY, 1: XZ, 2: YZ')
    parser.add_argument('--stride', '-s', type=int, required=False, default=None,
                        help='(Optional) Override stride used when tiling image. Set to 0 for no overlap.')
    parser.add_argument('--raw', '-R', action='store_true',
                        help='(Optional) Save raw predicted values instead of argmax/rounded ones.')
    parser.add_argument('--data', '-D', type=str, required=False, default=None,
                        help='(If multi data sources during training) '
                             'Either specify path to a yaml containing a *single* datasets configuration or an integer '
                             'index to the dataset in the original config of which the transformations should be used.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)  # Set desired log level

    main(args.model_path,
         args.data_path,
         args.out_path,
         args.module,
         args.index,
         args.axis,
         args.stride,
         args.raw,
         args.data)
