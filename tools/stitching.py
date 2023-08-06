import argparse
import glob
import os
import warnings

import skimage
import torch
import torchio as tio


def main(models_root, strategy, output_name, output_dir=None):
    # Setup
    output_path = os.path.join(models_root if output_dir is None else output_dir, output_name)

    # Load models
    axes_files = sorted(glob.glob(os.path.join(models_root, "*.tiff")))
    models = {}
    for a in axes_files:
        # TODO: this loop may be merged with the next (combining) loop to just keep two models in memory
        #  at a time and save RAM.
        model_name = os.path.splitext(os.path.basename(a))[0]
        if model_name == 'combined':
            warnings.warn('Found combined file in given root directory. Skipping it!')
            continue
        print("Loading model", model_name, "from", a)
        multi_img = skimage.io.MultiImage(a)
        models[model_name] = multi_img[0]

    # Combining
    print("Combining...")
    combined_model = None
    for m in models.values():
        print("...model", m)
        if combined_model is None:
            combined_model = m
            continue
        combined_model += m
    combined_model /= len(models)
    # Applying strategy
    print("Applying combination strategy.")
    if strategy == 'merge':
        combined_model = combined_model > 0
    elif strategy == 'majority':
        combined_model = combined_model > 0.5
    elif strategy == 'all':
        combined_model = combined_model == 1

    # Saving
    ## We use tio because it compresses the output file very well.
    print("Saving...")
    model_tensor = torch.from_numpy(combined_model)
    s_image = tio.ScalarImage(tensor=model_tensor.permute([2, 1, 0]).unsqueeze(0).float())
    s_image.save(output_path)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_root', '-R', type=str,
                        help='Path to the folder that contains all models that should be merged.')
    parser.add_argument('--strategy', '-S', type=str, default='merge', choices=['all', 'majority', 'merge'],
                        help='How models should be combined. all = Keep segmentation where all models agree. '
                             'majority = Keep segmentation where majority of models agree. '
                             'merge = Keep segmentation where at least one of the models segmented.')
    parser.add_argument('--output', '-O', type=str, default='combined.tiff',
                        help='Name of output file. Saved in models_root folder.')
    parser.add_argument('--output_dir', type=str,
                        help='If specified the stitched file is saved in this directory. '
                             'Otherwise, in the prediction root.')
    args = parser.parse_args()

    main(args.predictions_root, args.strategy, args.output, args.output_dir)
