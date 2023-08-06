Elephant Trunk Segmentation
=======
This repository contains code for AI automated *binary* segmentation of CT scans of elephant trunks.

CT scans are provided as black and white stacked tiffs (aka 3D tiffs). A deep learning model is trained on existing
(manually) annotated data. The trained model is then used to predict the segmentation in the whole file.

The project uses a U-Net++ architecture with configurable backbones.
This model takes 2D slices of the whole dataset and learns on them or predicts the segmentation of these.

Note, that the segmentation is binary.
This means that the model only predicts two classes (e.g., muscle and background), but not the instances (e.g., being able to distinguish different muscles) 

## Contents
1. [Setup](#setup)
1. [Usage](#usage)
2. [Requirements](#requirements)
3. [Description](#description)
3. [Authors and References](#authors-and-references)
4. [License](#license)
5. [Contributing](#contributing)

## Setup
Before you can use the project, you need to do some setting up.

### 1. Setup Environment
It is generally recommended to use a virtual environment in which to setup the project.
This is a Python project so any virtual python environment could be used.
Due to its popularity in science, we use conda here.
If not already done so, install follow the installation instructions for Anaconda on their [website](https://docs.anaconda.com/free/anaconda/install/).

Then create a new environment

    conda create -n trunkSeg python=3.10

Then, activate it.
Note, that you will usually always have to activate this conda environment when starting a new CLI session 
and you want to work with this project. All other steps only need to be performed once.

    conda activate trunkSeg

### 2. Setup Dependencies
For the next steps, make sure you are in the projects directory.

     cd PATH/TO/TRUNKSEG

Now, install the requirements for this project.
The `requirements.txt` specifies all requirements needed to run in the CLI.
However, due to additionaly dependencies, it is recommended to install PyTorch by first following the instructions on their website.

[**Install PyTorch**](https://pytorch.org/get-started/locally/)

Now you can run:

    pip install -r requirements.txt

### 3. Weights And Biases Logging
This project integrates with W&B logging. After running `pip install` above, `wandb` is installed in your environment.
If you have a [W&B account](https://wandb.ai) you can now connect to it by running:

    wandb login

You will then need to paste your W&B API key to log in. You can retrieve this in your account settings.
[Read their docs for more details.](https://docs.wandb.ai/quickstart)

Finally, in your W&B account you will need to add a project for usage with this python project.
You can then specify this name in the config file (more below) or just give it the default name `trunk-segmentation`
and not worry about it.

That's it. Now, you can observe your experiments (training progress) on W&B.

## Usage
This project can either be used via the CLI or by importing the respective packages into your own project (i.e. a jupyter notebook)
and experimenting there.
For examples of how to use jupyter notebooks, take a look into the `notebooks` directory.

### Data Preparation
The most important part of machine learning is having the right data.
So before training can be started, the data needs to be prepared accordingly.

The model expects three types of data:
1. `image` This is usually the CT scan that needs to be segmented.
2. `segmentation` This file contains the existing segmentation (either done manually or from previous runs).
The data should be binary, that is there should only be values 0 and 1 in the file (if you export the file as a tiff from a segmentation software with only one layer this is usually the case).
3. `mask` (*optional*) If your segmentation file is not complete, i.e. it does not segment all areas of the image correctly, you will want to
mark those regions that should be used for training. These masked regions are then used, all others ignored.
You may specify up to two masks: one for training, one for evaluation. These files should again be binary.

So in total you will provide up to five tiff files for training. During inference only an image file is necessary.

All data is expected as a multi-/3D-tiff with the same dimensions.

### Training
To train a model, you can use the `python train.py` command:

    train.py [-h] [--config CONFIG]
    options:
       -h, --help            show this help message and exit
      --config CONFIG, -C CONFIG, -c CONFIG
                            Path to the config file. Default: config.yaml

Everything important lives in the config file specified by the config parameter.
The config file is a yaml file where all hyperparameters of the model architecture, training, logging and everything else
are specified.
Here are the most important parameters. For examples, please refer to the `configs` directory.

    name:  # name of training run
    trainer: # either binary or multi, depending on the number of classes you have. Recommended: binary
    loss: # which loss to use. One of: ce, focal, combined. Not all may work with all trainers. Recommended: ce
    logging: # whether to disable logging on W&B. This is useful for debugging. Recommended: true
    tags:  # list of tags added to the training run on W&B. Recommended: [line break] - real
    
    # For training the Adam optimizer is used with a ReduceLROnPlateau schedule. Hyperparameters for these are:
    learning_rate: # initial learning rate
    lr_factor:
    lr_patience:
    
    # Further hyperparameters for training
    batch_size:
    epochs: # maximum number of epochs if no early stopping is triggered/configured
    min_epochs: # minimum number of epochs before early stopping can be invoked.
    
    patch_size: # width and height of a square patch
    stride: # if stride < patch_size there is an overlap between patches
    
    # Configure paths to data
    datasets:
      image_path: # path to scan data (as multi tiff)
      label_path: # path to segmentation mask (as multi tiff)
      # as the data is not fully annotated/segmented, we need to specify which regions can be used for training. 
      # We refer to this as masks.
      train:
        mask_path: # regions where patches for training can be extracted.
        cache_path: # path where cache can be stored. A directory will be created with this name. If you change the data, either delete the cache directory or change the name here.
      val:
        mask_path: # regions where patches for evaluation can be extracted.
        cache_path: # path where the extracted patches are cached. See notes in train/cache_path.
    cache: # whether to use caching. Highly recommended except for debugging.
    train_val_split_ration: # You can skip passing a validation mask and instead create a train/val split from the training data. Recommende: 1 when val mask is passed.
    pad_mode: # During patching and augmentation the data may be padded. This is the strategy to fill additional pixels. Recommended: 'reflect'
    pad_masked: # If set, adds padding with this number of pixels to each masked region. Used so that edge regions are not underrepresented. Value depends on your patch size.
    drop_until_equal: # If you have a massive skew between training classes, drop samples of the excessive class until all classes have the same number of samples. Recommended: false
    keep_extra: # If drop until equal is true, you can set a number >= 0 between specifying many more samples (in percent) may be kept of excessive classes. 
    
    encoder:
      name: # name of encoder to be used.

    # Because we train with little data, we need to augment the training data. Here, we specify transformation applied to achieve this.
    # You can specify any transforms from the albumentations module. Those specified here are meant as an example but are also used in our experiments.
    transforms:
      general: # always applied (before caching)
        - Clip: # A custom transform that clips the image values between 90 and 195 and then normalizes these between 0 and 1.
           min_val: 90
           max_val: 195
      train: # applied while sampling batches for training.
        - Blur:
            blur_limit: 5
        - GaussNoise:
            var_limit: 10.
        - HorizontalFlip:
        - VerticalFlip:
        - Rotate:
            limit: 45
            border_mode: 'BORDER_REFLECT_101'
    
    early_stopping: # If specified, configures early stopping during training
      patience: # how many steps to wait after no improvement has been observed until to actually stop.
      min_delta: # If metric does not improve more than this threshold for patience number of steps, stop training. Recommended: small value over 0.
      metric: # Which metric to track for early stopping. Usually: 'val_loss'
      mode: # Whether the metric should be small (min) or large (max). Usually: 'min'
    
    seeds:
      data: # Some arbitrary number used for seeding the validation-training-split if configured. This is seeded to allow reproducibility.

After successful training, the weights of the final model are stored on the disk.
If you used W&B logging, the path to the final model checkpoint is `PROJECT_ID/RUN_ID/checkpoints/last.ckpt` 
where `PROJECT_ID` is the W&B project name (default `trunk-segmentation`) and `RUN_ID` a unique string assigned to
this training run by W&B.

This checkpoint can be loaded for use in other projects or for inference with the `inference.py` script.

### Inference
The basic command for inference is:

    inference.py [-h] --model_path MODEL_PATH --data_path DATA_PATH [--out_path OUT_PATH] [--module {binary,multiclass}] [--index INDEX]
                    [--axis {0,1,2}] [--stride STRIDE] [--raw]
 
    options:
      -h, --help            show this help message and exit
      --model_path MODEL_PATH, -m MODEL_PATH
                            Path to the model checkpoint file
      --data_path DATA_PATH, -i DATA_PATH
                            Path to the input data
      --out_path OUT_PATH, -o OUT_PATH
                            (Optional) Path where output should be stored
      --module {binary,multiclass}, -M {binary,multiclass}
                            Which type of training module was used.
      --index INDEX, -idx INDEX
                            (Optional) Index of slice to predict.
      --axis {0,1,2}, -a {0,1,2}
                            Axis (i.e. plane) for the index. 0: XY, 1: XZ, 2: YZ
      --stride STRIDE, -s STRIDE
                            (Optional) Override stride used when tiling image. Set to 0 for no overlap.
      --raw, -R             (Optional) Save raw predicted values instead of argmax/rounded ones.

When predicting the segmentation for a whole file, your arguments will probably look similar to this example

    python inference.py -m trunk-segmentation/2ixkh869/checkpoints/last.ckpt -M binary -i ../data/ct_scan.tif -a 0 -o ../exports/2ixkh869/0.tiff -s 96

* `-m [...]/last.ckpt` is the path of the model you trained earlier and want to use for inference.
* `-M binary` means that you used the binary mode (which is also default) in the config.
* `-i ../data/ct_scan.tif` is the path to the image file that should be segmented.
* `-a 0` means that the 2D slices should be taken from the XY planes.
For best performance it is recommended to run inference over all three axes and potentially combining or at least comparing these. 
* `-o [...]/0.tiff` is the path where the predicted segmentation is saved.
If this is not given, the output is stored next to the specified model file.
* `-s 96` is the stride used during tiling each slice. If <=0 no overlap is used. 
However, it is recommended to have at least some overlap (stride <= patch_size in original config) to avoid rectangular
artifacts in the prediction.

The prediction is saved as a tiff file.
If you would like to continue working with the segmentation in a segmentation software, such as Amira or Dragonfly, you can import it normally.
One thing you should ensure is that the spacing of the voxels is the same as your original data.

### Post-Processing
If you created multiple predictions and would like to combine them to a merged prediction for better performance,
you can use the `tools/stitching.py` script.

It is common in machine learning to combine multiple models to create a new, ensemble model.
Here, instead of computing the output of all models at once, we save the prediction of every model first and combine their outputs afterward.
This allows us to get the same result while being able to experiment with different ensembles.

The command overview is:

    stitching.py [-h] [--predictions_root PREDICTIONS_ROOT] [--strategy {all,majority,merge}] [--output OUTPUT] [--output_dir OUTPUT_DIR]
    
    options:
      -h, --help            show this help message and exit
      --predictions_root PREDICTIONS_ROOT, -R PREDICTIONS_ROOT
                            Path to the folder that contains all models that should be merged.
      --strategy {all,majority,merge}, -S {all,majority,merge}
                            How models should be combined. all = Keep segmentation where all models agree. majority = Keep segmentation where majority of
                            models agree. merge = Keep segmentation where at least one of the models segmented.
      --output OUTPUT, -O OUTPUT
                            Name of output file. Saved in models_root folder.
      --output_dir OUTPUT_DIR
                            If specified the stitched file is saved in this directory. Otherwise, in the prediction root.

**NOTE** Please ensure that you have enough memory to fit all prediction into the memory at once and then still some spare memory.

## Requirements
Experiments have been conducted on a workstation with an NVIDIA RTX3060 GPU with 24GB and 512GB RAM.
The main limit of the program is the RAM.
In addition to needing to load the complete scan into memory, the scan will be split up into tiles as a single slice may
be too large for the model to process (i.e. requiring too much GPU memory).
Due to the tiling the memory usage is a multiple of the actual data size.

As this may quickly become a bottleneck during *inference* for large scans, there are two options:
1. (Recommended) Split the scans into multiple files.
2. Change the code to work for your use case.

In case of training, this should not necessarily be a problem.
This is because during training, you will usually provide a mask of the training regions that are already segmented.
Only these regions will be tiled and kept in memory during training.

## Description
In this section we might share a description of what we all tried in this repository.

## Tips and Tricks
In this section we might share some tips on what works best.

## Authors and References
This project was created by [Pius Ladenburger](https://github.com/GittiHab) during his time at the [Brecht Lab](https://www.activetouch.de/index.php) at the [BCCN Berlin](https://www.bccn-berlin.de/).

### References
* This repo uses the U-Net++ model ([UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165))
  * We use the [segmentation_models for PyTorch](https://github.com/qubvel/segmentation_models.pytorch) implementation by [qubvel](https://github.com/qubvel)
  * A variety of encoders (backbones) can be used, but we mainly used [EfficientNet](https://arxiv.org/abs/1905.11946) and experimented with [ResNet](https://arxiv.org/abs/1512.03385)
  * We tried using the Focal Loss as proposed by [Lin et al.](https://arxiv.org/abs/1708.02002) implemented in the [Kornia](https://github.com/kornia/kornia) library.
* For machine learning we used [PyTorch](https://pytorch.org/) in conjunction with [PyTorch Lightning](https://lightning.ai/) for training.
* Parts of the code were generated using ChatGPT (versions 3.5 and 4) and modified.

## License
UNLICENSED - This is a research project, you may build on it for research purpose.
For any other use please contact the authors.

Should you use this project for your research please cite this project and its authors.

## Contributing
If you decide to build on this project, please consider creating a fork with your improvements back to this repository.
This project tries to be as readable (clean code + documentation where required) as possible, so that it is maintainable (especially for new developers) and extensible.
Additionally, critical parts are tested using automated tests, found in the `tests` directory.
Please try to maintain this standard when continuing work on this project.

The code itself contains some todos that are planned (but may never be implemented without your help).
Further features that did not make it into this repo yet:
- [ ] Support models with 3D input (e.g. V-Net).
- [ ] Support predicting instances of muscles
(either by instance segmentation models or by predicting contours and then separating)
- [ ] Making everything more memory efficient to work with large input files.
