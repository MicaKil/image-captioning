# Image Captioning

This project implements an image captioning system using deep learning techniques. The system generates descriptive
captions for images by combining a convolutional neural network (CNN) or transformer-based encoder with a recurrent
neural network (RNN) or transformer-based decoder.

> _**Disclaimer:**_ This project was developed as part of a final assignment. Some features may be incomplete, and the
> code could benefit from further refinement.

## Features

- Support for multiple encoder-decoder architectures.
- Dataset handling, including splitting, saving, and logging.
- Integration with Weights & Biases (wandb) for experiment tracking.
- Support for custom tokenization and vocabulary generation.

## Directory Structure

```
├── models/
│   ├── encoders/
│   │   ├── base.py            # Base encoder class, all encoders inherit from
│   │   ├── basic.py           # Resnet50 based encoder for LSTM decoder
│   │   ├── intermediate.py    # Resnet50 based encoder for LSTM decoder
│   │   ├── transformer.py     # Resnet50 based encoder for transformer decoder
│   │   ├── swin.py            # Swin Transformer encoder for transformer decoder
│   ├── basic.py            # Basic encoder-LSTM decoder model
│   ├── intermediate.py     # Intermediate encoder-LSTM decoder model
│   ├── transformer.py      # Transformer-based model
│   ├── image_captioner.py  # Main class for image captioning model, Basic and Intermediate inherit from this. Has inference methods
├── runner/
│   ├── runner.py           # Main class for running the image captioning pipeline
│   ├── config.py           # Run configuration and hyperparameters
├── sweeper/
│   ├── sweeper.py          # Class for hyperparameter sweeping, inherits from Runner
│   ├── config.py           # Configuration for hyperparameter sweeping
├── datasets/
│   ├── dataset.py          # Dataset handling and preprocessing
│   ├── dataloader.py       # DataLoader for batching and shuffling
│   ├── vocabulary.py       # Vocabulary generation and tokenization
├── captioner.py   # Common interface for generating captions using different algorithms and models.
├── metrics.py     # Evaluation metrics for image captioning
├── runner_cli.py  # Command-line interface for running the image captioning pipeline
├── scheduler.py   # wrapper class for a learning rate scheduler
├── sweep.py       # Initializes wandb and runs the sweeper
├── test.py        # Evaluation script for testing the model
└── train.py       # Training script for the model
```

## Dataset

Any dataset can be used as long as it's presented in a DataFrame with image file paths and captions columns.

## Run the Project

Modify the `runner/config.py` file to set the up the run configuration and hyperparameters. The configuration file
contains parameters for the model, dataset, training, and evaluation.

To train the model, use the CLI:

```bash
# Example command to train and test the model
python runner_cli.py --use-wandb --train --test
```

> TODO feature: Add support for loading a config json file for the CLI.

## Gen Captions and Attention Maps

You can use the CLI at `plotter/caption.py` to generate captions and/or attention maps for a given image.

```bash
# For generating captions without attention maps
python plotter/caption.py --img_pth <path_to_image> --checkpoint_pth <path_to_checkpoint> --no-attn --save-name <output_filename> --save-dir <output_directory>
```

```bash
# For generating captions with attention maps
python plotter/caption.py --img_pth <path_to_image> --checkpoint_pth <path_to_checkpoint> --save-name <output_filename> --save-dir <output_directory>
```

# More Info

In `report/` you can find a pdf with an indepth analysis of the project (_in Spanish_), including the architectures,
training process, and results.
