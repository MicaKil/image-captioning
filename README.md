# Image Captioning

This project implements an image captioning system using deep learning techniques. The system generates descriptive
captions for images by combining a convolutional neural network (CNN) or transformer-based encoder with a recurrent
neural network (RNN) or transformer-based decoder.

## Features

- Support for multiple encoder-decoder architectures (e.g., basic, intermediate, transformer).
- Dataset handling, including splitting, saving, and logging.
- Integration with Weights & Biases (wandb) for experiment tracking.
- Support for custom tokenization and vocabulary generation.

## Directory Structure

```
├── runner/
│   ├── runner.py          # Main class for running the image captioning pipeline
│   ├── config.py          # Run configuration and hyperparameters
├── sweeper/
│   ├── sweeper.py         # Class for hyperparameter sweeping, inherits from Runner
│   ├── config.py          # Configuration for hyperparameter sweeping
├── models/
│   ├── basic.py           # Basic encoder-decoder model
│   ├── intermediate.py    # Intermediate encoder-decoder model
│   ├── transformer.py     # Transformer-based model
│   ├── swin.py            # Swin Transformer encoder
├── datasets/
│   ├── dataset.py         # Dataset handling and preprocessing
│   ├── vocabulary.py      # Vocabulary generation and tokenization
├── configs/
│   ├── example_config.json # Example JSON configuration file
├── results/
│   ├── *.csv              # Dataset splits (train, val, test)
│   ├── *.pt               # Saved datasets
├── plots/
│   ├── *.png              # Generated plots (e.g., boxplots)
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── main.py                # Entry point for running the project
```

## Getting Started

### 1. Install Dependencies

Make sure you have Python 3.8 or higher installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

- Place your dataset in the `datasets/` directory.
- Use `dataset.py` to preprocess and split the dataset into training, validation, and test sets.

### 3. Configure the Model

- Modify or create a JSON configuration file in the `configs/` directory.
- Example configurations can be found in `configs/example_config.json`.

### 4. Run the Project

To train the model, execute the following command:

```bash
python main.py --config configs/example_config.json
```

### 5. Hyperparameter Sweeping

To perform hyperparameter sweeps, use the `sweeper.py` script:

```bash
python sweeper/sweeper.py --config sweeper/config.py
```

## Results and Visualization

- Generated captions and evaluation metrics will be saved in the `results/` directory.
- Plots (e.g., loss curves, accuracy) will be saved in the `plots/` directory.
- Use Weights & Biases (wandb) for advanced experiment tracking and visualization.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
