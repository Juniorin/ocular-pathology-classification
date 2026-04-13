# Ocular Pathology Classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Deep learning–based classification of 10 ocular pathologies from fundus images using PyTorch.

## Disease Classes

| Class | Description |
|---|---|
| Central Serous Chorioretinopathy | Fluid buildup beneath the retina |
| Diabetic Retinopathy | Retinal damage caused by diabetes |
| Disc Edema | Swelling of the optic disc |
| Glaucoma | Progressive optic nerve damage |
| Healthy | No pathology detected |
| Macular Scar | Scarring of the macula |
| Myopia | Nearsightedness-related retinal changes |
| Pterygium | Abnormal tissue growth over the cornea |
| Retinal Detachment | Separation of the retina from underlying tissue |
| Retinitis Pigmentosa | Inherited degenerative retinal disease |

## Project Organization

```
├── LICENSE                        <- MIT open-source license
├── Makefile                       <- Convenience commands e.g. `make train` or `make test`
├── README.md                      <- Top-level README for developers
├── environment.yml                <- Conda environment file
├── pyproject.toml                 <- Project configuration and package metadata
├── requirements.txt               <- pip requirements file
│
├── data
│   └── raw
│       ├── Augmented_Dataset      <- Augmented training images (10 classes)
│       └── Original_Dataset       <- Original immutable fundus images (10 classes)
│
├── docs                           <- Project documentation
│
├── models
│   ├── baseline
│   │   ├── baseline_history.json  <- Training history for baseline model
│   │   └── baseline_model.pt      <- Saved baseline model weights
│   └── best
│       ├── best_run_history.json  <- Training history for best model
│       └── best_run_model.pt      <- Saved best model weights
│
├── notebooks
│   ├── 01_eda.ipynb               <- Exploratory data analysis
│   └── 02_results.ipynb           <- Results analysis and visualizations
│
├── references                     <- Data dictionaries, papers, and explanatory materials
│
├── reports
│   ├── best_run_report            <- Summary report for best run
│   └── figures
│       └── best_loss_curve.png    <- Training/validation loss curve
│
├── tests
│   └── test_data.py               <- Unit tests for data pipeline
│
└── ocular_path_classif            <- Source package
    │
    ├── __init__.py                <- Makes ocular_path_classif a Python package
    ├── config.py                  <- Global variables and configuration
    ├── dataset.py                 <- Dataset class and data loading utilities
    ├── transforms.py              <- Image augmentation and preprocessing transforms
    ├── features.py                <- Feature engineering and extraction
    ├── evaluate.py                <- Model evaluation and metrics
    ├── plots.py                   <- Visualization utilities
    │
    └── modeling
        ├── __init__.py
        ├── model.py               <- Model architecture definitions
        ├── train.py               <- Training loop and logic
        └── predict.py             <- Inference with trained models
```

## Setup

```bash
# Clone the repo
git clone https://github.com/Juniorin/ocular-pathology-classification.git
cd ocular-pathology-classification

# Create and activate environment
conda env create -f environment.yml
conda activate ocular-pathology-classification

# Install the package in editable mode
pip install -e .
```

## Usage

```bash
make train      # train the model
make evaluate   # evaluate on test set
make test       # run unit tests
make clean      # remove __pycache__ and .ipynb_checkpoints
```

--------