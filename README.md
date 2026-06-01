# MET Estimation with Machine Learning using CMS Open Data

This project explores the application of machine learning techniques to analyze CMS experimental data from CERN Open Data. The primary objective is to evaluate and compare **Missing Transverse Energy (MET)** estimation across different physics processes using a Keras-based neural network model.

## Setup
A working installation of the [ROOT framework](https://root.cern/install/) is required. To install the Python dependencies:
```bash
python3 -m venv .
source ./bin/activate
pip3 install -r requirements.txt
```

> **Note:** By default, the CPU version of TensorFlow is installed. If you have an NVIDIA GPU, you can enable GPU support by replacing `tensorflow` with `tensorflow[and-cuda]` in `requirements.txt` before installing.

## Background

**Missing Transverse Energy (MET)** is a key observable in collider physics. It quantifies the momentum imbalance in the transverse plane of a collision event, and is typically associated with the presence of particles that escape detection, such as neutrinos.

## Datasets

Two CMS Open Data simulations were used, representing different physics processes:

| Process | Record | File |
|--------|--------|------|
| [ZZTo2L2Nu](https://opendata.cern.ch/record/75567) | ZZ → 2 leptons + 2 neutrinos | `0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root` |
| [HToAATo2Mu2B](https://opendata.cern.ch/record/41341) | H → AA → 2 muons + 2 b-quarks | `6357E7BC-502C-2E45-A649-73A57B651715.root` |

The two processes serve as complementary cases: **ZZTo2L2Nu** involves neutrinos in the final state and is therefore expected to produce higher MET values, while **HToAATo2Mu2B** produces no neutrinos, so any observed MET is primarily attributable to pile-up and detector effects.

## Analysis Pipeline

### Data Understanding (`data_understanding/`)

The `data_understanding/` folder contains an exploratory analysis of single-value features, performed on both datasets to assess data quality and understand the distributions of relevant observables.

### Data Preparation (`data_preparation/`)

The `data_preparation/` folder contains the files used to clean and filter the raw datasets: only the events pertaining to each physics process were retained, while the other features were discarded. All selection criteria were deterministic: no statistical outlier removal was applied in order to preserve the generalization properties of the downstream ML model. Features used only during exploration were dropped at this stage.

### Model Training

Each cleaned dataset was passed to a Keras neural network model tailored to the respective process, with the goal of estimating MET from the remaining features.

The models share the following pipeline:
- the dataset is split into two non-overlapping partitions, 80% for training and 20% for testing, kept strictly separate throughout the entire process to prevent data leakage;
- hyperparameter selection is performed via grid search over 3-fold cross-validation on the training set. During this phase, training on each fold is interrupted by early stopping triggered by the validation loss;
- after identifying the best hyperparameters, a retraining is performed on the full training set. To avoid the underfitting that would otherwise arise from training on individual folds, we apply epoch matching: retraining stops once the average training loss achieved during model selection (for the same hyperparameters) is reached, allowing the model to fully exploit all available training data.

The main difference between the two datasets concerns the target variable: for the HToAATo2Mu2B process, the model is trained on the log-transformed GenMET rather than the raw value, since the original distribution is heavily skewed and the network showed poorer convergence without this transformation.

### Results (`results/`)

All outputs from each run are saved in the results/ folder, organized as follows:

a summary of the run for each dataset;
the learning curves from model selection (for the best hyperparameter configuration) and from the retraining phase;
the top 10 hyperparameter configurations found during the grid search;
the model predictions alongside the corresponding test set targets;
SHAP values for feature importance analysis.

A full evaluation of model performance on each dataset is reported directly in the notebook.

## View the notebooks
GitHub may fail to render the notebooks due to their size. Open them directly on `Google Colab`:

| Notebook | Open |
|----------|------|
| Data Understanding | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatildeBattisti/MET_estimation_CMSOpenData/blob/main/data_understanding/data_understanding.ipynb) |
| Data Preparation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatildeBattisti/MET_estimation_CMSOpenData/blob/main/data_preparation/data_preparation.ipynb) |
| Results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatildeBattisti/MET_estimation_CMSOpenData/blob/main/Results/results.ipynb) |