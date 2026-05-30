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

The models share some features such as:
- the dataset is divided in two parts: 80% is used for training the net while the remaining 20% is aimed at testing. They are kept separate at all times to void data leakage;
- model selection is performed with a grid search over a 3-fold Cross Validation on the training set. During this phase, the training over the folds is interrupted by early stopping triggered by the validation loss;
- a retraining is performed over the whole dataset using the best hyperparameters found during the grid search. During this phase, we perform an epoch matching: the retraining stops when the average training loss found in the model selection for the same parameters is reached. This method allows the model to use all the avaiable training data and balances the otherwise resulting underfitting that we would get by retraining a model that was before trained over single folds.

The main difference between the two datasets is that for the HToAATo2Mu2B we use as target the log value of the GenMET because it has a skewed distribution and we noticed that the model struggled with learning this configuration.

### Results

All the results of the run are saved in the `results/` folder. Specifically, we save:
- a run summary of each dataset;
- both the learning curves of the model selection for the best hyperparameters and the learning curve of the retraining;
- the best 10 hyperparameters configurations found during the grid search;
- the predicted data together with the testing data;
- shap results to determine the feature importance.

In the notebook is then reported an evaluation on the performance of the model over each dataset.




