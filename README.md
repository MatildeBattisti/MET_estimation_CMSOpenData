# MET Estimation with Machine Learning using CMS Open Data

This project explores the application of machine learning techniques to analyze CMS experimental data from CERN Open Data. The primary objective is to evaluate and compare **Missing Transverse Energy (MET)** estimation across different physics processes using a Keras-based neural network model.

## Setup
A working installation of the [ROOT framework](https://root.cern/install/) is required. To install the Python dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

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

An exploratory analysis of single-value features was performed on both datasets to assess data quality and understand the distributions of relevant observables.

### Data Preparation (`data_preparation/`)

The raw datasets were cleaned and filtered to retain only events relevant to each physics process. All selection criteria were deterministic: no statistical outlier removal was applied, in order to preserve the generalization properties of the downstream ML model. Features used only during exploration were dropped at this stage.

### Model Training

Each cleaned dataset was passed to a Keras neural network model tailored to the respective process, with the goal of estimating MET from the remaining features.



