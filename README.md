# Improving MET resolution with Machine Learning using CMS Open Data

This project explores the application of machine learning techniques to analyze CMS experimental data from CERN Open Data. The primary objective is to evaluate and compare **Missing Transverse Energy (MET)** resolution estimation across different physics processes using a single Keras-based neural network model. Missing Transverse Energy is a key observable in collider physics: it quantifies the momentum imbalance in the transverse plane of a collision event, and is typically associated with the presence of particles that escape detection, such as neutrinos. 

## Setup
A working installation of the [ROOT framework](https://root.cern/install/) is required. To install the Python dependencies:
```bash
python3 -m venv .
source ./bin/activate
pip3 install -r requirements.txt
```

> **Note:** By default, the CPU version of TensorFlow is installed. If you have an NVIDIA GPU, you can enable GPU support by replacing `tensorflow` with `tensorflow[and-cuda]` in `requirements.txt` before installing.

## View the notebooks
GitHub may fail to render the notebooks due to their size. Open them directly on `Google Colab`:

| Notebook | Open |
|----------|------|
| Data Understanding | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatildeBattisti/MET_estimation_CMSOpenData/blob/main/data_understanding/data_understanding.ipynb) |
| Data Preparation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatildeBattisti/MET_estimation_CMSOpenData/blob/main/data_preparation/data_preparation.ipynb) |
| Results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatildeBattisti/MET_estimation_CMSOpenData/blob/main/Results/) |

## Datasets

Three types of CMS Open Data simulations were used, representing different physics processes. Different datasets were used either to compose a training dataset or to serve as test:

| Process | Signature | File | Purpose |
|--------|--------|------|-------|
| [DYJetsToLL](https://opendata.cern.ch/record/35669) | Z/γ* → 2 leptons | `4578E947-084C-C946-9B8D-1B45A126DCED.root` | training |
| [HToAATo2Mu2B](https://opendata.cern.ch/record/41341) | H → AA → 2 muons + 2 b-quarks | `6357E7BC-502C-2E45-A649-73A57B651715.root` | training |
| [ZZTo2L2Nu](https://opendata.cern.ch/record/75567) | ZZ → 2 leptons + 2 neutrinos | `DC33D4B8-4AF1-C94A-8F03-EDB634488D2B.root` | training |
| | | | |
| [DYJetsToLL](https://opendata.cern.ch/record/35669) | Z/γ* → 2 leptons | `6C3CD8A5-A288-724E-BF9F-BAAC46A4C139.root` | testing |
| [HToAATo2Mu2B](https://opendata.cern.ch/record/41341) | H → AA → 2 muons + 2 b-quarks | `DB4AFAC8-16AD-AB48-82D2-1E9DAE8AB314.root` | testing |
| [ZZTo2L2Nu](https://opendata.cern.ch/record/75567) | ZZ → 2 leptons + 2 neutrinos | `0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root` | testing |

Two of this processes serve as complementary cases:
- **DYJetsToLL** does not produce neutrinos, consequently the reconstructed MET is expected to originate mainly from detector resolution, pile-up interactions, and object mismeasurements;
- **ZZTo2L2Nu** involves neutrinos in the final state resulting in genuine missing transverse energy and typically higher MET values.

Finally, the **HToAATo2Mu2B** process does not include neutrinos in its primary decay chain, but neutrinos can be produced in the semileptonic decays of hadrons originating from the two b quarks. As a result, this process exhibits a broad MET spectrum, intermediate between the previous two cases.

## Analysis Pipeline

### Data Understanding (`data_understanding/`)

The `data_understanding/` folder contains an exploratory analysis of single-value features, performed on each datasets to assess data quality and understand the distributions of relevant observables.

### Data Preparation (`data_preparation/`)

The `data_preparation/` folder contains the scripts used to clean and filter the raw datasets. Only the common features relevant to the selected physics processes are retained, while all others are discarded. The selection is entirely deterministic: no statistical outlier removal is performed, preserving the original event distributions and the generalization capability of the downstream machine learning model. Features used exclusively for exploratory analysis are removed at this stage.

The three training datasets are balanced and merged into a single training set. The corresponding testing datasets undergo the same preprocessing pipeline but are kept separate to enable process-wise evaluation of the trained model.

The same applies if we only use two datasets for training.

### Machine Learning Models (`MET_models`)

The `MET_models/` folder contains four Keras neural network models:
- **MET_pt** predicts `GenMET_pt` (and its transformed variants) using a MSE loss;
- **MET_pxpy** predicts the Cartesian components `GenMET_px` and `GenMET_py` (and their transformed variants) using a MSE loss;
- **MET_newloss** predicts `GenMET_pt` (and its transformed variants) using a combined loss function that augments the normalized MSE with an additional penalty on the response;
- **MET_pxpy_newloss** predicts `GenMET_px` and `GenMET_py` (and their transformed variants) using the same combined loss.

All models follow the same training pipeline:
- hyperparameters are optimized through a grid search with 3-fold cross-validation on the training dataset. During this stage, the training on each fold is stopped early when the validation loss no longer improves;
- the best hyperparameter configuration is then used to retrain the model on a dedicated training subset while monitoring performance on a validation subset;
- the trained model weights, the feature standardization parameters (mean and standard deviation), and a run summary containing the training configuration (e.g. feature names, target transformation, batch size, and other parameters) are saved for subsequent evaluation on the independent test datasets.

To run the `main.py` of each model use:
```bash
python3 MET_chosenmodel_main.py --transform chosentransform --results-dir ../Results/results_chosendir/
```
e.g. 
```bash
python3 MET_pt_main.py --transform residual --results-dir ../Results/results_residual/
```

To run the `testing.py` of each model use:
```bash
python3 MET_chosenmodel_testing.py --results-dir ../Results/results_chosendir/
```
e.g.
```bash
python3 MET_pt_testing.py --results-dir ../Results/results_residual/
```

### Results (`Results/`)

All outputs from each run are saved in the `Results/` folder. A full evaluation of the models performance on each dataset is reported in each respective notebook.
