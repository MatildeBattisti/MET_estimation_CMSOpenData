# CMS Open Data Analysis: MET Estimation with Machine Learning

## Introduction

This project explores the application of machine learning techniques to analyze CMS experimental data from CERN Open Data. The primary objective is to evaluate and compare Missing Transverse Energy (MET) estimation across different physics processes using two well-established ML models:
- **XGBoost**: a gradient boosting framework optimized for structured data;
- **Keras**: a high-level neural networks API for deep learning applications.

### Datasets

The project examines three distinct CMS datasets from CERN Open Data, each representing different physics processes:
1. **ZZZ**: triple Z boson production;
2. **HToAATo2Mu2B**: Higgs to pseudoscalar pairs decay;
3. **ZZTo2L2Nu**: Z boson pair production with leptonic decay.

You can find the specific datasets that I used here:
- **ZZZ** -> https://opendata.cern.ch/record/75600 -> 47348ED1-E550-CF48-9E94-BED2742AB141.root
- **HToAATo2Mu2B** -> https://opendata.cern.ch/record/41341 -> 6357E7BC-502C-2E45-A649-73A57B651715.root
- **ZZTo2L2Nu** -> https://opendata.cern.ch/record/75567 -> 0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root

### The analysis

The analysis follows a two-phase approach:

**Common Parameter Analysis**

Initially, only relevant parameters common to all three datasets are fed to the ML models. This ensures a fair comparison across different physics processes and establishes a baseline for MET estimation performance.

**Dataset-Specific Skimming**

In the second phase, dataset-specific feature selection and skimming are performed for each physics process. This targeted approach allows us to:
- optimize feature sets for each specific physics signature;
- investigate how dataset-specific variables influence MET estimation;
- compare the improvement in MET reconstruction accuracy when using process-optimized features.

### Goals

The goals for this analysis are:
- evaluate the performance of XGBoost vs. Keras models for MET estimation;
- quantify the impact of dataset-specific feature engineering on model accuracy;
- demonstrate reproducible machine learning workflows with CERN Open Data.
