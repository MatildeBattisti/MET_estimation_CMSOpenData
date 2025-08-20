import uproot
import numpy as np
import pandas as pd
import time
#import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import itertools

import matplotlib.pyplot as plt


"""
    Reads data from ROOT file and selects ML alorithm's parameters
"""
def read_data():
    file = uproot.open("../skimmed_datasets/skimmed1_specific.root")
    tree = file["Events"]

    features = [
        "nJet",
        "nMuon",
        "Jet_eta_bst",
        "Jet_pt_bst",
        "Jet_phi_bst",
        "Jet_mass_bst",
        "Jet_area_bst",
        "Jet_eta_bnd",
        "Jet_pt_bnd",
        "Jet_phi_bnd",
        "Jet_mass_bnd",
        "Jet_area_bnd",
        "Muon_eta_st",
        "Muon_pt_st",
        "Muon_phi_st",
        "Muon_mass_st",
        "Muon_charge_st",
        "Muon_eta_nd",
        "Muon_pt_nd",
        "Muon_phi_nd",
        "Muon_mass_nd",
        "Muon_charge_nd",
        "Muon_Deltaeta",
        "Muon_Deltaphi",
        "Muon_DeltaR",
        "Muon_InvMass",
        "Deltaphi_METJbest",
        "Deltaphi_METJbnd",
        "Deltaphi_METmst",
        "Deltaphi_METmnd",
        "MET_phi",
        "MET_pt",
        "MET_significance",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        "CaloMET_pt",
        "CaloMET_phi",
        "pt_sum",
        "PV_x",
        "PV_y",
        "PV_z",
        "PV_chi2",
    ]
    target = ["GenMET_pt"]

    data = tree.arrays(features + target, library="np")

    print(f'✅ Read data from .root file')
    return features, target, data


"""
    Data analysis on the target to later evaluate model accuracy
"""
def data_analysis(data):
    df = pd.DataFrame(data)
    target = data["GenMET_pt"]

    # Target analysis
    print(f"GenMET_pt data:\n {df['GenMET_pt'].describe()}")
    print(f"GenMET_pt MEDIAN: {np.median(target)}\n")

    # Correlation between features and target
    correlations = df.corr(method='pearson')["GenMET_pt"].sort_values()
    print(f"Correlations with target:\n{correlations}")
    return correlations


"""
    Data engineering
    Transforming uproot dictionary into a Numpy array
"""
def data_engineering(data):

    npFeatures = np.array([
        data["nJet"],
        data["nMuon"],
        data["Jet_eta_bst"],
        data["Jet_pt_bst"],
        data["Jet_phi_bst"],
        data["Jet_mass_bst"],
        data["Jet_area_bst"],
        data["Jet_eta_bnd"],
        data["Jet_pt_bnd"],
        data["Jet_phi_bnd"],
        data["Jet_mass_bnd"],
        data["Jet_area_bnd"],
        data["Muon_eta_st"],
        data["Muon_pt_st"],
        data["Muon_phi_st"],
        data["Muon_mass_st"],
        data["Muon_charge_st"],
        data["Muon_eta_nd"],
        data["Muon_pt_nd"],
        data["Muon_phi_nd"],
        data["Muon_mass_nd"],
        data["Muon_charge_nd"],
        data["Muon_Deltaeta"],
        data["Muon_Deltaphi"],
        data["Muon_DeltaR"],
        data["Muon_InvMass"],
        data["Deltaphi_METJbest"],
        data["Deltaphi_METJbnd"],
        data["Deltaphi_METmst"],
        data["Deltaphi_METmnd"],
        data["MET_phi"],
        data["MET_pt"],
        data["MET_significance"],
        data["MET_covXX"],
        data["MET_covXY"],
        data["MET_covYY"],
        data["CaloMET_pt"],
        data["CaloMET_phi"],
        data["pt_sum"],
        data["PV_x"],
        data["PV_y"],
        data["PV_z"],
        data["PV_chi2"],
    ]).T

    npTarget = data["GenMET_pt"]

    print(f'(N events, N features): {npFeatures.shape}')
    return npFeatures, npTarget


"""
    Defining the NN model
"""
def create_model(input_dim, n_layers, n_units, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for layer in range(n_layers):
        model.add(Dense(n_units, activation="relu"))

    model.add(Dense(1))

    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss = "mse"   
    )
    return model


"""
    Performing K-fold and Grid Search on the training and
    validation datasets
"""
def kfold_gridsearch(npFeatures, npTarget, hparam_grid):
    best_model = None
    best_val_loss = float("inf")
    best_params = None

    # Normalizing folds
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(npFeatures)

    # Hyper-parameters Grid Search
    for params in itertools.product(*hparam_grid.values()):
        param_dict = dict(zip(hparam_grid.keys(), params))
        print(f"\n🔍 Trying parameters: {param_dict}")
        val_losses = []

        # K-Fold inside the GS
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_index], X_scaled[val_index]
            y_train, y_val = npTarget[train_index], npTarget[val_index]

            model = create_model(
                input_dim=X_train.shape[1],
                n_layers=param_dict['n_layers'],
                n_units=param_dict['n_units'],
                #activation=param_dict['activation'],
                learning_rate=param_dict['learning_rate']
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      epochs=100,
                      batch_size=param_dict['batch_size'],
                      verbose=0,
                      callbacks=[early_stop])

            val_pred = model.predict(X_val)
            val_loss = mean_squared_error(y_val, val_pred)
            val_losses.append(val_loss)

        avg_val_loss = np.mean(val_losses)
        print(f"✅ Avg val MSE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = param_dict
            best_model = model

    print(f"\n🏆 Best params: {best_params}")
    return best_model, best_params, scaler


"""
    Evaluating model on test dataset
"""
def evaluate_model(X_test, y_test, best_model, scaler):
    X_test_scaled = scaler.transform(X_test)
    y_pred = best_model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = explained_variance_score(y_test, y_pred)

    print(f"RMSE on test dataset: {rmse}")
    print(f"MAE on test dataset: {mae}")
    print(f"R2 on test dataset: {r2}")
    return y_pred


"""
    Printing and plotting results of interest
"""



if __name__ == '__main__':
    features, target, data = read_data()

    data_analysis(data)

    npFeatures, npTarget = data_engineering(data)

    ## Split dataset (holdout test set)
    #X_trainval, X_test, y_trainval, y_test = train_test_split(npFeatures, npTarget, test_size=0.2, random_state=42)
#
    ## Hyper-parameters grid
    #hparam_grid = {
    #    'n_layers': [1, 2, 4],
    #    'n_units': [32, 64, 128],
    #    'activation': ['relu'],
    #    'batch_size': [64, 128],
    #    'learning_rate': [1e-3, 1e-4],
    #}
#
    ## Model Training
    #st = time.time()
#
    #best_model, best_params, scaler = kfold_gridsearch(npFeatures, npTarget, hparam_grid)
#
    #training_time = time.time() - st
    #print(f"✅Training executed in {training_time}s")
#
    #y_pred = evaluate_model(X_test, y_test, best_model, scaler)
#
    ## Scatter plot of test results
    #plt.figure(figsize=(8, 6))
    #plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
#
    #min_val = min(y_test.min(), y_pred.min())
    #max_val = max(y_test.max(), y_pred.max())
    #plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    #
    #plt.title('Scatter Plot of Predicted vs True')
    #plt.xlabel('TrueMET')
    #plt.ylabel('PredictedMET')
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()