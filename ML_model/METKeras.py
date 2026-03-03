from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from itertools import product
import os
import re
import uproot
import time



"""
    Opens ROOT file and selects desired branches.
    Extracts them with uproot method .arrays().
"""
def read_data(file_path: str):
    file = uproot.open(file_path)
    tree = file["Events"]

    filename = os.path.basename(file_path)

    # Dictionary with specific branches
    branch_map = {
        "ZZZ": [
            "MET_pt",
            "MET_phi",
            "MET_covXX",
            "MET_covXY",
            "MET_covYY",
            "MET_significance",
            "PV_chi2",
            "PV_score",
            "PV_x",
            "PV_y",
            "PV_z",
            "nSV",
            "SV_charge_st",
            "SV_chi2_st",
            "SV_dxy_st",
            "SV_pAngle_st",
            "Jet_area_st",
            "Jet_eta_st",
            "Jet_mass_st",
            "Jet_phi_st",
            "Jet_pt_st",
            "Jet_btag_st",
            "Jet_area_nd",
            "Jet_eta_nd",
            "Jet_mass_nd",
            "Jet_phi_nd",
            "Jet_pt_nd",
            "Jet_btag_nd",
            "GenMET_pt"
        ],
        "HToAATo2MU2B": [
            "MET_pt",
            "MET_phi",
            "MET_covXX",
            "MET_covXY",
            "MET_covYY",
            "MET_significance",
            "PV_chi2",
            "PV_score",
            "PV_x",
            "PV_y",
            "PV_z",
            "nSV",
            "nJet",
            "nMuon",
            "Jet_eta_bst",
            "Jet_eta_bnd",
            "Jet_pt_bst",
            "Jet_pt_bnd",
            "Jet_phi_bst",
            "Jet_phi_bnd",
            "Jet_mass_bst",
            "Jet_mass_bnd",
            "Jet_area_bst",
            "Jet_area_bnd",
            "Jet_btag_bst",
            "Jet_btag_bnd",
            "Muon_charge_st",
            "Muon_charge_nd",
            "Muon_dxy_st",
            "Muon_dxy_nd",
            "Muon_dz_st",
            "Muon_dz_nd",
            "Muon_eta_st",
            "Muon_eta_nd",
            "Muon_mass_st",
            "Muon_mass_nd",
            "Muon_phi_st",
            "Muon_phi_nd",
            "Muon_pt_st",
            "Muon_pt_nd",
            "GenMET_pt"
        ],
        "ZZTo2L2Nu": [
            "MET_pt",
            "MET_phi",
            "MET_covXX",
            "MET_covXY",
            "MET_covYY",
            "MET_significance",
            "PV_chi2",
            "PV_score",
            "PV_x",
            "PV_y",
            "PV_z",
            "nElectron",
            "Electron_charge_st",
            "Electron_charge_nd",
            "Electron_dxy_st",
            "Electron_dxy_nd",
            "Electron_dz_st",
            "Electron_dz_nd",
            "Electron_eta_st",
            "Electron_eta_nd",
            "Electron_mass_st",
            "Electron_mass_nd",
            "Electron_phi_st",
            "Electron_phi_nd",
            "Electron_pt_st",
            "Electron_pt_nd",
            "nMuon",
            "Muon_charge_st",
            "Muon_charge_nd",
            "Muon_dxy_st",
            "Muon_dxy_nd",
            "Muon_dz_st",
            "Muon_dz_nd",
            "Muon_eta_st",
            "Muon_eta_nd",
            "Muon_mass_st",
            "Muon_mass_nd",
            "Muon_phi_st",
            "Muon_phi_nd",
            "Muon_pt_st",
            "Muon_pt_nd",
            "GenMET_pt"
        ], 
    }

    # Default skimming branches
    default_branches = [
        "nJet",
        "MET_pt",
        "MET_phi",
        "MET_significance",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        "PV_chi2",
        "PV_score",
        "PV_x",
        "PV_y",
        "PV_z",
        "nSV",
        "GenMET_pt"
    ]

    # Automatic branch selection depending on the file name
    if filename.startswith("specific_skimmed_"):
        match = re.match(r"specific_skimmed_(.+)\.root", filename)
        if match:
            event = match.group(1)
            if event in branch_map:
                branches = branch_map[event]
                print(f"✅ Loaded dataset '{event}'.")
            else:
                raise ValueError(f"⚠️ No specific configuration for '{event}'.")
        else:
            raise ValueError("❌ File name not recognised.")
    elif filename.startswith("skimmed_"):
        branches = default_branches
        print(f"✅ Loaded dataset '{filename}' with default branches.")
    else:
        raise ValueError(f"❌ Filename '{filename}' doesn't wollow the expected configuration.")

    data = tree.arrays(branches, library="np")
    return branches, data



"""
    Data parsing to work with Keras.
"""
def data_parsing(branches, data):
    feature_names = [b for b in branches if b != "GenMET_pt"]

    features = np.column_stack([data[name] for name in feature_names])

    target = data["GenMET_pt"]

    print(f"✅ Shape of features and target:")
    print(f"(N events, N features): {features.shape}")
    print(f"(N events, N target): {target.shape}")
    return features, target



"""
    Data analysis on model's features: evaluates the relevant ones through correlation.
    Evaluates target's data for later analysis.
"""
def data_analysis(branches, features, target):
    # Using pandas
    df = pd.DataFrame(features, columns=branches[:-1])
    df["GenMET_pt"] = target

    # Pearson
    pearson_corr = df.corr(method='pearson')
    pearson_corr.to_csv("../results/results_Keras/ZZTo2L2Nu/skimmed/pearson_corr.csv")

    # Spearman
    spearman_corr = df.corr(method='spearman')
    spearman_corr.to_csv("../results/results_Keras/ZZTo2L2Nu/skimmed/spearman_corr.csv")

    # Pearson correlation
    Pcorrs = []
    for i in range(features.shape[1]):
        feature = features[:, i]
        corr = pearsonr(feature, target)[0]
        Pcorrs.append((branches[i], corr))
    
    sortedPcorrs = sorted(Pcorrs, key=lambda x: abs(x[1]), reverse=True)
    print("✅ Pearson correlations between features and target:")
    for name, corr in sortedPcorrs:
        print(f"{name:25s}: {corr:.4f}")
      
    # Spearman correlation
    Scorrs = []
    for i in range(features.shape[1]):
        feature = features[:, i]
        corr = spearmanr(feature, target)[0]
        Scorrs.append((branches[i], corr))
    
    sortedScorrs = sorted(Scorrs, key=lambda x: abs(x[1]), reverse=True)
    print("✅ Spearman correlations between features and target:")
    for name, corr in sortedScorrs:
        print(f"{name:25s}: {corr:.4f}")

    # Target data analysis
    print(f'✅ Target data analysis:')
    print(f"GenMET MIN: {np.min(target)}")
    print(f"GenMET MAX: {np.max(target)}")
    print(f"GenMET MEAN: {np.mean(target)}")
    print(f"GenMET MEDIAN: {np.median(target)}")
    print(f"GenMET STD DEVIATION: {np.std(target)}")
    return



"""
    Defines the NN model
"""
def create_model(input_dim, n_layers, n_units, learning_rate):
    # Defining model type
    model = Sequential()

    # Input layer
    model.add(Input(shape=(input_dim,)))

    # Adding hidden layers
    for layer in range(n_layers):
        model.add(Dense(n_units, activation="relu"))

    # Output layer for regression
    model.add(Dense(1, activation='linear'))

    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss = "mse",
        metrics = ['mse']
    )
    return model



"""
    Searches the best regression model:
    - divides dataset into training and testing;
    - normalizes features;
    - implements an hyper-parameters searching grid;
    - implements a K-fold;
    - implements early stopping;
    - selects the best hyper-parameters.
"""
def search_best_model(features, target):
    starting_time = time.time()

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Data normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyper-parameters searching grid
    hparam_grid = {
        'n_layers': [1, 2, 4],
        'n_units': [32, 64, 128],
        'batch_size': [32, 64, 128],
        'learning_rate': [1e-3, 1e-4],
    }

    # K-Fold CV
    kf = KFold(
        n_splits = 3,  # can be changed
        shuffle = True,
        random_state = 42
    )

    # Combinations of parameters
    param_combinations = list(product(*hparam_grid.values()))
    param_keys = list(hparam_grid.keys())

    best_score = float('inf')
    best_params = None

    # Iterate over each parameter combination
    for combo in param_combinations:
        params = dict(zip(param_keys, combo))
        print(f"🔍 Testing params: {params}")

        val_scores = []

        # K-Fold loop
        for train_idx, val_idx in kf.split(X_train_scaled):
            x_train, x_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_split, y_val = y_train[train_idx], y_train[val_idx]

            model = create_model(
                input_dim=x_train.shape[1],
                n_layers=params['n_layers'],
                n_units=params['n_units'],
                learning_rate=params['learning_rate']
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(
                x_train, y_train_split,
                batch_size=params['batch_size'],
                epochs=100,
                validation_data=(x_val, y_val),
                verbose=0,
                callbacks=[early_stop]
            )

            y_pred = model.predict(x_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_scores.append(rmse)

        avg_rmse = np.mean(val_scores)
        print(f"    Avg RMSE: {avg_rmse:.4f}\n")

        # Save best params
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params

    print(f"✅ Best Parameters Found:\n {best_params}")
    print(f"   Best Avg RMSE: {best_score:.4f}")


    elapsed_time = time.time() - starting_time
    print(f"✅ Best parameters search completed in {elapsed_time}s")
    return X_train, y_train, X_test_scaled, best_params, y_test



"""
    Retraines & tests the regression model:
    - uses a small fixed dataset for the validation;
    - implements the retrain with the best h-params found before;
    - implements early stopping;
    - finds the predicted target.
"""
def testing(X_train, y_train, X_test_scaled, best_params):
    starting_time = time.time()

    # Splitting training and validation datasets
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Data normalization
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    best_model = create_model(
        input_dim=X_tr_scaled.shape[1],
        n_layers=best_params['n_layers'],
        n_units=best_params['n_units'],
        learning_rate=best_params['learning_rate']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    best_model.fit(
        X_tr_scaled, y_tr,
        batch_size=best_params['batch_size'],
        epochs=100,
        validation_data=(X_val_scaled, y_val),
        verbose=1,
        callbacks=[early_stop]
    )

    y_test_pred = best_model.predict(X_test_scaled)

    retraining_time = time.time() - starting_time
    print(f"✅ Retraining and testing completed in {retraining_time}s")
    return y_test_pred



"""
    Evaluates the prediction made by the model over the test set.
"""
def results_evaluation(data, y_test, y_test_pred):
    MET_pt = data["MET_pt"]
    GenMET_pt = data["GenMET_pt"]

    original_rmse = np.sqrt(mean_squared_error(MET_pt, GenMET_pt))
    original_mae = mean_absolute_error(MET_pt, GenMET_pt)
    original_r2 = r2_score(MET_pt, GenMET_pt)

    predicted_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    predicted_mae = mean_absolute_error(y_test, y_test_pred)
    predicted_r2 = r2_score(y_test, y_test_pred)

    stats = {
        "RMSE_GenMET_vs_MET_pt": original_rmse,
        "MAE_GenMET_vs_MET_pt": original_mae,
        "R2_GenMET_vs_MET_pt": original_r2,
        "RMSE_GenMET_vs_Prediction": predicted_rmse,
        "MAE_GenMET_vs_Prediction": predicted_mae,
        "R2_GenMET_vs_Prediction": predicted_r2
    }

    pd.DataFrame([stats]).to_csv("../results/results_Keras/ZZTo2L2Nu/skimmed/summary_stats.csv", index=False)

    print(f"✅ Evaluation metrics for the model:")
    print(f"Target-Prediction RMSE: {predicted_rmse}")
    print(f"Target-Prediction MAE: {predicted_mae}")
    print(f"Target-Prediction R²: {predicted_r2}")
    return



"""
    Main. Executing all functions and plot relevant things.
"""
if __name__ == '__main__':
    branches, data = read_data("../skimmed_datasets/skimmed_ZZTo2L2Nu.root")

    features, target = data_parsing(branches, data)

    data_analysis(branches, features, target)

    X_train, y_train, X_test_scaled, best_params, y_test = search_best_model(features, target)

    y_test_pred = testing(X_train, y_train, X_test_scaled, best_params)

    results_evaluation(data, y_test, y_test_pred)

    # Save in files
    scatter_MET = pd.DataFrame({
        "MET_pt": data["MET_pt"],
        "GenMET_pt": target
    })
    scatter_MET.to_json("../results/results_Keras/ZZTo2L2Nu/skimmed/MET_vs_GenMET.json", orient="records")

    scatter_test = pd.DataFrame({
        "y_test": y_test,
        "y_test_pred": y_test_pred
    })
    scatter_test.to_json("../results/results_Keras/ZZTo2L2Nu/skimmed/METpred_vs_GenMET.json", orient="records")
    