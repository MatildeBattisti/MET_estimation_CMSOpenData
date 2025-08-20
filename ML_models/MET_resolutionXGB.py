# XGBoost has to be imported before ROOT to avoid crashes because of clashing
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from itertools import product
import uproot
import time

import matplotlib.pyplot as plt


"""
    Reads data from ROOT file and selects ML alorithm's parameters
"""
def read_data():
    file = uproot.open("../skimmed_datasets/general_skimmed0.root")
    tree = file["Events"]

    params = [
        "nJet",
        "Jet_eta_bst",
        "Jet_pt_bst",
        "Jet_phi_bst",
        "Jet_mass_bst",
        "Jet_eta_bnd",
        "Jet_pt_bnd",
        "Jet_phi_bnd",
        "Jet_mass_bnd",
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
    ]
    target = ["GenMET_pt"]

    data = tree.arrays(params + target, library="np")

    print(f'✅ Read data from .root file')
    return params, target, data


"""
    Data engineering
"""
def data_engineering(data):
    npFeatures = np.array([
        data["nJet"],
        data["Jet_eta_bst"],
        data["Jet_pt_bst"],
        data["Jet_phi_bst"],
        data["Jet_mass_bst"],
        data["Jet_eta_bnd"],
        data["Jet_pt_bnd"],
        data["Jet_phi_bnd"],
        data["Jet_mass_bnd"],
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
        #data["MET_pt"],
        data["MET_significance"]
    ]).T

    npTarget = data["GenMET_pt"]

    print(f'(N events, N features): {npFeatures.shape}')
    return npFeatures, npTarget

"""
    Correlation between each param and the target
"""
def feature_target_correlation(npFeatures, npTarget, params):
    correlations = []

    for i in range(npFeatures.shape[1]):
        feature = npFeatures[:, i]
        corr = np.corrcoef(feature, npTarget)[0, 1]
        correlations.append((params[i], corr))

    # Sort by absolute correlation value, descending
    correlations_sorted = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

    print("🔍 Pearson correlation of each feature with the target (GenMET_pt):")
    for name, corr in correlations_sorted:
        print(f"{name:25s}: {corr:.4f}")

    return correlations_sorted

"""
    Defines the difference between true MET and measured MET as
    the correction for this regression model
"""
def MET_correction(data):
    METcorr = data["MET_pt"] - data["GenMET_pt"]

    print(f"✅ Calculated MET correction for the regression model\nSome info on MET correction:")
    print(f"METcorr MIN: {np.min(METcorr)}")
    print(f"METcorr MAX: {np.max(METcorr)}")
    print(f"METcorr MEAN: {np.mean(METcorr)}")
    print(f"METcorr MEDIAN: {np.median(METcorr)}")
    print(f"METcorr STD DEVIATION: {np.std(METcorr)}")

    GenMET = data["GenMET_pt"]

    print(f"Some useful info on the skimmed GenMET:")
    print(f"GenMET MEAN: {np.mean(GenMET)}")
    print(f"GenMET STD DEVIATION: {np.std(GenMET)}")
    return METcorr

"""
    Training XGBoost model using all features
"""
def model_training(npFeatures, METcorr):
    train_start_time = time.time()

    # Splitting dataset in training and testing
    x_train_full, x_test, y_train_full, y_test = train_test_split(npFeatures, METcorr, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    scaler.fit(x_train_full)
    x_train_full = scaler.transform(x_train_full)
    x_test = scaler.transform(x_test)

    # Hyper-parameters searching grid
    hparam_grid = {
        'n_estimators': [200, 300, 500], #[100, 200, 300, 400, 500, 600, 800, 1000],
        'learning_rate': [0.05, 0.1], #[0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [2, 4, 6], #[2, 4, 6, 8, 10],
        'min_child_weight': [1, 2, 3], #[1, 2, 3, 4, 5],
        'subsample': [1], #[0.5, 0.8, 1],
        'colsample_bytree': [1],
        'reg_alpha': [1],
        'reg_lambda': [1]
    }

    # Defining KFold
    kf = KFold(
        n_splits = 3,
        shuffle = True,
        random_state = 42
    )

    # Create all combinations of parameters
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
        for train_idx, val_idx in kf.split(x_train_full):
            x_train, x_val = x_train_full[train_idx], x_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

            model = XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                early_stopping_rounds=20,
                random_state=42,
                **params
            )

            model.fit(
                x_train, y_train,
                eval_set=[(x_val, y_val)],
                verbose=False
            )

            y_pred = model.predict(x_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_scores.append(rmse)

        avg_rmse = np.mean(val_scores)
        print(f"    🔹 Avg RMSE: {avg_rmse:.4f}\n")

        # Save best params
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params

    print("✅ Best Parameters Found:")
    print(best_params)
    print(f"📉 Best Avg RMSE: {best_score:.4f}")

    # Training model with best hparams on full training set
    best_model = XGBRegressor(
        **best_params,
        eval_metric='rmse',
        objective='reg:squarederror',
        early_stopping_rounds=20
    )

    # Splitting training dataset in a smaller train and validation dataset
    x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

    best_model.fit(
        x_train_final, y_train_final,
        eval_set=[(x_val_final, y_val_final)],
        verbose=True
    )

    y_test_pred = best_model.predict(x_test)

    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    print(f"✅Training completed in {train_time}s")

    # Saving model
    #best_model.save_model('utils/bestmodel0_specific1.json')
    #
    #print(f"✅Saved best model, all features included")
    return best_model, y_test, y_test_pred


"""
    Model Training REDONE
"""
def model_training1(npFeatures, npTarget):
    start_time = time.time()

    # Splitting dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(npFeatures, npTarget, test_size=0.1, random_state=42)

    # Splitting training dataset in actual training dataset and validation
    X_true_train, X_val, y_true_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Scaling data
    scaler = MinMaxScaler()
    X_true_train = scaler.fit_transform(X_true_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Model
    model = XGBRegressor(
        eval_metric='rmse',
        objective="reg:squarederror",
        early_stopping_rounds=20
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose = True
    )

    y_test_pred = model.predict(X_test)

    training_time = time.time() - start_time
    print(f"✅Training completed in {training_time}s")
    return y_test, y_test_pred


"""
    Evaluating the regression model using different metrics
"""
def evaluate_regression(y_test, y_test_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f"Evaluation metrics for the model:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    return rmse, mae, r2

def select_top_features_cumulative(params, npFeatures, best_model):
    # Getting sorted features importance dictionary
    importances = best_model.feature_importances_
    feature_importance_dict = {
        feature: importance
        for feature, importance in sorted(zip(params, importances),
        key=lambda x: x[1],
        reverse=True)
    }
    print(f'Features importance dictionary:\n {feature_importance_dict}')

    # Select best features
    sorted_idx = np.argsort(importances)[::-1]
    selected_idx = sorted_idx[:10]

    # Print selected feature names
    selected_feature_names = [params[i] for i in selected_idx]
    print(f'Selected features ({len(selected_feature_names)}): {selected_feature_names}')

    features_selected = npFeatures[:, selected_idx]

    return features_selected

"""
    Adding to the experimental MET the ML predicted correction
"""
def apply_correction(npFeatures, best_model, data):
    correctedMET = data['MET_pt'] - best_model.predict(npFeatures)

    np.set_printoptions(threshold=np.inf)

    #print(f"Corrected MET: {correctedMET}")
    #print(f"True MET: {data['GenMET_pt']}")
    return correctedMET



"""
    Defining the MET resolution as the standard deviation between the
    experimental MET and the ML corrected one
"""
def MET_resolution(data, correctedMET):
    METres = np.std(data['MET_pt'] - correctedMET)

    print(f"MET resolution: {METres}")
    return METres



if __name__ == '__main__':
    params, target, data = read_data()

    npFeatures, npTarget = data_engineering(data)
        
    correlations_sorted = feature_target_correlation(npFeatures, npTarget, params)

    #METcorr = MET_correction(data)
    y_test, y_test_pred = model_training1(npFeatures, npTarget)

    #best_model, y_test, y_test_pred = model_training(npFeatures, npTarget)
    
    rmse, mae, r2 = evaluate_regression(y_test, y_test_pred)
    
    #features_selected = select_top_features_cumulative(params, npFeatures, best_model)
    
    #correctedMET = apply_correction(npFeatures, best_model, data)
    
    #METres = MET_resolution(data, correctedMET)

    #GenMET_pt = data["GenMET_pt"]
    #plt.figure(figsize=(8, 6))
    #plt.scatter(GenMET_pt, correctedMET, color='blue', alpha=0.6)
#
    #min_val = min(GenMET_pt.min(), correctedMET.min())
    #max_val = max(GenMET_pt.max(), correctedMET.max())
    #plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    #
    #plt.title('Scatter Plot of correctedMET vs GenMET_pt')
    #plt.xlabel('GenMET_pt')
    #plt.ylabel('correctedMET')
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6)

    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    
    plt.title('Scatter Plot of Predicted vs True')
    plt.xlabel('TrueMET')
    plt.ylabel('PredictedMET')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


