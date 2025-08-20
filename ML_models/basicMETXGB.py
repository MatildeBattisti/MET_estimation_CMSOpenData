# XGBoost has to be imported before ROOT to avoid crashes because of clashing
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import numpy as np
from itertools import product
import uproot
import time

import matplotlib.pyplot as plt



"""
    Opens ROOT file and selects desired branches.
    Extracts them with uproot method .arrays().
"""
def read_data():
    file = uproot.open("../skimmed_datasets/skimmed_ZZZ.root")
    tree = file["Events"]

    branches = [
        "nJet",
        "Jet_area_st",
        "Jet_eta_st",
        "Jet_pt_st",
        "Jet_phi_st",
        "Jet_mass_st",
        "Jet_area_nd",
        "Jet_eta_nd",
        "Jet_pt_nd",
        "Jet_phi_nd",
        "Jet_mass_nd",
        "Jet_area_rd",
        "Jet_eta_rd",
        "Jet_pt_rd",
        "Jet_phi_rd",
        "Jet_mass_rd",
        "MET_pt",
        "MET_phi",
        "MET_significance",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        "GenMET_pt"
    ]

    data = tree.arrays(branches, library="np")

    #Insert check for np arrays

    print(f'✅ Read data from .root file')
    return branches, data



"""
    Data parsing to work with XGB.
"""
def data_parsing(data):
    features = np.array([
        data["nJet"],
        data["Jet_area_st"],
        data["Jet_eta_st"],
        data["Jet_pt_st"],
        data["Jet_phi_st"],
        data["Jet_mass_st"],
        data["Jet_area_nd"],
        data["Jet_eta_nd"],
        data["Jet_pt_nd"],
        data["Jet_phi_nd"],
        data["Jet_mass_nd"],
        data["Jet_area_rd"],
        data["Jet_eta_rd"],
        data["Jet_pt_rd"],
        data["Jet_phi_rd"],
        data["Jet_mass_rd"],
        data["MET_pt"],
        data["MET_phi"],
        data["MET_significance"],
        data["MET_covXX"],
        data["MET_covXY"],
        data["MET_covYY"]
    ]).T

    target = data["GenMET_pt"]

    print(f'✅ Shape of features and target:')
    print(f"(N events, N features): {features.shape}")
    print(f"(N events, N target): {target.shape}")
    return features, target



"""
    Data analysis on model's features: evaluates the relevant ones through correlation.
    Evaluates target's data for later analysis.
"""
def data_analysis(branches, features, target):

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
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.05, 0.1, 0.3],
        'max_depth': [4, 6, 8],
        'min_child_weight': [1, 2, 4]
        #'subsample': [1],
        #'colsample_bytree': [1],
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

            model = XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                early_stopping_rounds=20,
                random_state=42,
                **params
            )

            model.fit(
                x_train, y_train_split,
                eval_set=[(x_val, y_val)],
                #eval_metric='rmse',
                #early_stopping_rounds=10,
                verbose=False
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
    print(f"✅ Training completed in {elapsed_time}s")
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

    best_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=20,
        random_state = 42,
        **best_params
    )

    best_model.fit(
        X_tr_scaled, y_tr,
        eval_set=[(X_val_scaled, y_val)],
        #eval_metric='rmse',
        #early_stopping_rounds=10,
        verbose=False
    )

    y_test_pred = best_model.predict(X_test_scaled)

    retraining_time = time.time() - starting_time
    print(f"✅ Retraining and testing completed in {retraining_time}s")
    return y_test_pred



"""
    Evaluates the prediction made by the model over the test set.
"""
def results_evaluation(y_test, y_test_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f"✅ Evaluation metrics for the model:")
    print(f"Target-Prediction RMSE: {rmse}")
    print(f"Target-Prediction MAE: {mae}")
    print(f"Target-Prediction R²: {r2}")
    return rmse, mae, r2



"""
    Evaluates the importance of the features used in the retraining.
"""
def top_features(branches, features, best_model):

    return



"""
    Main. Executing all functions and plot relevant things.
"""
if __name__ == '__main__':
    branches, data = read_data()

    features, target = data_parsing(data)

    data_analysis(branches, features, target)

    X_train, y_train, X_test_scaled, best_params, y_test = search_best_model(features, target)

    y_test_pred = testing(X_train, y_train, X_test_scaled, best_params)

    rmse, mae, r2 = results_evaluation(y_test, y_test_pred)

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