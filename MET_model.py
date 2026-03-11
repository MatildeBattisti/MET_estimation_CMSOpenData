import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization, ReLU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time


# Branch configuration per dataset
BRANCH_MAP = {
    "HToAATo2Mu2B": [
        "GenMET_pt", "MET_covXX", "MET_covXY", "MET_pt", "MET_significance",
        "fixedGridRhoFastjetAll", "PV_z", "PV_chi2", "PV_score",
        "Jet_eta_bst", "Jet_phi_bst", "Jet_btag_bst", "Jet_rawFactor_bst", "Jet_chHEF_bst", "Jet_neHEF_bst",
        "Jet_eta_bnd", "Jet_phi_bnd", "Jet_btag_bnd", "Jet_rawFactor_bnd", "Jet_chHEF_bnd", "Jet_neHEF_bnd",
        "Muon_eta_st", "Muon_phi_st", "Muon_pt_st", "Muon_eta_nd", "Muon_phi_nd", "Muon_pt_nd",
        "SV_dlenSig_bst", "SV_mass_bst", "M_mumu", "M_bb", "M_mumu_bb", "dR_MET_bb",
        "MET_projection_par", "MET_projection_perp", "dPhi_MET_mu1", "dPhi_MET_jet1", "HT",
        "nJet", "nMuon", "nSV", "Muon_charge_st", "Muon_charge_nd"
    ],
    "ZZTo2L2Nu": [
        "GenMET_pt", "MET_covXX", "MET_covXY", "MET_pt", "MET_significance",
        "PV_z", "PV_chi2", "PV_score",
        "Electron_dxy_st", "Electron_dz_st", "Electron_eta_st", "Electron_phi_st", "Electron_pt_st",
        "Electron_dxy_nd", "Electron_dz_nd", "Electron_eta_nd", "Electron_phi_nd", "Electron_pt_nd",
        "Muon_dxy_st", "Muon_dz_st", "Muon_eta_st", "Muon_phi_st", "Muon_pt_st",
        "Muon_dxy_nd", "Muon_dz_nd", "Muon_eta_nd", "Muon_phi_nd", "Muon_pt_nd",
        "nElectron", "nMuon", "nSV",
        "Electron_charge_st", "Electron_charge_nd", "Muon_charge_st", "Muon_charge_nd"
    ],
}

DEFAULT_BRANCHES = [
    "nJet", "MET_pt", "MET_significance",
    "MET_covXX", "MET_covXY",
    "PV_chi2", "PV_score", "PV_z",
    "nSV", "GenMET_pt"
]


def read_data(file_path: str):
    """
        Reads a .parquet file. The event name is inferred from the filename
    (   e.g. 'ZZTo2L2Nu.parquet' -> event 'ZZTo2L2Nu') to select the right branches.
    """
    filename = os.path.basename(file_path)
    event_name, ext = os.path.splitext(filename)

    if ext != ".parquet":
        raise ValueError(f"    Expected a .parquet file, got '{ext}'.")

    df = pd.read_parquet(file_path)

    if event_name in BRANCH_MAP:
        branches = BRANCH_MAP[event_name]
        print(f"    Loaded dataset '{event_name}' with specific branches.")
    else:
        branches = DEFAULT_BRANCHES
        print(f"    Loaded dataset '{event_name}' with default branches.")

    missing = [b for b in branches if b not in df.columns]
    if missing:
        raise ValueError(f"    Missing columns in parquet file: {missing}")

    data = {b: df[b].to_numpy() for b in branches}
    return branches, data


def data_parsing(branches, data):
    """
        Converts the branch dictionary into feature matrix and target vector.
        No global StandardScaler here: normalisation happens per-batch inside the model.
    """
    feature_names = [b for b in branches if b != "GenMET_pt"]
    features = np.column_stack([data[name] for name in feature_names])
    target = data["GenMET_pt"]

    print(f"Shape of features and target:")
    print(f"    (N events, N features): {features.shape}")
    print(f"    (N events, N target): {target.shape}")
    return features, target


def create_model(input_dim: int, n_layers: int, n_units: int, learning_rate: float):
    """
        Builds a Sequential model.

        BatchNormalization is placed after each hidden Dense layer and before the
        activation. During training it normalises the pre-activation values over
        the current mini-batch (zero mean, unit variance), learning per-feature
        scale (gamma) and shift (beta) parameters. At inference time it uses
        running statistics accumulated during training.

        This replaces the external StandardScaler: raw (unnormalised) features
        can be fed directly and the network learns the appropriate normalisation
        on its own, fold-by-fold.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for _ in range(n_layers):
        model.add(Dense(n_units))
        model.add(BatchNormalization())   # normalise activations per mini-batch
        model.add(keras.layers.ReLU())   # activation after BN (standard practice)

    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse"]
    )
    return model


def build_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int,
                     shuffle: bool = True) -> tf.data.Dataset:
    """
        Wraps numpy arrays into a tf.data.Dataset for mini-batch training.
        - shuffle = True for training splits (new shuffle order each epoch)
        - shuffle = False for validation / test splits
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.astype(np.float32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def search_best_model(features: np.ndarray, target: np.ndarray, hparam_grid, output_dir: str):
    """
        1. Splits data into train (80%) and test (20%). The test set is held-out
           and never seen during hyper-parameter search.
        2. Runs K-Fold CV exclusively on the training split.
        3. For each fold, data is fed as mini-batches via tf.data.Dataset.
        4. Selects best hyper-parameters by lowest average validation RMSE.
        5. Saves all hyper-parameter combinations and their RMSE to a CSV file.
    """
    starting_time = time.time()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # K-Fold CV on X_train only
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    param_combinations = list(product(*hparam_grid.values()))
    param_keys = list(hparam_grid.keys())

    best_score = float("inf")
    best_params = None
    all_results = []  # collect results for every combination

    print("GRID SEARCH OVER K-FOLD CV")
    print(f"    Total combinations: {len(param_combinations)}. Folds: {kf.n_splits}\n")

    for i, combo in enumerate(param_combinations, 1):
        params = dict(zip(param_keys, combo))
        print(f"    [{i}/{len(param_combinations)}] Testing params: {params}")
        val_scores = []

        fold_bar = tqdm(
            enumerate(kf.split(X_train), 1),
            total=kf.n_splits,
            desc="        Folds",
            leave=False,
            unit="fold"
        )

        for fold_idx, (train_idx, val_idx) in fold_bar:
            x_tr,  x_val  = X_train[train_idx], X_train[val_idx]
            y_tr,  y_val  = y_train[train_idx],  y_train[val_idx]

            # Mini-batch datasets
            train_ds = build_tf_dataset(x_tr,  y_tr,  params["batch_size"], shuffle=True)
            val_ds   = build_tf_dataset(x_val, y_val, params["batch_size"], shuffle=False)

            model = create_model(
                input_dim=x_tr.shape[1],
                n_layers=params["n_layers"],
                n_units=params["n_units"],
                learning_rate=params["learning_rate"]
            )

            early_stop = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )

            history = model.fit(
                train_ds,
                epochs=100,
                validation_data=val_ds,
                verbose=0,
                callbacks=[early_stop]
            )

            epochs_run = len(history.history["loss"])
            y_pred = model.predict(val_ds, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_scores.append(rmse)
            fold_bar.set_postfix({"fold_rmse": f"{rmse:.4f}", "epochs": epochs_run})

        avg_rmse = np.mean(val_scores)
        std_rmse = np.std(val_scores)
        print(f"        Avg RMSE: {avg_rmse:.4f} +/- {std_rmse:.4f}\n")

        # Collect result row
        row = {**params, "avg_rmse": avg_rmse, "std_rmse": std_rmse}
        for j, s in enumerate(val_scores, 1):
            row[f"rmse_fold{j}"] = s
        all_results.append(row)

        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params

    # Save all hyper-parameter results
    os.makedirs(output_dir, exist_ok=True)
    hparam_log_path = os.path.join(output_dir, "hparam_search_results.csv")
    pd.DataFrame(all_results).sort_values("avg_rmse").to_csv(hparam_log_path, index=False)
    print(f"Hyper-parameter search log saved -> {hparam_log_path}")

    elapsed = time.time() - starting_time

    print(f"BEST PARAMETERS: {best_params}")
    print(f"    Best Avg RMSE : {best_score:.4f}")
    print(f"    Completed in : {elapsed:.1f}s")

    return X_train, y_train, X_test, y_test, best_params


def testing(X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, best_params: dict):
    """
        Retrains with best hyper-parameters on the full training set (minus a
        small validation split used only for early stopping), then predicts on
        the held-out test set via mini-batch inference.
    """
    starting_time = time.time()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    train_ds = build_tf_dataset(X_tr,  y_tr,  best_params["batch_size"], shuffle=True)
    val_ds = build_tf_dataset(X_val, y_val, best_params["batch_size"], shuffle=False)
    # Dummy target (zeros) for the inference pipeline; predictions are used, not labels
    test_ds = build_tf_dataset(
        X_test, np.zeros(len(X_test), dtype=np.float32),
        best_params["batch_size"], shuffle=False
    )

    best_model = create_model(
        input_dim=X_tr.shape[1],
        n_layers=best_params["n_layers"],
        n_units=best_params["n_units"],
        learning_rate=best_params["learning_rate"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    best_model.fit(
        train_ds,
        epochs=100,
        validation_data=val_ds,
        verbose=1,
        callbacks=[early_stop]
    )

    y_test_pred = best_model.predict(test_ds, verbose=0)

    elapsed = time.time() - starting_time
    print(f"Retraining and testing completed in {elapsed:.1f}s")
    return y_test_pred


def results_evaluation(data: dict, y_test: np.ndarray, y_test_pred: np.ndarray):
    """
    Computes RMSE, MAE and R² for both the raw MET_pt baseline and the model prediction.
    """
    MET_pt    = data["MET_pt"]
    GenMET_pt = data["GenMET_pt"]

    def metrics(true, pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(true, pred)),
            "MAE":  mean_absolute_error(true, pred),
            "R2":   r2_score(true, pred),
        }

    baseline  = metrics(GenMET_pt, MET_pt)
    predicted = metrics(y_test, y_test_pred)

    print("Evaluation metrics:")
    print(f"    Baseline  RMSE={baseline['RMSE']:.4f}  MAE={baseline['MAE']:.4f}  R²={baseline['R2']:.4f}")
    print(f"    Predicted RMSE={predicted['RMSE']:.4f}  MAE={predicted['MAE']:.4f}  R²={predicted['R2']:.4f}")



if __name__ == "__main__":
    EVENT = "ZZTo2L2Nu"
    INPUT_FILE = f"CleanedDatasets/{EVENT}.parquet"
    OUTPUT_DIR = f"Results/"

    # Load data
    branches, data = read_data(INPUT_FILE)

    # Parse into feature matrix & target
    features, target = data_parsing(branches, data)

    # Grid-search with K-Fold CV (test set held-out throughout)
    hparam_grid = {
        "n_layers":      [1, 2],
        "n_units":       [32, 64, 128],
        "batch_size":    [32, 64],
        "learning_rate": [1e-3, 1e-4],
    }

    X_train, y_train, X_test, y_test, best_params = search_best_model(features, target, hparam_grid, OUTPUT_DIR)

    # Retrain best model and predict on test set
    y_test_pred = testing(X_train, y_train, X_test, best_params)

    # Evaluate and save metrics
    results_evaluation(data, y_test, y_test_pred)

    # Save scatter data for external plotting
    pd.DataFrame({
        "y_test":      y_test,
        "y_test_pred": y_test_pred.flatten(),
    }).to_parquet(os.path.join(OUTPUT_DIR, f"METpred_vs_GenMET_{EVENT}.parquet"), index=False)