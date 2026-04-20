import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Input, Dense, BatchNormalization, ReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import shap
import matplotlib.pyplot as plt


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

# Per-dataset training configuration
DATASET_CFG = {
    "ZZTo2L2Nu": {
        "clipnorm":            None,  # no clipping: larger natural gradients
        "lr_patience":         12,    # conservative LR reduction
        "es_patience_search":  15,
        "es_patience_retrain": 20,
        "l2_reg":              0.0,   # enough data, no L2 needed
    },
    "HToAATo2Mu2B": {
        "clipnorm":            1.0,   # clip gradients to stabilise small-dataset training
        "lr_patience":         5,
        "es_patience_search":  20,    # more patience: noisy loss with few events
        "es_patience_retrain": 30,
        "l2_reg":              1e-4,  # L2 regularisation to avoid overfitting
    },
}


def read_data(file_path: str):
    """
        Reads a .parquet file. The event name is inferred from the filename
        (e.g. 'ZZTo2L2Nu.parquet' -> event 'ZZTo2L2Nu') to select the right branches.
    """
    filename = os.path.basename(file_path)
    event_name, ext = os.path.splitext(filename)

    if ext != ".parquet":
        raise ValueError(f"Expected a .parquet file, got '{ext}'.")

    df = pd.read_parquet(file_path)

    if event_name in BRANCH_MAP:
        branches = BRANCH_MAP[event_name]
        print(f"Loaded dataset '{event_name}' with specific branches.")
    else:
        branches = DEFAULT_BRANCHES
        print(f"Loaded dataset '{event_name}' with default branches.")

    missing = [b for b in branches if b not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet file: {missing}")

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


def create_model(input_dim: int, architecture: tuple, learning_rate: float,
                 clipnorm: float = None, l2_reg: float = 0.0):
    """
        Builds a Sequential model.

        BatchNormalization after each Dense layer normalises activations per
        mini-batch (zero mean, unit variance) and replaces a global StandardScaler.

        l2_reg applies L2 weight regularisation to each Dense layer to penalise
        large weights and reduce overfitting (used for HToAATo2Mu2B).

        clipnorm in Adam clips the gradient norm to prevent large destabilising
        updates (used for HToAATo2Mu2B, disabled for ZZTo2L2Nu).
    """
    regularizer = l2(l2_reg) if l2_reg > 0.0 else None

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for n_units in architecture:
        model.add(Dense(n_units, kernel_regularizer=regularizer))
        model.add(BatchNormalization())
        model.add(keras.layers.ReLU())

    model.add(Dense(1, activation="linear"))

    optimizer_kwargs = {"learning_rate": learning_rate}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm

    model.compile(
        optimizer=Adam(**optimizer_kwargs),
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


def search_best_model(features: np.ndarray, target: np.ndarray, hparam_grid,
                      output_dir: str, EVENT: str, cfg: dict):
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
    all_results = []

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

            train_ds = build_tf_dataset(x_tr,  y_tr,  params["batch_size"], shuffle=True)
            val_ds   = build_tf_dataset(x_val, y_val, params["batch_size"], shuffle=False)

            model = create_model(
                input_dim=x_tr.shape[1],
                architecture=params["architecture"],
                learning_rate=params["learning_rate"],
                clipnorm=cfg["clipnorm"],
                l2_reg=cfg["l2_reg"],
            )

            early_stop = EarlyStopping(
                monitor="val_loss", patience=cfg["es_patience_search"],
                restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_delta=1.0, min_lr=1e-6, verbose=0
            )

            history = model.fit(
                train_ds,
                epochs=200,
                validation_data=val_ds,
                verbose=0,
                callbacks=[early_stop, reduce_lr]
            )

            epochs_run = len(history.history["loss"])
            y_pred = model.predict(val_ds, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_scores.append(rmse)
            fold_bar.set_postfix({"fold_rmse": f"{rmse:.4f}", "epochs": epochs_run})

        avg_rmse = np.mean(val_scores)
        std_rmse = np.std(val_scores)
        print(f"        Avg RMSE: {avg_rmse:.4f} +/- {std_rmse:.4f}\n")

        row = {**params, "avg_rmse": avg_rmse, "std_rmse": std_rmse}
        for j, s in enumerate(val_scores, 1):
            row[f"rmse_fold{j}"] = s
        all_results.append(row)

        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params

    os.makedirs(output_dir, exist_ok=True)
    hparam_log_path = os.path.join(output_dir, f"hparam_search_results_{EVENT}.csv")
    pd.DataFrame(all_results).sort_values("avg_rmse").to_csv(hparam_log_path, index=False)

    elapsed = time.time() - starting_time
    print(f"BEST PARAMETERS: {best_params}")
    print(f"    Best Avg RMSE : {best_score:.4f}")
    print(f"    Completed in : {elapsed:.1f}s")

    return X_train, y_train, X_test, y_test, best_params


def testing(X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, best_params: dict, cfg: dict):
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
    val_ds   = build_tf_dataset(X_val, y_val, best_params["batch_size"], shuffle=False)
    test_ds  = build_tf_dataset(
        X_test, np.zeros(len(X_test), dtype=np.float32),
        best_params["batch_size"], shuffle=False
    )

    best_model = create_model(
        input_dim=X_tr.shape[1],
        architecture=best_params["architecture"],
        learning_rate=best_params["learning_rate"],
        clipnorm=cfg["clipnorm"],
        l2_reg=cfg["l2_reg"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=cfg["es_patience_retrain"],
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=cfg["lr_patience"], min_delta=1.0, min_lr=1e-6, verbose=1
    )

    history = best_model.fit(
        train_ds,
        epochs=300,
        validation_data=val_ds,
        verbose=1,
        callbacks=[early_stop, reduce_lr]
    )

    y_test_pred = best_model.predict(test_ds, verbose=0)

    elapsed = time.time() - starting_time
    print(f"Retraining and testing completed in {elapsed:.1f}s")
    return y_test_pred, history, best_model


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


def feature_importance_shap(model, X_test: np.ndarray, feature_names: list,
                             event: str, output_dir: str):
    """
        Computes SHAP values using GradientExplainer (more compatible with Keras 3 + TF > 2.4).
        Plots:
        - beeswarm summary plot (global importance + direction of effect)
        - bar plot (mean |SHAP| per feature)
        Also saves the ranked importance to a CSV.
    """

    sample     = X_test[:500].astype(np.float32)
    background = X_test[:100].astype(np.float32)

    explainer   = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(sample)
    sv = np.array(shap_values)[..., 0]

    os.makedirs(output_dir, exist_ok=True)

    # Beeswarm plot
    shap.summary_plot(sv, sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — {event}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_summary_{event}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # Bar plot (mean |SHAP|)
    shap.summary_plot(sv, sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {event}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_bar_{event}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # Ranked importance
    mean_shap = np.abs(sv).mean(axis=0)
    ranked    = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)

    print(f"Feature importance (mean |SHAP|) for {event}:")
    for name, val in ranked:
        print(f"    {name:<35} {val:.4f}")

    pd.DataFrame(ranked, columns=["feature", "mean_abs_shap"]).to_csv(
        os.path.join(output_dir, f"shap_importance_{event}.csv"), index=False
    )


if __name__ == "__main__":
    EVENT = "HToAATo2Mu2B"
    INPUT_FILE = f"CleanedDatasets/{EVENT}.parquet"
    OUTPUT_DIR = f"Results/"

    print("GPUs available:", tf.config.list_physical_devices("GPU"))

    # Load per-dataset configuration
    cfg = DATASET_CFG.get(EVENT, DATASET_CFG["ZZTo2L2Nu"])
    print(f"    Using config for '{EVENT}': {cfg}\n")

    # Load data
    branches, data = read_data(INPUT_FILE)

    # Parse into feature matrix & target
    features, target = data_parsing(branches, data)

    # Grid-search with K-Fold CV (test set held-out throughout)
    #hparam_grid = {
    #    "architecture": [
    #        # shallow
    #        (32,), (64,), (128,), (256,),
    #        # 2 layer
    #        (128, 64), (256, 128), (128, 32), (64, 32),
    #        # 3 layer
    #        (256, 128, 64), (128, 64, 32), (256, 64, 32),
    #        # 4 layer
    #        (256, 128, 64, 32), (128, 128, 64, 32),
    #    ],
    #    "batch_size":    [32, 62, 128, 256],#[256, 512, 1024],
    #    "learning_rate": [1e-3, 5e-4, 1e-4],
    #}
    hparam_grid = {
        "architecture": [
            (32,), (64,), (128,), (256,),
            (128, 64), (256, 128), (128, 32), (64, 32),
        ],
        "batch_size":    [32, 64, 128, 256],
        "learning_rate": [1e-3, 5e-4, 1e-4],
    }

    X_train, y_train, X_test, y_test, best_params = search_best_model(
        features, target, hparam_grid, OUTPUT_DIR, EVENT, cfg
    )

    # Retrain best model and predict on test set
    y_test_pred, history, best_model = testing(X_train, y_train, X_test, best_params, cfg)

    # Evaluate and save metrics
    results_evaluation(data, y_test, y_test_pred)

    # Feature importance via SHAP
    feature_names = [b for b in branches if b != "GenMET_pt"]
    feature_importance_shap(best_model, X_test, feature_names, EVENT, OUTPUT_DIR)

    # Save learning curves for external plotting
    pd.DataFrame(history.history).to_parquet(
        os.path.join(OUTPUT_DIR, f"learning_curves_{EVENT}.parquet"), index=False
    )

    # Save scatter data for external plotting
    pd.DataFrame({
        "y_test":      y_test,
        "y_test_pred": y_test_pred.flatten(),
    }).to_parquet(os.path.join(OUTPUT_DIR, f"METpred_vs_GenMET_{EVENT}.parquet"), index=False)
