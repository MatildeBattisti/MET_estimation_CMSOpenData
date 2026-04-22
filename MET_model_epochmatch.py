import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
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
        "clipnorm":            None,
        "lr_patience":         12,
        "es_patience_search":  15,
        "es_patience_retrain": 20,
        "l2_reg":              0.0,
    },
    "HToAATo2Mu2B": {
        "clipnorm":            0.3,
        "lr_patience":         10,
        "es_patience_search":  40,
        "es_patience_retrain": 50,
        "l2_reg":              0.0,
    },
}


def _read_data(file_path: str):
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


def _data_parsing(branches, data):
    """
        Converts the branch dictionary into feature matrix and target vector.
        No global StandardScaler here: standardization happens per-batch inside the model.
    """
    feature_names = [b for b in branches if b != "GenMET_pt"]
    features = np.column_stack([data[name] for name in feature_names])
    target = data["GenMET_pt"]

    print(f"Shape of features and target:")
    print(f"    (N events, N features): {features.shape}")
    print(f"    (N events, N target): {target.shape}")
    return features, target


def _create_model(input_dim: int, architecture: tuple, learning_rate: float,
                 clipnorm: float = None, l2_reg: float = 0.0):
    """
        Builds a Sequential model.

        BatchNormalization after each Dense layer normalises activations per
        mini-batch (zero mean, unit variance) and replaces a global StandardScaler.

        l2_reg applies L2 weight regularisation to each Dense layer to penalise
        large weights and reduce overfitting.

        clipnorm in Adam clips the gradient norm to prevent large destabilising
        updates.
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


def _build_tensorflow_dataset(X: np.ndarray, y: np.ndarray, batch_size: int,
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


def _model_selection(features: np.ndarray, target: np.ndarray, hparam_grid,
                      output_dir: str, EVENT: str, cfg: dict):
    """
        1. Splits data into train (80%) and test (20%). The test set is held-out
           and never seen during hyper-parameter search.
        2. Runs K-Fold CV exclusively on the training split.
        3. For each fold, data is fed as mini-batches via tf.data.Dataset.
        4. Selects best hyper-parameters by lowest average validation RMSE.
        5. Saves all hyper-parameter combinations and their RMSE to a CSV file.
        6. Saves fold histories (loss, val_loss) for the best combo for plotting.
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
    target_train_loss = None
    best_fold_histories = None
    all_results = []

    print("GRID SEARCH OVER K-FOLD CV")
    print(f"    Total combinations: {len(param_combinations)}. Folds: {kf.n_splits}\n")

    for i, combo in enumerate(param_combinations, 1):
        params = dict(zip(param_keys, combo))
        print(f"    [{i}/{len(param_combinations)}] Testing params: {params}")

        val_scores = []
        best_train_loss_per_fold = []
        combo_fold_histories = []

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

            train_ds = _build_tensorflow_dataset(x_tr,  y_tr,  params["batch_size"], shuffle=True)
            val_ds   = _build_tensorflow_dataset(x_val, y_val, params["batch_size"], shuffle=False)

            model = _create_model(
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
                patience=5, min_delta=1e-2, min_lr=1e-6, verbose=0
            )

            history = model.fit(
                train_ds,
                epochs=200,
                validation_data=val_ds,
                verbose=0,
                callbacks=[early_stop, reduce_lr]
            )

            # Epoch at which val_loss was lowest
            best_ep_idx = int(np.argmin(history.history["val_loss"]))
            train_loss_at_best = history.history["loss"][best_ep_idx]
            best_train_loss_per_fold.append(train_loss_at_best)

            # Store fold history for plotting
            combo_fold_histories.append({
                "fold": fold_idx,
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
            })

            epochs_run = len(history.history["loss"])
            y_pred = model.predict(val_ds, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_scores.append(rmse)
            fold_bar.set_postfix({"fold_rmse": f"{rmse:.4f}", "epochs": epochs_run,
                                  "train_loss@best": f"{train_loss_at_best:.4f}"})

        avg_rmse = np.mean(val_scores)
        std_rmse = np.std(val_scores)
        avg_target_train_loss = float(np.mean(best_train_loss_per_fold))
        print(f"        Avg RMSE: {avg_rmse:.4f} +/- {std_rmse:.4f}  |  "
              f"Avg train loss at best val epoch: {avg_target_train_loss:.4f}  "
              f"{[f'{v:.4f}' for v in best_train_loss_per_fold]}\n")

        row = {**params, "avg_rmse": avg_rmse, "std_rmse": std_rmse,
               "avg_target_train_loss": avg_target_train_loss}
        for j, s in enumerate(val_scores, 1):
            row[f"rmse_fold{j}"] = s
        all_results.append(row)

        if avg_rmse < best_score:
            best_score            = avg_rmse
            best_params           = params
            target_train_loss     = avg_target_train_loss
            best_fold_histories = combo_fold_histories 

    os.makedirs(output_dir, exist_ok=True)
    hparam_log_path = os.path.join(output_dir, f"hparam_search_results_{EVENT}.csv")
    pd.DataFrame(all_results).sort_values("avg_rmse").to_csv(hparam_log_path, index=False)

    elapsed = time.time() - starting_time
    print(f"BEST PARAMETERS: {best_params}")
    print(f"    Best Avg RMSE : {best_score:.4f}")
    print(f"    Target train loss (retrain): {target_train_loss:.4f}")
    print(f"    Completed in : {elapsed:.1f}s")

    # CV metrics for the best hyperparameter combo
    best_row = next(r for r in all_results if r["avg_rmse"] == best_score)

    best_cv_metrics = {
        "cv_n_folds":          kf.n_splits,
        "cv_avg_train_mse":    target_train_loss,
        "cv_avg_train_rmse":   float(np.sqrt(target_train_loss)),
        "cv_avg_val_mse":      float(best_score ** 2),
        "cv_avg_val_rmse":     best_score,
        "cv_std_val_rmse":     float(best_row["std_rmse"]),
    }

    return X_train, y_train, X_test, y_test, best_params, target_train_loss, best_fold_histories, best_cv_metrics


class StopAtTrainLoss(keras.callbacks.Callback):
    """
    Stops training as soon as the training loss (end of epoch) drops at or
    below `target_loss`. This replaces early stopping based on a validation
    split: the target is derived from CV as the mean training loss recorded
    at the best validation epoch across folds.
    """
    def __init__(self, target_loss: float):
        super().__init__()
        self.target_loss = target_loss

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get("loss", float("inf"))
        if current <= self.target_loss:
            print(f"\nReached target train loss ({current:.4f} <= "
                  f"{self.target_loss:.4f}). Stopping.")
            self.model.stop_training = True


def _retraining(X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            best_params: dict, cfg: dict,
            target_train_loss: float):
    """
        Retrains with best hyper-parameters on the FULL training set.

        Training stops as soon as the training loss reaches `target_train_loss`,
        which is the mean training loss recorded at the best validation epoch
        across CV folds. No validation split is needed: the stopping criterion
        is entirely driven by the training loss on the full dataset.

        A large epoch ceiling (500) is set as a safety net; in practice the
        StopAtTrainLoss callback fires well before that.
    """
    starting_time = time.time()

    print(f"Retraining on full training set. "
          f"Target train loss: {target_train_loss:.4f}")

    train_ds = _build_tensorflow_dataset(X_train, y_train, best_params["batch_size"], shuffle=True)
    test_ds  = _build_tensorflow_dataset(
        X_test, np.zeros(len(X_test), dtype=np.float32),
        best_params["batch_size"], shuffle=False
    )

    best_model = _create_model(
        input_dim=X_train.shape[1],
        architecture=best_params["architecture"],
        learning_rate=best_params["learning_rate"],
        clipnorm=cfg["clipnorm"],
        l2_reg=cfg["l2_reg"],
    )

    stop_at_loss = StopAtTrainLoss(target_train_loss)

    reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.5,
        patience=cfg["lr_patience"], min_delta=1e-2, min_lr=1e-6, verbose=1
    )

    history = best_model.fit(
        train_ds,
        epochs=500,                        # safety ceiling
        verbose=1,
        callbacks=[stop_at_loss, reduce_lr]
    )

    # Metrics on the full training set
    y_train_pred       = best_model.predict(train_ds, verbose=0)
    retrain_train_mse  = float(mean_squared_error(y_train, y_train_pred))
    retrain_train_rmse = float(np.sqrt(retrain_train_mse))

    # Metrics on the held-out test set
    y_test_pred   = best_model.predict(test_ds, verbose=0)
    retrain_test_mse  = float(mean_squared_error(y_test, y_test_pred))
    retrain_test_rmse = float(np.sqrt(retrain_test_mse))


    retrain_metrics = {
        "retrain_train_mse":  retrain_train_mse,
        "retrain_train_rmse": retrain_train_rmse,
        "retrain_test_mse":   retrain_test_mse,
        "retrain_test_rmse":  retrain_test_rmse,
    }

    print(f"  Train  MSE={retrain_train_mse:.4f}  RMSE={retrain_train_rmse:.4f}")
    print(f"  Test   MSE={retrain_test_mse:.4f}   RMSE={retrain_test_rmse:.4f}")

    elapsed = time.time() - starting_time
    print(f"Retraining and testing completed in {elapsed:.1f}s")
    return y_test_pred, history, best_model, retrain_metrics


def _results_evaluation(data: dict, y_test: np.ndarray, y_test_pred: np.ndarray):
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
    return


def _feature_importance_shap(model, X_test: np.ndarray, feature_names: list,
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
    return


def _save_run_summary(event: str, best_params: dict, cfg: dict,
                      best_cv_metrics: dict, retrain_metrics: dict,
                      y_test: np.ndarray, y_test_pred: np.ndarray,
                      output_dir: str):
    """
    Saves a single-row CSV summarising the full run:
      - best hyperparameters
      - training config (clipnorm, lr_patience, etc.)
      - CV metrics for the best combo (avg train/val MSE, RMSE, n_folds)
      - retraining train MSE/RMSE
      - test set MSE/RMSE
    """
    test_mse  = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    row = {
        "event": event,
        # --- best hyperparameters ---
        "architecture":   str(best_params["architecture"]),
        "batch_size":     best_params["batch_size"],
        "learning_rate":  best_params["learning_rate"],
        # --- training config ---
        "clipnorm":            cfg["clipnorm"],
        "lr_patience":         cfg["lr_patience"],
        "es_patience_search":  cfg["es_patience_search"],
        "es_patience_retrain": cfg["es_patience_retrain"],
        "l2_reg":              cfg["l2_reg"],
        # --- CV metrics (best combo) ---
        "cv_n_folds":          best_cv_metrics["cv_n_folds"],
        "cv_avg_train_mse":    best_cv_metrics["cv_avg_train_mse"],
        "cv_avg_train_rmse":   best_cv_metrics["cv_avg_train_rmse"],
        "cv_avg_val_mse":      best_cv_metrics["cv_avg_val_mse"],
        "cv_avg_val_rmse":     best_cv_metrics["cv_avg_val_rmse"],
        "cv_std_val_rmse":     best_cv_metrics["cv_std_val_rmse"],
        # --- retraining metrics (full train set) ---
        "retrain_train_mse":   retrain_metrics["retrain_train_mse"],
        "retrain_train_rmse":  retrain_metrics["retrain_train_rmse"],
        # --- test set metrics ---
        "test_mse":   test_mse,
        "test_rmse":  test_rmse,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"run_summary_{event}.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"Run summary saved to {path}")
    return


if __name__ == "__main__":
    EVENT = "HToAATo2Mu2B"
    INPUT_FILE = f"CleanedDatasets/{EVENT}.parquet"
    OUTPUT_DIR = f"Results/"

    print("GPUs available:", tf.config.list_physical_devices("GPU"))

    # Load per-dataset configuration
    cfg = DATASET_CFG.get(EVENT, DATASET_CFG["ZZTo2L2Nu"])
    print(f"    Using config for '{EVENT}': {cfg}\n")

    # Load data
    branches, data = _read_data(INPUT_FILE)

    # Parse into feature matrix & target
    features, target = _data_parsing(branches, data)

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
            (128,), (256,),
            (256, 128), (512, 256), (256, 64),
            (128, 64, 32),# (256, 128, 64), (512, 256, 128),
        ],
        "batch_size":    [64, 128],
        "learning_rate": [1e-3, 5e-4],
    }

    X_train, y_train, X_test, y_test, best_params, target_train_loss, best_fold_histories, best_cv_metrics = _model_selection(
        features, target, hparam_grid, OUTPUT_DIR, EVENT, cfg
    )

    rows = []
    for fh in best_fold_histories:
        for ep, (tl, vl) in enumerate(zip(fh["loss"], fh["val_loss"]), 1):
            rows.append({"fold": fh["fold"], "epoch": ep, "loss": tl, "val_loss": vl})

    pd.DataFrame(rows).to_parquet(
        os.path.join(OUTPUT_DIR, f"fold_histories_{EVENT}.parquet"), index=False
    )

    # Retrain best model and predict on test set
    y_test_pred, history, best_model, retrain_metrics = _retraining(
        X_train, y_train, X_test, best_params, cfg, target_train_loss
    )

    # Evaluate and save metrics
    _results_evaluation(data, y_test, y_test_pred)

    _save_run_summary(
        event=EVENT,
        best_params=best_params,
        cfg=cfg,
        best_cv_metrics=best_cv_metrics,
        retrain_metrics=retrain_metrics,
        y_test=y_test,
        y_test_pred=y_test_pred,
        output_dir=OUTPUT_DIR,
    )

    # Feature importance via SHAP
    feature_names = [b for b in branches if b != "GenMET_pt"]
    _feature_importance_shap(best_model, X_test, feature_names, EVENT, OUTPUT_DIR)

    # Save learning curves for external plotting
    pd.DataFrame(history.history).to_parquet(
        os.path.join(OUTPUT_DIR, f"learning_curves_{EVENT}.parquet"), index=False
    )

    # Save scatter data for external plotting
    pd.DataFrame({
        "y_test":      y_test,
        "y_test_pred": y_test_pred.flatten(),
    }).to_parquet(os.path.join(OUTPUT_DIR, f"METpred_vs_GenMET_{EVENT}.parquet"), index=False)
