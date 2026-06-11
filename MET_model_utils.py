# CONTROLLA SE LASCIARE SOLO DENTRO PROCESS
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pyexpat import features
import warnings
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
#
import gc
import multiprocessing as mp
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Input, Dense, BatchNormalization, ReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import shap
import matplotlib.pyplot as plt


def _read_data(file_path: str, branch_map: dict, default_branches: list):
    """
        Reads a .parquet file. The event name is inferred from the filename
        (e.g. 'ZZTo2L2Nu.parquet' -> event 'ZZTo2L2Nu') to select the right branches.
    """
    filename = os.path.basename(file_path)
    event_name, ext = os.path.splitext(filename)

    if ext != ".parquet":
        raise ValueError(f"Expected a .parquet file, got '{ext}'.")

    df = pd.read_parquet(file_path)

    if event_name in branch_map:
        branches = branch_map[event_name]
        print(f"Loaded dataset '{event_name}' with specific branches.")
    else:
        branches = default_branches
        print(f"Loaded dataset '{event_name}' with default branches.")

    missing = [b for b in branches if b not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet file: {missing}")

    data = {b: df[b].to_numpy() for b in branches}
    return branches, data


def _data_parsing(branches, data):
    """
        Converts the branch dictionary into feature matrix and target vector.
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

        When non-zero, l2_reg applies L2 weight regularization to each dense layer to penalize
        large weights and reduce overfitting.

        When non-zero, clipnorm in Adam clips the gradient norm to prevent large destabilising
        updates.
    """
    regularizer = l2(l2_reg) if l2_reg > 0.0 else None

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for n_units in architecture:
        model.add(Dense(n_units, kernel_regularizer=regularizer))
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


def _standardize(x_tr: np.ndarray, x_val: np.ndarray):
    """
        Computes mean and std on x_tr only, then applies the same transformation
        to both x_tr and x_val. Columns with zero std are left unchanged (std=1).
    
        Returns the scaled arrays and the scaling parameters so that the caller
        can apply the same transform to new data (e.g. the test set at retraining).
    """
    mean = x_tr.mean(axis=0)
    std  = x_tr.std(axis=0)
    std  = np.where(std == 0, 1.0, std)
 
    x_tr_scaled  = (x_tr  - mean) / std
    x_val_scaled = (x_val - mean) / std
 
    return x_tr_scaled, x_val_scaled, mean, std


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


def _run_fold(queue, x_tr, y_tr, x_val, y_val, params, cfg, fold_idx):
    """
        Executed in a child process (spawn).

        The whole TF graph and the model weights are automatically deallocated
        when the process ends, solving the memory leak of ops.py found in
        Keras 3 + TensorFlow 2.19 and not solved by clear_session().

        With spawn a clean interpreter is started and the father's imports
        are not inherited. TensorFlow has to be imported after os.environ.

        The subprocess communicates with the father using mp.Queue using only
        primitive types (float, int, list). If an exception occours it still
        writes a dictionary with an 'error' key so that the father does not
        get killed on queue.get().
    """
    # CHECK QUALI USATI DAVVERO
    import os, sys
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    
    os.environ["GRPC_VERBOSITY"]       = "ERROR"
    os.environ["GLOG_minloglevel"]     = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
    os.environ["CUDA_MODULE_LOADING"]          = "LAZY"
    warnings.filterwarnings("ignore")

    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)
    ###
    
    try:
        import numpy as np
        import keras
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.metrics import mean_squared_error

        model = _create_model(
            input_dim     = x_tr.shape[1],
            architecture  = params["architecture"],
            learning_rate = params["learning_rate"],
            clipnorm      = cfg["clipnorm"],
            l2_reg        = cfg["l2_reg"],
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=cfg["es_patience_search"],
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg["rlrop_factor"],
            patience=cfg["lr_patience"],
            min_delta=cfg["rlrop_min_delta"],
            min_lr=cfg["rlrop_min_lr"],
            verbose=0
        )

        history = model.fit(
            x_tr, y_tr,
            batch_size      = params["batch_size"],
            epochs          = cfg["max_epochs_search"],
            validation_data = (x_val, y_val),
            verbose         = 0,
            callbacks       = [early_stop, reduce_lr],
            shuffle=True
        )

        best_ep_idx        = int(np.argmin(history.history["val_loss"]))
        train_loss_at_best = float(history.history["loss"][best_ep_idx])
        y_pred             = model.predict(x_val, batch_size=params["batch_size"], verbose=0)
        rmse               = float(np.sqrt(mean_squared_error(y_val, y_pred)))

        queue.put({
            "fold":               fold_idx,
            "rmse":               rmse,
            "train_loss_at_best": train_loss_at_best,
            "loss":     [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "epochs_run":         len(history.history["loss"]),
        })
        
    except Exception as e:
        # Garantees thst the father does not get blocked on queue.get() event if the child crashes.
        queue.put({"error": str(e), "fold": fold_idx})
    return
    

def _model_selection(features: np.ndarray, target: np.ndarray, hparam_grid,
                      output_dir: str, EVENT: str, cfg: dict):
    """
        1. Splits data into train (80%) and test (20%). The test set is held-out
           and never seen during hyper-parameter search.
        2. Standardizes each fold independently: mean and std are computed on
           x_tr only and applied to both x_tr and x_val, avoiding any leakage
           from validation data into the scaling parameters.
        3. Each fold is executed in a separate subprocess (spawn) to work around
           the Keras 3 + TF 2.19 memory leak in ops.py that clear_session()
           cannot resolve. The subprocess exits after the fold, freeing all TF
           memory automatically.
        4. Selects best hyper-parameters by lowest average validation RMSE.
        5. Saves only the top-10 hyper-parameter combinations to a CSV file.
        6. Keeps fold histories for the top-10 combos by avg validation RMSE.
        7. After the search, computes scaling parameters on the full X_train and
           applies them to both X_train and X_test for use in retraining. These
           parameters are returned so they can be saved and reused on new data.
    
        Memory management:
        - topk_buffer holds at most TOP_K (10) (avg_rmse, row, fold_histories)
          triples. When a new entry exceeds the limit the worst entry is popped
          and its histories freed, guarding against the edge case where it shares
          the same object as best_fold_histories.
        - all_results is never accumulated: each combo row goes straight into
          topk_buffer, keeping memory O(TOP_K) throughout.
    """
    starting_time = time.time()

    indices = np.arange(len(features))
    train_idx, test_idx = train_test_split(
        indices, test_size=cfg["test_size"], random_state=cfg["random_seed_split"]
    )
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = target[train_idx],   target[test_idx]
    
    # GenMET ha a skewed distribution
    if cfg["target_transform"] == "log1p":
        y_train = np.log1p(y_train)
        y_test  = np.log1p(y_test)
 
    kf = KFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["random_seed_kfold"])
    param_combinations = list(product(*hparam_grid.values()))
    param_keys         = list(hparam_grid.keys())
 
    best_score          = float("inf")
    best_params         = None
    target_train_loss   = None
    best_fold_histories = None
 
    TOP_K       = 10
    topk_buffer = []
 
    print("GRID SEARCH OVER K-FOLD CV")
    print(f"    Total combinations: {len(param_combinations)}. Folds: {kf.n_splits}\n")
 
    for i, combo in enumerate(param_combinations, 1):
        params = dict(zip(param_keys, combo))
        print(f"    [{i}/{len(param_combinations)}] Testing params: {params}")
 
        val_scores               = []
        best_train_loss_per_fold = []
        combo_fold_histories     = []
 
        fold_bar = tqdm(
            enumerate(kf.split(X_train), 1),
            total=kf.n_splits,
            desc="        Folds",
            leave=False,
            unit="fold"
        )
 
        for fold_idx, (fold_train_idx, fold_val_idx) in fold_bar:
            x_tr,  x_val = X_train[fold_train_idx], X_train[fold_val_idx]
            y_tr,  y_val = y_train[fold_train_idx],  y_train[fold_val_idx]
 
            # Per-fold standardization
            x_tr, x_val, _, _ = _standardize(x_tr, x_val)

            # Subprocess per fold
            queue = mp.Queue()
            p = mp.Process(
                target=_run_fold,
                args=(queue, x_tr, y_tr, x_val, y_val, params, cfg, fold_idx)
            )
            p.start()
            result = queue.get()
            p.join()
 
            if p.exitcode != 0:
                raise RuntimeError(
                    f"Fold {fold_idx} subprocess exited with code {p.exitcode}"
                )
            if "error" in result:
                raise RuntimeError(
                    f"Fold {fold_idx} failed: {result['error']}"
                )
 
            best_train_loss_per_fold.append(result["train_loss_at_best"])
            val_scores.append(result["rmse"])
            combo_fold_histories.append({
                "fold":     result["fold"],
                "loss":     result["loss"],
                "val_loss": result["val_loss"],
            })
            fold_bar.set_postfix({
                "fold_rmse":       f"{result['rmse']:.4f}",
                "epochs":          result["epochs_run"],
                "train_loss_at_best": f"{result['train_loss_at_best']:.4f}",
            })
 
        avg_rmse              = np.mean(val_scores)
        std_rmse              = np.std(val_scores)
        avg_target_train_loss = float(np.mean(best_train_loss_per_fold))
 
        print(f"        Avg RMSE: {avg_rmse:.4f} +/- {std_rmse:.4f}  |  "
              f"Avg train loss at best val epoch: {avg_target_train_loss:.4f}  "
              f"{[f'{v:.4f}' for v in best_train_loss_per_fold]}\n")
 
        row = {**params, "avg_rmse": avg_rmse, "std_rmse": std_rmse,
               "avg_target_train_loss": avg_target_train_loss}
        for j, s in enumerate(val_scores, 1):
            row[f"rmse_fold{j}"] = s
 
        # Top-K buffer
        topk_buffer.append((avg_rmse, row, combo_fold_histories))
        topk_buffer.sort(key=lambda x: x[0])
 
        if len(topk_buffer) > TOP_K:
            _, _, worst_histories = topk_buffer.pop()
            if worst_histories is not best_fold_histories:
                del worst_histories
 
        if avg_rmse < best_score:
            best_score          = avg_rmse
            best_params         = params
            target_train_loss   = avg_target_train_loss
            best_fold_histories = combo_fold_histories
 
        gc.collect()
 
    os.makedirs(output_dir, exist_ok=True)
    hparam_log_path = os.path.join(output_dir, f"hparam_search_results_{EVENT}.csv")
    topk_rows = [row for _, row, _ in topk_buffer]
    pd.DataFrame(topk_rows).sort_values("avg_rmse").to_csv(hparam_log_path, index=False)
 
    elapsed = time.time() - starting_time
    print(f"BEST PARAMETERS: {best_params}")
    print(f"    Best Avg RMSE : {best_score:.4f}")
    print(f"    Target train loss (retrain): {target_train_loss:.4f}")
    print(f"    Completed in : {elapsed:.1f}s")
 
    best_row = next(row for _, row, _ in topk_buffer if row["avg_rmse"] == best_score)
 
    best_cv_metrics = {
        "cv_n_folds":        kf.n_splits,
        "cv_avg_train_mse":  target_train_loss,
        "cv_avg_train_rmse": float(np.sqrt(target_train_loss)),
        "cv_avg_val_mse":    float(best_score ** 2),
        "cv_avg_val_rmse":   best_score,
        "cv_std_val_rmse":   float(best_row["std_rmse"]),
    }
 
    return (X_train, y_train, X_test, y_test,
            best_params, target_train_loss,
            best_fold_histories, best_cv_metrics,
            topk_buffer, test_idx)
    

class StopAtTrainLoss(keras.callbacks.Callback):
    """
        Stops training as soon as the training loss (end of epoch) drops at or
        below target_loss. This replaces early stopping based on a validation
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

    starting_time = time.time()

    X_train, X_test, _, _ = _standardize(X_train, X_test)

    print(f"Retraining on full training set. "
          f"Target train loss: {target_train_loss:.4f}")

    train_ds = _build_tensorflow_dataset(X_train, y_train, best_params["batch_size"], shuffle=True)

    best_model = _create_model(
        input_dim=X_train.shape[1],
        architecture=best_params["architecture"],
        learning_rate=best_params["learning_rate"],
        clipnorm=cfg["clipnorm"],
        l2_reg=cfg["l2_reg"],
    )

    stop_at_loss = StopAtTrainLoss(target_train_loss)
    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=cfg["rlrop_factor"],
        patience=cfg["lr_patience"],
        min_delta=cfg["rlrop_min_delta"],
        min_lr=cfg["rlrop_min_lr"],
        verbose=1
    )

    history = best_model.fit(
        train_ds,
        epochs=cfg["max_epochs_retrain"],
        verbose=1,
        callbacks=[stop_at_loss, reduce_lr]
    )

    # Predictions in the transformed space
    y_train_pred = best_model.predict(X_train, batch_size=best_params["batch_size"], verbose=0).flatten()
    y_test_pred  = best_model.predict(X_test,  batch_size=best_params["batch_size"], verbose=0).flatten()

    # De-transformation, conditional to the dataset used
    if cfg["target_transform"] == "log1p":
        y_train_GeV = np.expm1(y_train)
        y_test_GeV  = np.expm1(y_test)
        y_train_pred_GeV = np.expm1(y_train_pred)
        y_test_pred_GeV  = np.expm1(y_test_pred)
    else:
        y_train_GeV      = y_train
        y_test_GeV       = y_test
        y_train_pred_GeV = y_train_pred
        y_test_pred_GeV  = y_test_pred

    retrain_metrics = {
        "retrain_train_mse":  float(mean_squared_error(y_train_GeV, y_train_pred_GeV)),
        "retrain_train_rmse": float(np.sqrt(mean_squared_error(y_train_GeV, y_train_pred_GeV))),
        "retrain_test_mse":   float(mean_squared_error(y_test_GeV,  y_test_pred_GeV)),
        "retrain_test_rmse":  float(np.sqrt(mean_squared_error(y_test_GeV,  y_test_pred_GeV))),
    }

    print(f"  Train  MSE={retrain_metrics['retrain_train_mse']:.4f}  RMSE={retrain_metrics['retrain_train_rmse']:.4f}")
    print(f"  Test   MSE={retrain_metrics['retrain_test_mse']:.4f}   RMSE={retrain_metrics['retrain_test_rmse']:.4f}")

    elapsed = time.time() - starting_time
    print(f"Retraining and testing completed in {elapsed:.1f}s")

    return y_test_pred_GeV, y_test_GeV, history, best_model, retrain_metrics, X_test


def _feature_importance_shap(model, X_test: np.ndarray, feature_names: list,
                             event: str, output_dir: str):
    """
        Computes SHAP values using GradientExplainer.
        Plots:
        - beeswarm summary plot (global importance + direction of effect)
        - bar plot (mean |SHAP| per feature)
        Saves the ranked importance to a CSV.
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
        # best hyperparameters
        "architecture":   str(best_params["architecture"]),
        "batch_size":     best_params["batch_size"],
        "learning_rate":  best_params["learning_rate"],
        # training config
        **{f"cfg_{k}": v for k, v in cfg.items()},
        # CV metrics (best combo)
        "cv_n_folds":          best_cv_metrics["cv_n_folds"],
        "cv_avg_train_mse":    best_cv_metrics["cv_avg_train_mse"],
        "cv_avg_train_rmse":   best_cv_metrics["cv_avg_train_rmse"],
        "cv_avg_val_mse":      best_cv_metrics["cv_avg_val_mse"],
        "cv_avg_val_rmse":     best_cv_metrics["cv_avg_val_rmse"],
        "cv_std_val_rmse":     best_cv_metrics["cv_std_val_rmse"],
        # retraining metrics
        "retrain_train_mse":   retrain_metrics["retrain_train_mse"],
        "retrain_train_rmse":  retrain_metrics["retrain_train_rmse"],
        # test set metrics
        "test_mse":   test_mse,
        "test_rmse":  test_rmse,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"run_summary_{event}.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"Run summary saved to {path}")
    return