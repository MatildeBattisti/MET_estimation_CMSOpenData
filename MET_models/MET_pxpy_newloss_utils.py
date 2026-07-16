import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
import uproot
import gc
import json
import multiprocessing as mp
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Input, Dense, Dropout, ReLU
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


def _read_data(file_path: str):
    """
        Reads the 'Events' TTree from a .root file and returns
        all branches as a branch/data pair.
    """
    filename = os.path.basename(file_path)
    _, ext = os.path.splitext(filename)

    if ext != ".root":
        raise ValueError(f"Expected a .root file, got '{ext}'.")

    with uproot.open(file_path) as root_file:
        if "Events" not in root_file:
            raise ValueError("TTree 'Events' not found in the ROOT file.")

        tree = root_file["Events"]

        # Load all branches into a pandas DataFrame
        df = tree.arrays(library="pd")

    if "GenMET_px" not in df.columns or "GenMET_py" not in df.columns:
        raise ValueError(
            "Columns 'GenMET_px' and/or 'GenMET_py' (targets) not found in the ROOT file."
        )

    branches = list(df.columns)
    data = {b: df[b].to_numpy() for b in branches}

    print(
        f"Loaded dataset '{filename}' "
        f"(tree 'Events') - {len(branches)} branches, {len(df)} events."
    )
    return branches, data


def _data_parsing(branches: list, data: dict, cfg: dict):
    """
        Builds feature matrix X, target matrix y with shape (N, 2) for
        [GenMET_px, GenMET_py], and MET_px / MET_py arrays (used when
        target_transform requires them).

        cfg["target_transform"]:
          "none"     -> target = [GenMET_px, GenMET_py];      MET_p{x,y} are features
          "residual" -> target = MET_p{x,y} - GenMET_p{x,y};  MET_p{x,y} not a feature (leakage)
    """
    TARGET_COLS = ["GenMET_px", "GenMET_py"]
    MET_COLS    = ["MET_px",    "MET_py"]
    transform   = cfg.get("target_transform", "none")

    for col in TARGET_COLS + MET_COLS:
        if col not in branches:
            raise ValueError(f"Column '{col}' not found.")

    MET_used_in_target = transform in ("residual")
    exclude = set(TARGET_COLS) | (set(MET_COLS) if MET_used_in_target else set())

    feature_names = [b for b in branches if b not in exclude]
    features      = np.column_stack([data[name] for name in feature_names])
    GenMET_px = data["GenMET_px"]
    GenMET_py = data["GenMET_py"]
    MET_px    = data["MET_px"]
    MET_py    = data["MET_py"]

    if transform == "none":
        target = np.column_stack([GenMET_px, GenMET_py])
    elif transform == "residual":
        target = np.column_stack([
            MET_px - GenMET_px,
            MET_py - GenMET_py,
        ])
    else:
        raise ValueError(f"target_transform unknown: '{transform}'")

    MET_pxpy = np.column_stack([MET_px, MET_py])

    print(f"target_transform:     '{transform}'")
    print(f"MET_p(x,y) as feature: {not MET_used_in_target}")
    print(f"    features shape:    {features.shape}")
    print(f"    target   shape:    {target.shape}")
    return features, target, MET_pxpy, feature_names


def _pt_penalized_loss(genmet_threshold: float = 20.0,
                       lambda_pt: float = 1.0,
                       target_transform: str = "none",
                       target_variance: float = 1.0):
    """
        Defines a combined, dimensionless loss:

            loss = MSE(target, y_pred) / target_variance
                 + lambda_pt * 1{|GenMET| > genmet_threshold}
                             * (|MET_pred| / |GenMET| - 1)^2

        
        Supported target_transform: "none", "residual":
          - "none": the network target is [GenMET_px, GenMET_py], so y_true
            has shape (N, 2).
          - "residual": the network target is (MET - GenMET) packed as
            [residual_px, residual_py].
            Since the residual alone is not enough to recover the absolute GenMET,
            MET_px/MET_py must be packed as extra columns of y_true: shape (N, 4) =
            [residual_px, residual_py, MET_px, MET_py]. The first 2 columns
            are used for the MSE term, the last 2 are used only to
            reconstruct GenMET/MET_pred for the relative penalty.

        The genmet_threshold cut acts on GenMET_pt (in GeV), before the ratio is computed.
        Events below threshold have their GenMET replaced by a safe placeholder
        (1.0) prior to the division, so the ratio is never actually evaluated
        near a zero denominator; their contribution to rel_loss is then explicitly zeroed.
        This avoids 0 * inf = nan silently propagating into the loss.

        target_variance is the training-set variance averaged over both
        components (px, py), matching how the MSE term itself is averaged
        over both components; it makes the MSE term dimensionless and
        comparable to the (dimensionless) relative pT-penalty term.
    """
    if target_transform not in ("none", "residual"):
        raise ValueError(
            f"pt-penalized loss does not support target_transform='{target_transform}' "
            "(supported: 'none', 'residual')."
        )

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if target_transform == "none":
            target = y_true  # (N, 2)
            mse = tf.reduce_mean(tf.square(target - y_pred))

            gen_px, gen_py   = target[:, 0], target[:, 1]
            pred_px, pred_py = y_pred[:, 0], y_pred[:, 1]

        else:  # residual: y_true is (N, 4) = [residual_px, residual_py, MET_px, MET_py]
            target   = y_true[:, :2]
            MET_pxpy = y_true[:, 2:4]
            mse = tf.reduce_mean(tf.square(target - y_pred))

            # GenMET = MET - residual
            gen_px  = MET_pxpy[:, 0] - target[:, 0]
            gen_py  = MET_pxpy[:, 1] - target[:, 1]
            pred_px = MET_pxpy[:, 0] - y_pred[:, 0]
            pred_py = MET_pxpy[:, 1] - y_pred[:, 1]

        # Non-dimensionalize the MSE term (GeV^2 -> dimensionless)
        mse_dimless = mse / target_variance

        gen_pt  = tf.sqrt(tf.square(gen_px) + tf.square(gen_py))
        pred_pt = tf.sqrt(tf.square(pred_px) + tf.square(pred_py))

        # Cut on GenMET_pt
        mask        = gen_pt > genmet_threshold
        safe_gen_pt = tf.where(mask, gen_pt, tf.ones_like(gen_pt))
        rel_term = tf.where(
            mask,
            tf.square(pred_pt / safe_gen_pt - 1.0),
            tf.zeros_like(gen_pt),
        )

        n_uncut  = tf.reduce_sum(tf.cast(mask, tf.float32))
        rel_loss = tf.reduce_sum(rel_term) / tf.maximum(n_uncut, 1.0)

        return mse_dimless + lambda_pt * rel_loss

    return loss_fn


def _target_mse_metric(target_transform: str = "none"):
    """
        Defines the MSE metric restricted to the actual network target
        (first 2 columns of y_true when packed).

        It is kept separate from the loss so it's comparable across
        runs/transforms and unaffected by lambda_pt.
    """
    def target_mse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        target = y_true if target_transform == "none" else y_true[:, :2]
        return tf.reduce_mean(tf.square(target - y_pred))

    return target_mse


def _create_model(input_dim: int, architecture: tuple, learning_rate: float,
                  clipnorm: float = None, l2_reg: float = 0.0,
                  dropout_rate: float = 0.0,
                  genmet_threshold: float = 20.0,
                  lambda_pt: float = 1.0,
                  target_transform: str = "none",
                  target_variance: float = 1.0):
    """
        Builds a Sequential model.

        When non-zero, l2_reg applies L2 weight regularization to each dense layer.

        When non-zero, dropout_rate adds a Dropout layer after each hidden ReLU.

        When non-zero, clipnorm in Adam clips the gradient norm to prevent large
        destabilising updates.

        genmet_threshold, lambda_pt and target_transform configure the
        combined loss (MSE + relative pT-resolution penalty above
        genmet_threshold GeV).

        target_variance is the training-set variance of the network target,
        averaged over both components (px, py), used to make the MSE term
        of the combined loss dimensionless.
    """
    regularizer = l2(l2_reg) if l2_reg > 0.0 else None

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for n_units in architecture:
        model.add(Dense(n_units, kernel_regularizer=regularizer))
        model.add(ReLU())
        if dropout_rate > 0.0:
            model.add(Dropout(rate=dropout_rate))

    model.add(Dense(2, activation="linear"))

    optimizer_kwargs = {"learning_rate": learning_rate}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm

    model.compile(
        optimizer=Adam(**optimizer_kwargs),
        loss=_pt_penalized_loss(
            genmet_threshold=genmet_threshold,
            lambda_pt=lambda_pt,
            target_transform=target_transform,
            target_variance=target_variance,
        ),
        metrics=[_target_mse_metric(target_transform)]
    )
    return model


def _standardize(x_tr: np.ndarray, x_val: np.ndarray):
    """
        Computes mean and std on x_tr only, then applies the same transformation
        to both training and validation. Columns with zero std are left unchanged (std=1).

        Returns the scaled arrays and the scaling parameters so that the caller
        can apply the same transform to new data.
    """
    mean = x_tr.mean(axis=0)
    std  = x_tr.std(axis=0)
    std  = np.where(std == 0, 1.0, std)

    x_tr_scaled  = (x_tr  - mean) / std
    x_val_scaled = (x_val - mean) / std
    return x_tr_scaled, x_val_scaled, mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
        Applies a pre-computed (mean, std) standardization to a new array.
    """
    return (x - mean) / std


def _run_fold(queue, x_tr, y_tr, x_val, y_val, params, cfg, fold_idx):
    """
        Executed in a child process (spawn).

        The whole TF graph and the model weights are automatically deallocated
        when the process ends, solving the memory leak of ops.py found in
        Keras 3 + TensorFlow 2.19 and not solved by clear_session().

        With spawn a clean interpreter is started and the father's imports
        are not inherited. TensorFlow has to be imported after os.environ.

        The subprocess communicates with the father using mp.Queue using only
        primitive types (float, int, list). If an exception occurs it still
        writes a dictionary with an 'error' key so that the father does not
        get killed on queue.get().

        lambda_pt is now read from cfg (fixed across the whole search) rather
        than from params, since it is no longer part of the hyper-parameter grid.
    """
    import os, sys
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    #old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    os.environ["GRPC_VERBOSITY"]       = "ERROR"
    os.environ["GLOG_minloglevel"]     = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import warnings
    os.environ["CUDA_MODULE_LOADING"]  = "LAZY"
    warnings.filterwarnings("ignore")

    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)

    try:
        import numpy as np
        import keras
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.metrics import mean_squared_error

        model = _create_model(
            input_dim        = x_tr.shape[1],
            architecture     = params["architecture"],
            learning_rate    = params["learning_rate"],
            clipnorm         = cfg["clipnorm"],
            l2_reg           = cfg["l2_reg"],
            dropout_rate     = cfg["dropout_rate"],
            genmet_threshold = cfg.get("genmet_threshold", 20.0),
            lambda_pt        = cfg.get("lambda_pt", 1.0),
            target_transform = cfg.get("target_transform", "none"),
            target_variance  = cfg.get("target_variance", 1.0),
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
            shuffle         = True
        )

        best_ep_idx = int(np.argmin(history.history["val_loss"]))

        combined_train_loss_at_best = float(history.history["loss"][best_ep_idx])
        combined_val_loss_at_best   = float(history.history["val_loss"][best_ep_idx])
        train_mse_at_best           = float(history.history["target_mse"][best_ep_idx])
        val_mse_at_best             = float(history.history["val_target_mse"][best_ep_idx])

        y_pred = model.predict(x_val, batch_size=params["batch_size"], verbose=0)

        if cfg.get("target_transform", "none") == "residual":
            y_val_target = y_val[:, :2]
        else:
            y_val_target = y_val
        rmse = float(np.sqrt(mean_squared_error(y_val_target, y_pred)))

        queue.put({
            "fold":                         fold_idx,
            # selection criterion: combined
            "combined_val_loss":            combined_val_loss_at_best,
            "combined_train_loss":          combined_train_loss_at_best,
            # diagnostics: pure MSE-based quantities
            "train_mse_at_best":            train_mse_at_best,
            "val_mse_at_best":              val_mse_at_best,
            "rmse":                         rmse,
            # learning curves data
            "loss":                         [float(v) for v in history.history["loss"]],
            "val_loss":                     [float(v) for v in history.history["val_loss"]],
            "mse":                          [float(v) for v in history.history["target_mse"]],
            "val_mse":                      [float(v) for v in history.history["val_target_mse"]],
            "epochs_run":                   len(history.history["loss"]),
        })

    except Exception as e:
        # Garantees that the father does not get blocked on queue.get() event if the child crashes.
        queue.put({"error": str(e), "fold": fold_idx})
    return


def _model_selection(features: np.ndarray, target: np.ndarray,
                     MET_pxpy, hparam_grid,
                     output_dir: str, cfg: dict):
    """
        1. Standardizes each fold independently: mean and std are computed on
           x_tr only and applied to both x_tr and x_val, avoiding any leakage
           from validation data into the scaling parameters.
        2. Each fold is executed in a separate subprocess (spawn) to work around
           the Keras 3 + TF 2.19 memory leak in ops.py that clear_session()
           cannot resolve. The subprocess exits after the fold, freeing all TF
           memory automatically.
        3. Selects best hyper-parameters by lowest average validation combined
           loss (MSE + lambda_pt * relative pT term). The plain
           validation RMSE (in target-transformed space) is still tracked and
           logged purely as a physical diagnostic, but no longer drives the
           ranking.
        4. Saves only the top-10 hyper-parameter combinations to a CSV file.
        5. Keeps fold histories (both combined loss and MSE curves) for
           the top-10 combos by avg validation combined loss.
        6. After the search, computes scaling parameters on the full X_train and
           applies them to both X_train and X_test for use in retraining. These
           parameters are returned so they can be saved and reused on new data.
    """
    starting_time = time.time()

    X_train        = features
    y_train        = target.copy()

    cfg["target_variance"] = float(np.mean(np.var(y_train, axis=0)))

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
    print(f"    Total combinations: {len(param_combinations)}. Folds: {kf.n_splits}")

    for i, combo in enumerate(param_combinations, 1):
        params = dict(zip(param_keys, combo))
        print(f"    [{i}/{len(param_combinations)}] Testing params: {params}")

        combined_val_scores   = []
        combined_train_scores = []
        rmse_scores            = []
        mse_scores              = []
        combo_fold_histories   = []

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

            if cfg.get("target_transform", "none") == "residual":
                MET_tr  = MET_pxpy[fold_train_idx]
                MET_val = MET_pxpy[fold_val_idx]
                y_tr_packed  = np.concatenate([y_tr,  MET_tr],  axis=1)
                y_val_packed = np.concatenate([y_val, MET_val], axis=1)
            else:
                y_tr_packed, y_val_packed = y_tr, y_val

            # Subprocess per fold
            queue = mp.Queue()
            p = mp.Process(
                target=_run_fold,
                args=(queue, x_tr, y_tr_packed, x_val, y_val_packed, params, cfg, fold_idx)
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

            combined_val_scores.append(result["combined_val_loss"])
            combined_train_scores.append(result["combined_train_loss"])
            rmse_scores.append(result["rmse"])
            mse_scores.append(result["val_mse_at_best"])

            combo_fold_histories.append({
                "fold":     result["fold"],
                "loss":     result["loss"],
                "val_loss": result["val_loss"],
                "mse":      result["mse"],
                "val_mse":  result["val_mse"],
            })
            fold_bar.set_postfix({
                "val_combined_loss": f"{result['combined_val_loss']:.4f}",
                "epochs":            result["epochs_run"],
                "val_rmse":          f"{result['rmse']:.4f}",
            })

        avg_combined_val   = float(np.mean(combined_val_scores))
        std_combined_val   = float(np.std(combined_val_scores))
        avg_combined_train = float(np.mean(combined_train_scores))
        avg_rmse           = float(np.mean(rmse_scores))
        std_rmse           = float(np.std(rmse_scores))
        avg_val_mse        = float(np.mean(mse_scores))

        print(f"    FOLD STATS:\n       Avg combined val loss: {avg_combined_val:.4f} +/- {std_combined_val:.4f}  |  "
              f"Avg val RMSE: {avg_rmse:.4f} +/- {std_rmse:.4f}  |  "
              f"Avg combined train loss: {avg_combined_train:.4f}  "
              f"{[f'{v:.4f}' for v in combined_val_scores]}\n")

        row = {**params,
               "avg_combined_val_loss": avg_combined_val,
               "std_combined_val_loss": std_combined_val,
               "avg_combined_train_loss": avg_combined_train,
               "avg_rmse": avg_rmse,
               "std_rmse": std_rmse,
               "avg_val_mse": avg_val_mse}
        for j, s in enumerate(combined_val_scores, 1):
            row[f"combined_val_loss_fold{j}"] = s
        for j, s in enumerate(rmse_scores, 1):
            row[f"rmse_fold{j}"] = s

        topk_buffer.append((avg_combined_val, row, combo_fold_histories))
        topk_buffer.sort(key=lambda x: x[0])

        if len(topk_buffer) > TOP_K:
            _, _, worst_histories = topk_buffer.pop()
            if worst_histories is not best_fold_histories:
                del worst_histories

        if avg_combined_val < best_score:
            best_score          = avg_combined_val
            best_params         = params
            target_train_loss   = avg_combined_train
            best_fold_histories = combo_fold_histories

        gc.collect()

    os.makedirs(output_dir, exist_ok=True)
    hparam_log_path = os.path.join(output_dir, "hparam_search_results_pxpy.csv")
    topk_rows = [row for _, row, _ in topk_buffer]
    pd.DataFrame(topk_rows).sort_values("avg_combined_val_loss").to_csv(hparam_log_path, index=False)

    elapsed = time.time() - starting_time
    print(f"BEST PARAMETERS: {best_params}")
    print(f"    Best avg combined val loss:           {best_score:.4f}")
    print(f"    Combined train loss (retrain target): {target_train_loss:.4f}")
    print(f"    Completed in:                         {elapsed:.1f}s")

    best_row = next(row for _, row, _ in topk_buffer if row["avg_combined_val_loss"] == best_score)

    best_cv_metrics = {
        "cv_n_folds":                 kf.n_splits,
        "cv_avg_train_combined_loss": target_train_loss,
        "cv_avg_val_combined_loss":   best_score,
        "cv_std_val_combined_loss":   float(best_row["std_combined_val_loss"]),
        "cv_avg_val_rmse":            float(best_row["avg_rmse"]),
        "cv_std_val_rmse":            float(best_row["std_rmse"]),
        "cv_avg_val_mse":             float(best_row["avg_val_mse"]),
    }

    return (X_train, y_train,
            best_params, target_train_loss,
            best_fold_histories, best_cv_metrics,
            topk_buffer)


def _to_gev(y_true_transformed: np.ndarray,
            y_pred_transformed: np.ndarray,
            MET_pxpy: np.ndarray | None,
            transform: str):
    """
        Helper function to return both the target and prediction in unit measure [GeV].

        none     : y = [GenMET_px, GenMET_py]      -> identity;
        residual : y = MET_p{x,y} - GenMET_p{x,y}  -> GenMET_p{x,y} = MET_p{x,y} - y.
    """
    if transform == "none":
        return y_true_transformed, y_pred_transformed

    elif transform == "residual":
        if MET_pxpy is None:
            raise ValueError("MET_pxpy is None but target_transform='residual'")
        return MET_pxpy - y_true_transformed, MET_pxpy - y_pred_transformed

    else:
        raise ValueError(f"Unknown target_transform: '{transform}'")
    

def _retraining(X_train: np.ndarray, y_train: np.ndarray,
                MET_pxpy: np.ndarray,
                best_params: dict, cfg: dict,
                target_train_loss: float):
    """
        Retrains on the full training set using a held-out validation split
        (drawn from X_train) for early stopping.

        y_train has shape (N, 2).

        The scaler (mean, std) is fit on X_train before the val split is carved
        out, but after the train/val split so no leakage occurs.

        Returns the fitted scaler so it can be persisted alongside the model.
    """
    starting_time = time.time()

    # validation split for early stopping
    transform    = cfg.get("target_transform", "none")
    val_fraction = cfg.get("retrain_val_fraction", 0.1)
    n_val        = max(1, int(len(X_train) * val_fraction))
    rng          = np.random.default_rng(cfg.get("random_seed_split", 42))
    val_idx      = rng.choice(len(X_train), size=n_val, replace=False)
    train_mask   = np.ones(len(X_train), dtype=bool)
    train_mask[val_idx] = False

    X_tr, X_val = X_train[train_mask], X_train[val_idx]
    y_tr, y_val = y_train[train_mask], y_train[val_idx]

    # MET_pxpy for transforming back (None otherwise)
    MET_tr  = MET_pxpy[train_mask] if MET_pxpy is not None else None
    MET_val = MET_pxpy[val_idx]    if MET_pxpy is not None else None
    # standardize (fit on X_tr only)
    X_tr,  X_val,  scaler_mean, scaler_std = _standardize(X_tr, X_val)

    print(f"Retraining on full training set  "
          f"(train={len(X_tr)}, val={len(X_val)}).")
    print(f"    Target transform:                     '{transform}'")
    print(f"    Combined train loss target (from CV):  {target_train_loss:.4f}")

    if transform == "residual":
        y_tr_packed  = np.concatenate([y_tr,  MET_tr],  axis=1)
        y_val_packed = np.concatenate([y_val, MET_val], axis=1)
    else:
        y_tr_packed, y_val_packed = y_tr, y_val

    best_model = _create_model(
        input_dim        = X_tr.shape[1],
        architecture     = best_params["architecture"],
        learning_rate    = best_params["learning_rate"],
        clipnorm         = cfg["clipnorm"],
        l2_reg           = cfg["l2_reg"],
        dropout_rate     = cfg["dropout_rate"],
        genmet_threshold = cfg.get("genmet_threshold", 20.0),
        lambda_pt        = cfg.get("lambda_pt", 1.0),
        target_transform = transform,
        target_variance  = cfg.get("target_variance", float(np.mean(np.var(y_train, axis=0)))),
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg["es_patience_retrain"],
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=cfg["rlrop_factor"],
        patience=cfg["lr_patience"],
        min_delta=cfg["rlrop_min_delta"],
        min_lr=cfg["rlrop_min_lr"],
        verbose=1,
    )

    history = best_model.fit(
        X_tr, y_tr_packed,
        batch_size      = best_params["batch_size"],
        epochs          = cfg["max_epochs_retrain"],
        validation_data = (X_val, y_val_packed),
        verbose         = 1,
        callbacks       = [early_stop, reduce_lr],
        shuffle         = True,
    )

    # predictions in transformed space — shape (N, 2)
    y_train_pred = best_model.predict(X_tr, batch_size=best_params["batch_size"], verbose=0)

    # metrics at best validation epoch (Keras history)
    best_ep_idx = int(np.argmin(history.history["val_loss"]))

    # de-transforming to GeV for final metrics
    y_train_GeV, y_pred_GeV = _to_gev(y_tr, y_train_pred, MET_tr, transform)

    # MSE averaged over both components and all events
    train_mse_gev  = float(mean_squared_error(y_train_GeV, y_pred_GeV))
    train_rmse_gev = float(np.sqrt(train_mse_gev))

    # Per-component RMSE for diagnostics
    rmse_px_gev = float(np.sqrt(mean_squared_error(y_train_GeV[:, 0], y_pred_GeV[:, 0])))
    rmse_py_gev = float(np.sqrt(mean_squared_error(y_train_GeV[:, 1], y_pred_GeV[:, 1])))

    retrain_metrics = {
        "retrain_train_combined_loss": float(history.history["loss"][best_ep_idx]),
        "retrain_val_combined_loss":   float(history.history["val_loss"][best_ep_idx]),
        "retrain_train_target_mse":    float(history.history["target_mse"][best_ep_idx]),
        "retrain_val_target_mse":      float(history.history["val_target_mse"][best_ep_idx]),
        "retrain_train_mse_gev":       train_mse_gev,
        "retrain_train_rmse_gev":      train_rmse_gev,
        "retrain_train_rmse_px_gev":   rmse_px_gev,
        "retrain_train_rmse_py_gev":   rmse_py_gev,
    }

    print(f"\nRetrain    MSE={train_mse_gev:.4f}, RMSE={train_rmse_gev:.4f}  "
          f"(RMSE_px={rmse_px_gev:.4f}, RMSE_py={rmse_py_gev:.4f})")

    elapsed = time.time() - starting_time
    print(f"Retraining completed in {elapsed:.1f}s")

    scaler = {"mean": scaler_mean, "std": scaler_std}
    return (history, best_model, retrain_metrics, scaler)


def _save_model_and_scaler(model, scaler: dict, output_dir: str):
    """
        Saves everything needed to run inference on a new dataset without
        retraining:

            <output_dir>/
                model_pxpy.keras - full Keras model (weights + architecture)
                scaler_pxpy.npz  - mean and std arrays used for standardization
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path  = os.path.join(output_dir, "model_pxpy.keras")
    scaler_path = os.path.join(output_dir, "scaler_pxpy.npz")

    model.save(model_path)

    np.savez(scaler_path,
             mean=scaler["mean"],
             std=scaler["std"])

    print(f"Model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    return


def _save_run_summary(best_params: dict, cfg: dict,
                      best_cv_metrics: dict, retrain_metrics: dict,
                      feature_names: list, output_dir: str):
    """
        Saves a .json file summarising the full run:
          - best hyperparameters
          - training config (clipnorm, lr_patience, lambda_pt, etc.)
          - CV metrics for the best combo (avg train/val combined loss, RMSE, n_folds)
          - retraining train MSE/RMSE (global + per-component px/py)
    """
    run_summary_path = os.path.join(output_dir, "run_summary_pxpy.json")

    run_summary = {
        "feature_names":                 feature_names,
        "target_names":                  ["GenMET_px", "GenMET_py"],
        "architecture":                  str(best_params["architecture"]),
        "batch_size":                    best_params["batch_size"],
        "learning_rate":                 best_params["learning_rate"],
        **{f"cfg_{k}": v for k, v in cfg.items()},
        "cv_n_folds":                    best_cv_metrics["cv_n_folds"],
        "cv_avg_train_combined_loss":    best_cv_metrics["cv_avg_train_combined_loss"],
        "cv_avg_val_combined_loss":      best_cv_metrics["cv_avg_val_combined_loss"],
        "cv_std_val_combined_loss":      best_cv_metrics["cv_std_val_combined_loss"],
        "cv_avg_val_rmse":               best_cv_metrics["cv_avg_val_rmse"],
        "cv_std_val_rmse":               best_cv_metrics["cv_std_val_rmse"],
        "cv_avg_val_mse":                best_cv_metrics["cv_avg_val_mse"],
        "retrain_train_combined_loss":   retrain_metrics["retrain_train_combined_loss"],
        "retrain_val_combined_loss":     retrain_metrics["retrain_val_combined_loss"],
        "retrain_train_target_mse":      retrain_metrics["retrain_train_target_mse"],
        "retrain_val_target_mse":        retrain_metrics["retrain_val_target_mse"],
        "retrain_train_mse_gev":         retrain_metrics["retrain_train_mse_gev"],
        "retrain_train_rmse_gev":        retrain_metrics["retrain_train_rmse_gev"],
        "retrain_train_rmse_px_gev":     retrain_metrics["retrain_train_rmse_px_gev"],
        "retrain_train_rmse_py_gev":     retrain_metrics["retrain_train_rmse_py_gev"],
    }
    with open(run_summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Run summary saved to {run_summary_path}")
    return