import argparse
from MET_pt_utils import (
    _read_data, _data_parsing,
    _model_selection, _retraining,
    _save_model_and_scaler, _save_run_summary,
    mp, tf, keras, pd, os
)


VALID_TRANSFORMS = ("none", "response", "residual")


# Dataset training configuration
DATASET_CFG = {
    # model regularization
    "clipnorm":              1.0,
    "l2_reg":                1e-2,
    "dropout_rate":          0.1,
    # LR schedule / early stopping
    "lr_patience":           60,
    "es_patience_search":    150,
    "es_patience_retrain":   120,
    "rlrop_factor":          0.5,
    "rlrop_min_delta":       1e-5,
    "rlrop_min_lr":          1e-7,
    # training loop
    "max_epochs_search":     2000,
    "max_epochs_retrain":    2500,
    "retrain_val_fraction":  0.15,
    "target_transform":      "none",   # overridden by --transform
    "response_met_min":      10.0,
    # CV
    "random_seed_kfold":     42,
    "n_folds":               3,
}


HPARAM_GRID = {
    "architecture": [
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ],
    "batch_size":    [1024, 2048],
    "learning_rate": [3e-4, 1e-4],
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train MET_unified model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transform", "-t",
        choices=VALID_TRANSFORMS,
        default=None,
        help=(
            "Target transformation to apply. "
            "Overrides the value hardcoded in DATASET_CFG. "
            f"Choices: {VALID_TRANSFORMS}."
        ),
    )
    parser.add_argument(
        "--results-dir", "-o",
        default=None,
        dest="output_dir",
        help=(
            "Directory where results are saved. "
            "Overrides the default '../Results/results_unified'. "
            "If not given but --transform is set, defaults to "
            "'../Results/results_<transform>'."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = _parse_args()

    # Apply CLI overrides
    if args.transform is not None:
        DATASET_CFG["target_transform"] = args.transform

    transform = DATASET_CFG["target_transform"]

    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    elif args.transform is not None:
        OUTPUT_DIR = f"../Results/results_{transform}"
    else:
        OUTPUT_DIR = "../Results/results_unified"

    print(f"target_transform : '{transform}'")
    print(f"results_dir      : '{OUTPUT_DIR}'")

    import tracemalloc
    tracemalloc.start()

    print(f"Keras version:      {keras.__version__}")
    print(f"Tensorflow version: {tf.__version__}")

    INPUT_FILE = "../TrainingDatasets/training_2d.root"

    print("GPUs available:", tf.config.list_physical_devices("GPU"))

    # Load data
    branches, data = _read_data(INPUT_FILE)

    # Parse into feature matrix and target
    features, target, MET_pt, feature_names = _data_parsing(branches, data, DATASET_CFG)

    # Grid-search with K-Fold CV
    (X_train, y_train,
     best_params, target_train_loss,
     best_fold_histories, best_cv_metrics,
     topk_buffer) = _model_selection(
        features, target, HPARAM_GRID, OUTPUT_DIR, DATASET_CFG
    )

    # Saves CV folds learning curves
    rows = []
    for fh in best_fold_histories:
        for ep, (tl, vl) in enumerate(zip(fh["loss"], fh["val_loss"]), 1):
            rows.append({"fold": fh["fold"], "epoch": ep, "loss": tl, "val_loss": vl})
    pd.DataFrame(rows).to_parquet(
        os.path.join(OUTPUT_DIR, "fold_histories.parquet"), index=False
    )

    # Retrain best model
    history, best_model, retrain_metrics, scaler = _retraining(
        X_train, y_train, MET_pt, best_params, DATASET_CFG, target_train_loss
    )

    _save_model_and_scaler(
        model=best_model,
        scaler=scaler,
        output_dir=OUTPUT_DIR,
    )
    
    _save_run_summary(
        best_params=best_params,
        cfg=DATASET_CFG,
        best_cv_metrics=best_cv_metrics,
        retrain_metrics=retrain_metrics,
        feature_names=feature_names,
        output_dir=OUTPUT_DIR,
    )

    # Save retraining learning curve
    pd.DataFrame(history.history).to_parquet(
        os.path.join(OUTPUT_DIR, "learning_curves.parquet"), index=False
    )