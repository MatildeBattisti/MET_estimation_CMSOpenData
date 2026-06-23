from MET_pxpy_utils import (
    _read_data, _data_parsing, _create_model, _standardize,
    _apply_standardize, _build_tensorflow_dataset, _run_fold,
    _model_selection, _retraining,
    _save_model_and_scaler, _save_run_summary,
    mp, tf, keras, np, pd, os

)


# Dataset training configuration
DATASET_CFG = {
    # model regularisation
    "clipnorm":              1.0,
    "l2_reg":                1e-3,
    # LR schedule / early stopping
    "lr_patience":           25,  #20,
    "es_patience_search":    100,  #80
    "es_patience_retrain":   80,
    "rlrop_factor":          0.3,
    "rlrop_min_delta":       1e-5,  #5e-5,
    "rlrop_min_lr":          1e-7,  #1e-6,
    # training loop
    "max_epochs_search":     1500,
    "max_epochs_retrain":    1000,
    "retrain_val_fraction":  0.15,
    "target_transform":      "none",   # "log1p" only if px,py are strictly positive
    # CV
    "random_seed_kfold":     42,
    "n_folds":               3,
}


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    import tracemalloc
    tracemalloc.start()
    
    print(f"Keras version: {keras.__version__}")
    print(f"Tensorflow version: {tf.__version__}")
    
    INPUT_FILE = f"../TrainingDatasets/training_pxpy.root"
    OUTPUT_DIR = f"../Results/results_pxpy/"

    print("GPUs available:", tf.config.list_physical_devices("GPU"))
    
    # Load data
    branches, data = _read_data(file_path=INPUT_FILE)

    # Parse into feature matrix & target
    features, target, feature_names = _data_parsing(branches, data)

    # Grid-search with K-Fold CV (test set held-out throughout)
    HPARAM_GRID = {
        "architecture": [
            (128, 64, 32),
            (512, 256, 128),      # tieni il best
        ],
        "batch_size":    [4096, 8192],#[2048, 4096]e
        "learning_rate": [3e-4, 1e-4]#[3e-4, 1e-4, 5e-5],
    }
        
    hparam_grid = HPARAM_GRID

    X_train, y_train, best_params, target_train_loss, best_fold_histories, best_cv_metrics, topk_buffer = _model_selection(
        features, target, hparam_grid, OUTPUT_DIR, DATASET_CFG
    )

    rows = []
    for fh in best_fold_histories:
        for ep, (tl, vl) in enumerate(zip(fh["loss"], fh["val_loss"]), 1):
            rows.append({"fold": fh["fold"], "epoch": ep, "loss": tl, "val_loss": vl})

    pd.DataFrame(rows).to_parquet(
        os.path.join(OUTPUT_DIR, f"fold_histories_pxpy.parquet"), index=False
    )

    # Retrain best model and predict on test set
    history, best_model, retrain_metrics, scaler = _retraining(
        X_train, y_train, best_params, DATASET_CFG, target_train_loss
    )

    _save_model_and_scaler(
        model=best_model,
        scaler=scaler,
        output_dir=OUTPUT_DIR
    )

    _save_run_summary(
        best_params=best_params,
        cfg=DATASET_CFG,
        best_cv_metrics=best_cv_metrics,
        retrain_metrics=retrain_metrics,
        feature_names=feature_names,
        output_dir=OUTPUT_DIR
    )

    # Save retraining learning curve for external plotting
    pd.DataFrame(history.history).to_parquet(
        os.path.join(OUTPUT_DIR, f"learning_curves_pxpy.parquet"), index=False
    )