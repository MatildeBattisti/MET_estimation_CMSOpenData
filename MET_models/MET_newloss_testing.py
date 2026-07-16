import argparse
from MET_newloss_utils import (
    _read_data, _apply_standardize,
    os, json, keras,
    np, pd
)
from sklearn.metrics import mean_squared_error, r2_score


DATASETS = {
    "DYJetsToLL":   "../TestingDatasets/testing_DYJetsToLL.root",
    "HToAATo2Mu2B": "../TestingDatasets/testing_HToAATo2Mu2B.root",
    "ZZTo2L2Nu":    "../TestingDatasets/testing_ZZTo2L2Nu.root",
}


DEFAULT_RESULTS_DIR = "../Results/results_newloss_2d/"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained MET_newloss model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", "-r",
        default=DEFAULT_RESULTS_DIR,
        dest="results_dir",
        help="Directory containing model.keras, scaler.npz, and run_summary.json.",
    )
    return parser.parse_args()


def load_model_and_scaler(results_dir: str):
    """
        Loads a previously saved model, scaler, and run summary.

        Returns:
        model      : Keras model ready for .predict()
        scaler     : dict with keys 'mean' and 'std' (np.ndarray)
        run_summary: dict containing feature_names, target_transform, batch_size, etc
    """
    model_path       = os.path.join(results_dir, "model.keras")
    scaler_path      = os.path.join(results_dir, "scaler.npz")
    run_summary_path = os.path.join(results_dir, "run_summary.json")

    for path in (model_path, scaler_path, run_summary_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing saved artefact: {path}")
        
    model = keras.models.load_model(model_path, compile=False)

    npz    = np.load(scaler_path)
    scaler = {"mean": npz["mean"], "std": npz["std"]}

    with open(run_summary_path, "r") as f:
        run_summary = json.load(f)
    return model, scaler, run_summary


def predict_on_dataset(file_path: str, model, scaler: dict, run_summary: dict,
                       output_dir: str = None):
    """
        End-to-end inference over the .root dataset.

        1. Reads the .root file;
        2. Validates the expected features;
        3. Standardizes with the saved scaler;
        4. Predicts and de-transforms in GenMET [Gev] according to target_transform;
        5. Saves a .parquet with columns: MET_pt_pred, GenMET_pt, MET_pt.

        De-trasformazione:
          none     -> y_pred = GenMET_pt
          residual -> GenMET_pt = MET_pt - y_pred   (target = MET_pt - GenMET_pt)
    """
    feature_names    = run_summary["feature_names"]
    target_transform = run_summary.get("cfg_target_transform", "none")
    batch_size       = run_summary["batch_size"]

    branches, data = _read_data(file_path)

    missing = [f for f in feature_names if f not in data]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X_new        = np.column_stack([data[f] for f in feature_names]).astype(np.float32)
    X_new_scaled = _apply_standardize(X_new, scaler["mean"], scaler["std"])

    # Raw prediction in transformed space
    y_pred = model.predict(X_new_scaled, batch_size=batch_size, verbose=0).flatten()

    # MET_pt required by residual transforms
    MET_pt     = data.get("MET_pt", None)
    y_true_gev = data.get("GenMET_pt", None)

    needs_met = target_transform in ("response", "residual")
    if needs_met and MET_pt is None:
        raise ValueError(
            f"'MET_pt' not found in file but necessary for "
            f"target_transform='{target_transform}'."
        )

    # De-transforms in GenMET_pt [GeV]
    if target_transform == "none":
        y_pred_gev = y_pred
    elif target_transform == "residual":
        y_pred_gev = MET_pt - y_pred
    else:
        raise ValueError(f"target_transform unknown: '{target_transform}'")

    # Metrics (all in GeVs)
    if y_true_gev is not None:
        mse_pred  = mean_squared_error(y_true_gev, y_pred_gev)
        rmse_pred = np.sqrt(mse_pred)
        r2_pred   = r2_score(y_true_gev, y_pred_gev)

        print(f"\n  NN prediction vs GenMET_pt  [transform='{target_transform}']:")
        print(f"    MSE  = {mse_pred:.4f} GeV²")
        print(f"    RMSE = {rmse_pred:.4f} GeV")
        print(f"    R2   = {r2_pred:.4f}")

        if MET_pt is not None:
            mse_reco  = mean_squared_error(y_true_gev, MET_pt)
            rmse_reco = np.sqrt(mse_reco)
            r2_reco   = r2_score(y_true_gev, MET_pt)

            print(f"\n  Reconstructed MET_pt vs GenMET_pt:")
            print(f"    MSE  = {mse_reco:.4f} GeV²")
            print(f"    RMSE = {rmse_reco:.4f} GeV")
            print(f"    R2   = {r2_reco:.4f}")

            print(f"\n  Improvement:")
            print(f"    DELTA_MSE  = {mse_reco - mse_pred:.4f} GeV²")
            print(f"    DELTA_RMSE = {rmse_reco - rmse_pred:.4f} GeV")
            print(f"    DELTA_R2   = {r2_pred - r2_reco:.4f}")

    # Saves a .parquet
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        stem     = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, f"predictions_{stem}.parquet")

        result_df = pd.DataFrame({"MET_pt_pred": y_pred_gev})
        if y_true_gev is not None:
            result_df["GenMET_pt"] = y_true_gev
        if MET_pt is not None:
            result_df["MET_pt"] = MET_pt

        # saves X_test scaled for future SHAP
        x_df = pd.DataFrame(X_new_scaled, columns=feature_names)
        x_df.to_parquet(os.path.join(output_dir, f"X_test_scaled_{stem}.parquet"), index=False)

        result_df.to_parquet(out_path, index=False)
        print(f"\n  Saved in: {out_path}  |  columns: {list(result_df.columns)}")

    return y_pred_gev, y_true_gev, MET_pt


if __name__ == "__main__":
    args = _parse_args()
    RESULTS_DIR = args.results_dir

    print(f"results_dir : '{RESULTS_DIR}'")

    model, scaler, run_summary = load_model_and_scaler(RESULTS_DIR)
    feature_names = run_summary["feature_names"]

    for event_name, file_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {event_name}")
        print('='*60)

        if not os.path.exists(file_path):
            print(f"  [SKIP] File not found: {file_path}")
            continue

        y_pred, y_true, MET_pt = predict_on_dataset(
            file_path   = file_path,
            model       = model,
            scaler      = scaler,
            run_summary = run_summary,
            output_dir  = RESULTS_DIR,
        )