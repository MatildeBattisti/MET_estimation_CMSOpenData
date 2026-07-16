import argparse
from MET_pxpy_newloss_utils import (
    _read_data, _apply_standardize,
    os, json, keras,
    np, pd
)
from sklearn.metrics import mean_squared_error, r2_score


DATASETS = {
    "DYJetsToLL":   "../TestingDatasets/testing_pxpy_DYJetsToLL.root",
    "HToAATo2Mu2B": "../TestingDatasets/testing_pxpy_HToAATo2Mu2B.root",
    "ZZTo2L2Nu":    "../TestingDatasets/testing_pxpy_ZZTo2L2Nu.root",
}


DEFAULT_RESULTS_DIR  = "../Results/results_pxpy_newloss_2d/"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained MET_pxpy_newloss model.",
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
            model      : compiled Keras model ready for .predict()
            scaler     : dict with keys 'mean' and 'std' (np.ndarray)
            run_summary: dict containing feature_names, target_transform, batch_size, ...
    """
    model_path       = os.path.join(results_dir, "model_pxpy.keras")
    scaler_path      = os.path.join(results_dir, "scaler_pxpy.npz")
    run_summary_path = os.path.join(results_dir, "run_summary_pxpy.json")

    for path in (model_path, scaler_path, run_summary_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing saved artefact: {path}")

    model = keras.models.load_model(model_path, compile=False)

    npz    = np.load(scaler_path)
    scaler = {"mean": npz["mean"], "std": npz["std"]}

    with open(run_summary_path, "r") as f:
        run_summary = json.load(f)
    return model, scaler, run_summary


def _compute_pt_from_components(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
        Computes MET_pt from px and py components.
    """
    return np.sqrt(px**2 + py**2)


def _print_metrics(label_pred: str, label_true: str,
                   true_px, true_py, pred_px, pred_py):
    """
        Prints MSE, RMSE, R² for px, py and derived pt.
    """
    for comp, true_v, pred_v in [("px", true_px, pred_px), ("py", true_py, pred_py)]:
        mse  = mean_squared_error(true_v, pred_v)
        rmse = np.sqrt(mse)
        r2   = r2_score(true_v, pred_v)
        print(f"COMPONENT {comp}:\n  MSE={mse:.4f} GeV²\n  RMSE={rmse:.4f} GeV\n  R2={r2:.4f}")

    # derived pt
    pred_pt = _compute_pt_from_components(pred_px, pred_py)
    true_pt = _compute_pt_from_components(true_px, true_py)
    mse_pt  = mean_squared_error(true_pt, pred_pt)
    rmse_pt = np.sqrt(mse_pt)
    r2_pt   = r2_score(true_pt, pred_pt)
    print(f"COMPONENT pt:\n  MSE={mse_pt:.4f} GeV²\n  RMSE={rmse_pt:.4f} GeV\n  R2={r2_pt:.4f}")
    return pred_pt, true_pt


def predict_on_dataset(file_path: str, model, scaler: dict, run_summary: dict,
                       output_dir: str = None):
    """
        End-to-end inference on a new .root dataset.

        Steps:
            1. Reads the .root file;
            2. Validates that the expected features are present;
            3. Standardizes with the saved scaler;
            4. Predicts pred_MET_px and pred_MET_py (in transformed space);
            5. Reads reco MET_px/MET_py (needed to de-transform);
            6. De-transforms to GeV depending on target_transform:
                "none"     -> identity
                "residual" -> GenMET_pred = MET - y_pred
            7. Saves a parquet with columns:
                pred_MET_p{x,y,t}, GenMET_p{x,y,t}, MET_p{x,y,t}
    """
    feature_names    = run_summary["feature_names"]
    target_transform = run_summary.get("cfg_target_transform", "none")
    batch_size       = run_summary["batch_size"]

    stem = os.path.splitext(os.path.basename(file_path))[0]

    # load dataset
    branches, data = _read_data(file_path)

    missing = [f for f in feature_names if f not in data]
    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    X_new        = np.column_stack([data[f] for f in feature_names]).astype(np.float32)
    X_new_scaled = _apply_standardize(X_new, scaler["mean"], scaler["std"])

    # Raw prediction in transformed space
    y_pred = model.predict(X_new_scaled, batch_size=batch_size, verbose=0)

    # reconstructed MET required to transform
    has_reco = ("MET_px" in data) and ("MET_py" in data)
    if has_reco:
        reco_px  = data["MET_px"]
        reco_py  = data["MET_py"]
        MET_pxpy = np.column_stack([reco_px, reco_py])
    else:
        MET_pxpy = None
        print("    [WARNING] MET_px / MET_py not found - reco columns will be missing.")

    if target_transform == "residual":
        if not has_reco:
            raise ValueError(
                "target_transform='residual' requires MET_px/MET_py to invert "
                "the GenMET prediction, but they are not available."
            )
        y_pred_GeV = MET_pxpy - y_pred  # GenMET_pred = MET - residual_pred
    elif target_transform == "none":
        y_pred_GeV = y_pred
    else:
        raise ValueError(
            f"predict_on_dataset does not support target_transform='{target_transform}'."
        )

    pred_px = y_pred_GeV[:, 0]
    pred_py = y_pred_GeV[:, 1]

    # Ground truth (already in GeV)
    has_truth = ("GenMET_px" in data) and ("GenMET_py" in data)
    if has_truth:
        true_px = data["GenMET_px"]
        true_py = data["GenMET_py"]
        y_true_GeV = np.column_stack([true_px, true_py])
    else:
        y_true_GeV = None
        print("    [WARNING] GenMET_px / GenMET_py not found - skipping truth metrics.")

    # metrics
    if y_true_GeV is not None:

        print("\n   Predicted MET vs GenMET")
        pred_pt, true_pt = _print_metrics(
            "Predicted", "GenMET",
            true_px, true_py, pred_px, pred_py
        )

        if has_reco:
            print("\n   Reconstructed MET vs GenMET")
            reco_pt, _ = _print_metrics(
                "Reco", "GenMET",
                true_px, true_py, reco_px, reco_py
            )

            print("\n   Improvement")
            for comp, true_v, pred_v, reco_v in [
                ("px", true_px, pred_px, reco_px),
                ("py", true_py, pred_py, reco_py),
                ("pt", true_pt, pred_pt, reco_pt),
            ]:
                rmse_pred = np.sqrt(mean_squared_error(true_v, pred_v))
                rmse_reco = np.sqrt(mean_squared_error(true_v, reco_v))
                r2_pred   = r2_score(true_v, pred_v)
                r2_reco   = r2_score(true_v, reco_v)
                print(f"    {comp}  →  ΔRMSE={rmse_reco - rmse_pred:+.4f} GeV  "
                      f"ΔR²={r2_pred - r2_reco:+.4f}")

    # save parquet + scaled features for future SHAP (only if an output_dir was given)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"predictions_{stem}.parquet")

        result_df = pd.DataFrame({
            "pred_MET_px": pred_px,
            "pred_MET_py": pred_py,
            "pred_MET_pt": _compute_pt_from_components(pred_px, pred_py),
        })

        if y_true_GeV is not None:
            result_df["GenMET_px"] = true_px
            result_df["GenMET_py"] = true_py
            result_df["GenMET_pt"] = _compute_pt_from_components(true_px, true_py)

        if has_reco:
            result_df["MET_px"] = reco_px
            result_df["MET_py"] = reco_py
            result_df["MET_pt"] = _compute_pt_from_components(reco_px, reco_py)

        result_df.to_parquet(out_path, index=False)
        print(f"\n  Saved in: {out_path}  |  columns: {list(result_df.columns)}")

        # saves X_test scaled for future SHAP
        x_df = pd.DataFrame(X_new_scaled, columns=feature_names)
        x_df.to_parquet(os.path.join(output_dir, f"X_test_scaled_{stem}.parquet"), index=False)

    return y_pred_GeV, y_true_GeV, MET_pxpy


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

        y_pred, y_true, MET_pxpy = predict_on_dataset(
            file_path   = file_path,
            model       = model,
            scaler      = scaler,
            run_summary = run_summary,
            output_dir  = RESULTS_DIR,
        )