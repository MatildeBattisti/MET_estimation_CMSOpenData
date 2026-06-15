from MET_unified_utils import (
    _read_data, _apply_standardize,
    os, json, keras,
    np, pd, shap, plt
)
from sklearn.metrics import mean_squared_error, r2_score


DATASETS = {
    "DYJetsToLL":   "TestingDatasets/testing_DYJetsToLL.root",
    "HToAATo2Mu2B": "TestingDatasets/testing_HToAATo2Mu2B.root",
    "ZZTo2L2Nu":    "TestingDatasets/testing_ZZTo2L2Nu.root",
}


MODEL_DIR  = "Results/"
OUTPUT_DIR = "Results/"


def load_model_and_scaler(results_dir: str):
    """
        Loads a previously saved model, scaler, and run summary.

        Returns:
        model      : compiled Keras model ready for .predict()
        scaler     : dict with keys 'mean' and 'std' (np.ndarray)
        run_summary: dict containing feature_names, target_transform, batch_size, ...
    """
    model_path       = os.path.join(results_dir, "model.keras")
    scaler_path      = os.path.join(results_dir, "scaler.npz")
    run_summary_path = os.path.join(results_dir, "run_summary.json")

    for path in (model_path, scaler_path, run_summary_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing saved artefact: {path}")

    model = keras.models.load_model(model_path)

    npz    = np.load(scaler_path)
    scaler = {"mean": npz["mean"], "std": npz["std"]}

    with open(run_summary_path, "r") as f:
        run_summary = json.load(f)

    print(f"Loaded model + scaler from '{results_dir}'.")
    return model, scaler, run_summary


def predict_on_dataset(file_path: str, model, scaler: dict, run_summary: dict,
                       output_dir: str = None):
    """
        End-to-end inference on a new .root dataset using a previously loaded
        model and scaler. No retraining is performed.

        Steps:
        1. Read the .root file (all branches).
        2. Validate that the expected features are present.
        3. Standardize with the saved scaler.
        4. Predict and de-transform if target_transform == 'log1p'.
        5. Save a parquet with columns: MET_pt_pred, GenMET_pt, MET_pt.

        Returns:
        y_pred_GeV : np.ndarray         — NN predictions in GeV
        y_true_GeV : np.ndarray or None — GenMET_pt if present in the file
        met_pt     : np.ndarray or None — reconstructed MET_pt if present
    """
    feature_names    = run_summary["feature_names"]
    target_transform = run_summary.get("cfg_target_transform", "none")
    batch_size       = run_summary["batch_size"]

    # load testing dataset
    branches, data = _read_data(file_path)

    missing = [f for f in feature_names if f not in data]
    if missing:
        raise ValueError(
            f"New dataset is missing features expected by the model: {missing}"
        )

    X_new        = np.column_stack([data[f] for f in feature_names]).astype(np.float32)
    X_new_scaled = _apply_standardize(X_new, scaler["mean"], scaler["std"])

    # prediction
    y_pred = model.predict(X_new_scaled, batch_size=batch_size, verbose=0).flatten()

    if target_transform == "log1p":
        y_pred_GeV = np.expm1(y_pred)
    else:
        y_pred_GeV = y_pred

    # ground truth
    y_true_GeV = data.get("GenMET_pt", None)

    # reconstructed MET
    MET_pt = data.get("MET_pt", None)
    if MET_pt is None:
        print("  [WARNING] Branch 'MET_pt' not found — column will be missing.")

    # metrics
    if y_true_GeV is not None:

        # Predicted MET vs GenMET
        mse_pred = mean_squared_error(y_true_GeV, y_pred_GeV)
        rmse_pred = np.sqrt(mse_pred)
        r2_pred = r2_score(y_true_GeV, y_pred_GeV)

        print("\n  Predicted MET vs GenMET")
        print(f"    MSE  = {mse_pred:.4f} GeV²")
        print(f"    RMSE = {rmse_pred:.4f} GeV")
        print(f"    R²   = {r2_pred:.4f}")

        # Reconstructed MET vs GenMET
        if MET_pt is not None:
            mse_reco = mean_squared_error(y_true_GeV, MET_pt)
            rmse_reco = np.sqrt(mse_reco)
            r2_reco = r2_score(y_true_GeV, MET_pt)

            print("\nReconstructed MET vs GenMET")
            print(f"    MSE  = {mse_reco:.4f} GeV²")
            print(f"    RMSE = {rmse_reco:.4f} GeV")
            print(f"    R²   = {r2_reco:.4f}")

            print("\nImprovement")
            print(f"    ΔRMSE = {rmse_reco - rmse_pred:.4f} GeV")
            print(f"    ΔR²   = {r2_pred - r2_reco:.4f}")

    # save parquet
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        stem     = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, f"predictions_{stem}.parquet")

        result_df = pd.DataFrame({"MET_pt_pred": y_pred_GeV})
        if y_true_GeV is not None:
            result_df["GenMET_pt"] = y_true_GeV
        if MET_pt is not None:
            result_df["MET_pt"] = MET_pt

        result_df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}  |  columns: {list(result_df.columns)}")
    return y_pred_GeV, y_true_GeV, MET_pt


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

    shap.summary_plot(sv, sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — {event}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_summary_{event}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    shap.summary_plot(sv, sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {event}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_bar_{event}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    mean_shap = np.abs(sv).mean(axis=0)
    ranked    = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)

    print(f"Feature importance (mean |SHAP|) for {event}:")
    for name, val in ranked:
        print(f"    {name:<35} {val:.4f}")

    pd.DataFrame(ranked, columns=["feature", "mean_abs_shap"]).to_csv(
        os.path.join(output_dir, f"shap_importance_{event}.csv"), index=False
    )
    return


if __name__ == "__main__":
    model, scaler, run_summary = load_model_and_scaler(MODEL_DIR)
    feature_names = run_summary["feature_names"]

    for event_name, file_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {event_name}")
        print('='*60)

        if not os.path.exists(file_path):
            print(f"  [SKIP] File not found: {file_path}")
            continue

        y_pred, y_true, MET_pt = predict_on_dataset(
            file_path  = file_path,
            model      = model,
            scaler     = scaler,
            run_summary= run_summary,
            output_dir = OUTPUT_DIR,
        )

        # build X_test (scaled) for SHAP — same pipeline as predict_on_dataset
        _, data  = _read_data(file_path)
        X_test   = np.column_stack([data[f] for f in feature_names]).astype(np.float32)
        X_test   = _apply_standardize(X_test, scaler["mean"], scaler["std"])

        _feature_importance_shap(
            model         = model,
            X_test        = X_test,
            feature_names = feature_names,
            event         = event_name,
            output_dir    = OUTPUT_DIR,
        )
