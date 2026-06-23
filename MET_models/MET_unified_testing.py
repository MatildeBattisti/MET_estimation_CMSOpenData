from MET_unified_utils import (
    _read_data, _apply_standardize,
    os, json, keras,
    np, pd, shap, plt
)
from sklearn.metrics import mean_squared_error, r2_score


DATASETS = {
    "DYJetsToLL":   "../TestingDatasets/testing_DYJetsToLL.root",
    "HToAATo2Mu2B": "../TestingDatasets/testing_HToAATo2Mu2B.root",
    "ZZTo2L2Nu":    "../TestingDatasets/testing_ZZTo2L2Nu.root",
}


MODEL_DIR  = "Results/results_unified/"
OUTPUT_DIR = "Results/results_unified/"


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
        End-to-end inference over the .root dataset.

        1. Reads the .root file;
        2. Validates the expected features;
        3. Standardizes with the saved scaler;
        4. Predicts and de-transforms in GenMET [Gev] according to target_transform;
        5. Saves a .parquet with columns: MET_pt_pred, GenMET_pt, MET_pt.

        De-trasformazione:
          none     -> y_pred = GenMET_pt
          log1p    -> GenMET_pt = expm1(y_pred)
          response -> GenMET_pt = MET_pt / y_pred
          residual -> GenMET_pt = MET_pt - y_pred
    """
    feature_names    = run_summary["feature_names"]
    target_transform = run_summary.get("cfg_target_transform", "none")
    batch_size       = run_summary["batch_size"]

    branches, data = _read_data(file_path)

    missing = [f for f in feature_names if f not in data]
    if missing:
        raise ValueError(f"Feature mancanti nel dataset: {missing}")

    X_new        = np.column_stack([data[f] for f in feature_names]).astype(np.float32)
    X_new_scaled = _apply_standardize(X_new, scaler["mean"], scaler["std"])

    y_pred = model.predict(X_new_scaled, batch_size=batch_size, verbose=0).flatten()

    # MET_pt serve per de-trasformare response e residual
    MET_pt     = data.get("MET_pt", None)
    y_true_gev = data.get("GenMET_pt", None)

    needs_met = target_transform in ("response", "residual")
    if needs_met and MET_pt is None:
        raise ValueError(
            f"'MET_pt' non trovata nel file ma necessaria per "
            f"target_transform='{target_transform}'."
        )

    # De-transforms in GenMET_pt [GeV]
    if target_transform == "none":
        y_pred_gev = y_pred
    elif target_transform == "log1p":
        y_pred_gev = np.expm1(y_pred)
    elif target_transform == "response":
        y_pred_gev = MET_pt / np.where(y_pred == 0, 1e-9, y_pred)
    elif target_transform == "residual":
        y_pred_gev = MET_pt - y_pred
    else:
        raise ValueError(f"target_transform sconosciuto: '{target_transform}'")

    # Metrics (all in GeVs)
    if y_true_gev is not None:
        mse_pred  = mean_squared_error(y_true_gev, y_pred_gev)
        rmse_pred = np.sqrt(mse_pred)
        r2_pred   = r2_score(y_true_gev, y_pred_gev)

        print(f"\n  NN prediction vs GenMET_pt  [transform='{target_transform}']")
        print(f"    MSE  = {mse_pred:.4f} GeV²")
        print(f"    RMSE = {rmse_pred:.4f} GeV")
        print(f"    R²   = {r2_pred:.4f}")

        if MET_pt is not None:
            mse_reco  = mean_squared_error(y_true_gev, MET_pt)
            rmse_reco = np.sqrt(mse_reco)
            r2_reco   = r2_score(y_true_gev, MET_pt)

            print(f"\n  Reconstructed MET_pt vs GenMET_pt")
            print(f"    MSE  = {mse_reco:.4f} GeV²")
            print(f"    RMSE = {rmse_reco:.4f} GeV")
            print(f"    R²   = {r2_reco:.4f}")

            print(f"\n  Improvement")
            print(f"    DELTA_RMSE = {rmse_reco - rmse_pred:.4f} GeV")
            print(f"    DELTA_R²   = {r2_pred   - r2_reco:.4f}")

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

        result_df.to_parquet(out_path, index=False)
        print(f"\n  Saved → {out_path}  |  columns: {list(result_df.columns)}")

    return y_pred_gev, y_true_gev, MET_pt


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
