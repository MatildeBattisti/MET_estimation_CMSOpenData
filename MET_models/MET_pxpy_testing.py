from MET_pxpy_utils import (
    _read_data, _apply_standardize,
    os, json, keras,
    np, pd, shap, plt
)
from sklearn.metrics import mean_squared_error, r2_score


DATASETS = {
    "DYJetsToLL":   "../TestingDatasets/testing_pxpy_DYJetsToLL.root",
    "HToAATo2Mu2B": "../TestingDatasets/testing_pxpy_HToAATo2Mu2B.root",
    "ZZTo2L2Nu":    "../TestingDatasets/testing_pxpy_ZZTo2L2Nu.root",
}

MODEL_DIR  = "../Results/results_pxpy/"
OUTPUT_DIR = "../Results/results_pxpy/"


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

    model = keras.models.load_model(model_path)

    npz    = np.load(scaler_path)
    scaler = {"mean": npz["mean"], "std": npz["std"]}

    with open(run_summary_path, "r") as f:
        run_summary = json.load(f)

    print(f"Loaded model + scaler from '{results_dir}'.")
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
        print(f"    {comp} - MSE={mse:.4f} GeV²,  RMSE={rmse:.4f} GeV,  R²={r2:.4f}")

    # derived pt
    pred_pt = _compute_pt_from_components(pred_px, pred_py)
    true_pt = _compute_pt_from_components(true_px, true_py)
    mse_pt  = mean_squared_error(true_pt, pred_pt)
    rmse_pt = np.sqrt(mse_pt)
    r2_pt   = r2_score(true_pt, pred_pt)
    print(f"    pt - MSE={mse_pt:.4f} GeV²,  RMSE={rmse_pt:.4f} GeV,  R²={r2_pt:.4f}")

    return pred_pt, true_pt


def predict_on_dataset(file_path: str, model, scaler: dict, run_summary: dict,
                       output_dir: str = None):
    """
        End-to-end inference on a new .root dataset using a previously loaded
        model and scaler. No retraining is performed.

        Steps:
            1. Read the .root file (all branches).
            2. Validate that the expected features are present.
            3. Standardize with the saved scaler.
            4. Predict pred_MET_px and pred_MET_py.
            5. De-transform if target_transform == 'log1p' (applied per component).
            6. Save a parquet with columns:
               pred_MET_px, pred_MET_py, pred_MET_pt,
               GenMET_px, GenMET_py, GenMET_pt  (if available),
               MET_px, MET_py, MET_pt           (if available).

        Returns:
            y_pred : np.ndarray, shape (N, 2) — [pred_MET_px, pred_MET_py]
            y_true : np.ndarray or None       — [GenMET_px,   GenMET_py]
            MET_pxpy   : np.ndarray or None       — [MET_px,      MET_py]
    """
    feature_names    = run_summary["feature_names"]
    target_transform = run_summary.get("cfg_target_transform", "none")
    batch_size       = run_summary["batch_size"]

    # load dataset
    branches, data = _read_data(file_path)

    missing = [f for f in feature_names if f not in data]
    if missing:
        raise ValueError(
            f"New dataset is missing features expected by the model: {missing}"
        )

    X_new        = np.column_stack([data[f] for f in feature_names]).astype(np.float32)
    X_new_scaled = _apply_standardize(X_new, scaler["mean"], scaler["std"])

    # prediction: model output shape (N, 2). Columns [pred_MET_px, pred_MET_py]
    y_pred = model.predict(X_new_scaled, batch_size=batch_size, verbose=0)

    if target_transform == "log1p":
        # log1p applied per component during training; invert per component
        y_pred_GeV = np.sign(y_pred) * np.expm1(np.abs(y_pred))
    else:
        y_pred_GeV = y_pred

    pred_px = y_pred_GeV[:, 0]
    pred_py = y_pred_GeV[:, 1]

    # ground truth
    has_truth = ("GenMET_px" in data) and ("GenMET_py" in data)
    if has_truth:
        true_px = data["GenMET_px"]
        true_py = data["GenMET_py"]
        y_true_GeV = np.column_stack([true_px, true_py])
    else:
        y_true_GeV = None
        print("    [WARNING] GenMET_px / GenMET_py not found - skipping truth metrics.")

    # reconstructed MET
    has_reco = ("MET_px" in data) and ("MET_py" in data)
    if has_reco:
        reco_px  = data["MET_px"]
        reco_py  = data["MET_py"]
        met_pxpy = np.column_stack([reco_px, reco_py])
    else:
        met_pxpy = None
        print("    [WARNING] MET_px / MET_py not found - reco columns will be missing.")

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

    # save parquet
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        stem     = os.path.splitext(os.path.basename(file_path))[0]
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
        print(f"\n  Saved → {out_path}  |  columns: {list(result_df.columns)}")

    return y_pred_GeV, y_true_GeV, met_pxpy


def _feature_importance_shap(model, X_test: np.ndarray, feature_names: list,
                              event: str, output_dir: str):
    sample     = X_test[:500].astype(np.float32)
    background = X_test[:100].astype(np.float32)

    explainer   = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(sample)

    # normalise output format
    # GradientExplainer can return:
    #   (a) list of 2 arrays  shape (N, F)       → 2 separate outputs
    #   (b) single array      shape (N, F, 2)    → final Dense(2)
    #   (c) single array      shape (N, F)       → single output
    if isinstance(shap_values, list):
        # caso (a): già separati per output
        sv = np.stack(shap_values, axis=0)       # (2, N, F)
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3 and shap_values.shape[-1] == 2:
            # caso (b): (N, F, 2) → trasponi in (2, N, F)
            sv = shap_values.transpose(2, 0, 1)
        elif shap_values.ndim == 2:
            # caso (c): singolo output — wrappa in (1, N, F)
            sv = shap_values[np.newaxis, ...]
        else:
            raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")

    print(f"  SHAP sv shape: {sv.shape}  "
          f"(n_outputs={sv.shape[0]}, n_samples={sv.shape[1]}, n_features={sv.shape[2]})")

    os.makedirs(output_dir, exist_ok=True)
    target_labels = ["px", "py"]

    for i, label in enumerate(target_labels):
        if i >= sv.shape[0]:
            print(f"  [SKIP] No SHAP output for component '{label}'")
            continue

        sv_i = sv[i]                             # (N, F)

        shap.summary_plot(sv_i, sample, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary [{label}] — {event}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"shap_summary_pxpy_{label}_{event}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.show(); plt.close()

        shap.summary_plot(sv_i, sample, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance [{label}] — {event}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"shap_bar_pxpy_{label}_{event}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.show(); plt.close()

        mean_shap = np.abs(sv_i).mean(axis=0)
        ranked    = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)
        print(f"\n  Feature importance (mean |SHAP|) [{label}] for {event}:")
        for name, val in ranked:
            print(f"    {name:<35} {val:.4f}")

        pd.DataFrame(ranked, columns=["feature", "mean_abs_shap"]).to_csv(
            os.path.join(output_dir, f"shap_importance_pxpy_{label}_{event}.csv"), index=False
        )

    # ── combined (media sui due output) ───────────────────────────────────────
    sv_combined   = sv.mean(axis=0)              # (N, F)
    mean_shap_all = np.abs(sv_combined).mean(axis=0)
    ranked_all    = sorted(zip(feature_names, mean_shap_all),
                           key=lambda x: x[1], reverse=True)

    shap.summary_plot(sv_combined, sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance [combined] — {event}",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"shap_bar_pxpy_combined_{event}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.show(); plt.close()

    print(f"\n  Feature importance (mean |SHAP|) [combined] for {event}:")
    for name, val in ranked_all:
        print(f"    {name:<35} {val:.4f}")

    pd.DataFrame(ranked_all, columns=["feature", "mean_abs_shap"]).to_csv(
        os.path.join(output_dir, f"shap_importance_pxpy_combined_{event}.csv"), index=False
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

        y_pred, y_true, met_pxpy = predict_on_dataset(
            file_path   = file_path,
            model       = model,
            scaler      = scaler,
            run_summary = run_summary,
            output_dir  = OUTPUT_DIR,
        )

        # ── rebuild X_test (scaled) for SHAP ─────────────────────────────────
        _, data = _read_data(file_path)
        X_test  = np.column_stack([data[f] for f in feature_names]).astype(np.float32)
        X_test  = _apply_standardize(X_test, scaler["mean"], scaler["std"])

        _feature_importance_shap(
            model         = model,
            X_test        = X_test,
            feature_names = feature_names,
            event         = event_name,
            output_dir    = OUTPUT_DIR,
        )