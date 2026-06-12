import os
import json
#import tensorflow as tf
import keras
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def load_model_and_scaler(model_dir: str, event: str):
    """
    Loads a previously saved model, scaler, and inference config.

    Returns
    -------
    model          : compiled Keras model ready for .predict()
    scaler         : dict with keys 'mean' and 'std' (np.ndarray)
    inference_cfg  : dict with feature_names, target_transform, best_params
    """
    model_path  = os.path.join(model_dir, f"{event}_model.keras")
    scaler_path = os.path.join(model_dir, f"{event}_scaler.npz")
    cfg_path    = os.path.join(model_dir, f"{event}_inference_cfg.json")

    for path in (model_path, scaler_path, cfg_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing saved artefact: {path}")

    model = keras.models.load_model(model_path)

    npz    = np.load(scaler_path)
    scaler = {"mean": npz["mean"], "std": npz["std"]}

    with open(cfg_path) as f:
        inference_cfg = json.load(f)

    print(f"Loaded model + scaler for event '{event}' from '{model_dir}'.")
    return model, scaler, inference_cfg


def predict_on_dataset(file_path: str, model_dir: str, event: str,
                       output_dir: str = None):
    """
    End-to-end inference on a new parquet dataset using a previously saved
    model and scaler.  No retraining is performed.

    Steps
    -----
    1. Load model, scaler, and inference config from model_dir.
    2. Read the new parquet (all columns).
    3. Validate that the expected features are present.
    4. Standardize with the saved scaler.
    5. Predict and de-transform if necessary.
    6. Optionally save predictions as a parquet file.

    Returns
    -------
    y_pred_GeV : np.ndarray  — predictions in GeV
    y_true_GeV : np.ndarray or None — ground truth if 'GenMET_pt' is in the file
    """
    model, scaler, inference_cfg = load_model_and_scaler(model_dir, event)

    feature_names    = inference_cfg["feature_names"]
    target_transform = inference_cfg["target_transform"]
    batch_size       = inference_cfg["best_params"]["batch_size"]

    # ── load new dataset ──────────────────────────────────────────────────
    branches, data = _read_data(file_path)

    missing = [f for f in feature_names if f not in data]
    if missing:
        raise ValueError(
            f"New dataset is missing features expected by the model: {missing}"
        )

    X_new = np.column_stack([data[f] for f in feature_names]).astype(np.float32)
    X_new_scaled = _apply_standardize(X_new, scaler["mean"], scaler["std"])

    # ── predict ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_new_scaled, batch_size=batch_size, verbose=0).flatten()

    if target_transform == "log1p":
        y_pred_GeV = np.expm1(y_pred)
    else:
        y_pred_GeV = y_pred

    # ── ground truth (optional) ───────────────────────────────────────────
    y_true_GeV = None
    if "GenMET_pt" in data:
        y_true_GeV = data["GenMET_pt"]
        rmse = float(np.sqrt(mean_squared_error(y_true_GeV, y_pred_GeV)))
        print(f"Test RMSE on '{os.path.basename(file_path)}': {rmse:.4f} GeV")

    # ── save predictions ──────────────────────────────────────────────────
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        new_event = os.path.splitext(os.path.basename(file_path))[0]
        out_path  = os.path.join(output_dir, f"predictions_{new_event}.parquet")
        result_df = pd.DataFrame({"y_pred_GeV": y_pred_GeV})
        if y_true_GeV is not None:
            result_df["y_true_GeV"] = y_true_GeV
        result_df.to_parquet(out_path, index=False)
        print(f"Predictions saved → {out_path}")

    return y_pred_GeV, y_true_GeV


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

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
