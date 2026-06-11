from MET_model_utils import (
    _read_data, _data_parsing, _create_model, _standardize,
    _build_tensorflow_dataset, _run_fold, _model_selection,
    StopAtTrainLoss, _retraining,
    _feature_importance_shap, _save_run_summary,
    mp, tf, keras, np, pd, os
)


# Branch configuration per dataset
BRANCH_MAP = {
    "ZZTo2L2Nu": [
        "GenMET_pt", "MET_covXX", "MET_covXY", "MET_pt", "MET_significance",
        "PV_z", "PV_chi2", "PV_score",
        "Electron_dxy_st", "Electron_dz_st", "Electron_eta_st", "Electron_phi_st", "Electron_pt_st",
        "Electron_dxy_nd", "Electron_dz_nd", "Electron_eta_nd", "Electron_phi_nd", "Electron_pt_nd",
        "Muon_dxy_st", "Muon_dz_st", "Muon_eta_st", "Muon_phi_st", "Muon_pt_st",
        "Muon_dxy_nd", "Muon_dz_nd", "Muon_eta_nd", "Muon_phi_nd", "Muon_pt_nd",
        "nElectron", "nMuon", "nSV",
        "Electron_charge_st", "Electron_charge_nd", "Muon_charge_st", "Muon_charge_nd"
    ],
    "HToAATo2Mu2B": [
        "GenMET_pt", "MET_covXX", "MET_covXY", "MET_pt", "MET_significance",
        "fixedGridRhoFastjetAll", "PV_z", "PV_chi2", "PV_score",
        "Jet_eta_bst", "Jet_phi_bst", "Jet_btag_bst", "Jet_rawFactor_bst", "Jet_chHEF_bst", "Jet_neHEF_bst",
        "Jet_eta_bnd", "Jet_phi_bnd", "Jet_btag_bnd", "Jet_rawFactor_bnd", "Jet_chHEF_bnd", "Jet_neHEF_bnd",
        "Muon_eta_st", "Muon_phi_st", "Muon_pt_st", "Muon_eta_nd", "Muon_phi_nd", "Muon_pt_nd",
        "SV_dlenSig_bst", "SV_mass_bst", "M_mumu", "M_bb", "M_mumu_bb", "dR_MET_bb",
        "MET_projection_par", "MET_projection_perp", "dPhi_MET_mu1", "dPhi_MET_jet1", "HT",
        "nJet", "nMuon", "nSV", "Muon_charge_st", "Muon_charge_nd"
    ],
    "augmented_HToAATo2Mu2B": [
        "GenMET_pt", "MET_covXX", "MET_pt",
        "fixedGridRhoFastjetAll", "PV_z", "PV_score",
        "Jet_eta_bst", "Jet_phi_bst", "Jet_rawFactor_bst", "Jet_chHEF_bst", "Jet_neHEF_bst",
        "Jet_eta_bnd", "Jet_phi_bnd", "Jet_btag_bnd", "Jet_rawFactor_bnd", "Jet_chHEF_bnd", "Jet_neHEF_bnd",
        "Muon_eta_st", "Muon_phi_st", "Muon_pt_st", "Muon_eta_nd", "Muon_phi_nd", "Muon_pt_nd",
        "SV_mass_bst", "M_mumu", "M_bb", "M_mumu_bb", "dR_MET_bb",
        "MET_projection_par", "MET_projection_perp", "dPhi_MET_mu1", "dPhi_MET_jet1", "HT",
        "nJet", "nMuon", "nSV", "Muon_charge_st", "Muon_charge_nd"
    ],
}

DEFAULT_BRANCHES = [
    "nJet", "MET_pt", "MET_significance",
    "MET_covXX", "MET_covXY",
    "PV_chi2", "PV_score", "PV_z",
    "nSV", "GenMET_pt"
]

# Per-dataset training configuration
DATASET_CFG = {
    "ZZTo2L2Nu": {
        # model
        "clipnorm":            None,
        "l2_reg":              0.0,
        # callbacks
        "lr_patience":         20,
        "es_patience_search":  80,
        "rlrop_factor":        0.5,
        "rlrop_min_delta":     1e-4,
        "rlrop_min_lr":        1e-6,
        # training loop
        "max_epochs_search":   1500,
        "max_epochs_retrain":  1000,
        "target_transform":    "none",
        # data split
        "test_size":           0.2,
        "random_seed_split":   42,
        "random_seed_kfold":   42,
        "n_folds":             3,
    },
    "HToAATo2Mu2B": {
        # model
        "clipnorm":            0.5,
        "l2_reg":              1e-3,
        # callbacks
        "lr_patience":         10,
        "es_patience_search":  60,
        "rlrop_factor":        0.5,
        "rlrop_min_delta":     1e-3,
        "rlrop_min_lr":        1e-6,
        # training loop
        "max_epochs_search":   2000,
        "max_epochs_retrain":  3000,
        "target_transform":    "log1p",
        # data split
        "test_size":           0.2,
        "random_seed_split":   42,
        "random_seed_kfold":   42,
        "n_folds":             3,
    },
    "augmented_HToAATo2Mu2B": {
        # model
        "clipnorm":            0.5,
        "l2_reg":              1e-3,
        # callbacks
        "lr_patience":         10,
        "es_patience_search":  60,
        "rlrop_factor":        0.5,
        "rlrop_min_delta":     1e-3,
        "rlrop_min_lr":        1e-6,
        # training loop
        "max_epochs_search":   2000,
        "max_epochs_retrain":  3000,
        "target_transform":    "log1p",
        # data split
        "test_size":           0.2,
        "random_seed_split":   42,
        "random_seed_kfold":   42,
        "n_folds":             3,
    },
}


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    import tracemalloc
    tracemalloc.start()
    
    print(f"Keras version: {keras.__version__}")
    print(f"Tensorflow version: {tf.__version__}")
    
    EVENT = "HToAATo2Mu2B"
    INPUT_FILE = f"CleanedDatasets/{EVENT}.parquet"
    OUTPUT_DIR = f"Results/"

    print("GPUs available:", tf.config.list_physical_devices("GPU"))

    # Load per-dataset configuration
    cfg = DATASET_CFG.get(EVENT, DATASET_CFG["ZZTo2L2Nu"])
    print(f"    Using config for '{EVENT}': {cfg}\n")
    
    # Load data
    branches, data = _read_data(INPUT_FILE, BRANCH_MAP, DEFAULT_BRANCHES)

    # Parse into feature matrix & target
    features, target = _data_parsing(branches, data)

    # Grid-search with K-Fold CV (test set held-out throughout)
    HPARAM_GRIDS = {
        "ZZTo2L2Nu": {
            "architecture": [
                (256, 128),
                (512, 256),
                (512, 256, 128),
                (256, 128, 64),
            ],
            "batch_size":    [256, 512, 1024],
            "learning_rate": [1e-3, 5e-4],
        },
        "HToAATo2Mu2B": {
            "architecture": [
                (32,), (64,),# (128,),
                (64, 32),
            ],
            "batch_size":    [64, 128, 256],
            "learning_rate": [1e-3, 5e-4, 1e-4, 5e-5]#, 1e-5]
        },
        "augmented_HToAATo2Mu2B": {
            "architecture": [
                (32,), (64,), (128,),
                (64, 32),
            ],
            "batch_size":    [64, 128, 256],
            "learning_rate": [1e-3, 5e-4, 1e-4, 5e-5]#, 1e-5]
        },
    }

    if EVENT not in HPARAM_GRIDS:
        raise ValueError(f"No hparam_grid defined for event '{EVENT}'. "
                         f"Available: {list(HPARAM_GRIDS.keys())}")
        
    hparam_grid = HPARAM_GRIDS[EVENT]

    X_train, y_train, X_test, y_test, best_params, target_train_loss, best_fold_histories, best_cv_metrics, topk_buffer, test_idx = _model_selection(
        features, target, hparam_grid, OUTPUT_DIR, EVENT, cfg
    )

    rows = []
    for fh in best_fold_histories:
        for ep, (tl, vl) in enumerate(zip(fh["loss"], fh["val_loss"]), 1):
            rows.append({"fold": fh["fold"], "epoch": ep, "loss": tl, "val_loss": vl})

    pd.DataFrame(rows).to_parquet(
        os.path.join(OUTPUT_DIR, f"fold_histories_{EVENT}.parquet"), index=False
    )

    # Retrain best model and predict on test set
    y_test_pred_GeV, y_test_GeV, history, best_model, retrain_metrics, X_test_scaled = _retraining(
        X_train, y_train, X_test, y_test, best_params, cfg, target_train_loss
    )

    _save_run_summary(
        event=EVENT,
        best_params=best_params,
        cfg=cfg,
        best_cv_metrics=best_cv_metrics,
        retrain_metrics=retrain_metrics,
        y_test=y_test_GeV,
        y_test_pred=y_test_pred_GeV,
        output_dir=OUTPUT_DIR,
    )

    # Feature importance via SHAP
    feature_names = [b for b in branches if b != "GenMET_pt"]
    _feature_importance_shap(best_model, X_test_scaled, feature_names, EVENT, OUTPUT_DIR)

    # Save learning curves for external plotting
    pd.DataFrame(history.history).to_parquet(
        os.path.join(OUTPUT_DIR, f"learning_curves_{EVENT}.parquet"), index=False
    )

    # Save scatter data for external plotting
    pd.DataFrame({
        "y_test":      y_test_GeV,
        "y_test_pred": y_test_pred_GeV.flatten(),
    }).to_parquet(os.path.join(OUTPUT_DIR, f"METpred_vs_GenMET_{EVENT}.parquet"), index=False)
