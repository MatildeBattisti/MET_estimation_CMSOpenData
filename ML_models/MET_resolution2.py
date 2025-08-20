import uproot
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import matplotlib.pyplot as plt

# === Step 1: Caricamento dati ===
def read_data():
    file = uproot.open("../skimmed_datasets/skimmed1_specific.root")
    tree = file["Events"]

    params = [
        "nJet",
        #"Jet_eta_bst",
        "Jet_pt_bst",
        "Jet_phi_bst",
        "Jet_mass_bst",
        "Jet_eta_bnd",
        "Jet_pt_bnd",
        "Jet_phi_bnd",
        "Jet_mass_bnd",
        "Muon_eta_st",
        "Muon_pt_st",
        "Muon_phi_st",
        "Muon_mass_st",
        "Muon_charge_st",
        "Muon_eta_nd",
        "Muon_pt_nd",
        "Muon_phi_nd",
        "Muon_mass_nd",
        "Muon_charge_nd",
        "Muon_Deltaeta",
        #"Muon_Deltaphi",
        "Muon_DeltaR",
        "Muon_InvMass",
        "Deltaphi_METJbest",
        "Deltaphi_METJbnd",
        #"Deltaphi_METmst",
        #"Deltaphi_METmnd",
        "MET_phi",
        "MET_significance",
        "MET_pt"
    ]
    target = ["GenMET_pt"]
    data = tree.arrays(params + target, library="np")
    return params, target, data

def data_engineering(data):
    npFeatures = np.array([
        data[k] for k in data.keys() if k != "GenMET_pt"
    ]).T
    npTarget = data["GenMET_pt"]
    return npFeatures, npTarget

# === Step 2: Crea il modello ===
def create_model(input_dim, n_layers=2, n_units=64, activation='relu', learning_rate=1e-3):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for _ in range(n_layers):
        model.add(layers.Dense(n_units, activation=activation))

    model.add(layers.Dense(1))  # output layer (regression)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])
    return model

# === Step 3: K-Fold CV + Grid Search ===
def run_kfold_cv(X, y, param_grid, n_splits=5):
    best_model = None
    best_val_loss = float("inf")
    best_params = None

    # Normalizzazione costante su tutti i fold
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"\n🔍 Trying parameters: {param_dict}")
        val_losses = []

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_index], X_scaled[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = create_model(
                input_dim=X_train.shape[1],
                n_layers=param_dict['n_layers'],
                n_units=param_dict['n_units'],
                activation=param_dict['activation'],
                learning_rate=param_dict['learning_rate']
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      epochs=100,
                      batch_size=param_dict['batch_size'],
                      verbose=0,
                      callbacks=[early_stop])

            val_pred = model.predict(X_val)
            val_loss = mean_squared_error(y_val, val_pred)
            val_losses.append(val_loss)

        avg_val_loss = np.mean(val_losses)
        print(f"✅ Avg val MSE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = param_dict
            best_model = model

    print(f"\n🏆 Best params: {best_params}")
    return best_model, best_params, scaler_X

# === Step 4: Test finale ===
def evaluate_on_test(model, scaler_X, X_test, y_test):
    X_test_scaled = scaler_X.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n🧪 Final test MSE: {mse:.4f}")
    return mse, y_pred

# === MAIN PIPELINE ===
if __name__ == "__main__":
    params, target, raw_data = read_data()
    X, y = data_engineering(raw_data)

    # Split dataset (holdout test set)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Grid of hyperparameters
    param_grid = {
        'n_layers': [1, 2],
        'n_units': [32, 64],
        'activation': ['relu'],
        'batch_size': [64, 128],
        'learning_rate': [1e-3, 1e-4],
    }

    # Train and find best model
    model, best_params, scaler = run_kfold_cv(X_trainval, y_trainval, param_grid, n_splits=5)

    # Evaluate on holdout test set
    mse, y_pred = evaluate_on_test(model, scaler, X_test, y_test)

    # Scatter plot of test results
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    
    plt.title('Scatter Plot of Predicted vs True')
    plt.xlabel('TrueMET')
    plt.ylabel('PredictedMET')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
