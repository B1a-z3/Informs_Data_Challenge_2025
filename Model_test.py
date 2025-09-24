# run.py
# Consolidated script to train, predict, and evaluate both SARIMA and Seq2Seq models.

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ======================================================================================
# --- 1. CONFIGURATION & SETUP ---
# ======================================================================================

# --- Directory Paths ---
SPLIT_DIR = ""
OUT_DIR = "results/"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Seq2Seq Model Hyperparameters ---
SEQ_LEN = 24       # Use last 24 hours of data to predict the future
BATCH_SIZE = 64
EPOCHS = 5         # Number of training epochs
LR = 1e-3          # Learning rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DHIDDEN = 64       # Hidden dimension size of the LSTM
NLAYERS = 1        # Number of LSTM layers

# ======================================================================================
# --- 2. SARIMA MODEL PIPELINE ---
# (Based on demo_sarima.py)
# ======================================================================================

def safe_fit_sarimax(y: np.ndarray):
    """
    Fits a SARIMAX(1,0,1) model. Returns None if fitting fails or data is trivial.
    """
    y = np.asarray(y, dtype=float).flatten()
    # Require at least 8 data points and some variance
    if len(y) < 8 or np.allclose(y, y[0]):
        return None
    try:
        # Define a simple SARIMAX model
        model = SARIMAX(y, order=(1, 0, 1), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        return res
    except Exception:
        # Return None if any error occurs during fitting
        return None

def run_sarima_pipeline():
    """
    Trains a SARIMA model for each county and saves 24h and 48h predictions.
    """
    print("\n" + "="*50)
    print("--- Running SARIMA Pipeline ---")
    print("="*50)

    # --- Load Data ---
    try:
        ds_train = xr.open_dataset(os.path.join(SPLIT_DIR, "train.nc"))
        ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
        ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure data files are in the '{SPLIT_DIR}' directory.")
        return

    counties = list(ds_train.location.values)
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)

    rows24, rows48 = [], []

    print(f"Training one SARIMA model for each of the {len(counties)} counties...")
    for county in tqdm(counties, desc="Training SARIMA models"):
        y_train = ds_train.out.sel(location=county).values.astype(float).flatten()
        res = safe_fit_sarimax(y_train)

        # Generate forecasts; use zeros if model fitting failed
        if res is None:
            pred48 = np.zeros(len(ts48), dtype=float)
        else:
            pred48 = np.asarray(res.forecast(steps=len(ts48)), dtype=float)

        pred24 = pred48[:len(ts24)]

        # Collect predictions in long format
        rows24.append(pd.DataFrame({"timestamp": ts24, "location": county, "pred": pred24}))
        rows48.append(pd.DataFrame({"timestamp": ts48, "location": county, "pred": pred48}))

    # --- Concatenate & Save Predictions ---
    df24 = pd.concat(rows24, ignore_index=True)
    df48 = pd.concat(rows48, ignore_index=True)

    out24_path = os.path.join(OUT_DIR, "sarimax_pred_24h.csv")
    out48_path = os.path.join(OUT_DIR, "sarimax_pred_48h.csv")
    df24.to_csv(out24_path, index=False)
    df48.to_csv(out48_path, index=False)

    print("\nSARIMA predictions saved:")
    print(f"  - {out24_path}")
    print(f"  - {out48_path}")

    # --- Clean up ---
    ds_train.close()
    ds_test24.close()
    ds_test48.close()

# ======================================================================================
# --- 3. SEQ2SEQ MODEL PIPELINE ---
# (Based on demo_seq2seq.py)
# ======================================================================================

# --- Seq2Seq Model Definition ---
class SimpleSeq2Seq(nn.Module):
    def __init__(self, din, dh, nl, horizon):
        super().__init__()
        self.lstm = nn.LSTM(din, dh, nl, batch_first=True)
        self.head = nn.Linear(dh, horizon)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        _, (h, _) = self.lstm(x)
        # Get the last hidden state of the last layer
        h_last = h[-1]  # Shape: (batch_size, hidden_dim)
        return self.head(h_last)  # Shape: (batch_size, horizon)

# --- Data Preparation Helpers ---
def zfit(arr):
    """Calculates mean and standard deviation for Z-score normalization."""
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd == 0, 1.0, sd) # Avoid division by zero for constant features
    return mu, sd

def zapply(arr, mu, sd):
    """Applies Z-score normalization."""
    return (arr - mu) / sd

def build_windows(X_loc, y_loc, seq_len, horizon):
    """Builds sliding windows (X, y) for a single location."""
    n_samples = len(y_loc) - seq_len - horizon + 1
    if n_samples <= 0:
        return np.empty((0, seq_len, X_loc.shape[1])), np.empty((0, horizon))

    Xw, Yw = [], []
    for i in range(n_samples):
        Xw.append(X_loc[i : i + seq_len])
        Yw.append(y_loc[i + seq_len : i + seq_len + horizon])
    return np.asarray(Xw, dtype=float), np.asarray(Yw, dtype=float)

def prepare_training_data(ds_train, horizon):
    """Builds global scalers and training windows from all counties."""
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    T, L, F = w.shape

    # Global scalers over all time and counties
    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))

    # Apply scaling
    y_sc = zapply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, L)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, L, F)

    Din = 1 + F  # outage feature + weather features
    X_list, Y_list = [], []

    print(f"Building training windows for {L} counties...")
    for li in range(L):
        y_loc = y_sc[:, li]
        w_loc = w_sc[:, li, :]
        X_loc = np.concatenate([y_loc.reshape(-1, 1), w_loc], axis=1)
        Xw, Yw = build_windows(X_loc, y_loc, SEQ_LEN, horizon)
        if len(Xw) > 0:
            X_list.append(Xw)
            Y_list.append(Yw)

    X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, SEQ_LEN, Din))
    Y = np.concatenate(Y_list, axis=0) if Y_list else np.empty((0, horizon))
    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return X, Y, Din, scalers

# --- Model Training and Inference ---
def train_seq2seq_model(X, Y, din, horizon):
    """Trains a new Seq2Seq model."""
    if len(X) == 0:
        print("Warning: No training windows could be built. Skipping model training.")
        return None

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleSeq2Seq(din=din, dh=DHIDDEN, nl=NLAYERS, horizon=horizon).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Training Seq2Seq model for {horizon}h horizon on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(loader.dataset)
        elapsed = time.time() - start_time
        print(f"  [Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s")

    return model

@torch.no_grad()
def infer_seq2seq(model, Xin):
    """Performs inference with the trained model."""
    model.eval()
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))
    return out.cpu().numpy()

def predict_with_seq2seq(ds_train, model, scalers, horizon, ts_future):
    """Generates predictions for each county using the trained Seq2Seq model."""
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    T, L, F = w.shape
    Din = 1 + F

    # Apply scaling with pre-computed scalers
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, L)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, L, F)

    rows = []
    print(f"Generating predictions for {len(locs)} counties...")
    for li, county in enumerate(tqdm(locs, desc="Predicting per county")):
        if T < SEQ_LEN or model is None:
            pred = np.zeros(horizon)
        else:
            y_loc = y_sc[:, li]
            w_loc = w_sc[:, li, :]
            X_loc = np.concatenate([y_loc.reshape(-1, 1), w_loc], axis=1)
            Xin = X_loc[-SEQ_LEN:].reshape(1, SEQ_LEN, Din)
            pred_sc = infer_seq2seq(model, Xin)[0]

            # Invert scaling and clip to be non-negative
            pred = (pred_sc * scalers["y_sd"]) + scalers["y_mu"]
            pred = np.clip(np.nan_to_num(pred), 0.0, None).flatten()

        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred}))

    return pd.concat(rows, ignore_index=True)

def run_seq2seq_pipeline():
    """
    Trains two Seq2Seq models (24h and 48h) and saves their predictions.
    """
    print("\n" + "="*50)
    print("--- Running Seq2Seq Pipeline ---")
    print("="*50)

    # --- Load Data ---
    try:
        ds_train = xr.open_dataset(os.path.join(SPLIT_DIR, "train.nc"))
        ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
        ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure data files are in the '{SPLIT_DIR}' directory.")
        return

    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    # --- Train 24h Model & Predict ---
    X24, Y24, Din, scalers24 = prepare_training_data(ds_train, horizon=H24)
    model24 = train_seq2seq_model(X24, Y24, Din, H24)
    df24 = predict_with_seq2seq(ds_train, model24, scalers24, H24, ts24)

    # --- Train 48h Model & Predict ---
    X48, Y48, _, scalers48 = prepare_training_data(ds_train, horizon=H48)
    model48 = train_seq2seq_model(X48, Y48, Din, H48)
    df48 = predict_with_seq2seq(ds_train, model48, scalers48, H48, ts48)

    # --- Save Predictions ---
    out24_path = os.path.join(OUT_DIR, "seq2seq_pred_24h.csv")
    out48_path = os.path.join(OUT_DIR, "seq2seq_pred_48h.csv")
    df24.to_csv(out24_path, index=False)
    df48.to_csv(out48_path, index=False)

    print("\nSeq2Seq predictions saved:")
    print(f"  - {out24_path}")
    print(f"  - {out48_path}")

    # --- Clean up ---
    ds_train.close()
    ds_test24.close()
    ds_test48.close()

# ======================================================================================
# --- 4. EVALUATION PIPELINE (Corrected) ---
# (Based on demo_evaluate.py)
# ======================================================================================

def rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def load_truth_wide(nc_path: str) -> pd.DataFrame:
    """
    Loads ground truth data into a wide DataFrame, ensuring location IDs are strings.
    """
    if not os.path.exists(nc_path):
        raise FileNotFoundError(nc_path)
    ds = xr.open_dataset(nc_path)
    try:
        y = ds["out"].transpose("timestamp", "location").values.astype(float)
        ts = pd.to_datetime(ds["timestamp"].values)
        # CRITICAL FIX: Ensure location IDs are strings to match CSV data.
        locs = list(map(str, ds["location"].values))
    finally:
        ds.close()
    df = pd.DataFrame(y, columns=locs, index=ts).reset_index().rename(columns={"index": "timestamp"})
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def long_to_wide(df_long: pd.DataFrame, locs: list[str]) -> pd.DataFrame:
    """Pivots a long DataFrame to wide, ensuring all specified location columns exist."""
    if df_long.empty:
        return pd.DataFrame(columns=["timestamp", *locs])

    # CRITICAL FIX: Ensure location column is string type before pivoting.
    df_long["location"] = df_long["location"].astype(str)
    
    pivot = df_long.pivot_table(
        index="timestamp", columns="location", values="pred", aggfunc="sum", fill_value=0.0
    )
    # Reindex to ensure all target locations are present and in the correct order
    pivot = pivot.reindex(columns=locs, fill_value=0.0)
    pivot = pivot.sort_index().reset_index()
    pivot.columns.name = None
    return pivot

def per_county_rmse_mean(df_truth_wide: pd.DataFrame, df_pred_long: pd.DataFrame) -> float:
    """Calculates the mean of per-county RMSEs after robustly merging data."""
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]

    # Use the robust function to pivot predictions to a wide format
    df_pred_wide = long_to_wide(df_pred_long, locs)

    # Merge on timestamp. With matching string-based column names, suffixes will now be applied correctly.
    df = df_truth_wide.merge(df_pred_wide, on="timestamp", how="inner", suffixes=("_true", "_pred"))

    if df.empty:
        print("Warning: No overlapping timestamps found between truth and predictions.")
        return float("nan")

    rmses = []
    for loc in locs:
        true_col = f"{loc}_true"
        pred_col = f"{loc}_pred"
        
        if true_col in df.columns and pred_col in df.columns:
            yt = df[true_col].values
            yp = df[pred_col].values
            rmses.append(rmse(yt, yp))
        else:
            # This case is unlikely with the fixes, but provides a helpful warning
            print(f"Warning: Missing columns for location {loc} after merge. This may indicate an issue in the prediction file.")
            # Fallback to zero prediction for this county to avoid crashing
            yt = df_truth_wide[loc].values
            yp = np.zeros_like(yt)
            rmses.append(rmse(yt, yp))
            
    return float(np.mean(rmses)) if rmses else float("nan")


def zero_baseline_rmse(df_truth_wide: pd.DataFrame) -> float:
    """Computes the RMSE for an all-zero prediction baseline."""
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]
    if not locs:
        return float("nan")
    rmses = []
    for loc in locs:
        y = df_truth_wide[loc].values
        p = np.zeros_like(y, dtype=float)
        rmses.append(rmse(y, p))
    return float(np.mean(rmses))

def run_evaluation(model_name: str):
    """
    Runs the evaluation for a given model's prediction files.
    """
    print("\n" + "="*50)
    print(f"--- Evaluating {model_name.upper()} Model ---")
    print("="*50)

    pred_24_path = os.path.join(OUT_DIR, f"{model_name}_pred_24h.csv")
    pred_48_path = os.path.join(OUT_DIR, f"{model_name}_pred_48h.csv")

    # Prefer demo tests; fall back to real tests if demos missing
    t24_path = os.path.join(SPLIT_DIR, "test_24h_demo.nc")
    if not os.path.exists(t24_path):
        t24_path = os.path.join(SPLIT_DIR, "test_24h.nc")
    
    t48_path = os.path.join(SPLIT_DIR, "test_48h_demo.nc")
    if not os.path.exists(t48_path):
        t48_path = os.path.join(SPLIT_DIR, "test_48h.nc")

    try:
        # Load truth data
        df_t24 = load_truth_wide(t24_path)
        df_t48 = load_truth_wide(t48_path)
        
        # Load predictions
        df_p24 = pd.read_csv(pred_24_path, parse_dates=["timestamp"])
        df_p48 = pd.read_csv(pred_48_path, parse_dates=["timestamp"])
    except FileNotFoundError as e:
        print(f"Error: Cannot find data for evaluation: {e}. Skipping evaluation for {model_name}.")
        return

    # --- Calculate Model RMSE ---
    r24 = per_county_rmse_mean(df_t24, df_p24)
    r48 = per_county_rmse_mean(df_t48, df_p48)
    r_avg = np.nanmean([r24, r48])

    # --- Calculate Zero Baseline RMSE ---
    z24 = zero_baseline_rmse(df_t24)
    z48 = zero_baseline_rmse(df_t48)
    z_avg = np.nanmean([z24, z48])

    print("Per-county-first RMSE (mean across counties):")
    print(f"  24h Horizon: {r24:.4f} (Zero Baseline: {z24:.4f})")
    print(f"  48h Horizon: {r48:.4f} (Zero Baseline: {z48:.4f})")
    print(f"  Average:     {r_avg:.4f} (Zero Baseline: {z_avg:.4f})")

# ======================================================================================
# --- 5. MAIN EXECUTION BLOCK ---
# ======================================================================================
if __name__ == "__main__":
    # --- Step 1: Train SARIMA and generate predictions ---
    run_sarima_pipeline()

    # --- Step 2: Train Seq2Seq and generate predictions ---
    run_seq2seq_pipeline()

    # --- Step 3: Evaluate both models ---
    run_evaluation("sarimax")
    run_evaluation("seq2seq")

    print("\n" + "="*50)
    print("--- Pipeline Finished ---")
    print("="*50)