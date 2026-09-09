from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from models import MLPRegressor
from utils import save_json, set_seed

DATA_PATH = ROOT / "data" / "raw" / "boston_housing.csv"
MODEL_PATH = ROOT / "models" / "mlp_regressor.pth"
SCALER_PATH = ROOT / "models" / "standard_scaler.joblib"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
TARGET = "MEDV"


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "MSE": float(mse), "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y_true, y_pred))}


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Boston Housing dataset missing: {DATA_PATH}. "
            "Provide the real Boston Housing data file; no synthetic fallback is used."
        )
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    return df


def train_mlp(X_train, y_train, X_val, y_val, input_dim, device):
    model = MLPRegressor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1,1)), batch_size=32, shuffle=True)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1,1)
    history = []
    best_val = float("inf")
    best_state = None
    patience, wait, max_epochs = 20, 0, 300
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def main():
    set_seed(42)
    for p in [MODEL_PATH.parent, FIGURES, TABLES]: p.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = df[FEATURES].to_numpy(dtype=np.float32), df[TARGET].to_numpy(dtype=np.float32)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = train_mlp(X_train_s, y_train, X_val_s, y_val, X_train_s.shape[1], device)
    torch.save({"model_state_dict": model.state_dict(), "input_dim": len(FEATURES)}, MODEL_PATH)
    model.eval()
    with torch.no_grad():
        mlp_pred = model(torch.tensor(X_test_s, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    comparisons = {"MLP": metrics(y_test, mlp_pred)}
    baselines = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=42, max_depth=5),
        "Random Forest Regression": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    for name, est in baselines.items():
        est.fit(X_train_full, y_train_full)
        comparisons[name] = metrics(y_test, est.predict(X_test))
    pd.DataFrame(history).to_csv(TABLES / "training_history.csv", index=False)
    pd.DataFrame(comparisons).T.reset_index(names="Model").to_csv(TABLES / "model_comparison.csv", index=False)
    pd.DataFrame({"actual": y_test, "predicted": mlp_pred}).to_csv(TABLES / "mlp_test_predictions.csv", index=False)
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    plt.figure(); plt.plot(h.epoch, h.train_loss, label="train"); plt.plot(h.epoch, h.val_loss, label="validation"); plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES / "training_loss.png", dpi=150); plt.close()
    plt.figure(); plt.scatter(y_test, mlp_pred, alpha=0.7); lims=[min(y_test.min(), mlp_pred.min()), max(y_test.max(), mlp_pred.max())]; plt.plot(lims, lims); plt.xlabel("Actual Prices"); plt.ylabel("Predicted Prices"); plt.title("Actual vs. Predicted House Prices"); plt.tight_layout(); plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=150); plt.close()
    comp_df = pd.DataFrame(comparisons).T
    plt.figure(); plt.bar(comp_df.index, comp_df["RMSE"]); plt.ylabel("RMSE"); plt.title("Model RMSE Comparison"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(FIGURES / "model_rmse_comparison.png", dpi=150); plt.close()
    best_epoch = int(h.loc[h.val_loss.idxmin(), "epoch"])
    summary = {"dataset": "Boston Housing", "samples": int(len(df)), "features": len(FEATURES), "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}, "seed": 42, "architecture": "13 -> 32 -> 16 -> 1", "activation": "ReLU", "loss": "MSELoss", "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 32, "max_epochs": 300, "patience": 20, "best_epoch": best_epoch, "device": str(device), "metrics": comparisons}
    save_json(summary, RESULTS / "run_summary.json")
    print(pd.DataFrame(comparisons).T)
    print("ACTUAL TRAINING PASS")

if __name__ == "__main__": main()
