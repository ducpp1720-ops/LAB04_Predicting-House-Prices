"""End-to-end California Housing MLP regression training pipeline."""
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
from models import HousePriceMLP
from utils import save_json, set_seed

TRAIN_PATH = ROOT / "data" / "raw" / "california_housing_train.csv"
TEST_PATH = ROOT / "data" / "raw" / "california_housing_test.csv"
MODEL_PATH = ROOT / "models" / "mlp_house_price.pth"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
EDA_DIR = RESULTS / "eda"

FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
TARGET = "median_house_value"
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_data():
    for path in (TRAIN_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file is missing: {path}. "
                "Download the real California Housing train/test CSV files; no synthetic fallback is used."
            )
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    required = FEATURES + [TARGET]
    expected_columns = required
    for name, df in (("train", train_df), ("test", test_df)):
        if list(df.columns) != expected_columns:
            raise ValueError(f"{name} dataset columns do not match the required schema")
        if df[required].isna().any().any():
            raise ValueError(f"{name} dataset contains missing values in required columns")
    if len(train_df) != 17000 or len(test_df) != 3000:
        raise ValueError(f"Unexpected dataset sizes: train={len(train_df)}, test={len(test_df)}")
    return train_df, test_df


def train_mlp(X_train, y_train, X_val, y_val, input_dim, device):
    model = HousePriceMLP(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
    history, best_val, best_state = [], float("inf"), None
    for epoch in range(1, EPOCHS + 1):
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
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def run_baselines(X_train, y_train, X_test, y_test):
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=SEED, max_depth=10),
        "Random Forest Regression": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    results = {}
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        results[name] = regression_metrics(y_test, estimator.predict(X_test))
    return results


def save_eda(train_df):
    """Save auditable EDA tables and plots required by the lab rubric."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "shape": list(train_df.shape),
        "dtypes": {k: str(v) for k, v in train_df.dtypes.items()},
        "missing_values": train_df.isna().sum().to_dict(),
        "duplicate_rows": int(train_df.duplicated().sum()),
        "describe": train_df.describe().to_dict(),
    }
    save_json(summary, EDA_DIR / "eda_summary.json")
    train_df.describe().T.to_csv(EDA_DIR / "descriptive_statistics.csv")
    train_df.corr(numeric_only=True).to_csv(EDA_DIR / "correlation_matrix.csv")

    numeric = train_df[FEATURES + [TARGET]]
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))
    outlier_summary = pd.DataFrame({
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": q1 - 1.5 * iqr,
        "upper_bound": q3 + 1.5 * iqr,
        "outlier_count": outlier_mask.sum(),
        "outlier_pct": outlier_mask.mean() * 100,
    })
    outlier_summary.to_csv(EDA_DIR / "outlier_summary.csv")

    distribution_summary = numeric.agg(["min", "max", "mean", "median", "std", "skew"]).T
    distribution_summary.to_csv(EDA_DIR / "distribution_summary.csv")

    import matplotlib.pyplot as plt
    numeric.hist(figsize=(14, 10), bins=30)
    plt.suptitle("California Housing Feature Distributions")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "feature_distributions.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 7))
    numeric.boxplot(rot=30)
    plt.title("Outlier Inspection — IQR Boxplots")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "outlier_boxplots.png", dpi=150)
    plt.close()


def main():
    set_seed(SEED)
    for p in [MODEL_PATH.parent, FIGURES, TABLES, EDA_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data()
    save_eda(train_df)

    X_full = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_full = train_df[TARGET].to_numpy(dtype=np.float32)
    X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.20, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = train_mlp(
        X_train_s, y_train, X_val_s, y_val, len(FEATURES), device
    )
    torch.save({"model_state_dict": model.state_dict(), "input_dim": len(FEATURES)}, MODEL_PATH)

    model.eval()
    with torch.no_grad():
        mlp_pred = model(
            torch.tensor(X_test_s, dtype=torch.float32, device=device)
        ).cpu().numpy().ravel()

    comparisons = {"MLP": regression_metrics(y_test, mlp_pred)}
    comparisons.update(run_baselines(X_full, y_full, X_test, y_test))

    pd.DataFrame(history).to_csv(TABLES / "training_history.csv", index=False)
    pd.DataFrame(comparisons).T.reset_index(names="Model").to_csv(
        TABLES / "model_comparison.csv", index=False
    )
    pd.DataFrame({"actual": y_test, "predicted": mlp_pred}).to_csv(
        TABLES / "mlp_test_predictions.csv", index=False
    )

    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(h.epoch, h.train_loss, label="Train")
    plt.plot(h.epoch, h.val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MLP Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, mlp_pred, alpha=0.35)
    lims = [min(y_test.min(), mlp_pred.min()), max(y_test.max(), mlp_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("Actual House Value")
    plt.ylabel("Predicted House Value")
    plt.title("Actual vs Predicted — MLP")
    plt.tight_layout()
    plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=150)
    plt.close()

    comp_df = pd.DataFrame(comparisons).T
    plt.figure(figsize=(9, 5))
    plt.bar(comp_df.index, comp_df["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Regression Model RMSE Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_rmse_comparison.png", dpi=150)
    plt.close()

    save_json(
        {
            "dataset": "California Housing (Google MLCC train/test CSV)",
            "train_samples": int(len(train_df)),
            "official_test_samples": int(len(test_df)),
            "features": FEATURES,
            "target": TARGET,
            "internal_split": {"train": int(len(X_train)), "validation": int(len(X_val))},
            "test_source": "official california_housing_test.csv; never used for fitting or scaler fitting",
            "seed": SEED,
            "architecture": "8 -> 32 -> 16 -> 1",
            "activation": "ReLU hidden layers; linear output",
            "loss": "MSELoss",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "device": str(device),
            "metrics": comparisons,
        },
        RESULTS / "run_summary.json",
    )
    print(pd.DataFrame(comparisons).T)
    print("ACTUAL TRAINING PASS")


if __name__ == "__main__":
    main()
