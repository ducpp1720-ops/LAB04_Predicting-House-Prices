"""Predict a California house value from eight input features."""
from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import HousePriceMLP

FEATURES = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
MODEL_PATH = ROOT / "models" / "mlp_house_price.pth"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


def main():
    parser = argparse.ArgumentParser(description="Predict median house value with the trained PyTorch MLP")
    parser.add_argument("--values", nargs=8, type=float, required=True, metavar="FEATURE")
    args = parser.parse_args()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = HousePriceMLP(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = joblib.load(SCALER_PATH)
    x = scaler.transform(np.asarray(args.values, dtype=np.float32).reshape(1, -1))
    model.eval()
    with torch.no_grad(): pred = model(torch.tensor(x, dtype=torch.float32)).item()
    print(f"Predicted median house value: ${pred:,.2f}")


if __name__ == "__main__": main()
