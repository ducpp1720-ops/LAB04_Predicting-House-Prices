from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import MLPRegressor

FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
MODEL_PATH = ROOT / "models" / "mlp_regressor.pth"
SCALER_PATH = ROOT / "models" / "standard_scaler.joblib"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", nargs=13, type=float, required=True, metavar="FEATURE")
    args = parser.parse_args()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = MLPRegressor(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = joblib.load(SCALER_PATH)
    x = scaler.transform(np.asarray(args.values, dtype=np.float32).reshape(1, -1))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32)).item()
    print(f"Predicted house price (k$): {pred:.4f}")

if __name__ == "__main__": main()
