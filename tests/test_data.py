from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
TARGET = "median_house_value"


def test_dataset_schema_if_present():
    train_path = ROOT / "data/raw/california_housing_train.csv"
    test_path = ROOT / "data/raw/california_housing_test.csv"
    if not train_path.exists() or not test_path.exists():
        return
    for path in (train_path, test_path):
        df = pd.read_csv(path)
        assert list(df.columns) == FEATURES + [TARGET]
        assert not df[FEATURES + [TARGET]].isna().any().any()
