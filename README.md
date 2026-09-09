# Lab 04 — PyTorch MLP Regression

This project implements Practice Exercise 4: Multilayer Perceptron Regression for Boston Housing price prediction. The exercise specifies Boston Housing, feature/target separation, StandardScaler, an MLP with ReLU hidden layers and a linear output, Adam, MSE, evaluation on the test set, and comparison with Linear Regression, Decision Tree Regression, and Random Forest Regression.

## PyTorch implementation

The final implementation uses PyTorch. A validation split is kept separate from the test set so early stopping does not use the test set. The scaler is fitted only on the training split and reused with `transform()` for validation and test data.

## Run

```bash
python -m py_compile src/models.py src/utils.py src/train.py src/predict.py
python src/train.py
python src/predict.py --values <13-feature-values>
```

## Dataset

`data/raw/boston_housing.csv` must contain the real Boston Housing dataset with columns:
`CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV`.

The dataset is not synthesized. If it is absent, training fails explicitly rather than fabricating data.

## Artifacts

Training writes the model, scaler, metrics tables, prediction table, JSON summary, and three required figures under `models/` and `results/`.
