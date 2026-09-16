# Final Submission Checklist — LAB 04

## Data
- [x] `california_housing_train.csv` — 17,000 rows
- [x] `california_housing_test.csv` — 3,000 rows
- [x] 9 columns: 8 features + continuous target
- [x] Real dataset validated by GitHub Actions
- [x] No synthetic fallback

## EDA
- [x] shape
- [x] info / dtypes
- [x] missing values
- [x] duplicate rows
- [x] descriptive statistics
- [x] feature distributions
- [x] correlation matrix
- [x] IQR-based outlier inspection
- [x] EDA tables and plots saved under `results/eda/`

## Preprocessing
- [x] `train_test_split(test_size=0.2, random_state=42)`
- [x] `StandardScaler.fit()` only on training partition
- [x] validation/test transformed with the training-fitted scaler
- [x] official test set kept separate from training and validation

## PyTorch MLP
- [x] `HousePriceMLP(nn.Module)`
- [x] 8 → 32 → 16 → 1
- [x] ReLU hidden layers
- [x] linear output for regression
- [x] `Dataset` / `DataLoader`
- [x] Adam optimizer
- [x] MSELoss
- [x] `model.train()` / `model.eval()`
- [x] `torch.no_grad()` during validation/test inference
- [x] CPU/CUDA device support
- [x] best validation checkpoint restored before final test prediction

## Evaluation
- [x] MAE
- [x] MSE
- [x] RMSE
- [x] R²
- [x] Linear Regression baseline
- [x] Decision Tree Regression baseline
- [x] Random Forest Regression baseline
- [x] training/validation learning curve
- [x] actual-vs-predicted plot
- [x] RMSE comparison chart
- [x] test prediction CSV

## Reproducibility / Engineering
- [x] tests pass in GitHub Actions: 3 passed
- [x] model artifact saved: `models/mlp_house_price.pth`
- [x] scaler artifact saved: `models/scaler.pkl`
- [x] prediction CLI works
- [x] real dataset downloaded and schema/row counts validated automatically
- [x] GitHub Actions workflow passes end-to-end
- [x] generated artifacts committed to `main`
- [x] README and final report included

## Final Status

**READY FOR SUBMISSION**

Last verified workflow: GitHub Actions run #1, completed successfully on 2026-09-16. The workflow completed dependency installation, real-data validation, tests, training/evaluation, prediction smoke test, and artifact commit.
