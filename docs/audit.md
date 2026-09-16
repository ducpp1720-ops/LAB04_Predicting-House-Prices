# Final Audit — Lab 04

## Submission gate

The project is **READY only when the GitHub Actions training workflow completes successfully** and the repository contains real generated artifacts.

### Mandatory checks

- [x] California Housing schema is explicitly enforced.
- [x] Eight required input features and `median_house_value` target.
- [x] 80/20 train-validation split with `random_state=42`.
- [x] Official test CSV reserved for final evaluation.
- [x] StandardScaler fitted on training partition only.
- [x] PyTorch Dataset/DataLoader.
- [x] MLP `8 → 32 → 16 → 1` with ReLU hidden layers and linear output.
- [x] Adam + MSELoss, batch 32, learning rate 0.001, 100 epochs.
- [x] CPU/CUDA support.
- [x] MAE/MSE/RMSE/R².
- [x] Linear Regression, Decision Tree and Random Forest baselines.
- [x] EDA requirements covered.
- [x] Prediction smoke test.
- [x] No synthetic fallback.
- [x] Automated GitHub Actions training and artifact generation.

### Final status

Do not mark `READY` until the workflow log confirms dataset validation, tests, training, and prediction all pass.
