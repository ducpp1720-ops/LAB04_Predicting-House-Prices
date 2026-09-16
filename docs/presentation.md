# Presentation Flow — Lab 04

1. Problem: predict `median_house_value`.
2. Dataset: 17,000 train rows + 3,000 official test rows; 8 numerical features.
3. EDA: missing values, duplicates, distributions, correlation and outliers.
4. Preprocessing: 80/20 train-validation split and train-only StandardScaler fitting.
5. MLP: 8 → 32 → 16 → 1, ReLU, Adam, MSELoss.
6. Training: DataLoader, train/eval modes, `torch.no_grad()`, CPU/CUDA.
7. Evaluation: MAE, MSE, RMSE, R².
8. Baselines: Linear Regression, Decision Tree, Random Forest.
9. Visual evidence: loss curve, actual-vs-predicted, model comparison.
10. Reproducibility: tests + GitHub Actions.
