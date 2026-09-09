# Presentation Outline

1. Problem: predict Boston Housing median value.
2. Dataset: 506 observations, 13 input features, one continuous target.
3. Preprocessing: train/validation/test split; StandardScaler fitted only on training data.
4. MLP: 13 → 32 → 16 → 1, ReLU hidden activations, linear output.
5. Training: MSELoss, Adam, mini-batch DataLoader, backpropagation, validation monitoring, early stopping.
6. Regularization: Dropout and Adam weight decay.
7. Evaluation: MAE, MSE, RMSE, R².
8. Baselines: Linear Regression, Decision Tree Regression, Random Forest Regression.
9. Prediction: load saved MLP and scaler, transform input, evaluate in no-grad mode.
10. Conclusion: report only metrics produced by the final clean run.
