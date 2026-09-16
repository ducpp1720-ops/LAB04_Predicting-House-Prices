# Final Submission Checklist

## Data
- [ ] `california_housing_train.csv` — 17,000 rows
- [ ] `california_housing_test.csv` — 3,000 rows
- [ ] 9 columns with 8 features + target
- [ ] No synthetic fallback

## EDA
- [ ] shape
- [ ] info/dtypes
- [ ] missing values
- [ ] duplicates
- [ ] describe
- [ ] distributions
- [ ] correlation matrix
- [ ] outlier inspection

## MLP
- [ ] `HousePriceMLP`
- [ ] 8 → 32 → 16 → 1
- [ ] ReLU hidden layers
- [ ] linear output
- [ ] Dataset/DataLoader
- [ ] Adam
- [ ] MSELoss
- [ ] train/eval/no_grad
- [ ] CPU/CUDA

## Evaluation
- [ ] MAE
- [ ] MSE
- [ ] RMSE
- [ ] R²
- [ ] Linear Regression
- [ ] Decision Tree
- [ ] Random Forest
- [ ] learning curve
- [ ] actual vs predicted
- [ ] comparison chart

## Reproducibility
- [ ] tests pass
- [ ] model artifact saved
- [ ] scaler saved
- [ ] prediction command works
- [ ] GitHub Actions workflow passes
