# Lab 04 Final Report — VERIFIED

## 1. Recovery
The anti-false-missing recovery sequence was completed before finalizing the project. Current conversation attachments and file resources contained the latest `Lab4_PyTorch_MLP_Regression_FINAL.zip` and the supplied `BostonHousing.csv`; Library history also contained earlier Lab 04 ZIP versions. The latest FINAL artifact was used rather than resetting the project.

## 2. Dataset verification
The supplied `BostonHousing.csv` contains **506 rows and 14 columns**: 13 predictive features plus `MEDV`. No missing values were found. The dataset was copied into `data/raw/boston_housing.csv` without synthetic replacement. The loader normalizes header case/whitespace so the supplied lowercase headers are accepted while the project internally uses the assignment's uppercase feature names.

## 3. Execution pipeline
- Requirements installation using the available local package cache: PASS
- Python compilation: PASS
- Training: PASS (`ACTUAL TRAINING PASS`)
- Metrics generation: PASS
- Figure generation: PASS
- Model save: PASS
- Scaler save: PASS
- Independent model/scaler load: PASS
- Prediction CLI: PASS
- Notebook clean execution: PASS; no error outputs
- Artifact consistency: PASS

An unrelated pre-existing environment issue may remain in `pip check`: an installed `moviepy` package requires an older Pillow version. It is outside this project and does not affect the Lab 04 runtime.

## 4. Actual test results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| MLP | 2.506221 | 14.949417 | 3.866448 | 0.796146 |
| Linear Regression | 3.189087 | 24.291044 | 4.928595 | 0.668761 |
| Decision Tree Regression | 2.308157 | 8.553907 | 2.924706 | 0.883357 |
| Random Forest Regression | 2.031838 | 8.418204 | 2.901414 | 0.885207 |

The Random Forest is the strongest test-set model in this run. The MLP outperforms Linear Regression but does not outperform the tree-based baselines. This is reported as an observed result, not hidden or replaced.

## 5. Training configuration actually used
- Split: 353 train / 51 validation / 102 test
- Seed: 42
- Architecture: `13 -> 32 -> 16 -> 1`
- Activation: ReLU
- Loss: MSELoss
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 32
- Maximum epochs: 300
- Early-stopping patience: 20
- Best epoch: 114
- Device: CPU

## 6. Saved artifacts
- `models/mlp_regressor.pth`
- `models/standard_scaler.joblib`
- `results/run_summary.json`
- `results/tables/training_history.csv`
- `results/tables/model_comparison.csv`
- `results/tables/mlp_test_predictions.csv`
- `results/figures/training_loss.png`
- `results/figures/actual_vs_predicted.png`
- `results/figures/model_rmse_comparison.png`

## 7. Final status
**READY TO SUBMIT**

All mandatory runtime checks in the audit pipeline completed successfully. No synthetic dataset or fabricated metric was used.
