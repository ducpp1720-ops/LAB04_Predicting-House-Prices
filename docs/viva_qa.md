# Viva Q&A

**Why StandardScaler?** It puts features on comparable scales and helps gradient-based optimization converge more reliably.

**Why fit the scaler only on training data?** Fitting on validation/test data leaks evaluation-set information into preprocessing.

**Why is the output layer linear?** This is regression; the output is a continuous numeric value and should not be restricted by ReLU or Sigmoid.

**Why MSELoss?** It is a standard regression loss and penalizes squared prediction errors.

**Why keep the official test CSV untouched?** It provides a final evaluation after model-selection decisions are made using training/validation data.

**Why compare with Random Forest?** The lab requires classical baselines so the MLP can be interpreted against simpler/non-neural regressors.

**What happens if the CSV is missing?** The pipeline fails explicitly and never fabricates replacement data.
