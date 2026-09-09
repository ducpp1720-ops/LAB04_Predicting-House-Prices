# Viva Q&A — 20 Questions

1. What is the task? Regression: predict Boston Housing median value.
2. Why is MLP suitable? It can model nonlinear relationships through hidden layers and nonlinear activations.
3. Why ReLU? It introduces nonlinearity and is simple and efficient.
4. Why one output neuron? The target is a single continuous price value.
5. Why MSELoss? The course material identifies MSELoss as a common loss for regression.
6. Why Adam? It is a standard optimizer and is listed in the course material as a popular choice.
7. What is forward propagation? Inputs pass through the network to produce a prediction.
8. What is backpropagation? The error gradient is propagated backward to compute parameter gradients.
9. What does optimizer.step() do? It updates model parameters using the gradients.
10. Why zero_grad()? PyTorch accumulates gradients, so old gradients must be cleared before a new update.
11. Why model.train()? It enables training behavior such as active Dropout.
12. Why model.eval()? It switches the model to evaluation behavior.
13. Why torch.no_grad() for prediction? It disables gradient tracking and reduces memory/computation overhead.
14. Why StandardScaler? Features have different scales; standardization improves numerical behavior for optimization.
15. Where is the scaler fitted? Only on the training split.
16. Why keep validation separate from test? Validation supports model selection/early stopping without contaminating final test evaluation.
17. What is early stopping? Training stops when validation loss no longer improves for the patience window.
18. What is Dropout? A regularization technique that randomly deactivates neurons during training.
19. What does RMSE mean? The square root of mean squared error, in the target's units.
20. Why compare baselines? To determine whether the MLP provides useful performance relative to simpler regression models.
