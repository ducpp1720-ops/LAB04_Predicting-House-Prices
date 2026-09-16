"""PyTorch MLP model for California Housing regression."""
import torch
from torch import nn


class HousePriceMLP(nn.Module):
    """8 -> 32 -> 16 -> 1 MLP for house-price regression."""

    def __init__(self, input_dim: int = 8, hidden1: int = 32, hidden2: int = 16, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden1), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden1, hidden2), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden2, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Backward-compatible alias for older local scripts.
MLPRegressor = HousePriceMLP
