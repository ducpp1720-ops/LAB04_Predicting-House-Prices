import torch
from src.models import HousePriceMLP


def test_mlp_output_shape():
    model = HousePriceMLP(input_dim=8)
    assert model(torch.randn(4, 8)).shape == (4, 1)


def test_single_prediction_api():
    model = HousePriceMLP(input_dim=8)
    assert model(torch.randn(1, 8)).shape == (1, 1)
