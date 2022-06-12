from typing import List

import torch
from torch import nn


class WeightedSumLayer(nn.Module):
    def __init__(self, n_weights: int):
        """Weighted sum layer with learnable weights.

        Args:
            n_weights (int): Number of weights, i.e., number of hidden representations.
        """

        super().__init__()

        self.n_weights = n_weights
        self.weights = nn.Parameter(torch.zeros((n_weights,), dtype=torch.float))

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Weighted sum a list of tensors.

        Args:
            x (List[torch.Tensor]): Representations to be weighted summed.

        Returns:
            torch.Tensor: Weighted summed representations.
        """

        assert len(x) == self.n_weights, len(x)

        weights = torch.softmax(self.weights, dim=0)
        weights = weights.view(-1, *([1] * x[0].dim()))
        x = torch.stack(x, dim=0)
        x = (weights * x).sum(0)

        return x
