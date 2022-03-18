import torch
from torch import nn


class MeanPoolingLayer(nn.Module):
    def __init__(
        self,
        in_dim: int = 0,
        out_dim: int = 0,
        bias: bool = True,
        pre_proj: bool = True,
    ):
        super().__init__()

        if in_dim > 0 and out_dim > 0:
            self.proj = nn.Linear(in_dim, out_dim, bias=bias)
            self.pre_proj = pre_proj
        else:
            self.proj = None

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> torch.Tensor:
        if self.proj is not None and self.pre_proj:
            x = self.proj(x)

        x = x.sum(-1) / x_len.float()

        if self.proj is not None and not self.pre_proj:
            x = self.proj(x)

        return x
