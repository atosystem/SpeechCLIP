import torch
from torch import nn


class MeanPoolingLayer(nn.Module):
    def __init__(
        self,
        in_dim: int = 0,
        out_dim: int = 0,
        bias: bool = True,
        pre_proj: bool = True,
        post_proj: bool = True,
    ):
        """Mean pooling layer with linear layers.

        Args:
            in_dim (int, optional): Input dimension. Defaults to 0.
            out_dim (int, optional): Output dimension. Defaults to 0.
            bias (bool, optional): Linear layer bias. Defaults to True.
            pre_proj (bool, optional): Pre-projection layer. Defaults to True.
            post_proj (bool, optional): Post-projection layer. Defaults to True.
        """
        super().__init__()

        self.pre_proj = None
        self.post_proj = None

        if in_dim > 0 and out_dim > 0:
            if pre_proj:
                self.pre_proj = nn.Linear(in_dim, out_dim, bias=bias)
            if post_proj:
                self.post_proj = nn.Linear(
                    in_dim if not pre_proj else out_dim, out_dim, bias=bias
                )

    def forward(self, x: torch.Tensor, x_len: torch.Tensor = None) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): Input features. (B, T, D)
            x_len (torch.Tensor): Feature lengths. (B, )

        Returns:
            torch.Tensor: Mean pooled features.
        """
        if self.pre_proj is not None:
            x = self.pre_proj(x)

        if x_len is not None:
            x = [x[b, : x_len[b]].mean(0) for b in range(len(x))]
            x = torch.stack(x, dim=0)
        else:
            x = x.mean(1)

        if self.post_proj is not None:
            x = self.post_proj(x)

        return x
