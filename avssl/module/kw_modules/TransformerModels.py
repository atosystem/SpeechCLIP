import logging
from turtle import forward

from torch import nn
from torch.nn import TransformerEncoderLayer

logger = logging.getLogger(__name__)
import torch


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 1,
        d_model: int = 768,
        nhead: int = 8,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ) -> None:
        super().__init__()

        logger.info(f"Using {n_layers} layer transformer encoder")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.model = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

    def forward(self, src: torch.Tensor, key_padding_mask: torch.Tensor):
        return self.model(
            src=src,
            src_key_padding_mask=key_padding_mask,
        )


class MultiheadAttentionAndNorm(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.multihead_attn_layer = nn.MultiheadAttention(
            d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.attentionBlock_Norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src: torch.Tensor, key_padding_mask: torch.Tensor):
        return self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
