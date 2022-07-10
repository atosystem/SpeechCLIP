import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["TransformerEncoder", "MultiheadAttentionAndNorm"]


class nnTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)

    def extract_hidden_states(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn. (Return all hidden_states)

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        hidden_states = []

        for mod in self.layers:
            hidden_states.append(output)
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        hidden_states.append(output)
        if self.norm is not None:
            output = self.norm(output)

        return output, tuple(hidden_states)


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
        self.model = nnTransformerEncoder(encoder_layer, n_layers, encoder_norm)

    def forward(self, src: torch.Tensor, key_padding_mask: torch.Tensor):
        return self.model(
            src=src,
            src_key_padding_mask=key_padding_mask,
        )

    def extract_hidden_states(self, src: torch.Tensor, key_padding_mask: torch.Tensor):
        """Extract all hidden states

        Args:
            src (torch.Tensor): src
            key_padding_mask (torch.Tensor): key padding mask

        Returns:
            tuple: (output,hidden_states)
        """
        return self.model.extract_hidden_states(
            src=src,
            src_key_padding_mask=key_padding_mask,
        )[1]


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

    def extract_hidden_states(self, src: torch.Tensor, key_padding_mask: torch.Tensor):
        return tuple([src, self.forward(src, key_padding_mask)])

    def extract_attention_map(self, src: torch.Tensor, key_padding_mask: torch.Tensor):
        _out, _att_weight = self.multihead_attn_layer(
            src, src, src, key_padding_mask=key_padding_mask, average_attn_weights=False
        )
        _out = self.attentionBlock_Norm(_out + src)
        return _out, _att_weight
