import logging
import torch
from torch import nn
from s3prl import hub
from typing import Union, Tuple

from avssl.util import init_weights


class S3prlSpeechEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        trainable: bool = False,
        device: str = "cpu",
        feat_select_idx: Union[str, list] = "all",
        layer_drop: Union[str, float] = 0.0,
        **kwargs,
    ):
        """Speech Encoder with S3PRL (v0.3.1)

        Args:
            name (str): Name of speech encoder.
            pretrained (bool, optional): Init with pretrained model. Defaults to False.
            trainable (bool, optional): Whether to update the model while training. Defaults to False.
            device (str, optional): Device. Defaults to "cpu".
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to "all".
            layer_drop (Union[str, float], optional): Layer drop rate. Defaults to 0.0.
        """
        super().__init__()

        self.name = name
        self.pretrained = pretrained
        self.trainable = trainable
        self.device = device
        self.feat_select_idx = feat_select_idx

        self.encoder = getattr(hub, name)().to(device)
        if hasattr(self.encoder, "get_downsample_rates"):
            self.downsample_rate = self.encoder.get_downsample_rates("hidden_states")
        else:
            self.downsample_rate = 160

        if not pretrained:
            self.encoder.apply(init_weights)

        if not trainable:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if hasattr(self.encoder, "layer_drop"):
            if (
                isinstance(layer_drop, float)
                and layer_drop >= 0.0
                and layer_drop <= 1.0
            ):
                self.encoder.layer_drop = layer_drop
            elif layer_drop == "original":
                pass
            else:
                raise ValueError(f"layer_drop = {layer_drop} is not supported.")

        self.out_dim = 0
        with torch.no_grad():
            wav = [torch.randn(16000, dtype=torch.float)]
            feat_all, _ = self.encoder(wav)
            self.out_dim = feat_all["last_hidden_state"].shape[2]

        logging.info(f"Loaded s3prl speech encoder ({name}): out_dim = {self.out_dim}")

    def forward(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        feat_select_idx: Union[str, list] = None,
    ) -> Tuple[Union[torch.Tensor, list], torch.Tensor]:
        """Forward function for S3PRL speech encoder

        Args:
            wav (Union[torch.Tensor, list]): List of waveforms. (L, )
            wav_len (Union[torch.Tensor, list]): List of waveforms' lengths. Defaults to [].
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to None.

        Raises:
            KeyError: feat_select_idx is not "all", "hidden_states",
                      "last_hidden_state", or list.

        Returns:
            Tuple[Union[torch.Tensor, list], torch.Tensor]: Hidden features and their lengths.
        """

        if isinstance(wav, torch.Tensor):
            if wav.dim() == 2:
                wav = [wav[b, : wav_len[b]] for b in range(len(wav))]
            elif wav.dim() == 1:
                wav = [wav]

        if self.trainable:
            feat = self.encoder(wav)
        else:
            with torch.no_grad():
                feat = self.encoder(wav)

        if len(wav_len) == 0:
            wav_len = [len(w) for w in wav]

        feat_len = torch.LongTensor(
            [round(l / self.downsample_rate) for l in wav_len]
        ).to(feat["last_hidden_state"].device)

        if feat_select_idx is None:
            feat_select_idx = self.feat_select_idx

        if feat_select_idx == "all":
            return feat, feat_len
        elif isinstance(feat_select_idx, list):
            feat = [feat["hidden_states"][i] for i in feat_select_idx]
            return feat, feat_len
        elif feat_select_idx in feat:
            return feat[feat_select_idx], feat_len
        else:
            raise KeyError(feat_select_idx)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return self
