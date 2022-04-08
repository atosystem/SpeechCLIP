import imp
from re import sub
from typing import Tuple, Union
from unittest import result

import numpy as np
import torch
from torch import nn

from avssl.base import OrderedNamespace
from avssl.module import ClipModel, MeanPoolingLayer, S3prlSpeechEncoder
from avssl.module.speechclip_c_modules import (
    GumbelVectorQuantizer,
    KmeansVectorQuantizer,
)
from avssl.optim import get_scheduler

from .base_model import BaseLightningModel


class CascadedSpeechClip(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        # self.automatic_optimization = False
        self.audio_encoder_type = config.audio_encoder.type
        if self.audio_encoder_type == "s3prl":
            self.audio_encoder = S3prlSpeechEncoder(**config.audio_encoder)
            self.embd_dim = self.audio_encoder.out_dim
        else:
            raise NotImplementedError(
                f"Unknown audio encoder type {self.audio_encoder_type}"
            )

        self.clip = ClipModel(
            codebook_size=config.vq.num_vars,
            precision=config.trainer.precision,
            **config.clip,
        )
        self.text_embd = self.clip.text_embd
        self.text_embd_dim = self.text_embd.weight.size(-1)

        self.downsampling = nn.Sequential(
            nn.Conv1d(self.embd_dim, self.embd_dim, 2, 2, 0, 1),
            nn.AvgPool1d(2, 2, 0),
            nn.Conv1d(self.embd_dim, self.text_embd_dim, 2, 2, 0, 1),
        )

        self.vector_quantizer = None
        self.vq_type = config.vq.type

        if config.vq.activation == "relu":
            activation = nn.ReLU()
        elif config.vq.activation == "gelu":
            activation = nn.GELU()
        else:
            raise Exception("unknown activation " + config.activation)

        if self.vq_type == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=self.text_embd_dim,
                num_vars=config.vq.num_vars,
                temp=config.vq.temp,
                groups=config.vq.groups,
                combine_groups=config.vq.combine_groups,
                vq_dim=config.vq.vq_dim if config.vq.vq_dim > 0 else self.text_embd_dim,
                time_first=False,
                activation=activation,
                weight_proj_factor=2,
                init_codebook=self.text_embd.weight,
            )
        elif self.vq_type == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=self.text_embd_dim,
                num_vars=config.vq.num_vars,
                groups=config.vq.groups,
                combine_groups=config.vq.combine_groups,
                vq_dim=config.vq.vq_dim if config.vq.vq_dim > 0 else self.text_embd_dim,
                time_first=False,
                gamma=config.vq.gamma,
                init_codebook=self.text_embd.weight,
            )
        else:
            assert (
                config.vq_type == "none" or config.vq_type is None
            ), "Unknown quantizer type"

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.criterion = nn.CrossEntropyLoss()

    def forward_audio(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:
        audio_feat, audio_feat_len = self.audio_encoder(wav, wav_len)
        return audio_feat, audio_feat_len

    def forward_image(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(images, list):
            image_tensor = self.clip.prep_image(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            if images.dim() != 4 or images.shape[1] != 3:
                raise ValueError(f"Incorrect image tensor shape {images.shape}")
            image_tensor = images
        else:
            raise TypeError(f"Unknown image type {type(images)}")

        image_feat = self.clip.encode_image(image_tensor)
        return image_feat

    def forward_text(self, sents: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(sents, list):
            text_tensor = self.clip.prep_text(sents).to(self.device)
        elif isinstance(sents, torch.Tensor):
            if sents.dim() != 2:
                raise ValueError(f"Incorrect text tensor shape {sents.shape}")
            text_tensor = sents
        else:
            raise TypeError(f"Unknown text type {type(sents)}")

        text_feat = self.clip.encode_text(text_tensor)
        return text_feat

    def forward(
        self,
        batch,
        cal_loss: bool = False,
    ) -> dict:
        wav, wav_len, images, id = batch
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len)
        image_feat = self.forward_image(images)

        #  down sampling
        audio_feat = audio_feat.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        audio_feat = self.downsampling(audio_feat)

        # vector quantization
        result = self.vector_quantizer(audio_feat, produce_targets=True)

        if result["subword_prob"].size(1) > 77:
            result["subword_prob"] = result["subword_prob"][:, :77, :]

        text_feat = self.clip.encode_text(result)

        if cal_loss:
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_per_text = logit_scale * text_feat @ image_feat.t()
            logits_per_image = logits_per_text.t()

            labels = torch.arange(
                len(logits_per_text), device=logits_per_text.device, dtype=torch.long
            )
            loss_text = self.criterion(logits_per_text, labels)
            loss_image = self.criterion(logits_per_image, labels)

            loss = (result["loss"] + loss_text + loss_image) / 3

            return loss, text_feat, image_feat, result

        return text_feat, image_feat, result

    def validation_step(self, batch, batch_idx):
        loss, _, _, res = self.forward(batch, cal_loss=True)

        result = {"val_loss": loss}
        for key in res.keys():
            if (key == "code_cpx") | (key == "prob_cpx") | (key == "temp"):
                result[key] = res[key]

        self.log_dict(result, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    # def log_grad_norm(self, grad_norm_dict):
    #     print(grad_norm_dict)
    #     self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        if self.config.audio_encoder.trainable:
            audio_params = list(self.audio_encoder.parameters())

        audio_params = audio_params + list(self.downsampling.parameters())
        audio_params = audio_params + list(self.vector_quantizer.parameters())

        audio_optimizer = getattr(torch.optim, self.config.audio_encoder.optim.name)(
            audio_params,
            **self.config.audio_encoder.optim.args,
        )
        audio_scheduler = get_scheduler(
            optimizer=audio_optimizer,
            **self.config.audio_encoder.scheduler,
        )
        optimizers.append(audio_optimizer)
        schedulers.append(
            {
                "scheduler": audio_scheduler,
                "interval": "step",
            }
        )

        if self.config.clip.image_encoder_trainable:
            image_optimizer = getattr(torch.optim, self.config.clip.image_optim.name)(
                self.clip.model.visual.parameters(),
                **self.config.clip.image_optim.args,
            )
            image_scheduler = get_scheduler(
                optimizer=image_optimizer,
                **self.config.clip.scheduler,
            )
            optimizers.append(image_optimizer)
            schedulers.append(
                {
                    "scheduler": image_scheduler,
                    "interval": "step",
                }
            )

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        loss, text_feat, image_feat, res = self.forward(batch, cal_loss=True)

        result = {}
        for key in res.keys():
            if (key == "code_cpx") | (key == "prob_cpx") | (key == "temp"):
                result[key] = res[key]

        self.log_dict(result, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # opts, _ = self.configure_optimizers()
        # for opt in opts:
        #     opt.zero_grad()
        #     # automatically applies scaling, etc...
        #     self.manual_backward(loss)
        #     opt.step()

        return {"loss": loss}
