from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

from avssl.base import OrderedNamespace
from avssl.module import ClipModel, MeanPoolingLayer, S3prlSpeechEncoder
from avssl.optim import get_scheduler

from .base_model import BaseLightningModel


class ParallelSpeechClip(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.audio_encoder_type = config.audio_encoder.type
        if self.audio_encoder_type == "s3prl":
            self.audio_encoder = S3prlSpeechEncoder(**config.audio_encoder)
        else:
            raise NotImplementedError(
                f"Unknown audio encoder type {self.audio_encoder_type}"
            )

        self.clip = ClipModel(**config.clip)

        self.audio_pooling_type = config.audio_encoder.pooling
        if self.audio_pooling_type == "mean":
            self.audio_pooling = MeanPoolingLayer(
                self.audio_encoder.out_dim, self.clip.out_dim
            )
        else:
            raise NotImplementedError(
                f"Unknown audio pooling type {self.audio_pooling_type}"
            )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.criterion = nn.CrossEntropyLoss()

    def forward_audio(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        full_utt: bool = False,
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:
        audio_feat, audio_feat_len = self.audio_encoder(wav, wav_len)
        if full_utt:
            return audio_feat, audio_feat_len
        else:
            return self.audio_pooling(audio_feat, audio_feat_len), audio_feat_len

    def forward_image(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(images, list):
            image_tensor = self.clip.prep_image(paths)
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
            text_tensor = self.clip.prep_text(sents)
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
        full_utt: bool = False,
    ) -> dict:
        wav, wav_len, images = batch
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len, full_utt=full_utt)
        image_feat = self.forward_image(images)

        if cal_loss and not full_utt:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_per_audio = logit_scale * audio_feat @ image_feat.t()
            logits_per_image = logits_per_audio.t()

            labels = torch.arange(
                len(logits_per_audio), device=logits_per_audio.device, dtype=torch.long
            )
            loss_audio = self.criterion(logits_per_audio, labels)
            loss_image = self.criterion(logits_per_image, labels)
            loss = (loss_audio + loss_image) / 2

            return loss, audio_feat, image_feat

        return audio_feat, image_feat

    def training_step(self, batch, batch_idx):
        loss, audio_feat, image_feat = self.forward(batch, cal_loss=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.forward(batch, cal_loss=True)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optim_config = []

        audio_pooling_optimizer = getattr(nn, self.config.audio_encoder.optim.name)(
            self.audio_pooling.parameters(), **self.config.audio_encoder.optim
        )
        audio_pooling_scheduler = get_scheduler(
            self.config.audio_encoder.optim.scheduler.name,
            audio_pooling_optimizer,
            **self.config.audio_encoder.optim.scheduler,
        )
        optim_config.append(
            {
                "optimizer": audio_pooling_optimizer,
                "lr_scheduler": {
                    "scheduler": audio_pooling_scheduler,
                    "interval": "step",
                },
            }
        )

        if self.config.audio_encoder.trainable:
            audio_optimizer = getattr(nn, self.config.audio_encoder.optim.name)(
                self.audio_encoder.parameters(), **self.config.audio_encoder.optim
            )
            audio_scheduler = get_scheduler(
                self.config.audio_encoder.optim.scheduler.name,
                audio_optimizer,
                **self.config.audio_encoder.optim.scheduler,
            )
            optim_config.append(
                {
                    "optimizer": audio_optimizer,
                    "lr_scheduler": {
                        "scheduler": audio_scheduler,
                        "interval": "step",
                    },
                }
            )

        if self.config.clip.image_encoder_trainable:
            image_optimizer = getattr(nn, self.config.clip.image_optim.name)(
                self.clip.model.visual.parameters(), **self.config.clip.image_optim
            )
            image_scheduler = get_scheduler(
                self.config.clip.image_optim.scheduler.name,
                image_optimizer,
                **self.config.clip.image_optim.scheduler,
            )
            optim_config.append(
                {
                    "optimizer": image_optimizer,
                    "lr_scheduler": {
                        "scheduler": image_scheduler,
                        "interval": "step",
                    },
                }
            )

        # if self.config.clip.text_encoder_trainable:
        #     text_optimzer = getattr(nn, self.config.clip.text_optim.name)(
        #         self.clip.model.transformer.parameters(), **self.config.clip.text_optim
        #     )
        #     text_scheduler = get_scheduler(
        #         self.config.clip.text_optim.scheduler.name,
        #         text_optimizer,
        #         **self.config.clip.text_optim.scheduler,
        #     )
        #     optim_config.append(
        #         {
        #             "optimizer": text_optimizer,
        #             "lr_scheduler": {
        #                 "scheduler": text_scheduler,
        #                 "interval": "step",
        #             }
        #         }
        #     )
        return optim_config
