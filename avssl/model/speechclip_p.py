from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

from avssl.base import OrderedNamespace
from avssl.module import ClipModel, MeanPoolingLayer, S3prlSpeechEncoder, SupConLoss
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

        self.audio_pooling_type = config.audio_encoder.pooling.type
        if self.audio_pooling_type == "mean":
            self.audio_pooling = MeanPoolingLayer(
                self.audio_encoder.out_dim, self.clip.out_dim
            )
        else:
            raise NotImplementedError(
                f"Unknown audio pooling type {self.audio_pooling_type}"
            )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.criterion = SupConLoss(
            temperature=config.cl_loss.temperature,
            contrast_mode=config.cl_loss.contrast_mode,
            base_temperature=config.cl_loss.base_temperature,
        )

        self.recall_at = config.retrieval.recall_at

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
        full_utt: bool = False,
    ) -> dict:
        wav, wav_len, images, id = batch
        id = torch.cat(id, dim=0)
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len, full_utt=full_utt)
        image_feat = self.forward_image(images)

        if cal_loss and not full_utt:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            loss = self.criterion(
                features=torch.stack([audio_feat, image_feat], dim=1),
                labels=id,
            )

            return loss, audio_feat, image_feat, id

        return audio_feat, image_feat, id

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.forward(batch, cal_loss=True)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, audio_feat, image_feat, id = self.forward(batch, cal_loss=True)
        loss = loss.detach().cpu()
        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        id = id.detach().cpu()

        self.log("val_loss", loss)
        return {
            "id": id,
            "audio_feat": audio_feat,
            "image_feat": image_feat,
        }

    def validation_epoch_end(self, outputs):
        all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        all_imgs = torch.cat([x["image_feat"] for x in outputs], dim=0)
        id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_imgs)}

        del all_imgs

        all_audo_feats = torch.cat([x["audio_feat"] for x in outputs], dim=0)
        all_audo_feats_id = all_ids

        all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
        all_img_feats_id = torch.LongTensor(list(id_img_pairs.keys()))

        print(
            "Total #{} images, #{} audio".format(
                len(all_img_feats), len(all_audo_feats)
            )
        )

        # calculate dot product
        score_per_audio = torch.matmul(
            all_audo_feats.to(self.device), all_img_feats.T.to(self.device)
        )
        score_per_image = score_per_audio.T

        score_per_audio = torch.argsort(score_per_audio, dim=1, descending=True).cpu()
        score_per_image = torch.argsort(score_per_image, dim=1, descending=True).cpu()

        assert score_per_audio.shape == (len(all_audo_feats_id), len(all_img_feats_id))
        assert score_per_image.shape == (len(all_img_feats_id), len(all_audo_feats_id))

        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_img_feats_id

        rank_AI = all_img_feats_id.reshape(1, -1).repeat(AI_answers.shape[0], 1)
        rank_IA = all_audo_feats_id.reshape(1, -1).repeat(IA_answers.shape[0], 1)

        assert rank_AI.shape == score_per_audio.shape, (
            rank_AI.shape,
            score_per_audio.shape,
        )
        assert rank_IA.shape == score_per_image.shape, (
            rank_IA.shape,
            score_per_image.shape,
        )

        for r in range(AI_answers.shape[0]):
            rank_AI[r, :] = rank_AI[r, score_per_audio[r, :]]

        for r in range(IA_answers.shape[0]):
            rank_IA[r, :] = rank_IA[r, score_per_image[r, :]]

        rank_AI = rank_AI == AI_answers.unsqueeze(-1)
        rank_IA = rank_IA == IA_answers.unsqueeze(-1)

        recall_results_AI = {}
        recall_results_IA = {}
        # AI (many to one)
        for k in self.recall_at:
            recall_results_AI["recall@{}".format(k)] = (
                torch.sum(
                    torch.max(
                        rank_AI[:, :k].reshape(rank_AI.shape[0], k), dim=1, keepdim=True
                    )[0]
                )
                / rank_AI.shape[0]
            )
        recall_results_AI["recall_random@1"] = k / rank_AI.shape[0]

        # IA (one to many)
        for k in self.recall_at:
            recall_results_IA["recall@{}".format(k)] = (
                torch.sum(
                    torch.max(
                        rank_IA[:, :k].reshape(rank_IA.shape[0], k), dim=1, keepdim=True
                    )[0]
                )
                / rank_IA.shape[0]
            )
        # average one image corresponds to len(all_audo_feats) // len(all_img_feats) audio
        recall_results_IA["recall_random@1"] = 1
        _recall_at = 1
        for i in range(len(all_audo_feats) // len(all_img_feats)):
            recall_results_IA["recall_random@1"] *= (
                len(all_audo_feats) - _recall_at - i
            ) / (len(all_audo_feats) - i)

        recall_results_IA["recall_random@1"] = 1 - recall_results_IA["recall_random@1"]

        self.log("val_recall_AI", recall_results_AI)
        self.log("val_recall_IA", recall_results_IA)

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        audio_params = list(self.audio_pooling.parameters())
        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

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
