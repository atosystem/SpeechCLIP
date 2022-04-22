import logging
import types
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from avssl.base import OrderedNamespace
from avssl.module import (
    AttentivePoolingLayer,
    ClipModel,
    MeanPoolingLayer,
    S3prlSpeechEncoder,
    SupConLoss,
    mutualRetrieval,
)
from avssl.optim import get_scheduler

from .base_model import BaseLightningModel


class ParallelSpeechClipBase(BaseLightningModel):
    """ParallelSpeechClipBase"""

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

    def reportRetrieval(self, score_per_audio, score_per_image, AI_answers, IA_answers):
        recall_results_AI, recall_results_IA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_image,
            AB_answers=AI_answers,
            BA_answers=IA_answers,
            recall_at=self.recall_at,
        )

        self.log("val_recall_AI", recall_results_AI)
        self.log("val_recall_IA", recall_results_IA)
        self.log("val_recall_mean", recall_results_mean)
        self.log("val_recall_mean_1", recall_results_mean["recall@1"])

    def training_step(self, batch, batch_idx):
        # loss must return first
        loss = self.forward(batch, cal_loss=True)[0]
        self.log("train_loss", loss)
        return {"loss": loss}

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


class ParallelSpeechClip_MeanPool(ParallelSpeechClipBase):
    """ParallelSpeechClip_MeanPool

    Parallel CLIP with mean pooling
    """

    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

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

    def forward(
        self,
        batch,
        cal_loss: bool = False,
        full_utt: bool = False,
    ) -> dict:
        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]
        id = torch.cat(id, dim=0)
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len, full_utt=full_utt)
        image_feat = self.forward_image(image)

        if cal_loss and not full_utt:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            loss = self.criterion(
                features=torch.stack([audio_feat, image_feat], dim=1),
                labels=id,
            )

            return loss, audio_feat, image_feat, id

        return audio_feat, image_feat, id

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

        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_img_feats_id

        self.reportRetrieval(
            score_per_audio=score_per_audio,
            score_per_image=score_per_image,
            AI_answers=AI_answers,
            IA_answers=IA_answers,
        )

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


class ParallelSpeechClip_AttPool(ParallelSpeechClipBase):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.audio_feat_projection_type = (
            config.audio_encoder.pooling.audio_projection_type
        )
        self.audio_feat_projection = nn.Linear(
            self.audio_encoder.out_dim, self.clip.out_dim
        )

        self.audio_pooling_type = config.audio_encoder.pooling.type
        if hasattr(config.audio_encoder.pooling, "degraded"):
            # degraded : fixed U = I and make it non trainable
            self.audio_pooling_degraded = config.audio_encoder.pooling.degraded
        else:
            self.audio_pooling_degraded = False

        if self.audio_pooling_type == "attentive_pooling":
            self.audio_pooling = AttentivePoolingLayer(
                dim_A=self.audio_encoder.out_dim
                if not self.audio_feat_projection_type == "pre"
                else self.clip.out_dim,
                dim_B=self.clip.out_dim,
                degraded=self.audio_pooling_degraded,
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

    def pool_features(self, audio_feat, audio_feat_len, image_feat):
        # attentative pooling for audio_feat and image_feat
        if self.audio_feat_projection_type == "pre":
            audio_feat = self.audio_feat_projection(audio_feat)

        mask = self.audio_pooling.generate_input_msk(
            input_A_lens=audio_feat_len, max_Alen=audio_feat.shape[1]
        )

        audio_pool_feat, image_pool_feat = self.audio_pooling(
            input_A=audio_feat.permute(0, 2, 1),
            input_B=image_feat.unsqueeze(-1),
            intput_msk=mask,
        )

        if self.audio_feat_projection_type == "post":
            audio_pool_feat = self.audio_feat_projection(audio_pool_feat)

        audio_feat = audio_pool_feat
        image_feat = image_pool_feat

        return audio_feat, image_feat

    def forward(
        self,
        batch,
        cal_loss: bool = False,
        full_utt: bool = False,
        ret_pre_pooling: bool = False,
    ) -> dict:
        wav = batch["wav"]
        wav_len = batch["wav_len"]
        images = batch["image"]
        id = batch["id"]
        id = torch.cat(id, dim=0)
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len, full_utt=True)
        image_feat = self.forward_image(images)

        prePool_audio = audio_feat, audio_feat_len
        prePool_image = image_feat

        audio_feat, image_feat = self.pool_features(
            audio_feat, audio_feat_len, image_feat
        )

        # image_feat is actually the same, since each image is presented with only one vector

        if cal_loss:

            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            loss = self.criterion(
                features=torch.stack([audio_feat, image_feat], dim=1),
                labels=id,
            )

            if not ret_pre_pooling:
                return loss, audio_feat, image_feat, id
            else:
                return loss, prePool_audio, prePool_image, id

        return audio_feat, image_feat

    def validation_step(self, batch, batch_idx):
        # print("val")
        loss, prePool_audio, prePool_image, id = self.forward(
            batch, cal_loss=True, ret_pre_pooling=True
        )
        prePool_audio = prePool_audio[0].detach().cpu(), prePool_audio[1].detach().cpu()
        prePool_image = prePool_image.detach().cpu()
        id = id.detach().cpu()
        loss = loss.detach().cpu()
        self.log("val_loss", loss)
        return {
            "id": id,
            "prePool_audio": prePool_audio,
            "prePool_image": prePool_image,
        }

    def validation_epoch_end(self, validation_step_outputs):
        all_image_feat = []
        all_audio_feat = []
        all_audio_len = []
        results_scores = []
        for out in validation_step_outputs:
            audio_feat, audio_feat_len = out["prePool_audio"]
            image_feat = out["prePool_image"]
            all_image_feat.append(image_feat)
            all_audio_feat.append(audio_feat)
            all_audio_len.append(audio_feat_len)

        all_image_feat = torch.cat(all_image_feat, dim=0).float()
        all_ids = torch.cat([x["id"] for x in validation_step_outputs], dim=0)
        id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_image_feat)}
        del all_image_feat

        all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
        all_img_feats_id = torch.LongTensor(list(id_img_pairs.keys()))
        all_audo_feats_id = all_ids

        print(
            "Total #{} images, #{} audio".format(
                len(all_img_feats_id), len(all_audo_feats_id)
            )
        )

        for audio_feat, audio_feat_len in zip(all_audio_feat, all_audio_len):
            audio_feat = audio_feat.to(self.device)
            audio_feat_len = audio_feat_len.to(self.device)

            if self.audio_feat_projection_type == "pre":
                audio_feat = self.audio_feat_projection(audio_feat)

            mask = self.audio_pooling.generate_input_msk(
                input_A_lens=audio_feat_len, max_Alen=audio_feat.shape[1]
            )

            scores = []
            for image_offset in range(
                0, all_img_feats.shape[0], all_img_feats.shape[0] // 2
            ):
                _image_feats = all_img_feats[
                    image_offset : image_offset + all_img_feats.shape[0] // 2, :
                ].to(self.device)

                if len(_image_feats.shape) == 1:
                    _image_feats = _image_feats.unsqueeze(0)

                _audio_pooled_feats = self.audio_pooling.cal_batch_embedding(
                    input_A=audio_feat.permute(0, 2, 1),
                    input_B=_image_feats.permute(1, 0),
                    intput_msk=mask,
                )

                # audio_pooled_feats.shape = (bsz, audio_dim, # image sample)
                assert _audio_pooled_feats.shape == (
                    audio_feat.shape[0],
                    audio_feat.shape[2],
                    _image_feats.shape[0],
                ), (
                    _audio_pooled_feats.shape,
                    (audio_feat.shape[0], audio_feat.shape[2], _image_feats.shape[0]),
                )

                # audio_pooled_feats.shape = (bsz, #audioSamples, audio_dim)
                _audio_pooled_feats = _audio_pooled_feats.permute(0, 2, 1)

                if self.audio_feat_projection_type == "post":
                    _audio_pooled_feats = self._audio_pooled_feats(_audio_pooled_feats)

                _audio_pooled_feats = _audio_pooled_feats / _audio_pooled_feats.norm(
                    dim=-1, keepdim=True
                )

                _image_feats = _image_feats / _image_feats.norm(dim=-1, keepdim=True)

                _score = torch.matmul(_audio_pooled_feats, _image_feats.T)
                _score = torch.diagonal(_score, offset=0, dim1=1, dim2=2)

                # _score.shape (bsz, len_of_all_data_pairs // n )
                assert _score.shape == (
                    _audio_pooled_feats.shape[0],
                    _image_feats.shape[0],
                )
                scores.append(_score)

            scores = torch.cat(scores, dim=1)
            scores = scores.cpu()

            # score.shape (bsz, len_of_all_data_pairs )
            assert scores.shape == (
                audio_feat.shape[0],
                all_img_feats.shape[0],
            )
            results_scores.append(scores)

        results_scores = torch.cat(results_scores, dim=0)

        assert results_scores.shape == (len(all_audo_feats_id), len(all_img_feats_id))

        score_per_audio = results_scores
        score_per_image = score_per_audio.T
        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_img_feats_id

        self.reportRetrieval(
            score_per_audio=score_per_audio,
            score_per_image=score_per_image,
            AI_answers=AI_answers,
            IA_answers=IA_answers,
        )

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


class ParallelSpeechClip_AttPool_FineGrain_Base(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.pytorch_lightning_logger = logging.getLogger("pytorch_lightning.core")

        self.audio_encoder_type = config.audio_encoder.type
        if self.audio_encoder_type == "s3prl":
            self.audio_encoder = S3prlSpeechEncoder(**config.audio_encoder)
        else:
            raise NotImplementedError(
                f"Unknown audio encoder type {self.audio_encoder_type}"
            )

        self.clip = ClipModel(**config.clip)

        self.audio_feat_projection_type = (
            config.audio_encoder.pooling.audio_projection_type
        )
        self.audio_feat_projection = nn.Linear(
            self.audio_encoder.out_dim, self.clip.out_dim
        )

        self.image_feat_projection = nn.Linear(
            self.clip.model.visual.transformer.width, self.clip.out_dim
        )

        self.audio_pooling_type = config.audio_encoder.pooling.type
        if hasattr(config.audio_encoder.pooling, "degraded"):
            self.audio_pooling_degraded = config.audio_encoder.pooling.degraded
        else:
            self.audio_pooling_degraded = False

        if self.audio_pooling_type == "attentive_pooling":
            self.audio_pooling = AttentivePoolingLayer(
                dim_A=self.audio_encoder.out_dim
                if not self.audio_feat_projection_type == "pre"
                else self.clip.out_dim,
                dim_B=self.clip.model.visual.transformer.width
                if not self.audio_feat_projection_type == "pre"
                else self.clip.out_dim,
                degraded=self.audio_pooling_degraded,
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

        # evaluate
        self.recall_at = config.retrieval.recall_at

    def forward_audio(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        full_utt: bool = False,
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:
        audio_feat, audio_feat_len = self.audio_encoder(wav, wav_len)

        # audio_feat.shape = (bsz,seq_len,hid_dim)

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

    def reportRetrieval(self, score_per_audio, score_per_image, AI_answers, IA_answers):
        recall_results_AI, recall_results_IA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_image,
            AB_answers=AI_answers,
            BA_answers=IA_answers,
            recall_at=self.recall_at,
        )

        self.log("val_recall_AI", recall_results_AI)
        self.log("val_recall_IA", recall_results_IA)
        self.log("val_recall_mean", recall_results_mean)
        self.log("val_recall_mean_1", recall_results_mean["recall@1"])

    def pool_features(
        self, audio_feat, audio_feat_len, image_feat, batchwise_pooling=True
    ):
        """pool_features

        Args:
            audio_feat
            audio_feat_len
            image_feat
            batchwise_pooling (bool, optional): if true, each batch is calculated independently
                                                and the bsz of audio and image must be the same.
                                                if false, all pairs of attentive pooling is calculate betwen the audio and image inputs.
                                                Defaults to True.

        """
        assert len(audio_feat.shape) == 3
        if len(image_feat.shape) == 2:
            # single image (seqlen,dim)
            image_feat = image_feat.unsqueeze(0).repeat(audio_feat.shape[0], 1, 1)
            # single image (bsz,seqlen,dim)
        elif not len(image_feat.shape) == 3:
            raise ValueError("image_feat must be at leat 2 dimensional")

        # attentive pooling for audio_feat and image_feat
        if self.audio_feat_projection_type == "pre":
            audio_feat = self.audio_feat_projection(audio_feat)
            image_feat = self.image_feat_projection(image_feat)

        mask = self.audio_pooling.generate_input_msk(
            input_A_lens=audio_feat_len,
            max_Alen=audio_feat.shape[1],
            max_Blen=image_feat.shape[1],
        )

        if batchwise_pooling:
            # audio_pool_feat and image_pool_feat must have same batch size
            audio_pool_feat, image_pool_feat = self.audio_pooling(
                input_A=audio_feat.permute(0, 2, 1),
                input_B=image_feat.permute(0, 2, 1),
                intput_msk=mask,
            )
        else:
            audio_pool_feat, image_pool_feat = self.audio_pooling.batch_forward(
                input_A=audio_feat.permute(0, 2, 1),
                input_B=image_feat.permute(0, 2, 1),
                intput_msk=mask,
            )

        if self.audio_feat_projection_type == "post":
            audio_pool_feat = self.audio_feat_projection(audio_pool_feat)
            image_pool_feat = self.image_feat_projection(image_pool_feat)

        audio_feat = audio_pool_feat
        image_feat = image_pool_feat

        return audio_feat, image_feat

    def forward(
        self,
        batch,
        cal_loss: bool = False,
        full_utt: bool = False,
        ret_pre_pooling: bool = False,
    ) -> dict:
        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]
        id = torch.cat(id, dim=0)
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len, full_utt=True)
        image_feat = self.forward_image(image)

        # save prePooled features for further validation
        prePool_audio = audio_feat.detach().cpu(), audio_feat_len.detach().cpu()
        prePool_image = image_feat.detach().cpu()

        audio_feat, image_feat = self.pool_features(
            audio_feat, audio_feat_len, image_feat, False
        )

        if cal_loss:

            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            loss = self.criterion(
                features=torch.stack([audio_feat, image_feat], dim=1),
                labels=id,
            )

            if not ret_pre_pooling:
                return loss, audio_feat, image_feat, id
            else:
                return loss, prePool_audio, prePool_image, id

        return audio_feat, image_feat

    def log_grad_norm(self, grad_norm_dict):

        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.forward(batch, cal_loss=True)
        self.log("train_loss", loss)
        # return loss
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, prePool_audio, prePool_image, id = self.forward(
            batch, cal_loss=True, ret_pre_pooling=True
        )
        prePool_audio = prePool_audio[0].detach().cpu(), prePool_audio[1].detach().cpu()
        prePool_image = prePool_image.detach().cpu()
        id = id.detach().cpu()
        loss = loss.detach().cpu()
        self.log("val_loss", loss)
        return {
            "id": id,
            "prePool_audio": prePool_audio,
            "prePool_image": prePool_image,
        }

    def validation_epoch_end(self, validation_step_outputs):
        all_image_feat = []
        all_audio_feat = []
        all_audio_len = []
        results_scores = []
        for out in validation_step_outputs:
            audio_feat, audio_feat_len = out["prePool_audio"]
            image_feat = out["prePool_image"]
            all_image_feat.append(image_feat)
            all_audio_feat.append(audio_feat)
            all_audio_len.append(audio_feat_len)

        all_image_feat = torch.cat(all_image_feat, dim=0).float()
        # (#imgs,seqlen,dim)

        all_ids = torch.cat([x["id"] for x in validation_step_outputs], dim=0)
        id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_image_feat)}
        del all_image_feat

        all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
        all_img_feats_id = torch.LongTensor(list(id_img_pairs.keys()))
        all_audo_feats_id = all_ids

        print(
            "Total #{} images, #{} audio".format(
                len(all_img_feats_id), len(all_audo_feats_id)
            )
        )

        for audio_feat, audio_feat_len in zip(all_audio_feat, all_audio_len):
            audio_feat = audio_feat.to(self.device)
            audio_feat_len = audio_feat_len.to(self.device)

            scores = []

            for image_offset in range(
                0, all_img_feats.shape[0], all_img_feats.shape[0] // 8
            ):
                _image_feats = all_img_feats[
                    image_offset : image_offset + all_img_feats.shape[0] // 8, :
                ].to(self.device)

                if len(_image_feats.shape) == 2:
                    _image_feats = _image_feats.unsqueeze(0)

                _audio_pooled_feats, _image_pooled_feats = self.pool_features(
                    audio_feat=audio_feat,
                    audio_feat_len=audio_feat_len,
                    image_feat=_image_feats,
                    batchwise_pooling=False,
                )

                _image_feats = _image_feats.cpu()
                del _image_feats

                _audio_pooled_feats = _audio_pooled_feats / _audio_pooled_feats.norm(
                    dim=-1, keepdim=True
                )

                _image_pooled_feats = _image_pooled_feats / _image_pooled_feats.norm(
                    dim=-1, keepdim=True
                )

                _audio_pooled_feats = _audio_pooled_feats.unsqueeze(-1)
                _image_pooled_feats = _image_pooled_feats.unsqueeze(-1)

                _score = torch.matmul(
                    _audio_pooled_feats.permute(0, 1, 3, 2), _image_pooled_feats
                )
                assert _score.shape == (
                    _audio_pooled_feats.shape[0],
                    _audio_pooled_feats.shape[1],
                    1,
                    1,
                )

                _score = _score.reshape(
                    _audio_pooled_feats.shape[0], _audio_pooled_feats.shape[1]
                )

                _score = _score.cpu()

                scores.append(_score)

            audio_feat = audio_feat.cpu()
            audio_feat_len = audio_feat_len.cpu()

            scores = torch.cat(scores, dim=1)

            # score.shape (bsz, nums of images )
            assert scores.shape == (
                audio_feat.shape[0],
                all_img_feats.shape[0],
            )
            del audio_feat, audio_feat_len

            results_scores.append(scores)

        results_scores = torch.cat(results_scores, dim=0)

        assert results_scores.shape == (len(all_audo_feats_id), len(all_img_feats_id))

        score_per_audio = results_scores
        score_per_image = score_per_audio.T
        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_img_feats_id

        self.reportRetrieval(
            score_per_audio=score_per_audio,
            score_per_image=score_per_image,
            AI_answers=AI_answers,
            IA_answers=IA_answers,
        )

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


class ParallelSpeechClip_AttPool_FineGrain(ParallelSpeechClip_AttPool_FineGrain_Base):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        # to get output before pooling
        self.alterViT()

    def alterViT(self):
        # overwrite forward function in ViT
        def new_forward(self, x):
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    self.class_embedding.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            return x

        # replace old forward() with new forward()
        self.clip.model.visual.old_forward = self.clip.model.visual.forward
        self.clip.model.visual.forward = types.MethodType(
            new_forward, self.clip.model.visual
        )

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


class ParallelSpeechClip_AttPool_FineGrainHookResBlk(
    ParallelSpeechClip_AttPool_FineGrain_Base
):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        if not hasattr(config.clip, "hook_resBlk_id"):
            logging.warning(
                "config.clip.hook_resBlk_id not specified, using default = -1"
            )
            config.clip.hook_resBlk_id = -1
        else:
            logging.info("hook to ResBlk(Id={})".format(config.clip.hook_resBlk_id))

    def forward_image(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(images, list):
            image_tensor = self.clip.prep_image(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            if images.dim() != 4 or images.shape[1] != 3:
                raise ValueError(f"Incorrect image tensor shape {images.shape}")
            image_tensor = images
        else:
            raise TypeError(f"Unknown image type {type(images)}")

        # register hook to layer to get inputs of pooling layer
        hook_inps = []

        def layer_hook(module, inp, out):
            hook_inps.append(inp[0].data.cpu())

        # get the input of the last resblock
        hook = self.clip.model.visual.transformer.resblocks[
            self.config.clip.hook_resBlk_id
        ].register_forward_hook(layer_hook)

        self.clip.encode_image(image_tensor)
        hook.remove()
        hook_inps = torch.cat(hook_inps, dim=1)
        # hook_inps (seqlen, bsz, dim)
        hook_inps = hook_inps.permute(1, 0, 2).to(self.device)

        # get rid of [CLS]
        hook_inps = hook_inps[:, 1:, :]

        return hook_inps


# original text image encoder setting in CLIP
class ParallelClipTextImage(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.clip = ClipModel(**config.clip)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.criterion = SupConLoss(
            temperature=config.cl_loss.temperature,
            contrast_mode=config.cl_loss.contrast_mode,
            base_temperature=config.cl_loss.base_temperature,
        )

        self.recall_at = config.retrieval.recall_at

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
        image = batch["image"]
        text = batch["text"]
        id = batch["id"]
        id = torch.cat(id, dim=0)

        img_feat = self.forward_image(image)
        text_feat = self.forward_text(text)

        if cal_loss and not full_utt:
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            loss = self.criterion(
                features=torch.stack([img_feat, text_feat], dim=1),
                labels=id,
            )

            return loss, img_feat, text_feat, id

        return img_feat, text_feat, id

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.forward(batch, cal_loss=True)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, img_feat, text_feat, id = self.forward(batch, cal_loss=True)
        loss = loss.detach().cpu()
        img_feat = img_feat.detach().cpu()
        text_feat = text_feat.detach().cpu()
        id = id.detach().cpu()

        self.log("val_loss", loss)
        return {
            "id": id,
            "text_feat": text_feat,
            "image_feat": img_feat,
        }

    def validation_epoch_end(self, outputs):
        all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        all_imgs = torch.cat([x["image_feat"] for x in outputs], dim=0)
        id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_imgs)}

        del all_imgs

        all_text_feats = torch.cat([x["text_feat"] for x in outputs], dim=0)
        all_text_feats_id = all_ids

        all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
        all_img_feats_id = torch.LongTensor(list(id_img_pairs.keys()))

        print(
            "Total #{} image, #{} text".format(len(all_img_feats), len(all_text_feats))
        )

        # calculate dot product
        score_per_text = torch.matmul(
            all_text_feats.to(self.device), all_img_feats.T.to(self.device)
        )
        score_per_image = score_per_text.T

        # TI : Text -> Image, IT: Image -> Text
        TI_answers = all_text_feats_id
        IT_answers = all_img_feats_id

        recall_results_TI, recall_results_IT, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_text,
            score_per_B=score_per_image,
            AB_answers=TI_answers,
            BA_answers=IT_answers,
            recall_at=self.recall_at,
        )

        self.log("val_recall_TI", recall_results_TI)
        self.log("val_recall_IT", recall_results_IT)
        self.log("val_recall_mean", recall_results_mean)
        self.log("val_recall_mean_1", recall_results_mean["recall@1"])

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


# audio text retrieval
class ParallelSpeechClip_MeanPool_Text(ParallelSpeechClip_MeanPool):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

    def forward(
        self,
        batch,
        cal_loss: bool = False,
        full_utt: bool = False,
    ) -> dict:
        wav = batch["wav"]
        wav_len = batch["wav_len"]
        text = batch["text"]
        id = batch["id"]
        id = torch.cat(id, dim=0)
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len, full_utt=full_utt)
        text_feat = self.forward_text(text)

        if cal_loss and not full_utt:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            loss = self.criterion(
                features=torch.stack([audio_feat, text_feat], dim=1),
                labels=id,
            )

            return loss, audio_feat, text_feat, id

        return audio_feat, text_feat, id

    def validation_step(self, batch, batch_idx):
        loss, audio_feat, text_feat, id = self.forward(batch, cal_loss=True)
        loss = loss.detach().cpu()
        audio_feat = audio_feat.detach().cpu()
        text_feat = text_feat.detach().cpu()
        id = id.detach().cpu()
        self.log("val_loss", loss)
        return {
            "id": id,
            "audio_feat": audio_feat,
            "text_feat": text_feat,
        }

    def validation_epoch_end(self, outputs):
        all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        all_text_feats = torch.cat([x["text_feat"] for x in outputs], dim=0)
        all_audo_feats = torch.cat([x["audio_feat"] for x in outputs], dim=0)
        all_audo_feats_id = all_ids
        all_text_feats_id = all_ids

        print(
            "Total #{} text, #{} audio".format(len(all_text_feats), len(all_audo_feats))
        )

        # calculate dot product
        score_per_audio = torch.matmul(
            all_audo_feats.to(self.device), all_text_feats.T.to(self.device)
        )
        score_per_text = score_per_audio.T

        # AT : Audio -> Text, TA: Text -> Audio
        AT_answers = all_audo_feats_id
        TA_answers = all_text_feats_id

        recall_results_AT, recall_results_TA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_text,
            AB_answers=AT_answers,
            BA_answers=TA_answers,
            recall_at=self.recall_at,
        )

        self.log("val_recall_AT", recall_results_AT)
        self.log("val_recall_TA", recall_results_TA)
        self.log("val_recall_mean", recall_results_mean)
        self.log("val_recall_mean_1", recall_results_mean["recall@1"])

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


class ParallelSpeechClip_AttPoolText(ParallelSpeechClip_AttPool):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

    def forward(
        self,
        batch,
        cal_loss: bool = False,
        full_utt: bool = False,
        ret_pre_pooling: bool = False,
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        text = batch["text"]
        id = batch["id"]

        id = torch.cat(id, dim=0)
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len, full_utt=True)
        text_feat = self.forward_text(text)

        prePool_audio = audio_feat, audio_feat_len
        prePool_text = text_feat

        audio_feat, text_feat = self.pool_features(
            audio_feat, audio_feat_len, text_feat
        )

        # image_feat is actually the same, since each image is presented with only one vector

        if cal_loss:

            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            loss = self.criterion(
                features=torch.stack([audio_feat, text_feat], dim=1),
                labels=id,
            )

            if not ret_pre_pooling:
                return loss, audio_feat, text_feat, id
            else:
                return loss, prePool_audio, prePool_text, id

        return audio_feat, text_feat

    def validation_step(self, batch, batch_idx):
        loss, prePool_audio, prePool_text, id = self.forward(
            batch, cal_loss=True, ret_pre_pooling=True
        )
        prePool_audio = prePool_audio[0].detach().cpu(), prePool_audio[1].detach().cpu()
        prePool_text = prePool_text.detach().cpu()
        id = id.detach().cpu()
        loss = loss.detach().cpu()
        self.log("val_loss", loss)
        return {
            "id": id,
            "prePool_audio": prePool_audio,
            "prePool_text": prePool_text,
        }

    def validation_epoch_end(self, validation_step_outputs):
        all_text_feats = []
        all_audio_feat = []
        all_audio_len = []
        results_scores = []
        for out in validation_step_outputs:
            audio_feat, audio_feat_len = out["prePool_audio"]
            text_feat = out["prePool_text"]
            all_text_feats.append(text_feat)
            all_audio_feat.append(audio_feat)
            all_audio_len.append(audio_feat_len)

        all_text_feats = torch.cat(all_text_feats, dim=0).float()
        all_ids = torch.cat([x["id"] for x in validation_step_outputs], dim=0)
        all_audo_feats_id = all_ids
        all_text_feats_id = all_ids

        print(
            "Total #{} captions, #{} audio".format(
                len(all_text_feats_id), len(all_audo_feats_id)
            )
        )

        for audio_feat, audio_feat_len in zip(all_audio_feat, all_audio_len):
            audio_feat = audio_feat.to(self.device)
            audio_feat_len = audio_feat_len.to(self.device)

            if self.audio_feat_projection_type == "pre":
                audio_feat = self.audio_feat_projection(audio_feat)

            mask = self.audio_pooling.generate_input_msk(
                input_A_lens=audio_feat_len, max_Alen=audio_feat.shape[1]
            )

            scores = []
            for image_offset in range(
                0, all_text_feats.shape[0], all_text_feats.shape[0] // 2
            ):
                _text_feats = all_text_feats[
                    image_offset : image_offset + all_text_feats.shape[0] // 2, :
                ].to(self.device)

                if len(_text_feats.shape) == 1:
                    _text_feats = _text_feats.unsqueeze(0)

                _audio_pooled_feats = self.audio_pooling.cal_batch_embedding(
                    input_A=audio_feat.permute(0, 2, 1),
                    input_B=_text_feats.permute(1, 0),
                    intput_msk=mask,
                )

                # audio_pooled_feats.shape = (bsz, audio_dim, #audioSamples)
                assert _audio_pooled_feats.shape == (
                    audio_feat.shape[0],
                    audio_feat.shape[2],
                    _text_feats.shape[0],
                ), (
                    _audio_pooled_feats.shape,
                    (audio_feat.shape[0], audio_feat.shape[2], _text_feats.shape[0]),
                )

                # audio_pooled_feats.shape = (bsz, #audioSamples, audio_dim)
                _audio_pooled_feats = _audio_pooled_feats.permute(0, 2, 1)

                if self.audio_feat_projection_type == "post":
                    _audio_pooled_feats = self._audio_pooled_feats(_audio_pooled_feats)

                _audio_pooled_feats = _audio_pooled_feats / _audio_pooled_feats.norm(
                    dim=-1, keepdim=True
                )

                _text_feats = _text_feats / _text_feats.norm(dim=-1, keepdim=True)

                _score = torch.matmul(_audio_pooled_feats, _text_feats.T)
                _score = torch.diagonal(_score, offset=0, dim1=1, dim2=2)

                # _score.shape (bsz, len_of_all_data_pairs // n )
                assert _score.shape == (
                    _audio_pooled_feats.shape[0],
                    _text_feats.shape[0],
                )
                scores.append(_score)

            scores = torch.cat(scores, dim=1)
            scores = scores.cpu()

            # score.shape (bsz, len_of_all_data_pairs )
            assert scores.shape == (
                audio_feat.shape[0],
                all_text_feats.shape[0],
            )
            results_scores.append(scores)

        results_scores = torch.cat(results_scores, dim=0)

        assert results_scores.shape == (len(all_audo_feats_id), len(all_text_feats_id))

        score_per_audio = results_scores
        score_per_text = score_per_audio.T
        # AT : Audio -> Text, TA: Text -> Audio
        AT_answers = all_audo_feats_id
        TA_answers = all_text_feats_id

        recall_results_AT, recall_results_TA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_text,
            AB_answers=AT_answers,
            BA_answers=TA_answers,
            recall_at=self.recall_at,
        )

        self.log("val_recall_AT", recall_results_AT)
        self.log("val_recall_TA", recall_results_TA)
        self.log("val_recall_mean", recall_results_mean)
        self.log("val_recall_mean_1", recall_results_mean["recall@1"])

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
