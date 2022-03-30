from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

from avssl.base import OrderedNamespace
from avssl.module import ClipModel, MeanPoolingLayer, S3prlSpeechEncoder, SupConLoss, AttentativePoolingLayer
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


class ParallelSpeechClip_AttPool(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        # print(config)
        # print(config.to_dict())
        # print()
        # print(config._odict)
        # # print(config.keys())
        # # print(dir(config))
        # exit(1)

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

        self.audio_pooling_type = config.audio_encoder.pooling.type
        self.audio_pooling_degraded = config.audio_encoder.pooling.degraded

        if self.audio_pooling_type == "attentative_pooling":
            self.audio_pooling = AttentativePoolingLayer(
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

        self.criterion = nn.CrossEntropyLoss()

        # evaluate
        self.recall_at = config.evaluate.recall_at

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
        wav, wav_len, images = batch
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

            logit_scale = self.logit_scale.exp()
            logits_per_audio = logit_scale * audio_feat @ image_feat.t()
            logits_per_image = logits_per_audio.t()

            labels = torch.arange(
                len(logits_per_audio), device=logits_per_audio.device, dtype=torch.long
            )
            loss_audio = self.criterion(logits_per_audio, labels)
            loss_image = self.criterion(logits_per_image, labels)
            loss = (loss_audio + loss_image) / 2

            if not ret_pre_pooling:
                return loss, audio_feat, image_feat
            else:
                return loss, prePool_audio, prePool_image, audio_feat, image_feat

        return audio_feat, image_feat

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.forward(batch, cal_loss=True)
        self.log("train_loss", loss)
        # return loss
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # print("val")
        loss, prePool_audio, prePool_image, _, _ = self.forward(
            batch, cal_loss=True, ret_pre_pooling=True
        )
        prePool_audio = prePool_audio[0].cpu(), prePool_audio[1].cpu()
        prePool_image = prePool_image.cpu()
        self.log("val_loss", loss)
        return {
            "loss": loss,
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
        print("[validation_epoch] Total data pairs #{}".format(all_image_feat.shape[0]))
        # all_image_feat = all_image_feat.cuda()

        with torch.no_grad():
            for audio_feat, audio_feat_len in zip(all_audio_feat, all_audio_len):
                audio_feat = audio_feat.cuda()
                audio_feat_len = audio_feat_len.cuda()

                if self.audio_feat_projection_type == "pre":
                    audio_feat = self.audio_feat_projection(audio_feat)

                mask = self.audio_pooling.generate_input_msk(
                    input_A_lens=audio_feat_len, max_Alen=audio_feat.shape[1]
                )

                scores = []
                for image_offset in range(0,all_image_feat.shape[0],all_image_feat.shape[0]//2):
                    _image_feats = all_image_feat[image_offset:image_offset+all_image_feat.shape[0]//2,:].cuda()

                    _audio_pooled_feats = self.audio_pooling.cal_batch_embedding(
                        input_A=audio_feat.permute(0, 2, 1),
                        input_B=_image_feats.permute(1, 0),
                        intput_msk=mask,
                    )

                    # audio_pooled_feats.shape = (bsz, audio_dim, len_of_all_data_pairs)
                    assert _audio_pooled_feats.shape == (
                        audio_feat.shape[0],
                        audio_feat.shape[2],
                        _image_feats.shape[0],
                    )

                    # audio_pooled_feats.shape = (bsz, len_of_all_data_pairs, audio_dim)
                    _audio_pooled_feats = _audio_pooled_feats.permute(0, 2, 1)

                    if self.audio_feat_projection_type == "post":
                        _audio_pooled_feats = self._audio_pooled_feats(_audio_pooled_feats)

                    _audio_pooled_feats = _audio_pooled_feats / _audio_pooled_feats.norm(
                        dim=-1, keepdim=True
                    )
                
                    _image_feats = _image_feats / _image_feats.norm(
                        dim=-1, keepdim=True
                    )

                    _score = torch.matmul(_audio_pooled_feats, _image_feats.T)
                    _score = torch.diagonal(_score, offset=0, dim1=1, dim2=2)

                    # _score.shape (bsz, len_of_all_data_pairs // n )
                    assert _score.shape == (
                        _audio_pooled_feats.shape[0],
                        _image_feats.shape[0],
                    )
                    scores.append(_score)

                scores = torch.cat( scores, dim=1)
                scores = scores.cpu()
                # score.shape (bsz, len_of_all_data_pairs )
                assert scores.shape == (
                    audio_feat.shape[0],
                    all_image_feat.shape[0],
                )
                results_scores.append(scores)

        results_scores = torch.cat(results_scores, dim=0)

        score_rank = torch.argsort(results_scores, dim=1, descending=True)
        # print(score_rank)
        score_rank = score_rank == torch.arange(
            results_scores.shape[1], device=results_scores.device, dtype=torch.long
        ).reshape(-1, 1)
        # print(score_rank)

        recall = []

        for k in self.recall_at:
            _v = (
                torch.sum(score_rank[:, :k].reshape(-1, 1), dim=0).item()
                / all_image_feat.shape[0]
            )
            recall.append(_v)

        self.log( "val_recall", { "val_recall_{}".format(k): recall[i]  for i,k in enumerate(self.recall_at)})

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
