import json
import logging
import math
import pickle
from typing import Tuple, Union

import numpy as np
import torch
from importlib_metadata import distribution
from jiwer import cer, wer
from torch import nn
from torch.nn import functional as F

from avssl.base import OrderedNamespace
from avssl.module import (
    ClipModel,
    MeanPoolingLayer,
    S3prlSpeechEncoder,
    SupConLoss,
    mutualRetrieval,
)
from avssl.module.speechclip_c_modules import (
    GumbelVectorQuantizer,
    KmeansVectorQuantizer,
)
from avssl.module.speechclip_c_modules.cif import CIF
from avssl.optim import get_scheduler

from .base_model import BaseLightningModel


class CascadedSpeechClip(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        # self.automatic_optimization = False
        # self.device = config.clip.device
        self.audio_encoder_type = config.audio_encoder.type
        if self.audio_encoder_type == "s3prl":
            self.audio_encoder = S3prlSpeechEncoder(**config.audio_encoder)
            self.embd_dim = self.audio_encoder.out_dim
        else:
            raise NotImplementedError(
                f"Unknown audio encoder type {self.audio_encoder_type}"
            )

        self.clip = ClipModel(
            **config.clip,
            device=self.device,
        )

        self.text_embd_dim = self.clip.model.token_embedding.weight.size(-1)

        if hasattr(config, "downsampling"):
            self.downsampling_type = config.downsampling.type
        else:
            self.downsampling_type = "cnn"

        # self.downsampling = nn.Sequential(
        #     nn.Conv1d(self.embd_dim, self.embd_dim, 2, 2, 0, 1),
        #     nn.AvgPool1d(2, 2, 0),
        #     nn.Conv1d(self.embd_dim, self.text_embd_dim, 2, 2, 0, 1),
        # )

        # filter 1
        # self.downsampling = nn.Sequential(
        #     nn.Conv1d(self.embd_dim, self.embd_dim, 5, 5, 0, 1),
        #     nn.AvgPool1d(2, 2, 0),
        #     nn.Conv1d(self.embd_dim, self.text_embd_dim, 2, 2, 0, 1),
        # )
        if self.downsampling_type == "cnn":
            # filter 2
            self.downsampling = nn.Sequential(
                nn.Conv1d(self.embd_dim, self.embd_dim, 10, 5, 0, 1),
                nn.AvgPool1d(2, 2, 0),
                nn.Conv1d(self.embd_dim, self.text_embd_dim, 4, 2, 0, 1),
            )
        elif self.downsampling_type == "cif":
            self.downsampling = CIF(
                audio_feat_dim=self.embd_dim,
                beta=config.downsampling.cif.beta,
                scaling_stragety=config.downsampling.cif.scaling_stragety,
                cal_quantity_loss=config.downsampling.cif.cal_quantity_loss,
                tail_handling=config.downsampling.cif.tail_handling,
            )
            self.cif_lamda_c = config.downsampling.cif.lamda_c
        else:
            raise NotImplementedError()

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
                dim=self.text_embd_dim
                if not self.downsampling_type == "cif"
                else self.embd_dim,
                num_vars=self.clip.model.token_embedding.weight.size(
                    0
                ),  # config.vq.num_vars,
                temp=config.vq.temp,
                groups=config.vq.groups,
                combine_groups=config.vq.combine_groups,
                vq_dim=config.vq.vq_dim if config.vq.vq_dim > 0 else self.text_embd_dim,
                time_first=False,
                activation=activation,
                weight_proj_factor=2,
                # init_codebook=self.clip.model.token_embedding.weight.to(config.device),
                init_codebook=0,  # no codebook needed
                groundTruthPerplexity=config.vq.groundTruthPerplexity
                if hasattr(config.vq, "groundTruthPerplexity")
                else None,
            )
        elif self.vq_type == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=self.text_embd_dim
                if not self.downsampling_type == "cif"
                else self.embd_dim,
                num_vars=config.vq.num_vars,
                groups=config.vq.groups,
                combine_groups=config.vq.combine_groups,
                vq_dim=config.vq.vq_dim if config.vq.vq_dim > 0 else self.text_embd_dim,
                time_first=False,
                gamma=config.vq.gamma,
                init_codebook=self.clip.model.token_embedding,
            )
        else:
            assert (
                config.vq_type == "none" or config.vq_type is None
            ), "Unknown quantizer type"

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.recall_at = config.retrieval.recall_at

        self.beta = config.vq.beta

        self.criterion = SupConLoss(
            temperature=config.cl_loss.temperature,
            contrast_mode=config.cl_loss.contrast_mode,
            base_temperature=config.cl_loss.base_temperature,
        )

        self.log_detokenize_results = True
        if hasattr(config, "log_setting"):
            if hasattr(config.log_setting, "log_detokenize_results"):
                self.log_detokenize_results = config.log_setting.log_detokenize_results

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

    def forward(
        self,
        batch,
        cal_loss: bool = False,
    ) -> dict:
        def conv1d_length(
            length: Union[torch.Tensor, list],
            kernel: int,
            stride: int,
            pad: int,
            dilation: int,
        ):
            for i in range(length.size(0)):
                length[i] = math.floor(
                    (length[i] + 2 * pad - dilation * (kernel - 1)) / stride + 1
                )

        def mean_length(
            length: Union[torch.Tensor, list], kernel: int, stride: int, pad: int
        ):
            for i in range(length.size(0)):
                length[i] = math.floor((length[i] + 2 * pad - kernel) / stride + 1)

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]
        id = torch.cat(id, dim=0)

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        image_feat = self.forward_image(image)
        # print(image_feat.shape)
        # # exit(1)

        q_loss = None
        if self.downsampling_type == "cnn":
            #  down sampling
            audio_feat = audio_feat.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
            audio_feat = self.downsampling(audio_feat)

            # # compute audio length
            # conv1d_length(audio_len, 2, 2, 0, 1)
            # mean_length(audio_len, 2, 2, 0)
            # conv1d_length(audio_len, 2, 2, 0, 1)

            # # compute audio length
            # conv1d_length(audio_len, 5, 5, 0, 1)
            # mean_length(audio_len, 2, 2, 0)
            # conv1d_length(audio_len, 2, 2, 0, 1)

            # compute audio length
            conv1d_length(audio_len, 10, 5, 0, 1)
            mean_length(audio_len, 2, 2, 0)
            conv1d_length(audio_len, 4, 2, 0, 1)
        elif self.downsampling_type == "cif":
            text = batch["text"]
            text_toks = self.clip.prep_text(text).tolist()
            text_toks_len = []
            for t in text_toks:
                _x = t.index(self.clip.endOfTxt_reduced)
                assert _x > 1
                text_toks_len.append(_x - 1)
            text_toks_len = torch.tensor(text_toks_len).to(self.device)
            downsampling_out = self.downsampling(
                encoder_outputs=audio_feat,
                encoder_lens=audio_len,
                target_length=text_toks_len,
                # paddingTensor = self.clip.model.token_embedding(torch.tensor([0]).to(self.device)).squeeze()
            )
            if self.downsampling.cal_quantity_loss:
                audio_feat, audio_len, q_loss = downsampling_out
            else:
                audio_feat, audio_len = downsampling_out

            del downsampling_out
            audio_feat = audio_feat.permute(0, 2, 1)

        # vector quantization
        if self.vq_type == "gumbel":
            self.vector_quantizer.set_num_updates(self.global_step)
        vq_result = self.vector_quantizer(audio_feat, produce_targets=True)

        # mutliply subword distribution with clip text embeddings
        audio_feat = self.clip.encode_subword(vq_result, audio_len, self.vq_type)
        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape

            assert audio_feat.shape[0] == id.shape[0]

            cl_loss = self.criterion(
                features=torch.stack([audio_feat, image_feat], dim=1),
                labels=id,
            )
            if q_loss is not None:
                loss = (
                    vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
                )
            else:
                loss = vq_result["loss"] * self.beta + cl_loss
            losses = {
                "loss": loss,
                "vq_loss": vq_result["loss"].detach(),
                "cl_loss": cl_loss.detach(),
            }
            if q_loss is not None:
                losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, vq_result, id

        return audio_feat, image_feat, vq_result, id

    def training_step(self, batch, batch_idx):
        losses, _, _, res, _ = self.forward(batch, cal_loss=True)

        result = {}
        for key in res.keys():
            if key in ["code_perplexity", "prob_perplexity", "temp"]:
                result["train_{}".format(key)] = res[key]

        self.log_dict(result, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("loss", losses["loss"])
        self.log("train_cl_loss", losses["cl_loss"])
        self.log("train_vq_loss", losses["vq_loss"])
        if "q_loss" in losses:
            self.log("q_loss", losses["q_loss"])

        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        losses, audio_feat, image_feat, res, id = self.forward(batch, cal_loss=True)

        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        id = id.detach().cpu()

        result = {
            "val_loss": losses["loss"],
            "val_vq_loss": losses["vq_loss"],
            "val_cl_loss": losses["cl_loss"],
        }
        if "q_loss" in losses:
            result.update({"q_loss": losses["q_loss"]})

        for key in res.keys():
            if isinstance(res[key], torch.Tensor):
                res[key] = res[key].detach().cpu()
            if key in ["code_perplexity", "prob_perplexity", "temp"]:
                result["val_{}".format(key)] = res[key]

        detok_text = self.clip.deTokenize(res["targets"])

        wer_score = wer(batch["text"], detok_text)
        cer_score = cer(batch["text"], detok_text)

        result.update(
            {
                "val_wer": wer_score,
                "val_cer": cer_score,
            }
        )

        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {
            "id": id,
            "audio_feat": audio_feat,
            "image_feat": image_feat,
            "vq_targets": res["targets"].squeeze(),
            "gold_text": batch["text"],
            "detok_text": detok_text,
        }

    def validation_epoch_end(self, outputs):
        if self.log_detokenize_results:
            if not os.path.exists(os.path.join(self.logger.log_dir, "retokenizeText")):
                os.makedirs(
                    os.path.join(self.logger.log_dir, "retokenizeText"), exist_ok=True
                )
            retokenizeText_output = []

            for x in outputs:
                for _g, _d in zip(x["gold_text"], x["detok_text"]):
                    retokenizeText_output.append({"gold": _g, "detok": _d})

            with open(
                os.path.join(
                    self.logger.log_dir,
                    "retokenizeText/",
                    "ep{}.json".format(self.current_epoch),
                ),
                "w",
            ) as f:
                json.dump(retokenizeText_output, f)
            del retokenizeText_output

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

        # calculate KL between vq targets and ground truth dist.
        all_targets = torch.cat([x["vq_targets"].flatten() for x in outputs], dim=0)
        all_targets = all_targets.flatten()
        all_targets = all_targets.long()
        vq_dist = torch.bincount(all_targets)
        del all_targets
        if len(vq_dist) < len(self.clip.selected_text_emb_ids_dist):
            vq_dist = torch.cat(
                [
                    vq_dist,
                    torch.zeros(
                        len(self.clip.selected_text_emb_ids_dist) - len(vq_dist)
                    ),
                ]
            )
        codebook_usage = torch.sum(vq_dist > 0) / len(vq_dist)
        # smooth the prob. of vq_dist to calculate KL
        vq_dist = vq_dist + 1e-10
        vq_dist = vq_dist / vq_dist.sum()
        target_KL = F.kl_div(
            torch.log(vq_dist), self.clip.selected_text_emb_ids_dist, reduction="sum"
        )
        self.log("val_target_KL", target_KL)
        self.log("val_codebook_usage", codebook_usage)

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
