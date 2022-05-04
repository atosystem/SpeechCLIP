import json
import logging
import math
import os
import pickle
from ast import keyword
from typing import Tuple, Union

import numpy as np
import torch
from jiwer import cer, wer
from torch import nn
from torch.nn import functional as F

from ..base import OrderedNamespace
from ..module import (
    ClipModel,
    MeanPoolingLayer,
    S3prlSpeechEncoder,
    SupConLoss,
    mutualRetrieval,
)
from ..module.speechclip_c_modules import GumbelVectorQuantizer, KmeansVectorQuantizer
from ..module.speechclip_c_modules.cif import CIF
from ..optim import get_scheduler
from .base_model import BaseLightningModel
from ..module.speechclip_c_modules import vector_quantizers


class CascadedSpeechClip_Base(BaseLightningModel):
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
        # if self.downsampling_type == "cnn":
        #     # filter 2
        #     self.downsampling = nn.Sequential(
        #         nn.Conv1d(self.embd_dim, self.embd_dim, 10, 5, 0, 1),
        #         nn.AvgPool1d(2, 2, 0),
        #         nn.Conv1d(self.embd_dim, self.text_embd_dim, 4, 2, 0, 1),
        #     )
        # elif self.downsampling_type == "cif":
        #     self.downsampling = CIF(
        #         audio_feat_dim=self.embd_dim,
        #         beta=config.downsampling.cif.beta,
        #         scaling_stragety=config.downsampling.cif.scaling_stragety,
        #         cal_quantity_loss=config.downsampling.cif.cal_quantity_loss,
        #         tail_handling=config.downsampling.cif.tail_handling,
        #     )
        #     self.cif_lamda_c = config.downsampling.cif.lamda_c
        # else:
        #     raise NotImplementedError()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.recall_at = config.retrieval.recall_at

        self.criterion = SupConLoss(
            temperature=config.cl_loss.temperature,
            contrast_mode=config.cl_loss.contrast_mode,
            base_temperature=config.cl_loss.base_temperature,
        )

        self.log_detokenize_results = True
        if hasattr(config, "log_setting"):
            if hasattr(config.log_setting, "log_detokenize_results"):
                self.log_detokenize_results = config.log_setting.log_detokenize_results

    def feature_extractor_s3prl(self, wav):
        """feature_extractor_s3prl
        Implement for s3prl to get feature
        Args:
            wav ():
        """
        pass

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
        # print()
        # for _key in recall_results_AI:
        #     self.log("val_recall_AI/{}".format(_key), recall_results_AI[_key])
        # for _key in recall_results_IA:
        #     self.log("val_recall_IA/{}".format(_key), recall_results_IA[_key])
        # for _key in recall_results_mean:
        #     self.log("val_recall_mean/{}".format(_key), recall_results_mean[_key])
        # self.log("val_recall_mean_1", recall_results_mean["recall@1"])


        self.logger.experiment.add_scalars('val_recall_AI',recall_results_AI , self.global_step) 
        self.logger.experiment.add_scalars('val_recall_IA',recall_results_IA , self.global_step) 
        self.logger.experiment.add_scalars('val_recall_mean',recall_results_mean , self.global_step) 
        # self.logger.experiment.add_scalar('val_recall_mean_1',recall_results_mean["recall@1"] , self.global_step) 
        

        # self.log("val_recall_AI", recall_results_AI)
        # self.log("val_recall_IA", recall_results_IA)
        # self.log("val_recall_mean", recall_results_mean)
        self.log("val_recall_mean_1", recall_results_mean["recall@1"])


class VQCascadedSpeechClip(CascadedSpeechClip_Base):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.vector_quantizer = None
        self.vq_type = config.vq.type
        self.beta = config.vq.beta

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

    def forward(
        self,
        batch,
        cal_loss: bool = False,
    ) -> dict:
        max_len = 75

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
                if length[i] > max_len:
                    length[i] = max_len

        def mean_length(
            length: Union[torch.Tensor, list], kernel: int, stride: int, pad: int
        ):
            for i in range(length.size(0)):
                length[i] = math.floor((length[i] + 2 * pad - kernel) / stride + 1)
                if length[i] > max_len:
                    length[i] = max_len

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
            if isinstance(audio_feat, list):
                audio_feat = audio_feat
            audio_feat = audio_feat.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
            audio_feat = self.downsampling(audio_feat)

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
            self.log("train_q_loss", losses["q_loss"])

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
            result.update({"val_q_loss": losses["q_loss"]})

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
                "val_wer": wer_score * 100,
                "val_cer": cer_score * 100,
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
        audio_params = []

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


class KeywordCascadedSpeechClip(CascadedSpeechClip_Base):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.multihead_attn_layer = nn.MultiheadAttention(
            self.embd_dim, num_heads=1, dropout=0.1, batch_first=True
        )
        self.attentionBlock_Norm = nn.LayerNorm(self.embd_dim, eps=1e-5)
        self.keyword_num = 1
        self.downsampling_type = None
        if self.downsampling_type is None:
            self.linear_proj = nn.Linear(self.embd_dim, self.text_embd_dim)
        self.linear_proj
        self.log_detokenize_results = True

        logging.info("Start init [CLS]")
        self.cls = torch.nn.Parameter(torch.randn([1, self.keyword_num, self.embd_dim]))

    def feature_extractor_s3prl(self, wav):
        wav_len = [len(x) for x in wav]
        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]
        # for x in hidden_states:
        #     print(x.shape)
        # print()
        # print(audio_feat.shape)
        return audio_feat, hidden_states[:]

    def forward(
        self,
        batch,
        cal_loss: bool = False,
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]
        id = torch.cat(id, dim=0)

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        image_feat = self.forward_image(image)

        q_loss = None
        
        # Use multi-head attention layer to find keywords(cls)
        bsz = audio_feat.size(0)
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = torch.ones([bsz, audio_feat.size(1) + self.keyword_num])
        for mask, len in zip(key_padding_mask, audio_len):
            len += self.keyword_num  # add cls
            mask[:len] = torch.zeros(mask[:len].size())
        key_padding_mask = key_padding_mask.bool().to(self.device)
        # print(key_padding_mask[0])
        # print(key_padding_mask[1])
        # exit(11)

        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[0] + src
        )

        keywords = keywords[:, : self.keyword_num]
        if self.downsampling_type is None:
            keywords = self.linear_proj(keywords)

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape, "{} {}".format(audio_feat.shape , image_feat.shape )

            assert audio_feat.shape[0] == id.shape[0]

            cl_loss = self.criterion(
                features=torch.stack([audio_feat, image_feat], dim=1),
                labels=id,
            )
            # if q_loss is not None:
            #     loss = (
            #         vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
            #     )
            # else:
            #     loss = vq_result["loss"] * self.beta + cl_loss
            losses = {
                "cl_loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, id, keywords

        return audio_feat, image_feat, res, id
    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     pass
        # print("self.cls",self.multihead_attn_layer.in_proj_weight.flatten()[0])
        # if self.multihead_attn_layer.in_proj_weight.grad is not None:
        #     print("self.cls.grad",self.multihead_attn_layer.in_proj_weight.grad.flatten()[0])
        # else:
        #     print("self.cls.grad",None)

        # print("self.linear_proj",self.linear_proj.weight.flatten()[0])
        # if self.linear_proj.weight.grad is not None:
        #     print("self.linear_proj.grad",self.linear_proj.weight.grad.flatten()[0])
        # else:
        #     print("self.linear_proj.grad",None)
        # if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
        #     for k, v in self.named_parameters():
        #         self.logger.experiment.add_histogram(
        #             tag=k, values=v.grad, global_step=self.trainer.global_step
        #         )

    def training_step(self, batch, batch_idx):
        # print("self.cls",self.cls.squeeze()[:10])
        # if self.cls.squeeze().grad is not None:
        #     print("self.cls.grad",self.cls.squeeze().grad[:10])
        # else:
        #     print("self.cls.grad",None)

        # print("multihead_attn_layer",self.multihead_attn_layer.k_proj_weight.grad == None)
        losses, _, _, _, _ = self.forward(batch, cal_loss=True)
        result = {
            "train_loss": losses["cl_loss"],
            "cl_temp": self.criterion.get_temp(),
        }
        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # self.log("train_cl_loss", losses["cl_loss"])

        return {"loss": losses["cl_loss"]}

    def validation_step(self, batch, batch_idx):
        # print(self.cls.squeeze()[:10])
        # exit(1)
        losses, audio_feat, image_feat, id, keywords = self.forward(
            batch, cal_loss=True
        )

        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        keywords = keywords.detach().cpu().squeeze()
        id = id.detach().cpu()

        result = {
            "val_loss": losses["cl_loss"].item(),
        }

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
            "keywords": keywords,
            # "vq_targets": res["targets"].squeeze(),
            "gold_text": batch["text"],
            # "detok_text": detok_text,
        }

    def validation_epoch_end(self, outputs):
        if not os.path.exists(os.path.join(self.logger.log_dir, "retokenizeText")):
            os.makedirs(
                os.path.join(self.logger.log_dir, "retokenizeText"), exist_ok=True
            )

        gold_texts = []
        for x in outputs:
            gold_texts.extend(x["gold_text"])
        # gold_texts = [ x["gold_text"] for x in outputs]
        # gold_texts = [ x["gold_text"] for x in gold_texts]
        all_keyword_embeddings = torch.cat(
            [x["keywords"] for x in outputs], dim=0
        ).squeeze()

        assert all_keyword_embeddings.dim() == 2, all_keyword_embeddings.shape
        assert all_keyword_embeddings.shape[1] == 512, all_keyword_embeddings.shape
        all_retok_outputs = []
        K = 10
        tokenEmbeddings = self.clip.model.token_embedding.weight.detach().cpu()
        for i in range(len(gold_texts)):
            _k_values, _k_indices = torch.topk(
                F.cosine_similarity(all_keyword_embeddings[i], tokenEmbeddings), K
            )
            tmp_outputs = []
            for _ind, _dist in zip(_k_indices, _k_values):
                tmp_outputs.append(
                    [
                        self.clip.tokenizer.decoder[
                            # self.clip.reducedl2Original[_ind.item()]
                            _ind.item()
                        ],
                        _dist.item(),
                    ]
                )
            all_retok_outputs.append(
                {
                    "gold": gold_texts[i],
                    "neighbors": tmp_outputs,
                }
            )

        with open(
            os.path.join(
                self.logger.log_dir,
                "retokenizeText/",
                "keywords_ep{}.json".format(self.current_epoch),
            ),
            "w",
        ) as f:
            json.dump(all_retok_outputs, f)
        del all_retok_outputs

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
        audio_params = []

        if self.config.audio_encoder.trainable:
            audio_params = audio_params +  list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = audio_params + [self.cls] + list(self.attentionBlock_Norm.parameters())

        audio_params = audio_params + list(self.criterion.parameters())

        # audio_optimizer = torch.optim.Adam( audio_params, lr=1e-1)

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
            assert False
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


class KeywordCascadedSpeechClip_ProjVQ(KeywordCascadedSpeechClip):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.multihead_attn_layer = nn.MultiheadAttention(
            self.embd_dim, num_heads=1, dropout=0.1, batch_first=True
        )
        self.keyword_num = 1
        self.downsampling_type = None

        self.cls_embeddings = nn.Embedding(
            num_embeddings=self.keyword_num,
            embedding_dim=self.audio_encoder.out_dim
        )


        self.projection_network = nn.Sequential(
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),

            # torch.nn.GELU(),
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),

            # torch.nn.GELU(),
            torch.nn.Linear(self.audio_encoder.out_dim,self.clip.model.token_embedding.num_embeddings)
        )

        self.log_detokenize_results = True

        # codebook selection
        self.vector_quantizer = None
        self.vq_type = config.vq.type

        if not hasattr(vector_quantizers,config.vq.type):
            raise NotImplementedError("Vq ({}) not implemented".format(config.vq.type))

        self.vector_quantizer = getattr(vector_quantizers,self.vq_type)(**config.vq.args)

    def feature_extractor_s3prl(self, wav):
        wav_len = [len(x) for x in wav]
        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]
        # for x in hidden_states:
        #     print(x.shape)
        # print()
        # print(audio_feat.shape)
        return audio_feat, hidden_states[:]

    def forward(
        self,
        batch,
        cal_loss: bool = False,
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]
        id = torch.cat(id, dim=0)

        # update device information to clip model
        self.clip.update_device(self.device)

        image_feat = self.forward_image(image)
        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        
        
        # Use multi-head attention layer to find keywords(cls)
        bsz = audio_feat.size(0)

        # cls = torch.cat([self.cls] * bsz, dim=0)
        cls = self.cls_embeddings.weight[0].view(1,1,-1).repeat(bsz,1,1)
        src = torch.cat([cls, audio_feat], dim=1)
        keywords = (self.multihead_attn_layer(src, src, src)[0])[:, : self.keyword_num]
        
        keywords = self.projection_network(keywords)

        # VQ
        vq_results = self.vector_quantizer(
            x=keywords
        )
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        
        # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape

            assert audio_feat.shape[0] == id.shape[0]

            cl_loss = self.criterion(
                features=torch.stack([audio_feat, image_feat], dim=1),
                labels=id,
            )
            # if q_loss is not None:
            #     loss = (
            #         vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
            #     )
            # else:
            #     loss = vq_result["loss"] * self.beta + cl_loss
            losses = {
                "cl_loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})
            return losses, audio_feat, image_feat, id, vq_results,  keywords

        return audio_feat, image_feat, res, id

    def training_step(self, batch, batch_idx):
        losses, _, _, _, vq_results, _ = self.forward(batch, cal_loss=True)

        result = {
            "train_loss": losses["cl_loss"],
            "temp" : vq_results["temp"],
            "train_prob_ppl" : vq_results["prob_perplexity"].item(),
            "train_code_ppl" : vq_results["code_perplexity"].item(),
        }
        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # self.log("train_cl_loss", losses["cl_loss"])

        return {"loss": losses["cl_loss"]}

    def validation_step(self, batch, batch_idx):
        losses, audio_feat, image_feat, id,vq_results, keywords = self.forward(
            batch, cal_loss=True
        )

        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        keywords = keywords.detach().cpu().squeeze()
        id = id.detach().cpu()

        result = {
            "val_loss": losses["cl_loss"].item(),
            "val_temp" : vq_results["temp"],
            "val_prob_ppl" : vq_results["prob_perplexity"].item(),
            "val_code_ppl" : vq_results["code_perplexity"].item(),
        }

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
            "keywords": keywords,
            # "vq_targets": res["targets"].squeeze(),
            "gold_text": batch["text"],
            # "detok_text": detok_text,
        }

    def validation_epoch_end(self, outputs):
        if not os.path.exists(os.path.join(self.logger.log_dir, "retokenizeText")):
            os.makedirs(
                os.path.join(self.logger.log_dir, "retokenizeText"), exist_ok=True
            )

        gold_texts = []
        for x in outputs:
            gold_texts.extend(x["gold_text"])
        # gold_texts = [ x["gold_text"] for x in outputs]
        # gold_texts = [ x["gold_text"] for x in gold_texts]
        all_keyword_embeddings = torch.cat(
            [x["keywords"] for x in outputs], dim=0
        ).squeeze()

        assert all_keyword_embeddings.dim() == 2, all_keyword_embeddings.shape
        assert all_keyword_embeddings.shape[1] == 512, all_keyword_embeddings.shape
        all_retok_outputs = []
        K = 10
        tokenEmbeddings = self.clip.model.token_embedding.weight.detach().cpu()
        for i in range(len(gold_texts)):
            _k_values, _k_indices = torch.topk(
                F.cosine_similarity(all_keyword_embeddings[i], tokenEmbeddings), K
            )
            tmp_outputs = []
            for _ind, _dist in zip(_k_indices, _k_values):
                tmp_outputs.append(
                    [
                        self.clip.tokenizer.decoder[
                            self.clip.reducedl2Original[_ind.item()]
                            # _ind.item()
                        ],
                        _dist.item(),
                    ]
                )
            all_retok_outputs.append(
                {
                    "gold": gold_texts[i],
                    "neighbors": tmp_outputs,
                }
            )

        with open(
            os.path.join(
                self.logger.log_dir,
                "retokenizeText/",
                "keywords_ep{}.json".format(self.current_epoch),
            ),
            "w",
        ) as f:
            json.dump(all_retok_outputs, f)
        del all_retok_outputs

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
        audio_params = []


        if self.config.audio_encoder.trainable:
            audio_params = list(self.audio_encoder.parameters())


        audio_params = audio_params + list(self.cls_embeddings.parameters())
        audio_params = audio_params + list(self.vector_quantizer.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.projection_network.parameters())

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


class KeywordCascadedSpeechClip_ProjVQ_Cosine(KeywordCascadedSpeechClip_ProjVQ):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.multihead_attn_layer = nn.MultiheadAttention(
            self.embd_dim, num_heads=1, dropout=0.1, batch_first=True
        )
        self.keyword_num = 1
        self.downsampling_type = None

        self.cls_embeddings = nn.Embedding(
            num_embeddings=self.keyword_num,
            embedding_dim=self.audio_encoder.out_dim
        )


        self.projection_network = nn.Sequential(
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),

            # torch.nn.GELU(),
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),

            # torch.nn.GELU(),
            torch.nn.Linear(self.audio_encoder.out_dim,self.clip.model.token_embedding.embedding_dim)
        )

        self.log_detokenize_results = True

        # codebook selection
        self.vector_quantizer = None
        self.vq_type = config.vq.type

        if not hasattr(vector_quantizers,config.vq.type):
            raise NotImplementedError("Vq ({}) not implemented".format(config.vq.type))

        self.vector_quantizer = getattr(vector_quantizers,self.vq_type)(**config.vq.args)

    def feature_extractor_s3prl(self, wav):
        wav_len = [len(x) for x in wav]
        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]
        # for x in hidden_states:
        #     print(x.shape)
        # print()
        # print(audio_feat.shape)
        return audio_feat, hidden_states[:]

    def forward(
        self,
        batch,
        cal_loss: bool = False,
    ):

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]
        id = torch.cat(id, dim=0)

        # update device information to clip model
        self.clip.update_device(self.device)

        image_feat = self.forward_image(image)
        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        
        
        # Use multi-head attention layer to find keywords(cls)
        bsz = audio_feat.size(0)

        # cls = torch.cat([self.cls] * bsz, dim=0)
        cls = self.cls_embeddings.weight[0].view(1,1,-1).repeat(bsz,1,1)
        src = torch.cat([cls, audio_feat], dim=1)
        keywords = (self.multihead_attn_layer(src, src, src)[0])[:, : self.keyword_num]
        
        keywords = self.projection_network(keywords)

        # cosine
        cos_score = F.cosine_similarity(
            keywords.view(bsz,self.text_embd_dim,1), 
            self.clip.model.token_embedding.weight.transpose(0,1).unsqueeze(0),
            dim=1
        )

        cos_score = cos_score.view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        # VQ
        vq_results = self.vector_quantizer(
            x=cos_score
        )
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        
        # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape

            assert audio_feat.shape[0] == id.shape[0]

            cl_loss = self.criterion(
                features=torch.stack([audio_feat, image_feat], dim=1),
                labels=id,
            )
            # if q_loss is not None:
            #     loss = (
            #         vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
            #     )
            # else:
            #     loss = vq_result["loss"] * self.beta + cl_loss
            losses = {
                "cl_loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})
            return losses, audio_feat, image_feat, id, vq_results,  keywords

        return audio_feat, image_feat, res, id


    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []


        if self.config.audio_encoder.trainable:
            audio_params = list(self.audio_encoder.parameters())


        audio_params = audio_params + list(self.cls_embeddings.parameters())
        audio_params = audio_params + list(self.vector_quantizer.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.projection_network.parameters())

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

