import json
import logging
import math
import os
import pickle
from ast import keyword
from typing import List, Tuple, Union

import numpy as np
import torch
import tqdm
from jiwer import cer, wer
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.loggers.wandb import WandbLogger

from ..base import OrderedNamespace
from ..module import (
    ClipModel,
    MeanPoolingLayer,
    S3prlSpeechEncoder,
    SimpleCache,
    SupConLoss,
    losses,
    mutualRetrieval,
)
from ..module.speechclip_c_modules import (
    GumbelVectorQuantizer,
    KmeansVectorQuantizer,
    vector_quantizers,
)
from ..module.speechclip_c_modules.cif import CIF
from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm
from ..optim import get_scheduler
from ..util.embedding_visualization import draw_embedding_space_PCA
from ..util.penalty_scheduler import PenaltyScheduler
from .base_model import BaseLightningModel


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
            self.downsampling_type = None

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.recall_at = config.retrieval.recall_at

        # self.criterion = SupConLoss(
        #     temperature=config.cl_loss.temperature,
        #     contrast_mode=config.cl_loss.contrast_mode,
        #     base_temperature=config.cl_loss.base_temperature,
        #     learnable_temperature=config.cl_loss.learnable_temperature,
        # )

        self.criterion = getattr(losses, config.cl_loss.type)(**config.cl_loss.args)

        self.log_detokenize_results = True
        if hasattr(config, "log_setting"):
            if hasattr(config.log_setting, "log_detokenize_results"):
                self.log_detokenize_results = config.log_setting.log_detokenize_results
        self.feat_select_idx = config.audio_encoder.feat_select_idx
        if self.feat_select_idx == "all":
            logging.warning("init self.audio_encoder_layer_weights")
            self.audio_encoder_layer_weights = nn.parameter.Parameter(
                torch.randn(
                    13,
                )
            )

        # self.cache_mods = self.config.get("cache_mods", [])
        if not hasattr(self.config, "cache_mods"):
            self.config.cache_mods = []
        self.cache_mods = []
        # self.cache_mods = self.config.cache_mods
        if len(self.cache_mods) > 0:
            logging.warning(f"Caching modalities {self.cache_mods}")
            cache_keys = []
            if "audio" in self.cache_mods:
                cache_keys.append("audio_feat")
                cache_keys.append("audio_feat_len")
            if "image" in self.cache_mods:
                cache_keys.append("image_feat")
            if "text" in self.cache_mods:
                cache_keys.append("text_feat")

            self.cache = SimpleCache(cache_keys)
        
        print("Experiment root dir {}".format(self.config.trainer.default_root_dir))

    # delete v_num
    # def get_progress_bar_dict(self):
    #     tqdm_dict = super().get_progress_bar_dict()
    #     if 'v_num' in tqdm_dict:
    #         del tqdm_dict['v_num']
    #     return tqdm_dict

    # def log_grad_norm(self, grad_norm_dict) -> None:
    #     """Override this method to change the default behaviour of ``log_grad_norm``.
    #     If clipping gradients, the gradients will not have been clipped yet.
    #     Args:
    #         grad_norm_dict: Dictionary containing current grad norm metrics
    #     Example::
    #         # DEFAULT
    #         def log_grad_norm(self, grad_norm_dict):
    #             self.log_dict(grad_norm_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    #     """
    #     self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=True, logger=False)

    def feature_extractor_s3prl(self, wav: Union[Tuple[torch.Tensor],List[torch.Tensor]]) -> torch.Tensor:
        """feature_extractor_s3prl
        Implement for s3prl to get feature
        Args:
            wav ():
        """
        raise NotImplementedError()

    def feature_extractor_zerospeech(self, wav: Union[Tuple[torch.Tensor],List[torch.Tensor]]) -> torch.Tensor:
        """Extract features for zerospeech tasks

        Args:
            wav (list[torch.Tensor]): input list fo waveforms

        Returns:
            features (torch.Tensor)

        Example:
            You might want to start with the subsequent lines of codes

            wav_len = [len(x) for x in wav]
            audio_feat, audio_feat_len = self.forward_audio(wav, wav_len)

        """

        raise NotImplementedError()
        

    def forward_audio(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:
        audio_feat, audio_feat_len = self.audio_encoder(wav, wav_len)

        # if the feat_select_idx == "all", we use weighted sum
        if self.feat_select_idx == "all":
            audio_feat = torch.stack(audio_feat["hidden_states"], dim=0)
            n_layer, bsz, t_len, h_dim = audio_feat.shape
            audio_feat = F.softmax(self.audio_encoder_layer_weights, dim=0).view(
                n_layer, -1
            ) * audio_feat.view(n_layer, -1)
            audio_feat = audio_feat.view(n_layer, bsz, t_len, h_dim)
            audio_feat = audio_feat.sum(dim=0)
            assert audio_feat.shape == (bsz, t_len, h_dim)

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

    def cache_image(self, images: torch.Tensor, idx: List[int]):
        if self.cache.exists_list("image_feat", idx):
            return torch.stack(self.cache.get_batch("image_feat", idx), dim=0).to(
                images.device
            )
        else:
            image_feat = self.forward_image(images)
            for i, id in enumerate(idx):
                self.cache.save("image_feat", id, image_feat[i])
            return image_feat

    def reportRetrieval(self, score_per_audio, score_per_image, AI_answers, IA_answers):
        recall_results_AI, recall_results_IA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_image,
            AB_answers=AI_answers,
            BA_answers=IA_answers,
            recall_at=self.recall_at,
        )

        if isinstance(self.logger,WandbLogger):
            self.log("val_recall_AI", recall_results_AI)
            self.log("val_recall_IA", recall_results_IA)
            self.log("val_recall_mean", recall_results_mean)
        else:
            self.logger.experiment.add_scalars(
                "val_recall_AI", recall_results_AI, self.global_step
            )
            self.logger.experiment.add_scalars(
                "val_recall_IA", recall_results_IA, self.global_step
            )
            self.logger.experiment.add_scalars(
                "val_recall_mean", recall_results_mean, self.global_step
            )
        self.log("val_recall_mean_1", recall_results_mean["recall@1"])

# not updated, currently not used
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
            if not os.path.exists(os.path.join(self.config.trainer.default_root_dir, "retokenizeText")):
                os.makedirs(
                    os.path.join(self.config.trainer.default_root_dir, "retokenizeText"), exist_ok=True
                )
            retokenizeText_output = []

            for x in outputs:
                for _g, _d in zip(x["gold_text"], x["detok_text"]):
                    retokenizeText_output.append({"gold": _g, "detok": _d})

            with open(
                os.path.join(
                    self.config.trainer.default_root_dir,
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

# also served as the base class for all kw base experiments
class KeywordCascadedSpeechClip(CascadedSpeechClip_Base):
    def __init__(self, config: OrderedNamespace) -> None:
        super().__init__(config)

        # set `config.keyword.attention_heads`  default to 1
        if not hasattr(config.keyword, "attention_heads"):
            config.keyword.attention_heads = 1
        
        self.keyword_num = config.keyword.number
        print(
            f"Using {self.keyword_num} keyword, {config.keyword.attention_heads} heads for each keyword"
        )

        # multihead attention module and layer norm
        self.multihead_attn_layer = nn.MultiheadAttention(
            self.embd_dim,
            num_heads=config.keyword.attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attentionBlock_Norm = nn.LayerNorm(self.embd_dim, eps=1e-5)


        self.downsampling_type = None
        self.linear_proj = nn.Linear(self.embd_dim, self.text_embd_dim)

        self.log_detokenize_results = True
        if hasattr(config.log_setting, "log_detokenize_results_every_n_epoch"):
            self.log_detokenize_results_every_n_epoch = (
                config.log_setting.log_detokenize_results_every_n_epoch
            )
        
        logging.info("Start init [CLS]")
        self.cls = torch.nn.Parameter(torch.randn([1, self.keyword_num, self.embd_dim]))
        

    def feature_extractor_s3prl(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]
        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]

        if not include_last_attention:
            return audio_feat, hidden_states[:]
        # Use multi-head attention layer to find keywords(cls)
        bsz = audio_feat.size(0)
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = torch.ones([bsz, audio_feat.size(1) + self.keyword_num])
        for mask, _len in zip(key_padding_mask, audio_len):
            _len += self.keyword_num  # add cls
            mask[:_len] = torch.zeros(mask[:_len].size())

        key_padding_mask = key_padding_mask.bool().to(src.device)

        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        hidden_states = hidden_states + tuple([keywords[:, self.keyword_num :, :]])

        return audio_feat, hidden_states[:]

    def get_keypadding_mask(
        self, bsz: int, length: int, audio_len: torch.Tensor
    ) -> torch.Tensor:
        """Create keypadding mask for attention layers

        Args:
            bsz (int): size of batch
            length (int): the max sequence length of the batch
            audio_len (torch.Tensor): the lens for each data in the batch, shape = (bsz,)

        Returns:
            torch.Tensor: key_padding_mask, bool Tensor, True for padding
        """
        key_padding_mask = torch.ones([bsz, length])
        for mask, len in zip(key_padding_mask, audio_len):
            mask[:len] = 0.0
        key_padding_mask = key_padding_mask.bool().to(self.device)

        return key_padding_mask

    def get_attention_weights(self, wav: Union[Tuple[torch.Tensor],List[torch.Tensor]] ) -> List[torch.Tensor]:
        """Retrieve attention weights

        Args:
            wav (Union[Tuple[torch.Tensor],List[torch.Tensor]]): input list of waveforms

        Returns:
            List[torch.Tensor]: attention maps for each data in batch
        """
        raise NotImplementedError()

    def extract_kw_embeddings(self,wav: Union[Tuple[torch.Tensor],List[torch.Tensor]]):
        """Extrach keyword embeddings in batch

        Args:
            wav (Union[Tuple[torch.Tensor],List[torch.Tensor]]): input list of waveforms

        Returns:
            
        """
        raise NotImplementedError()

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

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        if "image" in self.cache_mods:
            image_feat = self.cache_image(image, id)
        else:
            image_feat = self.forward_image(image)

        # image_feat = self.forward_image(image)

        q_loss = None

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape, "{} {}".format(
                audio_feat.shape, image_feat.shape
            )

            assert audio_feat.shape[0] == id.shape[0]

            # cl_loss = self.criterion(
            #     features=torch.stack([audio_feat, image_feat], dim=1),
            #     labels=id,
            # )

            cl_loss = self.criterion(
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )

            # if q_loss is not None:
            #     loss = (
            #         vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
            #     )
            # else:
            #     loss = vq_result["loss"] * self.beta + cl_loss
            losses = {
                "cl_loss": cl_loss,
                "loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, id, keywords

        return audio_feat, image_feat, res, id

    def training_step(self, batch, batch_idx):
        losses, _, _, _, _ = self.forward(batch, cal_loss=True)

        result = {
            "train_loss": losses["loss"],
            "train_cl_loss": losses["cl_loss"],
            "cl_temp": self.criterion.current_temperature,
        }
        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        losses, audio_feat, image_feat, id, keywords = self.forward(
            batch, cal_loss=True
        )

        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        keywords = keywords.detach().cpu()
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
            "gold_text": batch["text"],
        }

    def validation_epoch_end(self, outputs):
        if not os.path.exists(os.path.join(self.config.trainer.default_root_dir, "retokenizeText")):
            os.makedirs(
                os.path.join(self.config.trainer.default_root_dir, "retokenizeText"), exist_ok=True
            )
        if not os.path.exists(os.path.join(self.config.trainer.default_root_dir, "visualization")):
            os.makedirs(
                os.path.join(self.config.trainer.default_root_dir, "visualization"), exist_ok=True
            )

        if (
            hasattr(self, "log_detokenize_results_every_n_epoch")
            and self.current_epoch % self.log_detokenize_results_every_n_epoch == 0
        ) or not (hasattr(self, "log_detokenize_results_every_n_epoch")):
            gold_texts = []
            for x in outputs:
                gold_texts.extend(x["gold_text"])
            # gold_texts = [ x["gold_text"] for x in outputs]
            # gold_texts = [ x["gold_text"] for x in gold_texts]
            all_keyword_embeddings = torch.cat([x["keywords"] for x in outputs], dim=0)
            all_keyword_embeddings = all_keyword_embeddings.view(
                all_keyword_embeddings.shape[0],
                self.keyword_num,
                all_keyword_embeddings.shape[-1],
            )

            # all_keyword_embeddings shape (total_audio, num_keywords, hid_dim)
            embeddings_stat_dict = {
                "mean": {},
                "std": {},
                "norm": {},
            }
            tokenEmbeddings = self.clip.model.token_embedding.weight.detach().cpu()

            # calculate mean, variance
            # torch.save(all_keyword_embeddings,"all_keyword_embeddings_8kw_1head_pen0.5.pt")

            # torch.norm(all_keyword_embeddings,dim=-1)
            for i in range(self.keyword_num):
                embeddings_stat_dict["mean"][f"kw_{i}"] = torch.mean(
                    torch.mean(all_keyword_embeddings[:, i, :], dim=0)
                )
                embeddings_stat_dict["std"][f"kw_{i}"] = torch.mean(
                    torch.std(all_keyword_embeddings[:, i, :], dim=0)
                )
                embeddings_stat_dict["norm"][f"kw_{i}"] = torch.mean(
                    torch.norm(all_keyword_embeddings[:, i, :], p=2, dim=-1)
                )

            embeddings_stat_dict["mean"]["pretrained"] = torch.mean(
                torch.mean(tokenEmbeddings, dim=0)
            )
            embeddings_stat_dict["std"]["pretrained"] = torch.mean(
                torch.std(tokenEmbeddings, dim=0)
            )
            embeddings_stat_dict["norm"]["pretrained"] = torch.mean(
                torch.norm(tokenEmbeddings, p=2, dim=-1)
            )

            self.log("embs_mean",embeddings_stat_dict["mean"])
            self.log("embs_std",embeddings_stat_dict["std"])
            self.log("embs_norm",embeddings_stat_dict["norm"])

            self.log(
                "kw_mean_mse",
                torch.norm(
                    torch.mean(
                        all_keyword_embeddings.view(-1, self.text_embd_dim), dim=0
                    )
                    - torch.mean(tokenEmbeddings, dim=0),
                    p=2,
                ),
            )
            # self.log("kw_std_mse",torch.std(
            #     torch.norm(
            #         torch.std(all_keyword_embeddings.view(-1,self.text_embd_dim),dim=0) - torch.std(tokenEmbeddings,dim=0),p=2
            #     )
            # ))

            if not hasattr(self.config.log_setting, "log_draw_pca_every_n_epoch"):
                self.config.log_setting.log_draw_pca_every_n_epoch = 0

            if self.config.log_setting.log_draw_pca_every_n_epoch > 0:
                if (
                    self.current_epoch
                    % self.config.log_setting.log_draw_pca_every_n_epoch
                    == 0
                ):
                    draw_embedding_space_PCA(
                        kw_embs=all_keyword_embeddings,
                        gold_embs=tokenEmbeddings,
                        output_path=os.path.join(
                            self.config.trainer.default_root_dir,
                            "visualization/",
                            "pca_ep{}.pdf".format(self.current_epoch),
                        ),
                    )

            assert all_keyword_embeddings.dim() == 3, all_keyword_embeddings.shape
            assert (
                all_keyword_embeddings.shape[2] == self.text_embd_dim
            ), all_keyword_embeddings.shape
            all_retok_outputs = []

            K = self.config.keyword.get("detokenized_K_neighbors",10)

            if not hasattr(self.config.keyword, "retrieve_method"):
                self.config.keyword.retrieve_method = "cosine"

            if self.config.keyword.retrieve_method == "pseudo_inverse":
                emb_pinv = torch.linalg.pinv(tokenEmbeddings.T).float()

            assert self.config.keyword.retrieve_method in ["cosine", "pseudo_inverse"]
            hit_rate = [0] * self.keyword_num
            # emb_pinv.shape (num of codes, dim)
            print("Detokenizing K={}".format((K)))
            for i in tqdm.tqdm(
                range(
                    0,
                    len(gold_texts) + self.config.data.dev_batch_size,
                    self.config.data.dev_batch_size,
                )
            ):
                _gold_texts = gold_texts[i : i + self.config.data.dev_batch_size]
                _bsz = len(_gold_texts)
                if len(_gold_texts) == 0:
                    break

                gold_subword_toks_set = [
                    set(self.clip.tokenizer.encode(_text)) for _text in _gold_texts
                ]


                _k_values, _k_indices = torch.topk(
                    (
                        emb_pinv.float()
                        @ all_keyword_embeddings[i : i + _bsz]
                        .view(-1, self.text_embd_dim)
                        .float()
                        .reshape(-1, self.text_embd_dim)
                        .permute(1, 0)
                    ).permute(1, 0)
                    if self.config.keyword.retrieve_method == "pseudo_inverse"
                    else F.cosine_similarity(
                        all_keyword_embeddings[i : i + _bsz].view(
                            -1, self.text_embd_dim, 1
                        ),
                        tokenEmbeddings.transpose(0, 1).unsqueeze(0),
                        dim=1,
                    ),
                    K,
                )
                assert _k_values.shape == (_bsz * self.keyword_num, K), _k_values.shape
                _k_indices = _k_indices.view(_bsz, self.keyword_num, K)
                _k_values = _k_values.view(_bsz, self.keyword_num, K)

                batch_tmp_outputs = []
                for x in range(_bsz):

                    tmp_outputs = {}
                    for _keyword_i in range(self.keyword_num):
                        tmp_outputs["keyword_{}".format(_keyword_i)] = []

                        # check if nearest K subword appears in gold text
                        top_k_toks = set(
                            [
                                self.clip.reducedl2Original[_ind.item()]
                                if self.clip.selected_text_emb_ids is not None
                                else _ind.item()
                                for _ind in _k_indices[x, _keyword_i]
                            ]
                        )
                        if bool(top_k_toks & gold_subword_toks_set[x]):
                            hit_rate[_keyword_i] += 1

                        for _ind, _dist in zip(
                            _k_indices[x, _keyword_i], _k_values[x, _keyword_i]
                        ):
                            tmp_outputs["keyword_{}".format(_keyword_i)].append(
                                [
                                    self.clip.tokenizer.decoder[
                                        self.clip.reducedl2Original[_ind.item()]
                                        if self.clip.selected_text_emb_ids is not None
                                        else _ind.item()
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

            hit_rate = torch.FloatTensor(hit_rate)
            hit_rate = hit_rate / len(gold_texts) * 100

            print("kw_hit_rate", hit_rate)
            
            self.log(
                "kw_hit_rate",
                {
                    "kw_{}".format(i): hit_rate[i].item()
                    for i in range(self.keyword_num)
                }
            )

            with open(
                os.path.join(
                    self.config.trainer.default_root_dir,
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

    def kw_retrieval(self,all_keyword_embeddings,top_k,retrieve_method,gold_texts=None,all_embeddings=None,all_embeddings_decoder=None):
        # all_embeddings.shape (num,dim)

        decoder = self.clip.tokenizer.decoder
        if all_embeddings is None:
            # all_embeddings = self.clip.model.token_embedding.weight.detach().cpu()
            all_embeddings = self.clip.original_text_emb_weight.cpu()
        else:
            decoder = all_embeddings_decoder

        assert len(decoder) == all_embeddings.size(0)
        
        assert retrieve_method in ["cosine", "pseudo_inverse"]

        if retrieve_method == "pseudo_inverse":
            emb_pinv = torch.linalg.pinv(all_embeddings.T).float()


        all_retok_outputs = []
        for i in tqdm.tqdm(
                range(
                    0,
                    all_keyword_embeddings.size(0) + self.config.data.dev_batch_size,
                    self.config.data.dev_batch_size,
                )
            ):
            _gold_texts = gold_texts[i : i + self.config.data.dev_batch_size]
            _bsz = len(_gold_texts)

            _k_values, _k_indices = torch.topk(
                    (
                        emb_pinv.float()
                        @ all_keyword_embeddings[i : i + _bsz]
                        .view(-1, self.text_embd_dim)
                        .float()
                        .reshape(-1, self.text_embd_dim)
                        .permute(1, 0)
                    ).permute(1, 0)
                    if self.config.keyword.retrieve_method == "pseudo_inverse"
                    else F.cosine_similarity(
                        all_keyword_embeddings[i : i + _bsz].view(
                            -1, self.text_embd_dim, 1
                        ),
                        all_embeddings.transpose(0, 1).unsqueeze(0),
                        dim=1,
                    ),
                    top_k,
                )
            assert _k_values.shape == (_bsz * self.keyword_num, top_k), _k_values.shape
            _k_indices = _k_indices.view(_bsz, self.keyword_num, top_k)
            _k_values = _k_values.view(_bsz, self.keyword_num, top_k)
            for x in range(_bsz):
                tmp_outputs = {}
                for _keyword_i in range(self.keyword_num):
                    tmp_outputs["keyword_{}".format(_keyword_i)] = []

                    # check if nearest K subword appears in gold text
                    # top_k_toks = set(
                    #     [
                    #         self.clip.reducedl2Original[_ind.item()]
                    #         if self.clip.selected_text_emb_ids is not None
                    #         else _ind.item()
                    #         for _ind in _k_indices[x, _keyword_i]
                    #     ]
                    # )
                    # if bool(top_k_toks & gold_subword_toks_set[x]):
                    #     hit_rate[_keyword_i] += 1
                    for _ind, _dist in zip(
                        _k_indices[x, _keyword_i], _k_values[x, _keyword_i]
                    ):
                        tmp_outputs["keyword_{}".format(_keyword_i)].append(
                            [
                                # self.clip.tokenizer.decoder[
                                #     # self.clip.reducedl2Original[_ind.item()]
                                #     # if self.clip.selected_text_emb_ids is not None
                                #     # else 
                                #     _ind.item()
                                # ],
                                decoder[_ind.item()],
                                _dist.item(),
                            ]
                        )

                all_retok_outputs.append(
                    {
                        "gold": gold_texts[i],
                        "neighbors": tmp_outputs,
                    }
                )

        return all_retok_outputs

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.attentionBlock_Norm.parameters())
        )

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


class KeywordCascadedSpeechClip_parallel_baseline(KeywordCascadedSpeechClip):
    def __init__(self, config: OrderedNamespace):
        config.keyword.number = 1
        super().__init__(config)

        assert self.keyword_num == 1, self.keyword_num

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

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        if "image" in self.cache_mods:
            image_feat = self.cache_image(image, id)
        else:
            image_feat = self.forward_image(image)

        # image_feat = self.forward_image(image)

        q_loss = None

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)
        audio_feat = keywords.view(-1, self.text_embd_dim)

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        # audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape, "{} {}".format(
                audio_feat.shape, image_feat.shape
            )

            assert audio_feat.shape[0] == id.shape[0]

            # cl_loss = self.criterion(
            #     features=torch.stack([audio_feat, image_feat], dim=1),
            #     labels=id,
            # )

            cl_loss = self.criterion(
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )

            # if q_loss is not None:
            #     loss = (
            #         vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
            #     )
            # else:
            #     loss = vq_result["loss"] * self.beta + cl_loss
            losses = {
                "cl_loss": cl_loss,
                "loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, id, keywords

        return audio_feat, image_feat, None, id


class KeywordCascadedSpeechClipBN(KeywordCascadedSpeechClip):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        if not hasattr(self.config.keyword, "batchnorms"):
            self.config.keyword.batchnorms.type = "eachKw"
            self.config.keyword.batchnorms.std_scale = 1

        self.bn_layer = Kw_BatchNorm(
            kw_num=self.keyword_num,
            kw_dim=self.text_embd_dim,
            batchnorm_type=self.config.keyword.batchnorms.type,
            init_bias=torch.mean(self.clip.model.token_embedding.weight, dim=0),
            init_scale=torch.std(self.clip.model.token_embedding.weight, dim=0),
            std_scale=self.config.keyword.batchnorms.std_scale,
            learnable=self.config.keyword.batchnorms.learnable
            if hasattr(self.config.keyword.batchnorms, "learnable")
            else True,
            parallel=self.config.keyword.batchnorms.parallel
            if hasattr(self.config.keyword.batchnorms, "parallel")
            else False,
        )

    def feature_extractor_zerospeech(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num).to(
            src.device
        )
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        keywords = self.bn_layer(keywords)

        return keywords

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        # audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

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

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        # image_feat = self.forward_image(image)

        if "image" in self.cache_mods:
            image_feat = self.cache_image(image, id)
        else:
            image_feat = self.forward_image(image)

        q_loss = None

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        keywords = self.bn_layer(keywords)

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape, "{} {}".format(
                audio_feat.shape, image_feat.shape
            )

            assert audio_feat.shape[0] == id.shape[0]

            # cl_loss = self.criterion(
            #     features=torch.stack([audio_feat, image_feat], dim=1),
            #     labels=id,
            # )

            cl_loss = self.criterion(
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )
            # if q_loss is not None:
            #     loss = (
            #         vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
            #     )
            # else:
            #     loss = vq_result["loss"] * self.beta + cl_loss
            losses = {
                "cl_loss": cl_loss,
                "loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, id, keywords

        return audio_feat, image_feat, res, id

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.attentionBlock_Norm.parameters())
        )

        audio_params = audio_params + list(self.criterion.parameters())

        audio_params = audio_params + list(
            [x for x in self.bn_layer.parameters() if x.requires_grad]
        )

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


class KeywordCascadedSpeechClip_ProjVQ(KeywordCascadedSpeechClipBN):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        # remove batch norm layer
        if not hasattr(self.config.vq, "bn_before_vq"):
            self.config.vq.bn_before_vq = False
        if not self.config.vq.bn_before_vq:
            self.bn_layer = None

        # override self.linear_proj
        self.linear_proj = nn.Sequential(
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),
            # torch.nn.GELU(),
            torch.nn.Linear(
                self.audio_encoder.out_dim,
                self.clip.model.token_embedding.num_embeddings,
            )
        )

        # codebook selection
        self.vector_quantizer = None
        self.vq_type = config.vq.type

        if not hasattr(vector_quantizers, config.vq.type):
            raise NotImplementedError("Vq ({}) not implemented".format(config.vq.type))

        self.vector_quantizer = getattr(vector_quantizers, self.vq_type)(
            **config.vq.args
        )

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

        if "image" in self.cache_mods:
            image_feat = self.cache_image(image, id)
        else:
            image_feat = self.forward_image(image)

        # image_feat = self.forward_image(image)

        q_loss = None

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        # VQ
        vq_results = self.vector_quantizer(x=keywords)
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
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )
            losses = {
                "cl_loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})
            return losses, audio_feat, image_feat, id, vq_results, keywords

        return audio_feat, image_feat, res, id

    def training_step(self, batch, batch_idx):
        losses, _, _, _, vq_results, _ = self.forward(batch, cal_loss=True)

        result = {
            "train_loss": losses["cl_loss"],
            "cl_temp": self.criterion.current_temperature,
            "softmax_temp": vq_results["temp"],
            "train_prob_ppl": vq_results["prob_perplexity"].item(),
            "train_code_ppl": vq_results["code_perplexity"].item(),
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
        losses, audio_feat, image_feat, id, vq_results, keywords = self.forward(
            batch, cal_loss=True
        )

        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        keywords = keywords.detach().cpu()
        id = id.detach().cpu()

        result = {
            "val_loss": losses["cl_loss"].item(),
            "val_temp": vq_results["temp"],
            "val_prob_ppl": vq_results["prob_perplexity"].item(),
            "val_code_ppl": vq_results["code_perplexity"].item(),
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

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.attentionBlock_Norm.parameters())
        )

        audio_params = audio_params + list(self.criterion.parameters())

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


class KeywordCascadedSpeechClip_ProjVQ_Cosine(KeywordCascadedSpeechClip_ProjVQ):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.linear_proj = nn.Sequential(
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),
            # torch.nn.GELU(),
            torch.nn.Linear(
                self.audio_encoder.out_dim,
                self.clip.model.token_embedding.embedding_dim,
            )
        )

        # codebook selection
        self.vector_quantizer = None
        self.vq_type = config.vq.type

        if not hasattr(vector_quantizers, config.vq.type):
            raise NotImplementedError("Vq ({}) not implemented".format(config.vq.type))

        self.vector_quantizer = getattr(vector_quantizers, self.vq_type)(
            **config.vq.args
        )
        
    def get_attention_weights(self, wav: Union[Tuple[torch.Tensor],List[torch.Tensor]] ) -> List[torch.Tensor]:
        """Retrieve attention weights

        Args:
            wav (Union[Tuple[torch.Tensor],List[torch.Tensor]]): input list of waveforms

        Returns:
            List[torch.Tensor]: attention maps for each data in batch
        """
        wav_len = [len(x) for x in wav]
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        # audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        # hidden_states = audio_feat["hidden_states"]
        # audio_feat = audio_feat["last_hidden_state"]

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num).to(
            src.device
        )

        key_padding_mask = key_padding_mask.bool()

        _, attn_output_weights = self.multihead_attn_layer(
            src, src, src, key_padding_mask=key_padding_mask, average_attn_weights=False
        )
        # (bsz,num_head,target_L,source_L)

        # get [CLS] token attention weights
        # cls_weights = attn_output_weights[:, 0, 0,:]
        cls_weights = []
        for i in range(attn_output_weights.shape[0]):
            cls_weights.append(
                attn_output_weights[
                    i, :, : self.keyword_num, : audio_len[i] + self.keyword_num
                ]
            )

        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        print(keywords.shape)

        if self.config.vq.bn_before_vq:
            keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False

        top1_kw = [ ["" for _ in range(self.keyword_num)] for _ in range(bsz)]
        top1_kw_id = torch.argmax(vq_results["subword_prob"],dim=-1)
        for bsz_i in range(bsz):
            for kw_i in range(self.keyword_num):
                top1_kw[bsz_i][kw_i] = self.clip.tokenizer.decoder[
                    self.clip.reducedl2Original[top1_kw_id[bsz_i,kw_i].item()]
                ]        

        # keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight
        

        return cls_weights, top1_kw

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

    def feature_extractor_zerospeech(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        # audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        # hidden_states = audio_feat["hidden_states"]
        # audio_feat = audio_feat["last_hidden_state"]

        # return hidden_states[-2]

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num).to(
            src.device
        )
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        # return keywords[:, self.keyword_num:]
        # return keywords[:, self.keyword_num :]
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        # return keywords[]
        keywords = self.linear_proj(keywords)

        keywords = self.bn_layer(keywords)
        return keywords
        # return keywords[:,2].view(bsz,1,-1).repeat(1,2,1)
        

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        # audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)
    def extract_kw_embeddings(self,wav):
        print("here")
        wav_len = [len(x) for x in wav]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        if self.config.vq.bn_before_vq:
            keywords = self.bn_layer(keywords)

        return None, keywords.contiguous()

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        return audio_feat, keywords
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

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        # image_feat = self.forward_image(image)
        if "image" in self.cache_mods:
            image_feat = self.cache_image(image, id)
        else:
            image_feat = self.forward_image(image)

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        if self.config.vq.bn_before_vq:
            keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
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
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )
            losses = {
                "cl_loss": cl_loss,
            }
            
            # losses = {
            #     # "cl_loss": cl_loss,
            #     "feat_A" : audio_feat,
            #     # "feat_B" : image_feat,
            #     # "index" : id,
            # }


            return losses, audio_feat, image_feat, id, vq_results, keywords

        return audio_feat, image_feat, res, id, keywords

    def training_step(self, batch, batch_idx):
        losses, _, _, _, vq_results, _ = self.forward(batch, cal_loss=True)

        if "ent_per_t" in vq_results:
            ent_per_t_dict = { "kw_{}".format(i): vq_results["ent_per_t"][i].item() for i in range(self.keyword_num) }
            self.log(
                "train_ent_per_kw",
                ent_per_t_dict
            )

        result = {
            "train_loss": losses["cl_loss"],
            "cl_temp": self.criterion.current_temperature,
            "softmax_temp": vq_results["temp"],
            "train_prob_ppl": vq_results["prob_perplexity"].item(),
            "train_code_ppl": vq_results["code_perplexity"].item(),
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

    # def training_step(self, batch, batch_idx):
    #     losses, _, _, _, vq_results, _ = self.forward(batch, cal_loss=True)
    #     print("training_step",losses["feat_A"].shape)
    #     return losses
    
    # def training_step_end(self, step_output):
    #     print("training_step_end",step_output["feat_A"].shape)
    # #     cl_loss = self.criterion(
    # #             feat_A=audio_feat,
    # #             feat_B=image_feat,
    # #             index=id,
    # #     )
    #     exit(1)

    def validation_step(self, batch, batch_idx):
        losses, audio_feat, image_feat, id, vq_results, keywords = self.forward(
            batch, cal_loss=True
        )

        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        keywords = keywords.detach().cpu()
        id = id.detach().cpu()

        if "ent_per_t" in vq_results:
            ent_per_t_dict = { "kw_{}".format(i): vq_results["ent_per_t"][i].item() for i in range(self.keyword_num) }
            self.log(
                "val_ent_per_kw",
                ent_per_t_dict
            )

        result = {
            "val_loss": losses["cl_loss"].item(),
            "val_temp": vq_results["temp"],
            "val_prob_ppl": vq_results["prob_perplexity"].item(),
            "val_code_ppl": vq_results["code_perplexity"].item(),
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


    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.attentionBlock_Norm.parameters())
        )

        audio_params = audio_params + list(self.criterion.parameters())

        audio_params = audio_params + list(self.vector_quantizer.parameters())

        if self.config.vq.bn_before_vq:
            logging.warning("Using BatchNorm before Cosine Similarity code selection")
            audio_params = audio_params + list(
                [x for x in self.bn_layer.parameters() if x.requires_grad]
            )

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


class KeywordCascadedSpeechClip_ProjVQ_Cosine_AttMap_Constraint(KeywordCascadedSpeechClip_ProjVQ_Cosine):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        assert config.keyword.attention_heads == 1, config.keyword.attention_heads
        
    def get_attention_weights(self, wav: Union[Tuple[torch.Tensor],List[torch.Tensor]] ) -> List[torch.Tensor]:
        """Retrieve attention weights

        Args:
            wav (Union[Tuple[torch.Tensor],List[torch.Tensor]]): input list of waveforms

        Returns:
            List[torch.Tensor]: attention maps for each data in batch
        """
        wav_len = [len(x) for x in wav]
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        # audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        # hidden_states = audio_feat["hidden_states"]
        # audio_feat = audio_feat["last_hidden_state"]

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num).to(
            src.device
        )

        key_padding_mask = key_padding_mask.bool()

        _, attn_output_weights = self.multihead_attn_layer(
            src, src, src, key_padding_mask=key_padding_mask, average_attn_weights=False
        )
        # (bsz,num_head,target_L,source_L)

        # get [CLS] token attention weights
        # cls_weights = attn_output_weights[:, 0, 0,:]
        cls_weights = []
        for i in range(attn_output_weights.shape[0]):
            cls_weights.append(
                attn_output_weights[
                    i, :, : self.keyword_num, : audio_len[i] + self.keyword_num
                ]
            )

        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        print(keywords.shape)

        if self.config.vq.bn_before_vq:
            keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False

        top1_kw = [ ["" for _ in range(self.keyword_num)] for _ in range(bsz)]
        top1_kw_id = torch.argmax(vq_results["subword_prob"],dim=-1)
        for bsz_i in range(bsz):
            for kw_i in range(self.keyword_num):
                top1_kw[bsz_i][kw_i] = self.clip.tokenizer.decoder[
                    self.clip.reducedl2Original[top1_kw_id[bsz_i,kw_i].item()]
                ]        

        # keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight
        

        return cls_weights, top1_kw

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

    def feature_extractor_zerospeech(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        # audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        # hidden_states = audio_feat["hidden_states"]
        # audio_feat = audio_feat["last_hidden_state"]

        # return hidden_states[-2]

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num).to(
            src.device
        )
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        # return keywords[:, self.keyword_num:]
        # return keywords[:, self.keyword_num :]
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        # return keywords[]
        keywords = self.linear_proj(keywords)

        keywords = self.bn_layer(keywords)
        return keywords
        # return keywords[:,2].view(bsz,1,-1).repeat(1,2,1)
        

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        # audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)
    def extract_kw_embeddings(self,wav):
        print("here")
        wav_len = [len(x) for x in wav]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        if self.config.vq.bn_before_vq:
            keywords = self.bn_layer(keywords)

        return None, keywords.contiguous()

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        return audio_feat, keywords
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

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        # image_feat = self.forward_image(image)
        if "image" in self.cache_mods:
            image_feat = self.cache_image(image, id)
        else:
            image_feat = self.forward_image(image)

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        
        keywords, att_map = self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask,average_attn_weights=False)

        # att_map bsz, num_heads, query_len, kv_len
        # config.keyword.attention_heads = 1

        att_map = att_map[:,0]
        # att_map bsz, query_len, kv_len


        # diversity_per_kw_loss
        probs_per_kw =  att_map[:,:self.keyword_num,self.keyword_num:]
        probs_per_kw = probs_per_kw / probs_per_kw.sum(-1,keepdim=True)
        # probs_per_kw shape bsz, self.keyword_num, frame_len

        # mask out padding
        probs_per_kw = probs_per_kw.permute(1,0,2)
        # probs_per_kw shape self.keyword_num, bsz, frame_len
        msk = self.get_keypadding_mask(bsz=bsz,length=probs_per_kw.shape[-1],audio_len = audio_len)
        # msk shape bsz, frame_len
        probs_per_kw[:,msk] = 0.0
        probs_per_kw = probs_per_kw.permute(1,0,2)


        ent_probs_per_kw = torch.mean( -probs_per_kw * torch.log(probs_per_kw+1e-12),dim = -1 )
        # ent_probs_per_kw shape : bsz, self.keyword_num
        ent_probs_per_kw = torch.mean(ent_probs_per_kw)
        # ent_probs_per_kw shape : self.keyword_num


        # diversity_per_frame_loss
        probs_per_frame =  att_map[:,:self.keyword_num,self.keyword_num:]
        probs_per_frame = probs_per_frame.permute(0,2,1)
        # probs_per_frame shape bsz, frame_len, self.keyword_num

        # normalize
        probs_per_frame = probs_per_frame / probs_per_frame.sum(-1,keepdim=True)

        # mask out padding
        probs_per_frame[msk,:]= 0.0

        ent_probs_per_frame = torch.mean( -probs_per_frame * torch.log(probs_per_frame+1e-12),dim = -1 )
        # ent_probs_per_kw shape : bsz, frame_len
        ent_probs_per_frame = torch.mean(ent_probs_per_frame)

        # smoothness_per_frame_loss_weight
        smoothness_per_frame =  F.mse_loss(probs_per_frame[:,:-1], probs_per_frame[:,1:])      


        keywords = self.attentionBlock_Norm(
            keywords + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        if self.config.vq.bn_before_vq:
            keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
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
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )
            

            
            diversity_per_kw_loss_weight = self.config.keyword.attention_constraints.diversity_per_kw_loss_weight
            diversity_per_frame_loss_weight = self.config.keyword.attention_constraints.diversity_per_frame_loss_weight
            smoothness_per_frame_loss_weight = self.config.keyword.attention_constraints.smoothness_per_frame_loss_weight

            total_loss = cl_loss \
                + diversity_per_kw_loss_weight * ent_probs_per_kw \
                + diversity_per_frame_loss_weight * ent_probs_per_frame \
                + smoothness_per_frame_loss_weight * smoothness_per_frame
            print(total_loss)
            exit(1)
            losses = {
                "cl_loss": cl_loss,
                "diversity_per_kw_loss": ent_probs_per_kw   ,
                "diversity_per_frame_loss": ent_probs_per_frame  ,
                "smoothness_per_frame_loss": smoothness_per_frame ,
                "total_loss":total_loss,

            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})
            return losses, audio_feat, image_feat, id, vq_results, keywords

        return audio_feat, image_feat, res, id, keywords

    def training_step(self, batch, batch_idx):
        losses, _, _, _, vq_results, _ = self.forward(batch, cal_loss=True)

        if "ent_per_t" in vq_results:
            ent_per_t_dict = { "kw_{}".format(i): vq_results["ent_per_t"][i].item() for i in range(self.keyword_num) }
            self.log(
                "train_ent_per_kw",
                ent_per_t_dict
            )

        result = {
            "train_loss": losses["total_loss"],
            "train_cl_loss" : losses["cl_loss"],
            "cl_temp": self.criterion.current_temperature,
            "softmax_temp": vq_results["temp"],
            "train_prob_ppl": vq_results["prob_perplexity"].item(),
            "train_code_ppl": vq_results["code_perplexity"].item(),
            "train_diversity_per_kw_loss" : losses["diversity_per_kw_loss"].item(),
            "train_diversity_per_frame_loss" : losses["diversity_per_frame_loss"].item(),
            "train_smoothness_per_frame_loss" : losses["smoothness_per_frame_loss"].item(),

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

        return {"loss": losses["total_loss"]}

    def validation_step(self, batch, batch_idx):
        losses, audio_feat, image_feat, id, vq_results, keywords = self.forward(
            batch, cal_loss=True
        )

        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        keywords = keywords.detach().cpu()
        id = id.detach().cpu()

        if "ent_per_t" in vq_results:
            ent_per_t_dict = { "kw_{}".format(i): vq_results["ent_per_t"][i].item() for i in range(self.keyword_num) }
            self.log(
                "val_ent_per_kw",
                ent_per_t_dict
            )

        result = {
            "val_loss": losses["total_loss"].item(),
            "val_cl_loss" : losses["cl_loss"],
            "val_temp": vq_results["temp"],
            "val_prob_ppl": vq_results["prob_perplexity"].item(),
            "val_code_ppl": vq_results["code_perplexity"].item(),
            "val_diversity_per_kw_loss" : losses["diversity_per_kw_loss"].item(),
            "val_diversity_per_frame_loss" : losses["diversity_per_frame_loss"].item(),
            "val_smoothness_per_frame_loss" : losses["smoothness_per_frame_loss"].item(),
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


    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.attentionBlock_Norm.parameters())
        )

        audio_params = audio_params + list(self.criterion.parameters())

        audio_params = audio_params + list(self.vector_quantizer.parameters())

        if self.config.vq.bn_before_vq:
            logging.warning("Using BatchNorm before Cosine Similarity code selection")
            audio_params = audio_params + list(
                [x for x in self.bn_layer.parameters() if x.requires_grad]
            )

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


class KeywordCascadedSpeechClip_ProjVQ_Cosine_w_Parallel(KeywordCascadedSpeechClip_ProjVQ):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        # parallel predict head
        self.multihead_attn_layer_for_parallel = nn.MultiheadAttention(
            self.embd_dim,
            **config.parallel_objective.attention_args,
        )
        self.attentionBlock_Norm_for_parallel = nn.LayerNorm(self.embd_dim, eps=1e-5)
        self.parallel_linear_proj = nn.Linear(self.embd_dim, self.clip.out_dim)
        self.global_cls = nn.parameter.Parameter(torch.FloatTensor(torch.randn(1,1,self.embd_dim)))


        self.linear_proj = nn.Sequential(
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),
            # torch.nn.GELU(),
            # torch.nn.Linear(self.audio_encoder.out_dim,self.audio_encoder.out_dim),
            # torch.nn.GELU(),
            torch.nn.Linear(
                self.audio_encoder.out_dim,
                self.clip.model.token_embedding.embedding_dim,
            )
        )

        # codebook selection
        self.vector_quantizer = None
        self.vq_type = config.vq.type

        if not hasattr(vector_quantizers, config.vq.type):
            raise NotImplementedError("Vq ({}) not implemented".format(config.vq.type))

        self.vector_quantizer = getattr(vector_quantizers, self.vq_type)(
            **config.vq.args
        )

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

    def feature_extractor_zerospeech(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]

        # update device information to clip model
        self.clip.update_device(self.device)

        # audio_feat, audio_len = self.forward_audio(wav, wav_len)

        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]

        return hidden_states[-2]

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num).to(
            src.device
        )
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        return keywords[:, self.keyword_num :]
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        # return keywords[:,6].view(bsz,1,-1).repeat(1,2,1)
        keywords = self.linear_proj(keywords)

        keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        # audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

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

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        # image_feat = self.forward_image(image)
        if "image" in self.cache_mods:
            image_feat = self.cache_image(image, id)
        else:
            image_feat = self.forward_image(image)


        bsz = audio_feat.size(0)
        # parallel objective representation extraction
        src_parallel = torch.cat( ( self.global_cls.repeat(bsz,1,1), audio_feat), dim=1)
        assert src_parallel.shape == (bsz, 1 + audio_feat.size(1),self.embd_dim), (src_parallel.shape,(bsz, 1 + audio_feat.size(1),self.embd_dim))

        key_padding_mask = self.get_keypadding_mask(bsz, audio_feat.size(1)+1, audio_len+1)
        assert key_padding_mask.shape == (bsz, audio_feat.size(1)+1), (key_padding_mask.shape,(bsz, audio_feat.size(1)+1) )

        audio_parallel_feats =  self.attentionBlock_Norm_for_parallel(
            self.multihead_attn_layer_for_parallel(
                src_parallel,src_parallel,src_parallel,key_padding_mask=key_padding_mask
            )[0] + src_parallel
        )[:,0].view(bsz,self.embd_dim)

        audio_parallel_feats = self.parallel_linear_proj(audio_parallel_feats)
        assert audio_parallel_feats.shape == (bsz, self.clip.out_dim)



        # Use multi-head attention layer to find keywords(cls)
        # total_len = audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, audio_feat.size(1) + self.keyword_num, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        if self.config.vq.bn_before_vq:
            keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_embd_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
            # .view(bsz,1,self.clip.model.token_embedding.num_embeddings)

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # cos_score = cos_score.view(
        #     bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings
        # )

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            audio_parallel_feats = audio_parallel_feats / audio_parallel_feats.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape

            assert audio_feat.shape[0] == id.shape[0]

            cl_loss = self.criterion(
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )

            cl_loss_parallel = self.criterion(
                feat_A=audio_parallel_feats,
                feat_B=image_feat,
                index=id,
            )


            losses = {
                "loss" : cl_loss + self.config.parallel_objective.loss_weight * cl_loss_parallel,
                "cl_loss": cl_loss,
                "cl_loss_parallel" : cl_loss_parallel,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})
            return losses, audio_feat, image_feat, id, vq_results, keywords

        return audio_feat, image_feat, res, id

    def training_step(self, batch, batch_idx):
        losses, _, _, _, vq_results, _ = self.forward(batch, cal_loss=True)

        result = {
            "train_loss": losses["loss"] ,
            "train_cl_loss_cascaded" : losses["cl_loss"],
            "train_cl_loss_parallel" : losses["cl_loss_parallel"],
            "cl_temp": self.criterion.current_temperature,
            "softmax_temp": vq_results["temp"],
            "train_prob_ppl": vq_results["prob_perplexity"].item(),
            "train_code_ppl": vq_results["code_perplexity"].item(),
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

        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        losses, audio_feat, image_feat, id, vq_results, keywords = self.forward(
            batch, cal_loss=True
        )

        audio_feat = audio_feat.detach().cpu()
        image_feat = image_feat.detach().cpu()
        keywords = keywords.detach().cpu()
        id = id.detach().cpu()

        result = {
            "val_loss": losses["loss"].item(),
            "val_cl_loss_cascaded" : losses["cl_loss"].item(),
            "val_cl_loss_parallel" : losses["cl_loss_parallel"].item(),
            "val_temp": vq_results["temp"],
            "val_prob_ppl": vq_results["prob_perplexity"].item(),
            "val_code_ppl": vq_results["code_perplexity"].item(),
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

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.attentionBlock_Norm.parameters())
        )

        audio_params = audio_params + list(self.criterion.parameters())

        audio_params = audio_params + list(self.vector_quantizer.parameters())

        if self.config.vq.bn_before_vq:
            logging.warning("Using BatchNorm before Cosine Similarity code selection")
            audio_params = audio_params + list(
                [x for x in self.bn_layer.parameters() if x.requires_grad]
            )

        # parallel objective

        audio_params = audio_params +  list(self.multihead_attn_layer_for_parallel.parameters())
        audio_params = audio_params +  list(self.attentionBlock_Norm_for_parallel.parameters())
        audio_params = audio_params +  list(self.parallel_linear_proj.parameters())
        audio_params = audio_params +  [self.global_cls]



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

# not updated
class KeywordCascadedSpeechClip_CodeBookPenalty(KeywordCascadedSpeechClipBN):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        # remove batch norm layer
        if not hasattr(self.config.vq, "bn_before_vq"):
            self.config.vq.bn_before_vq = False
        if not self.config.vq.bn_before_vq:
            self.bn_layer = None

        self.codebook_penalty_type = config.codebook_penalty.type
        logging.warning("Using Penalty Scheduler")
        self.penalty_scheduler = PenaltyScheduler(
            weights=config.codebook_penalty.loss_weight,
            keypoints=config.codebook_penalty.keypoints,
        )

        assert self.codebook_penalty_type in [
            "cosine",
            "l2",
        ], self.codebook_penalty_type

        if not hasattr(self.config.codebook_penalty, "k_neighbors"):
            self.config.codebook_penalty.k_neighbors = 1

        logging.warning(
            f"[Penalty Scheduler] {self.config.codebook_penalty.k_neighbors} neighbors"
        )

    def feature_extractor_s3prl(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]
        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]

        if not include_last_attention:
            return audio_feat, hidden_states[:]
        # Use multi-head attention layer to find keywords(cls)
        bsz = audio_feat.size(0)
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = torch.ones([bsz, audio_feat.size(1) + self.keyword_num])
        for mask, _len in zip(key_padding_mask, audio_len):
            _len += self.keyword_num  # add cls
            mask[:_len] = torch.zeros(mask[:_len].size())

        key_padding_mask = key_padding_mask.bool().to(src.device)

        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        hidden_states = hidden_states + tuple([keywords[:, self.keyword_num :, :]])

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

        self.penalty_scheduler.update(self.global_step)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        image_feat = self.forward_image(image)

        q_loss = None

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        # keywords bsz, keyword_num, embd_dim

        assert self.clip.model.token_embedding.weight.requires_grad == False
        # add penalty
        if self.codebook_penalty_type == "cosine":
            # cosine
            self.config.codebook_penalty.k_neighbors
            cos_score = F.cosine_similarity(
                keywords.view(-1, self.text_embd_dim, 1),
                self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                dim=1,
            )
            cos_score = torch.topk(
                cos_score, k=self.config.codebook_penalty.k_neighbors, dim=-1
            ).values
            codebook_penalty_loss = (1 - cos_score).mean()
            # print(codebook_penalty_loss)
            # exit(1)
        elif self.codebook_penalty_type == "l2":
            pass
            # l2_dists = F.mse_loss(
            #     keywords.view(bsz,self.text_embd_dim,1),
            #     self.clip.model.token_embedding.weight.transpose(0,1).unsqueeze(0),
            #     dim=1
            # ).max(-1)[0]
            # cosine_loss = 1 - cos_score

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape, "{} {}".format(
                audio_feat.shape, image_feat.shape
            )

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
            loss = cl_loss + self.penalty_scheduler.get_value() * codebook_penalty_loss
            losses = {
                "cbk_pen_loss": codebook_penalty_loss,
                "cl_loss": cl_loss,
                "loss": loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, id, keywords

        return audio_feat, image_feat, res, id

    def training_step(self, batch, batch_idx):
        losses, _, _, _, _ = self.forward(batch, cal_loss=True)
        result = {
            "train_loss": losses["loss"],
            "train_cl_loss": losses["cl_loss"],
            "train_cbk_pen_loss": losses["cbk_pen_loss"],
            "cl_temp": self.criterion.get_temp(),
            "penalty_weight": self.penalty_scheduler.get_value(),
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

        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        # self.feature_extractor_s3prl(batch["wav"])
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
            "val_loss": losses["loss"].item(),
            "val_cl_loss": losses["cl_loss"],
            "val_cbk_pen_loss": losses["cbk_pen_loss"],
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

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())
        else:
            audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.multihead_attn_layer.parameters())
        )

        audio_params = audio_params + list(self.attentionBlock_Norm.parameters())

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

# not updated
class KeywordCascadedSpeechClip_CodeBookPenaltyBN(KeywordCascadedSpeechClipBN):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.codebook_penalty_type = config.codebook_penalty.type
        logging.warning("Using Penalty Scheduler")
        self.penalty_scheduler = PenaltyScheduler(
            weights=config.codebook_penalty.loss_weight,
            keypoints=config.codebook_penalty.keypoints,
        )

        assert self.codebook_penalty_type in [
            "cosine",
            "l2",
        ], self.codebook_penalty_type

        if not hasattr(self.config.codebook_penalty, "k_neighbors"):
            self.config.codebook_penalty.k_neighbors = 1

        logging.warning(
            f"[Penalty Scheduler] {self.config.codebook_penalty.k_neighbors} neighbors"
        )

    def feature_extractor_s3prl(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]
        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]

        if not include_last_attention:
            return audio_feat, hidden_states[:]
        # Use multi-head attention layer to find keywords(cls)
        bsz = audio_feat.size(0)
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = torch.ones([bsz, audio_feat.size(1) + self.keyword_num])
        for mask, _len in zip(key_padding_mask, audio_len):
            _len += self.keyword_num  # add cls
            mask[:_len] = torch.zeros(mask[:_len].size())

        key_padding_mask = key_padding_mask.bool().to(src.device)

        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        hidden_states = hidden_states + tuple([keywords[:, self.keyword_num :, :]])

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

        self.penalty_scheduler.update(self.global_step)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        image_feat = self.forward_image(image)

        q_loss = None

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        keywords = self.bn_layer(
            # (B,T,D) -> (B,D,T)
            keywords.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # keywords bsz, keyword_num, embd_dim

        assert self.clip.model.token_embedding.weight.requires_grad == False
        # add penalty
        if self.codebook_penalty_type == "cosine":
            # cosine
            self.config.codebook_penalty.k_neighbors
            cos_score = F.cosine_similarity(
                keywords.view(-1, self.text_embd_dim, 1),
                self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                dim=1,
            )
            cos_score = torch.topk(
                cos_score, k=self.config.codebook_penalty.k_neighbors, dim=-1
            ).values
            codebook_penalty_loss = (1 - cos_score).mean()
            # print(codebook_penalty_loss)
            # exit(1)
        elif self.codebook_penalty_type == "l2":
            pass
            # l2_dists = F.mse_loss(
            #     keywords.view(bsz,self.text_embd_dim,1),
            #     self.clip.model.token_embedding.weight.transpose(0,1).unsqueeze(0),
            #     dim=1
            # ).max(-1)[0]
            # cosine_loss = 1 - cos_score

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape, "{} {}".format(
                audio_feat.shape, image_feat.shape
            )

            assert audio_feat.shape[0] == id.shape[0]

            cl_loss = self.criterion(
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )
            # if q_loss is not None:
            #     loss = (
            #         vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
            #     )
            # else:
            #     loss = vq_result["loss"] * self.beta + cl_loss
            loss = cl_loss + self.penalty_scheduler.get_value() * codebook_penalty_loss
            losses = {
                "cbk_pen_loss": codebook_penalty_loss,
                "cl_loss": cl_loss,
                "loss": loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, id, keywords

        return audio_feat, image_feat, res, id

    def training_step(self, batch, batch_idx):
        losses, _, _, _, _ = self.forward(batch, cal_loss=True)
        result = {
            "train_loss": losses["loss"],
            "train_cl_loss": losses["cl_loss"],
            "train_cbk_pen_loss": losses["cbk_pen_loss"],
            "cl_temp": self.criterion.current_temperature,
            "penalty_weight": self.penalty_scheduler.get_value(),
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

        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        # self.feature_extractor_s3prl(batch["wav"])
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
            "val_loss": losses["loss"].item(),
            "val_cl_loss": losses["cl_loss"],
            "val_cbk_pen_loss": losses["cbk_pen_loss"],
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

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())
        else:
            audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.multihead_attn_layer.parameters())
        )

        audio_params = audio_params + list(self.attentionBlock_Norm.parameters())

        audio_params = audio_params + list(self.criterion.parameters())

        audio_params = audio_params + list(self.bn_layer.parameters())

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

# not updated , used for experiments
class KeywordCascadedSpeechClipBNEachKw(KeywordCascadedSpeechClip):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(self.text_embd_dim) for _ in range(self.keyword_num)]
        )

        # init
        if hasattr(self.config.keyword, "batchnorms"):
            _std_scale = self.config.keyword.batchnorms.std_scale
        else:
            _std_scale = 1
        logging.warning(
            "Initialize BatchNorm weight and bias with token embeddings w/ scale={}".format(
                _std_scale
            )
        )

        for _bn_layer in self.bn_layers:
            _bn_layer.weight.data.copy_(
                torch.std(self.clip.model.token_embedding.weight, dim=0) * _std_scale
            )
            _bn_layer.bias.data.copy_(
                torch.mean(self.clip.model.token_embedding.weight, dim=0)
            )

    def feature_extractor_zerospeech(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num).to(
            src.device
        )
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        return keywords[:, self.keyword_num :, :]
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)
        keywords_bns = []
        for i in range(self.keyword_num):
            keywords_bns.append(
                self.bn_layers[i](
                    # (B,#kw,D)
                    keywords[:, i]
                )
            )

        keywords = torch.stack(keywords_bns, dim=1)
        del keywords_bns

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

        audio_feat, audio_len = self.forward_audio(wav, wav_len)
        image_feat = self.forward_image(image)

        q_loss = None

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)
        keywords_bns = []
        for i in range(self.keyword_num):
            keywords_bns.append(
                self.bn_layers[i](
                    # (B,#kw,D)
                    keywords[:, i]
                )
            )

        keywords = torch.stack(keywords_bns, dim=1)
        del keywords_bns

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape, "{} {}".format(
                audio_feat.shape, image_feat.shape
            )

            assert audio_feat.shape[0] == id.shape[0]

            # cl_loss = self.criterion(
            #     features=torch.stack([audio_feat, image_feat], dim=1),
            #     labels=id,
            # )

            cl_loss = self.criterion(
                feat_A=audio_feat,
                feat_B=image_feat,
                index=id,
            )
            # if q_loss is not None:
            #     loss = (
            #         vq_result["loss"] * self.beta + cl_loss + self.cif_lamda_c * q_loss
            #     )
            # else:
            #     loss = vq_result["loss"] * self.beta + cl_loss
            losses = {
                "cl_loss": cl_loss,
                "loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, id, keywords

        return audio_feat, image_feat, res, id

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        audio_params = []

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.multihead_attn_layer.parameters())
        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = (
            audio_params + [self.cls] + list(self.attentionBlock_Norm.parameters())
        )

        audio_params = audio_params + list(self.criterion.parameters())

        audio_params = audio_params + list(self.bn_layers.parameters())

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

# not updated
class KeywordCascadedSpeechClipNLayer(CascadedSpeechClip_Base):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        if not hasattr(config.keyword, "attention_heads"):
            config.keyword.attention_heads = 8

        num_encoder_layers = config.keyword.n_encoder_layers
        print(f"Using {num_encoder_layers} layer transformer decoder")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embd_dim,
            nhead=config.keyword.attention_heads,
            dim_feedforward=self.embd_dim * 4,
            dropout=0.1,
            activation="gelu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
        )
        encoder_norm = nn.LayerNorm(self.embd_dim, eps=1e-5)
        self.transformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.keyword_num = config.keyword.number
        print(
            f"Using {self.keyword_num} keyword, {config.keyword.attention_heads} heads for each keyword"
        )

        self.downsampling_type = None
        self.linear_proj = nn.Linear(self.embd_dim, self.text_embd_dim)

        self.log_detokenize_results = True
        if hasattr(config.log_setting, "log_detokenize_results_every_n_epoch"):
            self.log_detokenize_results_every_n_epoch = (
                config.log_setting.log_detokenize_results_every_n_epoch
            )

        logging.info("Start init [CLS]")
        self.cls = torch.nn.Parameter(torch.randn([1, self.keyword_num, self.embd_dim]))

    def feature_extractor_s3prl(self, wav, include_last_attention=True):
        wav_len = [len(x) for x in wav]
        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]

        if not include_last_attention:
            return audio_feat, hidden_states[:]
        # Use multi-head attention layer to find keywords(cls)
        bsz = audio_feat.size(0)
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = torch.ones([bsz, audio_feat.size(1) + self.keyword_num])
        for mask, _len in zip(key_padding_mask, audio_len):
            _len += self.keyword_num  # add cls
            mask[:_len] = torch.zeros(mask[:_len].size())

        key_padding_mask = key_padding_mask.bool().to(src.device)

        keywords = self.attentionBlock_Norm(
            self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[
                0
            ]
            + src
        )
        hidden_states = hidden_states + tuple([keywords[:, self.keyword_num :, :]])

        return audio_feat, hidden_states[:]

    def get_attention_weights(self, wav):
        wav_len = [len(x) for x in wav]
        audio_feat, audio_len = self.audio_encoder(wav, wav_len, feat_select_idx="all")
        hidden_states = audio_feat["hidden_states"]
        audio_feat = audio_feat["last_hidden_state"]

        # Use multi-head attention layer to find keywords(cls)
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num).to(
            src.device
        )

        key_padding_mask = key_padding_mask.bool()

        _, attn_output_weights = self.multihead_attn_layer(
            src, src, src, key_padding_mask=key_padding_mask, average_attn_weights=False
        )
        # (bsz,num_head,target_L,source_L)

        # get [CLS] token attention weights
        # cls_weights = attn_output_weights[:, 0, 0,:]
        cls_weights = []
        for i in range(attn_output_weights.shape[0]):
            cls_weights.append(
                attn_output_weights[
                    i, :, : self.keyword_num, : audio_len[i] + self.keyword_num
                ]
            )

        return cls_weights

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
        bsz, total_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = self.get_keypadding_mask(bsz, total_len, audio_len + self.keyword_num)
        keywords = self.transformerEncoder(
            src=src,
            src_key_padding_mask=key_padding_mask,
        )

        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.embd_dim
        )

        keywords = self.linear_proj(keywords)

        # audio_feat = keywords
        # audio_feat = audio_feat.squeeze()

        # # Feed keyword into clip text encoder
        audio_feat, res = self.clip.encode_keywords(keywords, self.keyword_num)

        if cal_loss:
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            assert audio_feat.shape == image_feat.shape, "{} {}".format(
                audio_feat.shape, image_feat.shape
            )

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
                "loss": cl_loss,
            }
            # if q_loss is not None:
            #     losses.update({"q_loss": q_loss.detach()})

            return losses, audio_feat, image_feat, id, keywords

        return audio_feat, image_feat, res, id

    def training_step(self, batch, batch_idx):
        losses, _, _, _, _ = self.forward(batch, cal_loss=True)

        result = {
            "train_loss": losses["loss"],
            "train_cl_loss": losses["cl_loss"],
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

        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        # self.feature_extractor_s3prl(batch["wav"])
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
        if not os.path.exists(os.path.join(self.config.trainer.default_root_dir, "retokenizeText")):
            os.makedirs(
                os.path.join(self.config.trainer.default_root_dir, "retokenizeText"), exist_ok=True
            )

        if (
            hasattr(self, "log_detokenize_results_every_n_epoch")
            and self.current_epoch % self.log_detokenize_results_every_n_epoch == 0
        ) or not (hasattr(self, "log_detokenize_results_every_n_epoch")):
            gold_texts = []
            for x in outputs:
                gold_texts.extend(x["gold_text"])
            # gold_texts = [ x["gold_text"] for x in outputs]
            # gold_texts = [ x["gold_text"] for x in gold_texts]
            all_keyword_embeddings = torch.cat([x["keywords"] for x in outputs], dim=0)
            all_keyword_embeddings = all_keyword_embeddings.view(
                all_keyword_embeddings.shape[0],
                self.keyword_num,
                all_keyword_embeddings.shape[-1],
            )

            # all_keyword_embeddings shape (total_audio, num_keywords, hid_dim)

            assert all_keyword_embeddings.dim() == 3, all_keyword_embeddings.shape
            assert (
                all_keyword_embeddings.shape[2] == self.text_embd_dim
            ), all_keyword_embeddings.shape
            all_retok_outputs = []
            K = self.config.keyword.detokenized_K_neighbors
            tokenEmbeddings = self.clip.model.token_embedding.weight.detach().cpu()

            hit_rate = [0] * self.keyword_num
            for i in range(len(gold_texts)):
                gold_subword_toks_set = set(self.clip.tokenizer.encode(gold_texts[i]))

                _k_values, _k_indices = torch.topk(
                    F.cosine_similarity(
                        all_keyword_embeddings[i].view(
                            self.keyword_num, self.text_embd_dim, 1
                        ),
                        tokenEmbeddings.transpose(0, 1).unsqueeze(0),
                        dim=1,
                    ),
                    K,
                )
                assert _k_values.shape == (self.keyword_num, K)

                tmp_outputs = {}
                for _keyword_i in range(self.keyword_num):
                    tmp_outputs["keyword_{}".format(_keyword_i)] = []

                    # check if nearest K subword appears in gold text
                    top_k_toks = set(
                        [
                            self.clip.reducedl2Original[_ind.item()]
                            if self.clip.selected_text_emb_ids is not None
                            else _ind.item()
                            for _ind in _k_indices[_keyword_i]
                        ]
                    )
                    if bool(top_k_toks & gold_subword_toks_set):
                        hit_rate[_keyword_i] += 1

                    for _ind, _dist in zip(
                        _k_indices[_keyword_i], _k_values[_keyword_i]
                    ):
                        tmp_outputs["keyword_{}".format(_keyword_i)].append(
                            [
                                self.clip.tokenizer.decoder[
                                    self.clip.reducedl2Original[_ind.item()]
                                    if self.clip.selected_text_emb_ids is not None
                                    else _ind.item()
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

            hit_rate = torch.FloatTensor(hit_rate)
            hit_rate = hit_rate / len(gold_texts) * 100

            print("kw_hit_rate", hit_rate)
            self.logger.experiment.add_scalars(
                "kw_hit_rate",
                {
                    "kw_{}".format(i): hit_rate[i].item()
                    for i in range(self.keyword_num)
                },
                self.global_step,
            )

            with open(
                os.path.join(
                    self.config.trainer.default_root_dir,
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

        if self.feat_select_idx == "all":
            logging.warning("add self.audio_encoder_layer_weights to optimizer")
            audio_params = audio_params + [self.audio_encoder_layer_weights]

        if self.config.audio_encoder.trainable:
            audio_params = audio_params + list(self.audio_encoder.parameters())

        if self.downsampling_type is not None:
            audio_params = audio_params + list(self.downsampling.parameters())

        audio_params = audio_params + list(self.linear_proj.parameters())

        audio_params = audio_params + list(self.transformerEncoder.parameters())

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

