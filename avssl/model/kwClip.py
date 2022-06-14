import json
import logging

logger = logging.getLogger(__name__)
import math
import os
import pickle
from ast import keyword
from typing import List, Tuple, Union

import numpy as np
import torch
import tqdm
from jiwer import cer, wer
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ..base import OrderedNamespace
from ..data import random_crop_max_length
from ..module import (
    ClipModel,
    FairseqSpeechEncoder_Hubert,
    S3prlSpeechEncoder,
    S3prlSpeechEncoderPlus,
    SimpleCache,
    SupConLoss,
    losses,
    mutualRetrieval,
)
from ..module.fast_vgs_modules import DualEncoder, Wav2Vec2Model_cls
from ..module.kw_modules import TransformerModels
from ..module.speechclip_c_modules import vector_quantizers
from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm
from ..optim import get_scheduler
from ..util import freeze_model, get_keypadding_mask
from ..util.embedding_visualization import draw_embedding_space_PCA
from .base_model import BaseLightningModel

__all__ = ["KWClip_GeneralTransformer"]

METRIC_REDUCEFN_MAPPING = {
    torch.Tensor: lambda x: torch.mean(x),
    float: lambda x: x,
    int: lambda x: x,
    str: lambda x: x,
}


def load_fastvgs(
    model_type: str = "fast-vgs-plus",
    pretrained: bool = True,
    trainable: bool = False,
    superb: bool = False,
):
    model_path = (
        f"/work/vjsalt22/hsuanfu/audio-visual-ssl/avssl/model/{model_type}_coco"
    )
    # load args
    with open(f"{model_path}/args.pkl", "rb") as f:
        args = pickle.load(f)

    if superb:
        print("Using Wav2Vec2Model_cls model")
        model = Wav2Vec2Model_cls(args)
    else:
        print("Using DualEncoder model")
        model = DualEncoder(args)

    if pretrained:
        # load weights
        weights = torch.load(os.path.join(model_path, "best_bundle.pth"))
        model.carefully_load_state_dict(weights["dual_encoder"])

    if not trainable:
        freeze_model(model)

    return model


class KWClipBase(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.audio_encoder_type = config.audio_encoder.type
        if self.audio_encoder_type == "s3prl":
            raise DeprecationWarning("Please use s3prl_plus")
            self.audio_encoder = S3prlSpeechEncoder(**config.audio_encoder)
        elif self.audio_encoder_type == "s3prl_plus":
            self.audio_encoder = S3prlSpeechEncoderPlus(**config.audio_encoder)
        elif self.audio_encoder_type == "FairseqHubert":
            self.audio_encoder = FairseqSpeechEncoder_Hubert(**config.audio_encoder)
        self.clip = ClipModel(
            **config.clip,
        )

        self.audio_embd_dim = self.audio_encoder.out_dim
        self.subword_embd_dim = self.clip.model.token_embedding.weight.size(-1)

        self.recall_at = config.retrieval.recall_at

        self.criterion = getattr(losses, config.cl_loss.type)(**config.cl_loss.args)

        self.log_detokenize_results = config.log_setting.get(
            "log_detokenize_results", True
        )

        self.keyword_num = self.config.model_settings.cascaded_branch.keyword.number

    def forward_audio(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        return_hidden_states: bool = False,
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:

        if self.audio_encoder_type in ["s3prl_plus", "FairseqHubert"]:
            return self.audio_encoder(
                wav, wav_len, return_hidden_states=return_hidden_states
            )
        else:
            raise NotImplementedError("Unknown type:{}".format(self.audio_encoder_type))

    def forward(self, batch, cal_loss: bool = True):
        raise NotImplementedError()

    def compute_loss(self, input_feats):
        """compute the loss here

        Args:
            input_feats (Any): the feats required for computing loss
        """
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        # ss = "[{}]".format(self.device)
        # ss += "\nwav : {}".format(len(batch["wav"]))
        # ss += "\nwav_len : {}".format(len(batch["wav_len"]))
        # ss += "\nimage : {}".format(len(batch["image"]))
        # ss += "\nid : {}".format(len(batch["id"]))
        # print(ss)
        losses, log_metrics = self.forward(batch)[:2]

        return {"loss_feats": losses, "log_metrics": log_metrics}

    def training_step_end(self, outputs):
        if isinstance(outputs, dict):
            if "loss" in outputs:
                # training_step has already calculated the loss
                return torch.mean(outputs["loss"])
            elif "loss_feats" in outputs and "log_metrics" in outputs:
                losses = self.compute_loss(outputs["loss_feats"])
                log_metrics = outputs["log_metrics"]
                result = {
                    **{f"train_{k}": losses[k] for k in losses},
                    **{
                        f"train_{k}": METRIC_REDUCEFN_MAPPING[type(log_metrics[k])](
                            log_metrics[k]
                        )
                        for k in log_metrics
                    },
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
            else:
                print("outputs", outputs)
                raise NotImplementedError()
        else:
            print("outputs", outputs)
            raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        losses, log_metrics, others = self.forward(batch)

        audio_feat = (
            others["cascaded_audio_feat"]
            if self.config.retrieval.audio_feat_src == "cascaded"
            else others["parallel_audio_feat"]
        )

        image_feat = others["image_feat"]
        id = others["id"]
        return_dict = {
            "id": id,
            "audio_feat": audio_feat,
            "image_feat": image_feat,
        }

        if "keywords" in others and others["keywords"] is not None:
            keywords = others["keywords"]
            return_dict["keywords"] = keywords
            return_dict["gold_text"] = batch["text"]

        return {"loss_feats": losses, "log_metrics": log_metrics, "others": return_dict}

    def validation_step_end(self, outputs):
        assert isinstance(outputs, dict)
        losses = self.compute_loss(outputs["loss_feats"])

        log_metrics = outputs["log_metrics"]
        result = {
            **{f"val_{k}": losses[k] for k in losses},
            **{
                f"val_{k}": METRIC_REDUCEFN_MAPPING[type(log_metrics[k])](
                    log_metrics[k]
                )
                for k in log_metrics
            },
        }
        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for k in outputs["others"]:
            if isinstance(outputs["others"][k], torch.Tensor):
                outputs["others"][k] = outputs["others"][k].detach().cpu()
        return outputs["others"]

    def validation_epoch_end(self, outputs):
        if "keywords" in outputs[0].keys():
            if not os.path.exists(
                os.path.join(self.config.trainer.default_root_dir, "retokenizeText")
            ):
                os.makedirs(
                    os.path.join(
                        self.config.trainer.default_root_dir, "retokenizeText"
                    ),
                    exist_ok=True,
                )
            if not os.path.exists(
                os.path.join(self.config.trainer.default_root_dir, "visualization")
            ):
                os.makedirs(
                    os.path.join(self.config.trainer.default_root_dir, "visualization"),
                    exist_ok=True,
                )

            if (
                hasattr(self, "log_detokenize_results_every_n_epoch")
                and self.current_epoch % self.log_detokenize_results_every_n_epoch == 0
            ) or not (hasattr(self, "log_detokenize_results_every_n_epoch")):
                gold_texts = []
                for x in outputs:
                    for sent in x["gold_text"]:
                        # exit(1)
                        gold_texts.append(
                            self.clip.tokenizer.decode(sent.squeeze().tolist())
                        )
                        # gold_texts.extend(self.clip.deTokenize(x["gold_text"]))
                # gold_texts = [ x["gold_text"] for x in outputs]
                # gold_texts = [ x["gold_text"] for x in gold_texts]
                all_keyword_embeddings = torch.cat(
                    [x["keywords"] for x in outputs], dim=0
                )
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

                # self.log("embs_mean", embeddings_stat_dict["mean"])
                # self.log("embs_std", embeddings_stat_dict["std"])
                # self.log("embs_norm", embeddings_stat_dict["norm"])

                self.log(
                    "kw_mean_mse",
                    torch.norm(
                        torch.mean(
                            all_keyword_embeddings.view(-1, self.subword_embd_dim),
                            dim=0,
                        )
                        - torch.mean(tokenEmbeddings, dim=0),
                        p=2,
                    ),
                    sync_dist=True,
                )
                # self.log("kw_std_mse",torch.std(
                #     torch.norm(
                #         torch.std(all_keyword_embeddings.view(-1,self.subword_embd_dim),dim=0) - torch.std(tokenEmbeddings,dim=0),p=2
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
                    all_keyword_embeddings.shape[2] == self.subword_embd_dim
                ), all_keyword_embeddings.shape
                all_retok_outputs = []

                K = self.config.model_settings.cascaded_branch.keyword.get(
                    "detokenized_K_neighbors", 10
                )

                if not hasattr(
                    self.config.model_settings.cascaded_branch.keyword,
                    "retrieve_method",
                ):
                    self.config.model_settings.cascaded_branch.keyword.retrieve_method = (
                        "cosine"
                    )

                if (
                    self.config.model_settings.cascaded_branch.keyword.retrieve_method
                    == "pseudo_inverse"
                ):
                    emb_pinv = torch.linalg.pinv(tokenEmbeddings.T).float()

                assert (
                    self.config.model_settings.cascaded_branch.keyword.retrieve_method
                    in ["cosine", "pseudo_inverse"]
                )
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
                            .view(-1, self.subword_embd_dim)
                            .float()
                            .reshape(-1, self.subword_embd_dim)
                            .permute(1, 0)
                        ).permute(1, 0)
                        if self.config.model_settings.cascaded_branch.keyword.retrieve_method
                        == "pseudo_inverse"
                        else F.cosine_similarity(
                            all_keyword_embeddings[i : i + _bsz].view(
                                -1, self.subword_embd_dim, 1
                            ),
                            tokenEmbeddings.transpose(0, 1).unsqueeze(0),
                            dim=1,
                        ),
                        K,
                    )
                    assert _k_values.shape == (
                        _bsz * self.keyword_num,
                        K,
                    ), _k_values.shape
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
                                            if self.clip.selected_text_emb_ids
                                            is not None
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
                    },
                    sync_dist=True,
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
            all_audo_feats.float().to(self.device),
            all_img_feats.float().T.to(self.device),
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

        if isinstance(self.logger, WandbLogger):
            self.log("val_recall_AI", recall_results_AI, sync_dist=True)
            self.log("val_recall_IA", recall_results_IA, sync_dist=True)
            self.log("val_recall_mean", recall_results_mean, sync_dist=True)
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
        self.log("val_recall_mean_1", recall_results_mean["recall@1"], sync_dist=True)

    def processWavs(self, wav):
        wav_len = [len(x) for x in wav]
        if isinstance(wav, torch.Tensor):
            wav_len = torch.LongTensor(wav_len, device=wav.device)
        return wav, wav_len

    def feature_extractor_s3prl(
        self, wav: Union[Tuple[torch.Tensor], List[torch.Tensor]]
    ) -> torch.Tensor:
        """feature_extractor_s3prl
        Implement for s3prl to get feature
        Args:
            wav ():
        """
        raise NotImplementedError()

    def getTrainableParams(self) -> list:
        """getTrainableParams

        return trainable parameter list
        children class should return their additional trainable parameters

        Returns:
            list: list of trainable parameters
        """
        my_params = []

        my_params += self.audio_encoder.trainable_params()
        my_params += list(self.criterion.parameters())

        return my_params

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        my_params = self.getTrainableParams()

        audio_optimizer = getattr(torch.optim, self.config.audio_encoder.optim.name)(
            my_params,
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

        return optimizers, schedulers


class KW_CascadedBranch(nn.Module):
    def __init__(self, config, audio_dim: int, text_dim: int, clip: ClipModel) -> None:
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.clip = clip
        self.config = config

        logger.info("Using KW_CascadedBranch")
        self.keyword_num = config.model_settings.cascaded_branch.keyword.number

        self.cls = self._create_cls()
        logger.info("Start init [CLS] {}".format(self.cls.shape))

        assert hasattr(
            TransformerModels, config.model_settings.cascaded_branch.transformer_type
        )
        logger.info(
            f"Using {config.model_settings.cascaded_branch.transformer_type} as KW_CascadedBranch"
        )
        self.self_att = getattr(
            TransformerModels, config.model_settings.cascaded_branch.transformer_type
        )(**config.model_settings.cascaded_branch.transformer_args)

        self.linear_proj = nn.Linear(
            self.config.model_settings.cascaded_branch.transformer_args.d_model,
            self.text_dim,
        )

        # codebook selection
        self.vector_quantizer = None
        self.vq_type = config.model_settings.cascaded_branch.vq.type

        if not hasattr(
            vector_quantizers, config.model_settings.cascaded_branch.vq.type
        ):
            raise NotImplementedError(
                "Vq ({}) not implemented".format(
                    config.model_settings.cascaded_branch.vq.type
                )
            )

        self.vector_quantizer = getattr(vector_quantizers, self.vq_type)(
            **config.model_settings.cascaded_branch.vq.args
        )

        self.bn_layer = Kw_BatchNorm(
            kw_num=self.keyword_num,
            kw_dim=self.text_dim,
            batchnorm_type=config.model_settings.cascaded_branch.keyword.batchnorms.type,
            init_bias=torch.mean(self.clip.model.token_embedding.weight, dim=0),
            init_scale=torch.std(self.clip.model.token_embedding.weight, dim=0),
            std_scale=config.model_settings.cascaded_branch.keyword.batchnorms.std_scale,
            learnable=config.model_settings.cascaded_branch.keyword.batchnorms.learnable
            if hasattr(
                config.model_settings.cascaded_branch.keyword.batchnorms, "learnable"
            )
            else True,
            parallel=config.model_settings.cascaded_branch.keyword.batchnorms.parallel
            if hasattr(
                config.model_settings.cascaded_branch.keyword.batchnorms, "parallel"
            )
            else False,
        )

    def _create_cls(self):
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    self.keyword_num,
                    self.config.model_settings.cascaded_branch.transformer_args.d_model,
                ]
            )
        )

    def extract_hidden_states(self, audio_feat, audio_len):
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + self.keyword_num
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        hidden_states = [x[:, self.keyword_num, ...] for x in hidden_states]

        return tuple(hidden_states)

    def forward(self, audio_feat, audio_len):
        # Use multi-head attention layer to find keywords(cls)
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + self.keyword_num
        )

        keywords = self.self_att(src=src, key_padding_mask=key_padding_mask)

        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.audio_dim
        )

        keywords = self.linear_proj(keywords)

        keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_dim, 1),
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

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # Feed keyword into clip text encoder
        audio_feat = self.clip.encode_keywords(keywords, self.keyword_num)

        return audio_feat, vq_results, keywords


class KW_CascadedBranch_Integrated(KW_CascadedBranch):
    def __init__(
        self, config, audio_dim: int, text_dim: int, out_dim: int, clip: ClipModel
    ) -> None:
        super().__init__(config, audio_dim, text_dim, clip)
        self.out_dim = out_dim
        self.parallel_proj = nn.Linear(self.audio_dim, self.out_dim)

    def _create_cls(self):
        # first cls for parallel objective
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    self.keyword_num + 1,
                    self.config.model_settings.cascaded_branch.transformer_args.d_model,
                ]
            )
        )

    def extract_hidden_states(self, audio_feat, audio_len):
        bsz, total_max_len = (
            audio_feat.size(0),
            audio_feat.size(1) + self.keyword_num + 1,
        )
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + self.keyword_num + 1
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        hidden_states = [x[:, self.keyword_num + 1 :, ...] for x in hidden_states]
        return tuple(hidden_states)

    def forward(self, audio_feat, audio_len):
        # Use multi-head attention layer to find keywords(cls)
        bsz, total_max_len = (
            audio_feat.size(0),
            audio_feat.size(1) + self.keyword_num + 1,
        )
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len,
            data_lens=audio_len + self.keyword_num + 1,
        )

        keywords = self.self_att(src=src, key_padding_mask=key_padding_mask)

        parallel_cls = keywords[:, :1].reshape(-1, self.audio_dim)

        parallel_cls = self.parallel_proj(parallel_cls)

        keywords = keywords[:, 1 : self.keyword_num + 1].reshape(
            -1, self.keyword_num, self.audio_dim
        )

        keywords = self.linear_proj(keywords)

        keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_dim, 1),
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

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # Feed keyword into clip text encoder
        audio_feat = self.clip.encode_keywords(keywords, self.keyword_num)

        return audio_feat, vq_results, keywords, parallel_cls


class KW_ParallelBranch(nn.Module):
    def __init__(self, config, audio_dim: int, out_dim: int) -> None:
        super().__init__()
        self.config = config
        self.audio_dim = audio_dim
        self.out_dim = out_dim

        assert hasattr(
            TransformerModels, config.model_settings.parallel_branch.transformer_type
        )
        logger.info(
            f"Using {config.model_settings.parallel_branch.transformer_type} as KW_ParallelBranch"
        )
        self.self_att = getattr(
            TransformerModels, config.model_settings.parallel_branch.transformer_type
        )(**config.model_settings.parallel_branch.transformer_args)

        self.cls = self._create_cls()
        logger.info("Start init [CLS] {}".format(self.cls.shape))

        self.linear_proj = nn.Linear(self.audio_dim, self.out_dim)

    def _create_cls(self):
        # first cls for parallel objective
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    1,
                    self.config.model_settings.parallel_branch.transformer_args.d_model,
                ]
            )
        )

    def extract_hidden_states(self, audio_feat, audio_len):
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + 1
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + 1
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        hidden_states = [x[:, 1:, ...] for x in hidden_states]
        return tuple(hidden_states)

    def forward(self, audio_feat, audio_len):
        # Use multi-head attention layer to find keywords(cls)
        bsz, total_max_len = (
            audio_feat.size(0),
            audio_feat.size(1) + 1,
        )
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len,
            data_lens=audio_len + 1,
        )

        out = self.self_att(src=src, key_padding_mask=key_padding_mask)

        out = out[:, :1].reshape(-1, self.audio_dim)

        out = self.linear_proj(out)

        return out


class KWClip_GeneralTransformer(KWClipBase):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.cascaded_branch = None
        self.parallel_branch = None
        if self.config.model_settings.cascaded_objective_weight > 0:
            logger.info("Create Cascaded Branch")
            # cascaded_branch
            if self.config.model_settings.cascaded_branch.type == "KW_CascadedBranch":
                self.cascaded_branch = KW_CascadedBranch(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif (
                self.config.model_settings.cascaded_branch.type
                == "KW_CascadedBranch_Integrated"
            ):
                assert self.config.model_settings.parallel_objective_weight > 0
                logger.info("Using Parallel Objective (Integrated w/ cascaded_branch)")
                self.cascaded_branch = KW_CascadedBranch_Integrated(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    out_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            else:
                raise NotImplementedError()

        if (
            self.config.model_settings.parallel_objective_weight > 0
            and not self.config.model_settings.cascaded_branch.type
            == "KW_CascadedBranch_Integrated"
        ):
            logger.info("Create Parallel Branch")
            self.parallel_branch = KW_ParallelBranch(
                config=self.config,
                audio_dim=self.audio_embd_dim,
                out_dim=self.subword_embd_dim,
            )

    def getTrainableParams(self):
        _params = super().getTrainableParams()
        if self.cascaded_branch is not None:
            logger.info("Add cascaded_branch parameters")
            _params += list(self.cascaded_branch.parameters())

        if self.parallel_branch is not None:
            logger.info("Add parallel_branch parameters")
            _params += list(self.parallel_branch.parameters())

        return _params

    def feature_extractor_s3prl(self, wav):
        wav, wav_len = self.processWavs(wav)

        audio_feat, audio_len, hidden_states = self.forward_audio(
            wav, wav_len, return_hidden_states=True
        )
        assert isinstance(hidden_states, tuple)

        cascaded_hidden_states = None
        parallel_hidden_states = None
        if self.cascaded_branch is not None:
            cascaded_hidden_states = self.cascaded_branch.extract_hidden_states(
                audio_feat, audio_len
            )
            assert isinstance(cascaded_hidden_states, tuple)
            hidden_states = hidden_states + tuple(cascaded_hidden_states[1:])
        if self.parallel_branch is not None:
            parallel_hidden_states = self.parallel_branch.extract_hidden_states(
                audio_feat, audio_len
            )
            assert isinstance(parallel_hidden_states, tuple)
            hidden_states = hidden_states + tuple(parallel_hidden_states[1:])

        return None, hidden_states

    def compute_loss(self, input_feats):
        """compute the loss here

        Args:
            input_feats (Any): the feats required for computing loss
        """
        assert isinstance(input_feats, dict)
        assert "id" in input_feats
        assert (
            "cascaded_audio_feat" in input_feats or "parallel_audio_feat" in input_feats
        )
        assert "image_feat" in input_feats

        cascaded_audio_feat = (
            input_feats["cascaded_audio_feat"].float()
            if "cascaded_audio_feat" in input_feats
            else None
        )
        parallel_audio_feat = (
            input_feats["parallel_audio_feat"].float()
            if "parallel_audio_feat" in input_feats
            else None
        )
        image_feat = input_feats["image_feat"].float()
        id = input_feats["id"]

        losses = {"loss": 0}
        if self.config.model_settings.cascaded_objective_weight > 0:
            losses["c_cl_loss"] = self.criterion(
                feat_A=cascaded_audio_feat,
                feat_B=image_feat,
                index=id,
            )
            losses["loss"] += (
                self.config.model_settings.cascaded_objective_weight
                * losses["c_cl_loss"]
            )

        if self.config.model_settings.parallel_objective_weight > 0:
            losses["p_cl_loss"] = self.criterion(
                feat_A=parallel_audio_feat,
                feat_B=image_feat,
                index=id,
            )
            losses["loss"] += (
                self.config.model_settings.parallel_objective_weight
                * losses["p_cl_loss"]
            )

        return losses

    def forward(
        self,
        batch,
        cal_loss: bool = False,
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        image_feat = self.forward_image(image)

        cascaded_audio_feat = None
        parallel_audio_feat = None
        vq_results = None
        keywords = None
        if self.cascaded_branch is not None:
            if (
                self.config.model_settings.cascaded_branch.type
                == "KW_CascadedBranch_Integrated"
            ):
                (
                    cascaded_audio_feat,
                    vq_results,
                    keywords,
                    parallel_audio_feat,
                ) = self.cascaded_branch(
                    audio_feat=audio_feat,
                    audio_len=audio_len,
                )
            else:
                cascaded_audio_feat, vq_results, keywords = self.cascaded_branch(
                    audio_feat=audio_feat,
                    audio_len=audio_len,
                )

        if self.parallel_branch is not None:
            parallel_audio_feat = self.parallel_branch(
                audio_feat=audio_feat,
                audio_len=audio_len,
            )

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        losses = {
            "id": id,
            "image_feat": image_feat,
        }
        log_metrics = {}

        if cascaded_audio_feat is not None:
            cascaded_audio_feat = cascaded_audio_feat / cascaded_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["cascaded_audio_feat"] = cascaded_audio_feat

        if parallel_audio_feat is not None:
            parallel_audio_feat = parallel_audio_feat / parallel_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["parallel_audio_feat"] = parallel_audio_feat

        # losses = {"loss": 0}
        if self.config.model_settings.cascaded_objective_weight > 0:
            log_metrics["softmax_temp"] = vq_results["temp"]

        if self.config.model_settings.parallel_objective_weight > 0:
            pass

        # losses.update(
        log_metrics.update(
            {
                "cl_temp": self.criterion.current_temperature,
            }
        )
        return (
            losses,
            log_metrics,
            {
                "cascaded_audio_feat": cascaded_audio_feat,
                "parallel_audio_feat": parallel_audio_feat,
                "image_feat": image_feat,
                "id": id,
                "vq_results": vq_results,
                "keywords": keywords,
            },
        )
