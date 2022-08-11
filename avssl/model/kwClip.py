import json
import logging

logger = logging.getLogger(__name__)

import os
from typing import List, Tuple, Union

import numpy as np
import torch
import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F

from ..base import OrderedNamespace
from ..module import (
    ClipModel,
    FairseqSpeechEncoder_Hubert,
    MLPLayers,
    S3prlSpeechEncoderPlus,
    losses,
    mutualRetrieval,
)
from ..module.kw_modules import TransformerModels
from ..module.speechclip_c_modules import vector_quantizers
from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm
from ..optim import get_scheduler
from ..util import get_keypadding_mask
from ..util.embedding_visualization import draw_embedding_space_PCA
from .base_model import BaseLightningModel

__all__ = [
    "KWClip_GeneralTransformer",
    "KWClip_SpeechText",
    "KWClip_CLIP_Original",
    "KWClip_GeneralTransformer_SpeechText",
]

"""METRIC_REDUCEFN_MAPPING
define the reduction function for each data type when reducing from multiple GPUs

"""
METRIC_REDUCEFN_MAPPING = {
    torch.Tensor: lambda x: torch.mean(x),
    float: lambda x: x,
    int: lambda x: x,
    str: lambda x: x,
}


class KWClipBase(BaseLightningModel):
    """Base Class for SpeechCLIP"""

    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        # select audio_encoder type
        self.audio_encoder_type = config.audio_encoder.type
        if self.audio_encoder_type == "s3prl":
            raise DeprecationWarning("Please use s3prl_plus")
            self.audio_encoder = S3prlSpeechEncoder(**config.audio_encoder)
        elif self.audio_encoder_type == "s3prl_plus":
            self.audio_encoder = S3prlSpeechEncoderPlus(**config.audio_encoder)
        elif self.audio_encoder_type == "FairseqHubert":
            self.audio_encoder = FairseqSpeechEncoder_Hubert(**config.audio_encoder)
        elif self.audio_encoder_type == "ChimeraSpeechEncoder":
            self.audio_encoder = ChimeraSpeechEncoder(**config.audio_encoder)
        else:
            logger.warning("No audio encoder loaded")

        # define ClipModel
        self.clip = ClipModel(
            **config.clip,
        )

        if hasattr(self, "audio_encoder"):
            self.audio_embd_dim = self.audio_encoder.out_dim
        # dimension of the CLIP Text Encoder's subword embedding
        self.subword_embd_dim = self.clip.model.token_embedding.weight.size(-1)

        # the recall to calculate
        self.recall_at = config.retrieval.recall_at

        # define loss function
        self.criterion = getattr(losses, config.cl_loss.type)(**config.cl_loss.args)

        # whether or not to log detokenize subwords of keywords
        self.log_detokenize_results = config.log_setting.get(
            "log_detokenize_results", True
        )

        # the number of keywords in cascaded branch
        self.keyword_num = self.config.model_settings.cascaded_branch.keyword.number

    def forward_audio(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        return_hidden_states: bool = False,
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:
        """Get the representations of audio wav files after passing through the audio encoder

        Args:
            wav (Union[torch.Tensor, list]): wav files
            wav_len (Union[torch.Tensor, list], optional): lengths of each wavform. Defaults to [].
            return_hidden_states (bool, optional): return the hidden representations in the audio encoder. Defaults to False.

        Raises:
            NotImplementedError: if the audio encoder is not implemented in the code

        Returns:
            Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]: return the representations of waveforms (and also the hidden_states)
        """
        if self.audio_encoder_type in [
            "s3prl_plus",
            "FairseqHubert",
            "ChimeraSpeechEncoder",
        ]:
            return self.audio_encoder(
                wav, wav_len, return_hidden_states=return_hidden_states
            )
        else:
            raise NotImplementedError("Unknown type:{}".format(self.audio_encoder_type))

    def forward(self, batch: dict) -> tuple:
        """the main forward function for our model (should be implemented in child class)

        Args:
            batch (dict): the input data in a batch

        Returns:
            tuple: return model output : (losses, log_metric, other_feats)
                losses: features required for calculating loss (pass into comput_loss)
                        if loss is calulated on each GPU individually, "loss" should exist in lossess
                log_metric: the calculated metric to log
                other_feats: other features required for validation
        """
        raise NotImplementedError()

    def compute_loss(self, input_feats):
        """compute the loss here

        Args:
            input_feats (Any): the feats required for computing loss (gathered from model forward output)
        """
        raise NotImplementedError()

    def training_step(self, batch: dict) -> dict:
        losses, log_metrics = self.forward(batch)[:2]
        return {"loss_feats": losses, "log_metrics": log_metrics}

    def training_step_end(self, outputs: dict) -> dict:
        """training_step_end

        Collect results from all GPUs

        Args:
            outputs (dict): output from trainin_step

        Raises:
            NotImplementedError: if the outputs' format collected from GPU(s) is not correct

        Returns:
            dict: loss (return to pytorch lightning for updating params)
        """
        if isinstance(outputs, dict):
            if "loss" in outputs:
                # training_step has already calculated the loss
                # we simply just average the loss on GPU(s)
                return {"loss": torch.mean(outputs["loss"])}
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
                # log training loss(es) and metrics
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
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """validation_step

        Args:
            batch (dict): input data

        Returns:
            dict: output features
        """
        losses, log_metrics, others = self.forward(batch)

        # select cascaded or parallel branch's output for contrastive loss calculation
        audio_feat = (
            others["cascaded_audio_feat"]
            if self.config.retrieval.audio_feat_src == "cascaded"
            else others["parallel_audio_feat"]
        )

        image_feat = others["image_feat"] if "image_feat" in others else None
        text_feat = others["text_feat"] if "text_feat" in others else None
        id = others["id"]

        # collect features
        return_dict = {
            "id": id,
            "audio_feat": audio_feat,
        }
        if image_feat is not None:
            return_dict["image_feat"] = image_feat
        if text_feat is not None:
            return_dict["text_feat"] = text_feat

        if "keywords" in others and others["keywords"] is not None:
            keywords = others["keywords"]
            return_dict["keywords"] = keywords
            return_dict["gold_text"] = batch["text"]

        return {"loss_feats": losses, "log_metrics": log_metrics, "others": return_dict}

    def validation_step_end(self, outputs: dict) -> dict:
        """validation_step_end

        Collect features from all GPU(s) and calculate loss

        Args:
            outputs (dict): output from GPU(s)

        Returns:
            dict: features required for validation
        """

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

    def validation_epoch_end(self, outputs: list):
        """validation_epoch_end

        Args:
            outputs (list): list of aggregated results
        """
        # if keywords is in the input, calculate keyword related metrics
        if "keywords" in outputs[0].keys():
            # create detokenize text dir
            if not os.path.exists(
                os.path.join(self.config.trainer.default_root_dir, "detokenizeText")
            ):
                os.makedirs(
                    os.path.join(
                        self.config.trainer.default_root_dir, "detokenizeText"
                    ),
                    exist_ok=True,
                )

            if (
                hasattr(self, "log_detokenize_results_every_n_epoch")
                and self.current_epoch % self.log_detokenize_results_every_n_epoch == 0
            ) or not (hasattr(self, "log_detokenize_results_every_n_epoch")):
                # collect glden texts
                gold_texts = []
                for x in outputs:
                    for sent in x["gold_text"]:
                        gold_texts.append(
                            self.clip.tokenizer.decode(sent.squeeze().tolist())
                        )
                all_keyword_embeddings = torch.cat(
                    [x["keywords"] for x in outputs], dim=0
                )
                all_keyword_embeddings = all_keyword_embeddings.view(
                    all_keyword_embeddings.shape[0],
                    self.keyword_num,
                    all_keyword_embeddings.shape[-1],
                )

                tokenEmbeddings = self.clip.model.token_embedding.weight.detach().cpu()

                assert all_keyword_embeddings.dim() == 3, all_keyword_embeddings.shape
                assert (
                    all_keyword_embeddings.shape[2] == self.subword_embd_dim
                ), all_keyword_embeddings.shape
                all_retok_outputs = []

                # list the semantically related subwords to the selected keyword
                K = self.config.model_settings.cascaded_branch.keyword.get(
                    "detokenized_K_neighbors", 10
                )

                # get retrieval method : either via cosine or pseudo inverse (default cosine similarity)
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
                kw_top_ret = [[] for _ in range(self.keyword_num)]
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
                                hit_token_id = int(
                                    list(top_k_toks & gold_subword_toks_set[x])[0]
                                )
                                kw_top_ret[_keyword_i].append(hit_token_id)

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
                        "detokenizeText/",
                        "kw_hit_ep{}.json".format(self.current_epoch),
                    ),
                    "w",
                ) as f:
                    json.dump(kw_top_ret, f)

                with open(
                    os.path.join(
                        self.config.trainer.default_root_dir,
                        "detokenizeText/",
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
            score_per_A=score_per_audio,
            score_per_B=score_per_image,
            AB_answers=AI_answers,
            BA_answers=IA_answers,
        )

    def forward_image(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        """forward_image

        Args:
            images (Union[list, torch.Tensor]): image input

        Raises:
            ValueError: image tensor shape error
            TypeError: image type should be either list or torch.Tensor

        Returns:
            torch.Tensor: image representations
        """
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
        if hasattr(self.clip, "original2Reduced"):
            # if reduced embedding is used, we need to convert original ids to reduced ids
            for i in range(text_tensor.shape[0]):
                for j in range(text_tensor.shape[1]):
                    text_tensor[i, j] = self.clip.original2Reduced[
                        text_tensor[i, j].item()
                    ]

        text_feat = self.clip.encode_text(text_tensor)
        return text_feat

    def reportRetrieval(
        self,
        score_per_A: torch.Tensor,
        score_per_B: torch.Tensor,
        AB_answers: torch.Tensor,
        BA_answers: torch.Tensor,
        metadata: dict = {
            "modality_A_title": "audio",
            "modality_B_title": "image",
            "modality_A_logAbbr": "A",
            "modality_B_logAbbr": "I",
        },
    ):
        """reportRetrieval

        Args:
            score_per_A (torch.Tensor): the similarity score per modality A sample
            score_per_B (torch.Tensor): the similarity score per modality B sample
            AB_answers (torch.Tensor): the golden answer (pair ID) for each audio sample
            BA_answers (torch.Tensor): the golden answer (pair ID) for each image sample
            metadata (dict): metadata should include modality the title for A, B and the abbreviation for A and B
        """

        # metadata should include modality the title for A, B and the abbreviation for A and B
        assert "modality_A_title" in metadata
        assert "modality_B_title" in metadata
        assert "modality_A_logAbbr" in metadata
        assert "modality_B_logAbbr" in metadata

        recall_results_AB, recall_results_BA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_A,
            score_per_B=score_per_B,
            AB_answers=AB_answers,
            BA_answers=BA_answers,
            recall_at=self.recall_at,
            modality_A_title=metadata["modality_A_title"],
            modality_B_title=metadata["modality_B_title"],
        )

        log_AB_abbr = "{}{}".format(
            metadata["modality_A_logAbbr"], metadata["modality_B_logAbbr"]
        )
        log_BA_abbr = "{}{}".format(
            metadata["modality_B_logAbbr"], metadata["modality_A_logAbbr"]
        )

        print(f"val_recall_{log_AB_abbr}", recall_results_AB)
        print(f"val_recall_{log_BA_abbr}", recall_results_BA)
        print("val_recall_mean", recall_results_mean)

        if isinstance(self.logger, WandbLogger):
            # when using wandb
            self.log(f"val_recall_{log_AB_abbr}", recall_results_AB, sync_dist=True)
            self.log(f"val_recall_{log_BA_abbr}", recall_results_BA, sync_dist=True)
            self.log("val_recall_mean", recall_results_mean, sync_dist=True)
        elif isinstance(self.logger, TensorBoardLogger):
            # when using tensorboard
            self.logger.experiment.add_scalars(
                f"val_recall_{log_AB_abbr}", recall_results_AB, self.global_step
            )
            self.logger.experiment.add_scalars(
                f"val_recall_{log_BA_abbr}", recall_results_BA, self.global_step
            )
            self.logger.experiment.add_scalars(
                "val_recall_mean", recall_results_mean, self.global_step
            )
        if self.logger is not None:
            self.log(
                "val_recall_mean_10", recall_results_mean["recall@10"], sync_dist=True
            )

    def processWavs(
        self, wav: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """processWavs

        Args:
            wav (torch.LongTensor): wav input

        Returns:
            Tuple[torch.Tensor,torch.LongTensor]: wavs, wav_lens
        """

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

        if hasattr(self, "audio_encoder"):
            my_params += self.audio_encoder.trainable_params()
            my_params += list(self.criterion.parameters())

        my_params += self.clip.trainable_params()

        return my_params

    def configure_optimizers(self) -> Tuple[list, list]:
        """configure_optimizers

        Returns:
            Tuple[list,list]: (optimizer_list,scheduler_list)
        """
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
    """KW_CascadedBranch

    Cascaded Branch for SpeechCLIP

    """

    def __init__(
        self, config: OrderedNamespace, audio_dim: int, text_dim: int, clip: ClipModel
    ) -> None:
        """init

        Args:
            config (OrderedNamespace): config of the model
            audio_dim (int): dimension for audio features
            text_dim (int): dimension for subword embeddings
            clip (ClipModel): the CLIP model

        """
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.clip = clip
        self.config = config

        # projection network for (before BatchNorm Layer)
        self.kw_projection_config = (
            self.config.model_settings.cascaded_branch.keyword.get(
                "kw_projection", None
            )
        )

        logger.info("Using KW_CascadedBranch")
        self.keyword_num = config.model_settings.cascaded_branch.keyword.number

        self.cls = self._create_cls()
        logger.info("Start init [CLS] {}".format(self.cls.shape))

        # select the main structure for transformer encoder layer
        assert hasattr(
            TransformerModels, config.model_settings.cascaded_branch.transformer_type
        ), "transformer structure '{}' not supported".format(
            config.model_settings.cascaded_branch.transformer_type
        )
        logger.info(
            f"Using {config.model_settings.cascaded_branch.transformer_type} as KW_CascadedBranch"
        )
        self.self_att = getattr(
            TransformerModels, config.model_settings.cascaded_branch.transformer_type
        )(**config.model_settings.cascaded_branch.transformer_args)

        if self.kw_projection_config is None:
            logger.info(
                "kw_projection not specified, using single linear layer as default"
            )
            self.linear_proj = nn.Linear(
                self.config.model_settings.cascaded_branch.transformer_args.d_model,
                self.text_dim,
            )
        else:
            logger.info(
                f"kw_projection dims:{self.kw_projection_config.dimensions} droupout:{self.kw_projection_config.dropout}"
            )
            assert (
                self.kw_projection_config.dimensions[0]
                == self.config.model_settings.cascaded_branch.transformer_args.d_model
            ), f"first dim({self.kw_projection_config.dimensions[0]}) should match the audio encoder dim({self.config.model_settings.cascaded_branch.transformer_args.d_model})"
            assert (
                self.kw_projection_config.dimensions[-1] == self.text_dim
            ), f"last dim({self.kw_projection_config.dimensions[-1]}) should match the text encoder dim({self.text_dim})"
            self.linear_proj = MLPLayers(
                units=self.kw_projection_config.dimensions,
                dropout=self.kw_projection_config.dropout,
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

        # batchnorms
        if hasattr(config.model_settings.cascaded_branch.keyword, "batchnorms"):
            self.bn_layer = Kw_BatchNorm(
                kw_num=self.keyword_num,
                kw_dim=self.text_dim,
                batchnorm_type=config.model_settings.cascaded_branch.keyword.batchnorms.type,
                init_bias=torch.mean(self.clip.model.token_embedding.weight, dim=0),
                init_scale=torch.std(self.clip.model.token_embedding.weight, dim=0),
                std_scale=config.model_settings.cascaded_branch.keyword.batchnorms.std_scale,
                learnable=config.model_settings.cascaded_branch.keyword.batchnorms.learnable
                if hasattr(
                    config.model_settings.cascaded_branch.keyword.batchnorms,
                    "learnable",
                )
                else True,
                parallel=config.model_settings.cascaded_branch.keyword.batchnorms.parallel
                if hasattr(
                    config.model_settings.cascaded_branch.keyword.batchnorms, "parallel"
                )
                else False,
            )

    def _create_cls(self) -> torch.nn.Parameter:
        """Create CLS

        Returns:
            torch.nn.Parameter: the params for CLS(s)
        """
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    self.keyword_num,
                    self.config.model_settings.cascaded_branch.transformer_args.d_model,
                ]
            )
        )

    def extract_hidden_states(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple:
        """extract_hidden_states
        Extracting hidden representation of each layers

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: tuples of hiddenstates
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + self.keyword_num
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        # exclude the cls positions
        hidden_states = [x[:, self.keyword_num :, ...] for x in hidden_states]

        return tuple(hidden_states)

    def forward(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """forward

        Args:
            audio_feat (torch.Tensor)
            audio_len (torch.Tensor)

        Returns:
            Tuple: (audio_feat, vq_results, keywords)
        """
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

        if hasattr(self, "bn_layer"):
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

    def getAttentionMap(self, audio_feat: torch.Tensor, audio_len: torch.Tensor):
        """getAttentionMap

        return attention maps for visualization

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: cls_weights, topk_kw, None
        """
        # Use multi-head attention layer to find keywords(cls)
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + self.keyword_num
        )

        _, attn_output_weights = self.self_att.extract_attention_map(
            src=src, key_padding_mask=key_padding_mask
        )

        cls_weights = []
        for i in range(attn_output_weights.shape[0]):
            cls_weights.append(
                attn_output_weights[
                    i, :, : self.keyword_num, : audio_len[i] + self.keyword_num
                ]
            )

        keywords = self.self_att(src=src, key_padding_mask=key_padding_mask)

        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.audio_dim
        )

        keywords = self.linear_proj(keywords)

        if hasattr(self, "bn_layer"):
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

        cos_score = torch.stack(cos_score, dim=1)
        # disallow special tokens
        cos_score[..., 0] -= 100
        cos_score[..., 2] -= 100
        cos_score[..., 3] -= 100

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        # VQ
        # vq_results = self.vector_quantizer(x=cos_score)
        # assert self.clip.model.token_embedding.weight.requires_grad == False

        topk_kw = [[[] for _ in range(self.keyword_num)] for _ in range(bsz)]
        # print(vq_results["subword_prob"].shape)
        _, topk_kw_ids = torch.topk(cos_score, dim=-1, k=10)
        for bsz_i in range(bsz):
            for kw_i in range(self.keyword_num):
                topk_kw[bsz_i][kw_i] = [
                    self.clip.tokenizer.decoder[
                        self.clip.reducedl2Original[x.item()]
                        # top1_kw_id[bsz_i, kw_i].item()
                    ].replace("</w>", "")
                    for x in topk_kw_ids[bsz_i, kw_i]
                ]
        return cls_weights, topk_kw, None


class KW_ParallelBranch(nn.Module):
    """KW_ParallelBranch

    The parallel branch of SpeechCLIP

    """

    def __init__(self, config: OrderedNamespace, audio_dim: int, out_dim: int) -> None:
        super().__init__()
        self.config = config
        self.audio_dim = audio_dim
        self.out_dim = out_dim
        self.need_projection = self.config.model_settings.parallel_branch.get(
            "need_projection", True
        )

        # select the transformer structure for main architecture for parallel branch
        assert hasattr(
            TransformerModels, config.model_settings.parallel_branch.transformer_type
        )
        logger.info(
            f"Using {config.model_settings.parallel_branch.transformer_type} as KW_ParallelBranch (projection={self.need_projection})"
        )
        self.self_att = getattr(
            TransformerModels, config.model_settings.parallel_branch.transformer_type
        )(**config.model_settings.parallel_branch.transformer_args)

        self.cls = self._create_cls()
        logger.info("Start init [CLS] {}".format(self.cls.shape))

        if self.need_projection:
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

    def extract_hidden_states(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple:
        """extract_hidden_states
        Extract hiddenstates of parallel branch
        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: hidden representation of each layers
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + 1
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + 1
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        # exclude CLS position
        hidden_states = [x[:, 1:, ...] for x in hidden_states]
        return tuple(hidden_states)

    def forward(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> torch.Tensor:
        """forward

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            torch.Tensor: output
        """
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

        if hasattr(self, "linear_proj"):
            out = self.linear_proj(out)

        return out


class KWClip_GeneralTransformer(KWClipBase):
    """KWClip_GeneralTransformer
    Main class for SpeechCLIP
    """

    def __init__(self, config: OrderedNamespace) -> None:
        """init

        Args:
            config (OrderedNamespace): _description_
        """
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
            else:
                raise NotImplementedError()

        if self.config.model_settings.parallel_objective_weight > 0:
            logger.info("Create Parallel Branch")
            self.parallel_branch = KW_ParallelBranch(
                config=self.config,
                audio_dim=self.audio_embd_dim,
                out_dim=self.subword_embd_dim,
            )

        # projection network after CLIP image encoder
        self.img_enc_proj_net = None
        image_encoder_projection = self.config.model_settings.get(
            "image_encoder_projection", None
        )
        if image_encoder_projection is not None:
            logger.info(
                f"image_encoder_projection dims:{image_encoder_projection.dimensions} droupout:{image_encoder_projection.dropout}"
            )
            self.img_enc_proj_net = MLPLayers(
                units=image_encoder_projection.dimensions,
                dropout=image_encoder_projection.dropout,
            )

        # projection network after parallel branch
        self.p_branch_proj_net = None
        parallel_branch_projection = self.config.model_settings.get(
            "parallel_branch_projection", None
        )
        if parallel_branch_projection is not None:
            logger.info(
                f"parallel_branch_projection dims:{parallel_branch_projection.dimensions} droupout:{parallel_branch_projection.dropout}"
            )
            self.p_branch_proj_net = MLPLayers(
                units=parallel_branch_projection.dimensions,
                dropout=parallel_branch_projection.dropout,
            )

        # projection network after cascaded branch
        self.c_branch_proj_net = None
        cascaded_branch_projection = self.config.model_settings.get(
            "cascaded_branch_projection", None
        )
        if parallel_branch_projection is not None:
            logger.info(
                f"cascaded_branch_projection dims:{cascaded_branch_projection.dimensions} droupout:{cascaded_branch_projection.dropout}"
            )
            self.c_branch_proj_net = MLPLayers(
                units=cascaded_branch_projection.dimensions,
                dropout=cascaded_branch_projection.dropout,
            )

    def getTrainableParams(self) -> list:
        """getTrainableParams

        Returns:
            list: list of trainable params in this class
        """
        _params = super().getTrainableParams()
        if self.cascaded_branch is not None:
            logger.info("Add cascaded_branch parameters")
            _params += list(self.cascaded_branch.parameters())

        if self.parallel_branch is not None:
            logger.info("Add parallel_branch parameters")
            _params += list(self.parallel_branch.parameters())

        if self.img_enc_proj_net is not None:
            logger.info("Add img_enc_proj_net parameters")
            _params += list(self.img_enc_proj_net.parameters())

        if self.p_branch_proj_net is not None:
            logger.info("Add parallel_branch_projection parameters")
            _params += list(self.p_branch_proj_net.parameters())

        return _params

    def feature_extractor_s3prl(self, wav, featrure_layer_norm=True):
        """feature_extractor_s3prl

        Function for extracting features for s3prl

        Args:
            wav (_type_): _description_
            featrure_layer_norm (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
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

        # assert len(hidden_states) == 15
        # print(hidden_states[0].shape)
        # print(hidden_states[-1].shape)
        # if hidden_states[0].shape[0] > 1:
        # assert hidden_states[0].shape[0] == 1
        # import uuid
        # import glob

        # current_files_num = len(list(glob.glob("/work/twsezjg982/atosystem/audio-visual-ssl/slurms/KS_hidstates/KW_bsz256_WS_p1_flickr/*.pt")))
        # if current_files_num >= 51094:
        #     print("Finish")
        #     exit(1)

        # hubert_states = torch.stack(hidden_states).view(14,-1,768)
        # hubert_states = torch.mean(torch.norm(hubert_states,dim=-1),dim=-1)
        # assert hubert_states.shape == (14,)
        # # gap = torch.mean(torch.norm(hubert_states[:-1,...] - hubert_states[-1,...],dim=-1),dim=-1)
        # # print(hubert_states.shape)
        # # exit(1)
        # torch.save(hubert_states.cpu(),f"/work/twsezjg982/atosystem/audio-visual-ssl/slurms/KS_hidstates/KW_bsz256_WS_p1_flickr/{uuid.uuid4()}.pt")
        assert featrure_layer_norm == True
        if featrure_layer_norm:
            hidden_states = torch.stack(hidden_states, dim=0)
            hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
            hidden_states = [x for x in hidden_states]

        return hidden_states[-1], hidden_states

    def compute_loss(self, input_feats: dict):
        """compute the loss here

        Args:
            input_feats (dict): the feats required for computing loss
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
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        image_feat = self.forward_image(image)
        if self.img_enc_proj_net is not None:
            image_feat = self.img_enc_proj_net(image_feat)

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
            if self.p_branch_proj_net is not None:
                parallel_audio_feat = self.p_branch_proj_net(parallel_audio_feat)

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

        if self.config.model_settings.cascaded_objective_weight > 0:
            log_metrics["softmax_temp"] = vq_results["temp"]

        if self.config.model_settings.parallel_objective_weight > 0:
            pass

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

    def get_attention_weights(
        self, wav: Union[Tuple[torch.Tensor], List[torch.Tensor]]
    ):
        """get_attention_weights

        For attention map visualization
        Args:
            wav (Union[Tuple[torch.Tensor], List[torch.Tensor]]):

        Returns:
            attention weights
        """
        wav_len = [len(x) for x in wav]
        self.clip.update_device(self.device)
        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        return self.cascaded_branch.getAttentionMap(audio_feat, audio_len)


class KWClip_CLIP_Original(KWClipBase):
    """KWClip_CLIP_Original

    The original CLIP Text Encoder + Image Encoder

    """

    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        # the original CLIP doesn't require parallel and cascaded branch
        assert self.cascaded_branch is None
        assert self.parallel_branch is None

    def getTrainableParams(self):
        _params = super().getTrainableParams()
        return _params

    def reportRetrieval(
        self,
        score_per_A: torch.Tensor,
        score_per_B: torch.Tensor,
        AB_answers: torch.Tensor,
        BA_answers: torch.Tensor,
        metadata: dict = {
            "modality_A_title": "audio",
            "modality_B_title": "image",
            "modality_A_logAbbr": "A",
            "modality_B_logAbbr": "I",
        },
    ):
        return super().reportRetrieval(
            score_per_A,
            score_per_B,
            AB_answers,
            BA_answers,
            {
                "modality_A_title": "text",
                "modality_B_title": "image",
                "modality_A_logAbbr": "T",
                "modality_B_logAbbr": "I",
            },
        )

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
    ) -> dict:
        image = batch["image"]
        id = batch["id"]
        text = batch["text"]

        # update device information to clip model
        self.clip.update_device(self.device)

        image_feat = self.forward_image(image)
        text_feat = self.forward_text(text.view(-1, 77))

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        losses = {
            "id": id,
            "image_feat": image_feat,
        }
        log_metrics = {}

        losses["parallel_audio_feat"] = text_feat

        log_metrics.update(
            {
                "cl_temp": self.criterion.current_temperature,
            }
        )
        return (
            losses,
            log_metrics,
            {
                "parallel_audio_feat": text_feat,
                "image_feat": image_feat,
                "id": id,
                "vq_results": None,
                "keywords": None,
            },
        )


class KWClip_GeneralTransformer_SpeechText(KWClip_GeneralTransformer):
    """KWClip_GeneralTransformer_SpeechText

    Use this class for load pretrained Parallel SpeechCLIP for Zeroshot Spech-Text Retrieval
    Using the parallel branch only

    """

    def __init__(self, config: OrderedNamespace):
        config.retrieval.audio_feat_src = "parallel"
        super().__init__(config)
        assert config.model_settings.parallel_objective_weight > 0

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
        text_feat = input_feats["text_feat"].float()
        id = input_feats["id"]

        losses = {"loss": 0}
        if self.config.model_settings.cascaded_objective_weight > 0:
            losses["c_cl_loss"] = self.criterion(
                feat_A=cascaded_audio_feat,
                feat_B=text_feat,
                index=id,
            )
            losses["loss"] += (
                self.config.model_settings.cascaded_objective_weight
                * losses["c_cl_loss"]
            )

        if self.config.model_settings.parallel_objective_weight > 0:
            losses["p_cl_loss"] = self.criterion(
                feat_A=parallel_audio_feat,
                feat_B=text_feat,
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
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        id = batch["id"]
        text = batch["text"]

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        text_feat = self.forward_text(text.view(-1, 77))

        parallel_audio_feat = None
        vq_results = None
        keywords = None

        if self.parallel_branch is not None:
            parallel_audio_feat = self.parallel_branch(
                audio_feat=audio_feat,
                audio_len=audio_len,
            )
            if self.p_branch_proj_net is not None:
                parallel_audio_feat = self.p_branch_proj_net(parallel_audio_feat)
        else:
            logger.error("No parallel branch found")
            exit(1)

        losses = {
            "id": id,
            "text_feat": text_feat,
        }
        log_metrics = {}

        if parallel_audio_feat is not None:
            parallel_audio_feat = parallel_audio_feat / parallel_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["parallel_audio_feat"] = parallel_audio_feat

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
                "parallel_audio_feat": parallel_audio_feat,
                "text_feat": text_feat,
                "id": id,
                "vq_results": vq_results,
                "keywords": keywords,
            },
        )

    def validation_epoch_end(self, outputs: list) -> None:
        """validation_epoch_end

        Override for conducting Speech-Text Retrieval

        """
        all_text_feats = torch.cat([x["text_feat"] for x in outputs], dim=0)
        if "id" in outputs[0] and outputs[0]["id"] is not None:
            all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        else:
            all_ids = torch.arange(len(all_text_feats))

        all_audo_feats = torch.cat([x["audio_feat"] for x in outputs], dim=0)
        all_audo_feats_id = all_ids
        all_text_feats_id = all_ids

        print(
            "Total #{} text, #{} audio".format(len(all_text_feats), len(all_audo_feats))
        )
        assert len(all_text_feats) == len(all_audo_feats)

        # calculate dot product
        score_per_audio = torch.matmul(
            all_audo_feats.float(),
            all_text_feats.float().T,
        ).cpu()

        score_per_text = score_per_audio.T

        # AT : Audio -> Text, TA: Text -> Audio
        AT_answers = all_audo_feats_id
        TA_answers = all_text_feats_id

        self.reportRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_text,
            AB_answers=AT_answers,
            BA_answers=TA_answers,
        )

    def reportRetrieval(
        self,
        score_per_A: torch.Tensor,
        score_per_B: torch.Tensor,
        AB_answers: torch.Tensor,
        BA_answers: torch.Tensor,
    ):
        return super().reportRetrieval(
            score_per_A,
            score_per_B,
            AB_answers,
            BA_answers,
            {
                {
                    "modality_A_title": "audio",
                    "modality_B_title": "text",
                    "modality_A_logAbbr": "A",
                    "modality_B_logAbbr": "T",
                }
            },
        )


class KWClip_SpeechText(KWClipBase):
    """KWClip_SpeechText

    Train Speech-Text model using SSL model and CLIP Text Encoder

    """

    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        # exactly means that we do not serve the captions correspond to the same image as the same
        if self.config.retrieval.exactly:
            logger.warning("Retrieval = (Exactly)")
        self.parallel_branch = None
        if self.config.model_settings.parallel_objective_weight > 0:
            logger.info("Create Parallel Branch")
            self.parallel_branch = KW_ParallelBranch(
                config=self.config,
                audio_dim=self.audio_embd_dim,
                out_dim=self.subword_embd_dim,
            )

        self.img_enc_proj_net = None
        image_encoder_projection = self.config.model_settings.get(
            "image_encoder_projection", None
        )
        if image_encoder_projection is not None:
            logger.info(
                f"image_encoder_projection dims:{image_encoder_projection.dimensions} droupout:{image_encoder_projection.dropout}"
            )
            self.img_enc_proj_net = MLPLayers(
                units=image_encoder_projection.dimensions,
                dropout=image_encoder_projection.dropout,
            )

        self.p_branch_proj_net = None
        parallel_branch_projection = self.config.model_settings.get(
            "parallel_branch_projection", None
        )
        if parallel_branch_projection is not None:
            logger.info(
                f"parallel_branch_projection dims:{parallel_branch_projection.dimensions} droupout:{parallel_branch_projection.dropout}"
            )
            self.p_branch_proj_net = MLPLayers(
                units=parallel_branch_projection.dimensions,
                dropout=parallel_branch_projection.dropout,
            )

    def getTrainableParams(self):
        _params = super().getTrainableParams()
        if self.parallel_branch is not None:
            logger.info("Add parallel_branch parameters")
            _params += list(self.parallel_branch.parameters())

        if self.img_enc_proj_net is not None:
            logger.info("Add img_enc_proj_net parameters")
            _params += list(self.img_enc_proj_net.parameters())

        if self.p_branch_proj_net is not None:
            logger.info("Add parallel_branch_projection parameters")
            _params += list(self.p_branch_proj_net.parameters())

        return _params

    def compute_loss(self, input_feats: dict):
        """compute the loss here

        Args:
            input_feats (dict): the feats required for computing loss
        """
        assert isinstance(input_feats, dict)
        assert "id" in input_feats
        assert "parallel_audio_feat" in input_feats
        assert "text_feat" in input_feats

        parallel_audio_feat = (
            input_feats["parallel_audio_feat"].float()
            if "parallel_audio_feat" in input_feats
            else None
        )
        text_feat = input_feats["text_feat"].float()
        id = input_feats["id"]

        losses = {"loss": 0}
        if self.config.model_settings.parallel_objective_weight > 0:
            losses["p_cl_loss"] = self.criterion(
                feat_A=parallel_audio_feat,
                feat_B=text_feat,
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
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        text = batch["text"]
        id = batch["id"]

        if self.config.retrieval.exactly:
            id = None

        # update device information to clip model
        self.clip.update_device(self.device)

        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        text_feat = self.forward_text(text.view(-1, 77))

        if self.parallel_branch is not None:
            parallel_audio_feat = self.parallel_branch(
                audio_feat=audio_feat,
                audio_len=audio_len,
            )
            if self.p_branch_proj_net is not None:
                parallel_audio_feat = self.p_branch_proj_net(parallel_audio_feat)

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        losses = {
            "id": id,
            "text_feat": text_feat,
        }
        log_metrics = {}

        if parallel_audio_feat is not None:
            parallel_audio_feat = parallel_audio_feat / parallel_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["parallel_audio_feat"] = parallel_audio_feat

        log_metrics.update(
            {
                "cl_temp": self.criterion.current_temperature,
            }
        )
        return (
            losses,
            log_metrics,
            {
                "text_feat": text_feat,
                "parallel_audio_feat": parallel_audio_feat,
                "id": id,
            },
        )

    def validation_epoch_end(self, outputs):

        all_text_feats = torch.cat([x["text_feat"] for x in outputs], dim=0)
        if "id" in outputs[0] and outputs[0]["id"] is not None:
            all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        else:
            all_ids = torch.arange(len(all_text_feats))

        all_audo_feats = torch.cat([x["audio_feat"] for x in outputs], dim=0)
        all_audo_feats_id = all_ids
        all_text_feats_id = all_ids

        print(
            "Total #{} text, #{} audio".format(len(all_text_feats), len(all_audo_feats))
        )
        assert len(all_text_feats) == len(all_audo_feats)

        # calculate dot product
        score_per_audio = torch.matmul(
            all_audo_feats.float(),  # .to(self.device),
            all_text_feats.float().T,  # .to(self.device),
        ).cpu()
        # score_per_audio = score_per_audio
        score_per_text = score_per_audio.T

        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_text_feats_id

        self.reportRetrieval(
            score_per_audio=score_per_audio,
            score_per_image=score_per_text,
            AI_answers=AI_answers,
            IA_answers=IA_answers,
        )

    def reportRetrieval(
        self,
        score_per_A: torch.Tensor,
        score_per_B: torch.Tensor,
        AB_answers: torch.Tensor,
        BA_answers: torch.Tensor,
    ):
        return super().reportRetrieval(
            score_per_A,
            score_per_B,
            AB_answers,
            BA_answers,
            {
                "modality_A_title": "audio",
                "modality_B_title": "text",
                "modality_A_logAbbr": "A",
                "modality_B_logAbbr": "T",
            },
        )
