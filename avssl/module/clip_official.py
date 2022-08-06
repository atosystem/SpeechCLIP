import logging

logger = logging.getLogger(__name__)
import os
import string

import clip
import numpy as np
import torch
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image
from torch import nn

_clip_models = {
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
}


class ClipModel(nn.Module):
    def __init__(
        self,
        name: str,
        device: str = "cpu",
        image_encoder_trainable: bool = False,
        text_encoder_trainable: bool = False,
        reduce_subword_embbedding: str = None,
        **kwargs,
    ):
        """Official CLIP model.

        Args:
            name (str): Name of CLIP model.
            device (str, optional): Device. Defaults to "cpu".
            image_encoder_trainable (bool, optional): Whether to train the image encoder. Defaults to False.
            text_encoder_trainable (bool, optional): Whether to train the text encoder. Defaults to False.
            reduce_subword_embbedding (str, optional): The reduced vocabulary. Defaults to False
        """
        super().__init__()
        assert name in _clip_models
        self.name = name
        self.device = device

        self.model, self.image_preprocess = clip.load(name, device)

        self.image_encoder_trainable = image_encoder_trainable
        self.text_encoder_trainable = text_encoder_trainable

        self.out_dim = self.model.transformer.width

        self.tokenizer = SimpleTokenizer()

        self.freeze_models()

        self.selected_text_emb_ids = None
        if reduce_subword_embbedding is not None:
            if not os.path.exists(reduce_subword_embbedding):
                logger.error(f"File not found {reduce_subword_embbedding}")
                exit(1)

            _data = np.load(reduce_subword_embbedding)
            self.selected_text_emb_ids = _data[:, 0]
            self.selected_text_emb_ids_dist = _data[:, 1]
            self.selected_text_emb_ids_dist = torch.from_numpy(
                self.selected_text_emb_ids_dist
                / np.sum(self.selected_text_emb_ids_dist)
            )
            del _data
            logger.warning(
                "Reduce text embedding to size of {}".format(
                    len(self.selected_text_emb_ids)
                )
            )
            # use tensor to save original weights
            self.original_text_emb_weight = self.model.token_embedding.weight
            reduced_embedding_weight = self.model.token_embedding.weight[
                self.selected_text_emb_ids
            ]
            # reduced embedding
            self.model.token_embedding = nn.Embedding.from_pretrained(
                reduced_embedding_weight
            )
            if not self.text_encoder_trainable:
                self.model.token_embedding.weight.requires_grad = False
            self.original2Reduced = {
                old_id: _new_id
                for (_new_id, old_id) in enumerate(self.selected_text_emb_ids)
            }
            self.reducedl2Original = {
                _new_id: old_id
                for (_new_id, old_id) in enumerate(self.selected_text_emb_ids)
            }

            self.startOfTxt_reduced = self.original2Reduced[
                self.tokenizer.encoder["<|startoftext|>"]
            ]

            self.endOfTxt_reduced = self.original2Reduced[
                self.tokenizer.encoder["<|endoftext|>"]
            ]
        else:
            # use original CLIP Subword Embedding
            pass

    def freeze_models(self):
        """Freeze Models if required"""

        if not self.image_encoder_trainable:
            # freeze visual
            for p in self.model.visual.parameters():
                p.requires_grad = False

        if not self.text_encoder_trainable:
            for p in self.model.token_embedding.parameters():
                p.requires_grad = False

            self.model.positional_embedding.requires_grad = False

            for p in self.model.transformer.parameters():
                p.requires_grad = False

            for p in self.model.ln_final.parameters():
                p.requires_grad = False

            self.model.text_projection.requires_grad = False
            self.model.logit_scale.requires_grad = False

    def trainable_params(self) -> list:
        params = []
        if self.image_encoder_trainable:
            params += list(self.model.visual.parameters())
        if self.text_encoder_trainable:
            params += list(self.model.token_embedding.parameters())
            params += [self.model.positional_embedding]
            params += list(self.model.transformer.parameters())
            params += list(self.model.ln_final.parameters())
            params += [self.model.text_projection]

        return params

    def update_device(self, device):
        # since it is a pure nn.Module, it won't update itself
        self.device = device

    def prep_image(self, paths: list) -> torch.Tensor:
        """Prepare image tensor

        Args:
            paths (list): Paths to multiple images

        Returns:
            torch.Tensor: Preprocessed image tensor (B, 3, H, W)
        """
        image_list = []
        for p in paths:
            img = Image.open(p)
            image_list.append(self.image_preprocess(img))
        return torch.stack(image_list, dim=0).to(self.device)

    def prep_text(self, sents: list) -> torch.Tensor:
        """Tokenize text

        Args:
            sents (list): Sentences

        Returns:
            torch.Tensor: _description_
        """
        res = clip.tokenize(sents)
        if self.selected_text_emb_ids is not None:
            for sent in res:
                for i in range(len(sent)):
                    sent[i] = self.original2Reduced[sent[i].item()]
        return res

    def deTokenize(self, sents):
        if isinstance(sents, torch.Tensor):
            # print(sents.shape)
            sents = sents.view(*sents.shape[:2]).tolist()
        res = []
        for sent in sents:
            if self.selected_text_emb_ids is not None:
                for i in range(len(sent)):
                    sent[i] = self.reducedl2Original[sent[i]]
            res.append(
                self.tokenizer.decode(sent)
                .replace("<|startoftext|>", "")
                .replace("<|endoftext|>", "")
                .strip()
            )

        return res

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            image (torch.Tensor): Images. (B, 3, H, W)

        Returns:
            torch.Tensor: Image features. (B, D)
        """
        return self.model.encode_image(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sentences.
        Args:
            text (torch.Tensor): Sentences. (B, L)
        Returns:
            torch.Tensor: Text features. (B, D)
        """
        return self.model.encode_text(text)

    def encode_keywords(self, keywords: torch.Tensor, keyword_num: int) -> torch.Tensor:
        """encode_keywords

        Args:
            keywords (torch.Tensor): keywords input
            keyword_num (int): number of keywords

        Returns:
            torch.Tensor: output of CLIP Text Encoder
        """
        if isinstance(keywords, torch.Tensor):
            bsz = keywords.size(0)
        else:
            raise TypeError(f"Unknown keywords type {type(keywords)}")

        text = torch.zeros([bsz, 77], device=self.device, dtype=int)
        if self.selected_text_emb_ids is None:
            sot_token, eot_token = (
                self.tokenizer.encoder["<|startoftext|>"],
                self.tokenizer.encoder["<|endoftext|>"],
            )
        else:
            sot_token, eot_token = self.startOfTxt_reduced, self.endOfTxt_reduced

        text[:, 0] = torch.full(text[:, 0].size(), sot_token, device=self.device)
        text[:, keyword_num + 1] = torch.full(
            text[:, keyword_num + 1].size(), eot_token, device=self.device
        )

        x = self.model.token_embedding(text)
        x[:, 1 : 1 + keyword_num] = keywords
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection

        # take features from the eot embedding
        x = x[:, 1 + keyword_num] @ self.model.text_projection

        return x

    def encode_subword(
        self, prob: torch.Tensor, audio_len: torch.Tensor, vq_type: string
    ) -> torch.Tensor:
        """Encode a batch of subwords.

        Args:
            text (torch.Tensor): Sentences. (B, L)

        Returns:
            torch.Tensor: Text features. (B, D)
        """
        return self.encode_subword_prob(prob, audio_len, vq_type)

    def get_scores(self, image: torch.Tensor, text: torch.Tensor) -> tuple:
        """Get logit scores between the images and text sentences.

        Args:
            image (torch.Tensor): Images. (B_image, 3, H, W)
            text (torch.Tensor): Sentences. (B_text, L)

        Returns:
            tuple: (logits_per_image, logits_per_text) ((B_image, B_text), (B_text, B_image))
        """
        return self.model(image, text)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.model.token_embedding.weight.device
        return self
