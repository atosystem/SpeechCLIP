import logging
import pickle
import string

import clip
import numpy as np
import torch
from clip.simple_tokenizer import SimpleTokenizer
from importlib_metadata import distribution
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
        reduce_subword_embbedding=None,
        **kwargs,
    ):
        """Official CLIP model.

        Args:
            name (str): Name of CLIP model.
            device (str, optional): Device. Defaults to "cpu".
            image_encoder_trainable (bool, optional): Whether to train the image encoder. Defaults to False.
            text_encoder_trainable (bool, optional): Whether to train the text encoder. Defaults to False.
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
            _data = np.load(reduce_subword_embbedding)
            self.selected_text_emb_ids = _data[:, 0]
            self.selected_text_emb_ids_dist = _data[:, 1]
            self.selected_text_emb_ids_dist = torch.from_numpy(
                self.selected_text_emb_ids_dist
                / np.sum(self.selected_text_emb_ids_dist)
            )
            del _data
            logging.warning(
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

            # delete original token embedding to save memory
            # del self.clip.model.token_embedding
            # self.clip.model.token_embedding = None
            # self.original_text_embs_weights = self.clip.model.token_embedding.weight.detach()
        else:
            # self.reduced_embedding_weight = None
            pass
        #     exit(1)

        # with open('./avssl/data/flickr_stat/token_mapping.p', 'rb') as fp:
        #     self.token_mapping = pickle.load(fp)
        # ids = torch.tensor( list(self.token_mapping.keys()) ).to(self.device)
        # self.used_text_embd_weight = self.model.token_embedding(ids).detach()

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
            print(sents.shape)
            sents = sents.view(*sents.shape[:2]).tolist()
        res = []
        if self.selected_text_emb_ids is not None:
            for sent in sents:
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

    def encode_subword_prob(
        self, result: dict, audio_len: torch.Tensor
    ) -> torch.Tensor:
        # start token embd = 49406, end token embd = 49407
        prob, idx = result["subword_prob"], result["targets"].squeeze(-1)
        bsz, seq_len, max_len = prob.size(0), prob.size(1), 75
        paddings = torch.zeros(bsz, max_len - seq_len).int().to(self.device)
        # assert paddings.device == self.model.token_embedding.weight.device, "{} {}".format(paddings.device , self.model.token_embedding.weight.device)

        # paddings, pad_embd_idx = [], self.token_mapping[0]
        # for i in range( bsz* (max_len - seq_len) ):
        #     paddings.append(self.used_text_embd_weight[pad_embd_idx])
        # paddings = torch.stack(paddings, dim=0)
        # paddings = paddings.view(bsz, (max_len - seq_len), -1)

        # if self.reduced_embedding_weight is not None:
        #     paddings = paddings @ self.reduced_embedding_weight
        #     weighted_embd = prob @ self.reduced_embedding_weight
        # else:
        paddings = self.model.token_embedding(paddings)
        weighted_embd = prob @ self.model.token_embedding.weight

        sot_idx, eot_idx = torch.tensor([49406]).unsqueeze(0).to(
            self.device
        ), torch.tensor([49407]).unsqueeze(0).to(self.device)
        sot = self.original_text_emb_weight[sot_idx]
        eot = self.original_text_emb_weight[eot_idx]

        new_idx, new_weighted_embd = [], []
        for len, i, embd in zip(audio_len, idx, weighted_embd):
            i, embd = i.unsqueeze(0), embd.unsqueeze(0)
            cat_i, cat_embd = [sot_idx, i[:len], eot_idx], [sot, embd[:len], eot]
            if i[len:].size(0) > 0:
                cat_i.append(i[len:])
            if embd[len:].size(0) > 0:
                cat_embd.append(embd[len:])
            new_idx.append(torch.cat(cat_i, dim=1))
            new_weighted_embd.append(torch.cat(cat_embd, dim=1))

        idx, weighted_embd = torch.cat(new_idx, dim=0), torch.cat(
            new_weighted_embd, dim=0
        )
        x = torch.cat((weighted_embd, paddings), dim=1)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        x = (
            x[
                torch.arange(x.shape[0]),
                torch.argmax((token_idx == self.endOfTxt_reduced).long(), dim=-1),
            ]
            @ self.model.text_projection
        )
        # x = x[torch.arange(x.shape[0]), idx.argmax(dim=-1)] @ self.model.text_projection
        return x

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sentences.
        Args:
            text (torch.Tensor): Sentences. (B, L)
        Returns:
            torch.Tensor: Text features. (B, D)
        """
        return self.model.encode_text(text)

    def encode_subword(
        self, prob: torch.Tensor, audio_len: torch.Tensor
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
        # if self.text_encoder_trainable and self.image_encoder_trainable:
        #     return self.model(image, text)
        # else:
        #     with torch.no_grad():
        #         return self.model(image, text)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.model.token_embedding.weight.device
        return self
