import clip
import torch
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

        self.model = self.model.float()

        self.image_encoder_trainable = image_encoder_trainable
        self.text_encoder_trainable = text_encoder_trainable

        self.out_dim = self.model.transformer.width
        self.text_embd = self.model.token_embedding

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
        return clip.tokenize(sents).to(self.device)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            image (torch.Tensor): Images. (B, 3, H, W)

        Returns:
            torch.Tensor: Image features. (B, D)
        """
        if self.image_encoder_trainable:
            return self.model.encode_image(image)
        else:
            with torch.no_grad():
                return self.model.encode_image(image)

    def encode_subword_prob(self, result: dict) -> torch.Tensor:
        # start token embd = 49406, end token embd = 49407
        # self.model.to(self.device)
        # prob, self.text_embd = prob.to(self.device), self.text_embd.half()
        prob, idx = result["subword_prob"], result["targets"].squeeze(-1)
        bsz, seq_len, max_len = prob.size(0), prob.size(1), 77
        paddings = torch.zeros(bsz, max_len - seq_len).int().to(self.device)
        weighted_embd = prob @ self.text_embd.weight
        x = torch.cat( (weighted_embd, self.text_embd(paddings)), dim=1 ) # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), idx.argmax(dim=-1)] @ self.model.text_projection 
        return x

    def encode_text(self, prob: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sentences.

        Args:
            text (torch.Tensor): Sentences. (B, L)

        Returns:
            torch.Tensor: Text features. (B, D)
        """
        if self.text_encoder_trainable:
            # return self.model.encode_text(text)
            return self.encode_subword_prob(prob)
        else:
            with torch.no_grad():
                # return self.model.encode_text(text)
                return self.encode_subword_prob(prob)

    def get_scores(self, image: torch.Tensor, text: torch.Tensor) -> tuple:
        """Get logit scores between the images and text sentences.

        Args:
            image (torch.Tensor): Images. (B_image, 3, H, W)
            text (torch.Tensor): Sentences. (B_text, L)

        Returns:
            tuple: (logits_per_image, logits_per_text) ((B_image, B_text), (B_text, B_image))
        """
        if self.text_encoder_trainable and self.image_encoder_trainable:
            return self.model(image, text)
        else:
            with torch.no_grad():
                return self.model(image, text)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.model.token_embedding.weight.device
        return self
