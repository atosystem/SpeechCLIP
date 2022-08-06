import logging

logger = logging.getLogger(__name__)
import torch
from torch import nn


class Kw_BatchNorm(nn.Module):
    """Kw_BatchNorm

    BatchNorm Layer for keywords

    """

    def __init__(
        self,
        kw_num: int,
        kw_dim: int,
        batchnorm_type: str,
        init_bias: torch.Tensor,
        init_scale: torch.Tensor,
        std_scale: int = 1,
        learnable: bool = True,
        parallel: bool = False,
    ) -> None:
        """init

        Args:
            kw_num (int): number of keywords
            kw_dim (int): dimension of keywords
            batchnorm_type (str): type for BatchNorm (`eachKw`: each kw has it's own params or `same`: all kw shared the same params)
            init_bias (torch.Tensor): initialized bias for BatchNorm
            init_scale (torch.Tensor): initialized scale for BatchNorm
            std_scale (int, optional): scale for init scale. Defaults to 1.
            learnable (bool, optional): if gamma and beta is learnable in BatchNoem. Defaults to True.
            parallel (bool, optional): (in eachKw mode) if each kw is BatchNorm parallelly. Defaults to False.

        """
        super().__init__()
        self.batchnorm_type = batchnorm_type
        self.kw_num = kw_num
        self.kw_dim = kw_dim
        self.std_scale = std_scale
        self.learnable = learnable
        self.parallel = parallel

        if self.batchnorm_type == "eachKw":
            if self.parallel:
                self.bn_layer = nn.BatchNorm1d(kw_dim * self.kw_num)
            else:
                self.bn_layers = nn.ModuleList(
                    [nn.BatchNorm1d(kw_dim) for _ in range(self.kw_num)]
                )
        elif self.batchnorm_type == "same":
            self.bn_layer = nn.BatchNorm1d(kw_dim)
        else:
            raise NotImplementedError()

        if not isinstance(self.std_scale, list):
            self.std_scale = [self.std_scale] * self.kw_num

        self.init_bn(init_bias, init_scale)
        logger.info(
            "Initialize BatchNorm({}) weight and bias learnable=({}) with token embeddings w/ scale={}, parallel=({})".format(
                self.batchnorm_type, self.learnable, self.std_scale, self.parallel
            )
        )

    def init_bn(self, init_bias: torch.Tensor, init_scale: torch.Tensor) -> None:
        """init_bn
        Initialize batchnorm's params
        Args:
            init_bias (torch.Tensor): bias
            init_scale (torch.Tensor): scale
        """
        if self.batchnorm_type == "eachKw":
            if self.parallel:
                self.bn_layer.weight.data.copy_(
                    (init_scale * self.std_scale[0]).repeat(self.kw_num)
                )
                self.bn_layer.bias.data.copy_(init_bias.repeat(self.kw_num))
                self.bn_layer.weight.requires_grad = self.learnable
                self.bn_layer.bias.requires_grad = self.learnable
            else:
                for i, _bn_layer in enumerate(self.bn_layers):
                    _bn_layer.weight.data.copy_(init_scale * self.std_scale[i])
                    _bn_layer.bias.data.copy_(init_bias)
                    _bn_layer.weight.requires_grad = self.learnable
                    _bn_layer.bias.requires_grad = self.learnable

        elif self.batchnorm_type == "same":
            self.bn_layer.weight.data.copy_(init_scale * self.std_scale[0])
            self.bn_layer.bias.data.copy_(init_bias)
            self.bn_layer.weight.requires_grad = self.learnable
            self.bn_layer.bias.requires_grad = self.learnable

    def forward(
        self, keywords: torch.Tensor, seq_lens: torch.Tensor = None
    ) -> torch.Tensor:
        """forward

        Args:
            keywords (torch.Tensor): the input keywords embeddings
            seq_lens (torch.Tensor, optional): lengths for input keywords tensor. Defaults to None.

        Returns:
            torch.Tensor: batchnormed output
        """
        assert keywords.dim() == 3
        assert keywords.shape[2] == self.kw_dim
        if seq_lens is None:
            assert keywords.shape[1] == self.kw_num
        bsz = keywords.shape[0]

        if self.batchnorm_type == "eachKw":
            if self.parallel:
                keywords = keywords.permute(0, 2, 1)
                # (B,dim,kw_num)
                keywords = keywords.reshape(bsz, -1)

                keywords = self.bn_layer(keywords)
                keywords = keywords.reshape(bsz, self.kw_dim, self.kw_num)
                keywords = keywords.permute(0, 2, 1)
            else:
                keywords_bns = []
                for i in range(self.kw_num):
                    keywords_bns.append(
                        self.bn_layers[i](
                            # (B,#kw,D)
                            keywords[:, i]
                        )
                    )

                keywords = torch.stack(keywords_bns, dim=1)
                del keywords_bns
        elif self.batchnorm_type == "same":
            if seq_lens is None:
                keywords = self.bn_layer(keywords.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                kw_flatten = []
                assert seq_lens.dim() == 1

                seq_lens = seq_lens.tolist()

                offsets = [0]
                for b_i in range(keywords.size(0)):
                    kw_flatten.append(
                        keywords[b_i, : seq_lens[b_i]].view(seq_lens[b_i], -1)
                    )
                    offsets.append(offsets[-1] + seq_lens[b_i])
                kw_flatten = torch.cat(kw_flatten, dim=0)
                assert kw_flatten.size(0) == sum(seq_lens)
                # kw_flatten shape [#total kws, kw_dim]
                kw_flatten = self.bn_layer(kw_flatten)

                for b_i, (st_i, ed_i) in enumerate(zip(offsets[:-1], offsets[1:])):
                    assert seq_lens[b_i] == ed_i - st_i
                    keywords[b_i, : seq_lens[b_i]] = kw_flatten[st_i:ed_i]

                # keywords = self.bn_layer(keywords.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise NotImplementedError()

        return keywords
