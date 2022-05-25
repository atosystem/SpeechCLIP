import logging

import torch
from torch import nn


class Kw_BatchNorm(nn.Module):
    def __init__(
        self,
        kw_num,
        kw_dim,
        batchnorm_type,
        init_bias,
        init_scale,
        std_scale=1,
        learnable=True,
        parallel=False,
    ):
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
        logging.warning(
            "Initialize BatchNorm({}) weight and bias learnable=({}) with token embeddings w/ scale={}, parallel=({})".format(
                self.batchnorm_type, self.learnable, self.std_scale, self.parallel
            )
        )

    def init_bn(self, init_bias, init_scale):
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

    def forward(self, keywords):
        assert keywords.dim() == 3
        assert keywords.shape[1] == self.kw_num
        assert keywords.shape[2] == self.kw_dim
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
            keywords = self.bn_layer(keywords.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise NotImplementedError()

        return keywords