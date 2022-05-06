# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
        init_codebook=None,
        groundTruthPerplexity=None,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.time_first = time_first
        self.num_vars = num_vars

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        if init_codebook is not None:
            if isinstance(init_codebook, torch.Tensor):
                # init self.vars with init_codebook
                vq_dim = init_codebook.size(-1)
                num_vars = init_codebook.size(0)
                self.vars = nn.Parameter(
                    init_codebook.view(1, num_groups * num_vars, var_dim)
                )
            elif init_codebook == 0:
                # no codebook needed
                self.vars = None
            else:
                raise NotImplementedError()
        else:
            self.vars = nn.Parameter(
                torch.FloatTensor(1, num_groups * num_vars, var_dim)
            )
            nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

        self.groundTruthPerplexity = groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            self.perplexity_criteria = nn.MSELoss()

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars ** self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
            n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        # x.shape = (bsz, tsz, grps * num_vars)

        x = x.view(bsz * tsz * self.groups, -1)
        # x.shape = (bsz * tsz * grps, num_vars)

        # k is the indices of the largest logits among num_vars
        _, k = x.max(-1)

        # hard_x: one hot for the choosen codewords ( bsz * tsz, self.groups, num_vars )
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )

        # hard_probs: probs for all codewords in each codebook group : (grp, num_vars) (use one-hot as prob)
        hard_probs = torch.mean(hard_x.float(), dim=0)

        # codebook perplexity sigma {e^(entropy for codebook group)} for all codebook groups
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        # average over minibatch and all timesteps of the codewords logits and get their prob by softmax (grp, num_vars)
        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)

        # prob_cpx : probs for all codewords in each codebook group : (grp, num_vars) (use softmax as prob)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)
        # x (bsz * tsz, group * num_vars)

        # add gumbel softmax hard target
        result["subword_prob"] = x.view(bsz, tsz, -1)

        # if groundTruthPerplexity is given, minimized the l2 norm with groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            result["loss"] = (
                self.perplexity_criteria(
                    result["prob_perplexity"],
                    torch.tensor(self.groundTruthPerplexity).type_as(x),
                )
                / (result["num_vars"] - self.groundTruthPerplexity) ** 2
            )
        else:
            result["loss"] = (result["num_vars"] - result["prob_perplexity"]) / result[
                "num_vars"
            ]

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        vars = self.vars
        if vars is not None:
            # calculate the following only if codebook exists
            if self.combine_groups:
                # codebook groups shared same set of parameters
                vars = vars.repeat(1, self.groups, 1)

            x = x.unsqueeze(-1) * vars
            # print(x.dtype)
            x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
            x = x.sum(-2).type_as(x)
            x = x.view(bsz, tsz, -1)

            if not self.time_first:
                x = x.transpose(1, 2)  # BTC -> BCT

            result["x"] = x

        return result


class SimpleVectorQuantizer(nn.Module):
    def __init__(
        self,
        temp,
        groundTruthPerplexity=None,
        time_first=True,
        use_gumbel=False,
        hard=True,
    ):
        super().__init__()
        self.time_first = time_first
        self.use_gumbel = use_gumbel
        self.hard = hard

        if isinstance(temp, str):
            import ast

            if temp.startswith("learnable="):
                self.temp_type = "learnable"
                temp = temp.replace("learnable=", "")
                temp = ast.literal_eval(temp)
                self.curr_temp = nn.parameter.Parameter(torch.FloatTensor([temp]))
                logging.warning("Setting vq temp learnable (init={})".format(temp))
            elif temp.startswith("fixed="):
                self.temp_type = "fixed"
                temp = temp.replace("fixed=", "")
                temp = ast.literal_eval(temp)
                self.curr_temp = torch.FloatTensor([temp])
                logging.warning("Setting vq temp fixed={}".format(temp))
            else:
                self.temp_type = "scheduled"
                temp = ast.literal_eval(temp)
                assert len(temp) == 3, f"{temp}, {len(temp)}"

                self.max_temp, self.min_temp, self.temp_decay = temp
                logging.warning("Setting vq temp scheduled = ({},{},{})".format(*temp))
                self.curr_temp = self.max_temp
        self.codebook_indices = None

        self.groundTruthPerplexity = groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            self.perplexity_criteria = nn.MSELoss()

    def set_num_updates(self, num_updates):
        if self.temp_type == "scheduled":
            self.curr_temp = max(
                self.max_temp * self.temp_decay ** num_updates, self.min_temp
            )

    def forward(self, x, prob_msk=[0, 2, 3], produce_targets=True):

        if not self.time_first:
            x = x.transpose(1, 2)

        result = {"num_vars": x.shape[-1]}
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        # x.shape = (bsz, tsz, grps * num_vars)

        x = x.view(bsz * tsz * 1, -1)
        # x.shape = (bsz * tsz * grps, num_vars)

        # exclude special token
        for i in prob_msk:
            x[:, i] += float("-inf")

        # k is the indices of the largest logits among num_vars
        _, k = x.max(-1)

        # hard_x: one hot for the choosen codewords ( bsz * tsz, 1, num_vars )
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, 1, -1)
        )

        hard_x = hard_x.squeeze()

        # hard_probs: probs for all codewords in each codebook group : (grp, num_vars) (use one-hot as prob)
        hard_probs = torch.mean(hard_x.float(), dim=0)

        # codebook perplexity sigma {e^(entropy for codebook group)} for all codebook groups
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        # average over minibatch and all timesteps of the codewords logits and get their prob by softmax (grp, num_vars)
        avg_probs = torch.softmax(x.view(bsz * tsz, 1, -1).float(), dim=-1).mean(dim=0)

        # prob_cpx : probs for all codewords in each codebook group : (grp, num_vars) (use softmax as prob)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp.item()
        if self.training:
            if self.use_gumbel:
                x = F.gumbel_softmax(
                    x.float(), tau=self.curr_temp, hard=self.hard
                ).type_as(x)
            else:
                x = x / self.curr_temp
                x = F.softmax(x, dim=-1).type_as(x)
                if self.hard:
                    # print(x.shape)
                    # print(hard_x.shape)
                    # exit(1)
                    x = hard_x + x - x.detach()

        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)
        # x (bsz * tsz, group * num_vars)

        # add gumbel softmax hard target
        result["subword_prob"] = x.view(bsz, tsz, -1)
        # if groundTruthPerplexity is given, minimized the l2 norm with groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            result["diversity_loss"] = (
                self.perplexity_criteria(
                    result["prob_perplexity"],
                    torch.tensor(self.groundTruthPerplexity).type_as(x),
                )
                / (result["num_vars"] - self.groundTruthPerplexity) ** 2
            )
        else:
            result["diversity_loss"] = (
                result["num_vars"] - result["prob_perplexity"]
            ) / result["num_vars"]

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * 1, -1).argmax(dim=-1).view(bsz, tsz, 1).detach()
            )

        return result
