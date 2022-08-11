import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SimpleVectorQuantizer"]


class SimpleVectorQuantizer(nn.Module):
    """SimpleVectorQuantizer"""

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

        # tenperature for vector quantizer
        if isinstance(temp, str):
            import ast

            if temp.startswith("learnable="):
                self.temp_type = "learnable"
                temp = temp.replace("learnable=", "")
                temp = ast.literal_eval(temp)
                self.curr_temp = nn.parameter.Parameter(torch.FloatTensor([temp]))
                logger.info("Setting vq temp learnable (init={})".format(temp))
            elif temp.startswith("fixed="):
                self.temp_type = "fixed"
                temp = temp.replace("fixed=", "")
                temp = ast.literal_eval(temp)
                self.register_buffer("curr_temp", torch.FloatTensor([temp]))
                logger.info("Setting vq temp fixed={}".format(temp))
            else:
                self.temp_type = "scheduled"
                temp = ast.literal_eval(temp)
                assert len(temp) == 3, f"{temp}, {len(temp)}"

                self.max_temp, self.min_temp, self.temp_decay = temp
                logger.info("Setting vq temp scheduled = ({},{},{})".format(*temp))
                self.curr_temp = self.max_temp
        self.codebook_indices = None

        self.groundTruthPerplexity = groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            self.perplexity_criteria = nn.MSELoss()

    def set_num_updates(self, num_updates):
        if self.temp_type == "scheduled":
            self.curr_temp = max(
                self.max_temp * self.temp_decay**num_updates, self.min_temp
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

        probs_per_t = (
            torch.softmax(x.view(bsz, tsz, -1), dim=-1).contiguous().permute(1, 0, 2)
        )
        # probs_per_t shape (tsz,bsz,num_vars)
        assert probs_per_t.shape[0] == tsz
        assert probs_per_t.shape[1] == bsz

        ent_per_t = -torch.sum(probs_per_t * torch.log(probs_per_t + 1e-9), dim=-1)
        # ent_per_t shape (tsz,bsz)
        ent_per_t = ent_per_t.mean(dim=-1)
        # ent_per_t shape (tsz,)
        del probs_per_t
        result["ent_per_t"] = ent_per_t

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
