from typing import Tuple

import torch


def mutualRetrieval(
    score_per_A: torch.Tensor,
    score_per_B: torch.Tensor,
    AB_answers: torch.Tensor,
    BA_answers: torch.Tensor,
    recall_at: list,
) -> Tuple[dict, dict, dict]:
    """mutualRetrieval
    A to B and B to A retrieval


    Args:
        score_per_A (torch.Tensor): tensor shape = (#modalityA_samples,#modalityAB)
        score_per_B (torch.Tensor): tensor shape = (#modalityAB,#modalityA_samples)
        AB_answers (torch.Tensor): tensor shape = (#modalityA_samples,)
        BA_answers (torch.Tensor): tensor shape = (#modalityAB

    Return:
        Tuple( dict, dict) : recall_results_AB, recall_results_BA, recall_results_mean
    """

    assert len(score_per_A.shape) == 2
    assert len(score_per_B.shape) == 2
    assert len(AB_answers.shape) == 1
    assert len(BA_answers.shape) == 1

    assert score_per_A.shape == (
        len(AB_answers),
        len(BA_answers),
    ), "{} , {}".format(score_per_A.shape, (len(AB_answers), len(BA_answers)))
    assert score_per_B.shape == (
        len(BA_answers),
        len(AB_answers),
    ), "{} , {}".format(score_per_B.shape, (len(BA_answers), len(AB_answers)))

    score_per_A = torch.argsort(score_per_A, dim=1, descending=True).cpu()
    score_per_B = torch.argsort(score_per_B, dim=1, descending=True).cpu()

    # AI : Audio -> Image, IA: Image -> Audio
    rank_AI = BA_answers.reshape(1, -1).repeat(AB_answers.shape[0], 1)
    rank_IA = AB_answers.reshape(1, -1).repeat(BA_answers.shape[0], 1)

    assert rank_AI.shape == score_per_A.shape, (
        rank_AI.shape,
        score_per_A.shape,
    )
    assert rank_IA.shape == score_per_B.shape, (
        rank_IA.shape,
        score_per_B.shape,
    )

    for r in range(AB_answers.shape[0]):
        rank_AI[r, :] = rank_AI[r, score_per_A[r, :]]

    for r in range(BA_answers.shape[0]):
        rank_IA[r, :] = rank_IA[r, score_per_B[r, :]]

    rank_AI = rank_AI == AB_answers.unsqueeze(-1)
    rank_IA = rank_IA == BA_answers.unsqueeze(-1)

    recall_results_AI = {}
    recall_results_IA = {}
    recall_results_mean = {}

    # AI (many to one)
    for k in recall_at:
        if k > rank_AI.shape[1]:
            print(
                "recall@{} is not eligible for #{} image samples".format(
                    k, rank_AI.shape[1]
                )
            )
        recall_results_AI["recall@{}".format(k)] = (
            torch.sum(
                torch.max(
                    rank_AI[:, : min(k, rank_AI.shape[1])].reshape(
                        rank_AI.shape[0], min(k, rank_AI.shape[1])
                    ),
                    dim=1,
                    keepdim=True,
                )[0]
            )
            / rank_AI.shape[0]
        ).item()
    recall_results_AI["recall_random@1"] = k / rank_AI.shape[0]

    # IA (one to many)
    for k in recall_at:
        if k > rank_IA.shape[1]:
            print(
                "recall@{} is not eligible for #{} audio samples".format(
                    k, rank_IA.shape[1]
                )
            )
        recall_results_IA["recall@{}".format(k)] = (
            torch.sum(
                torch.max(
                    rank_IA[:, : min(k, rank_IA.shape[1])].reshape(
                        rank_IA.shape[0], min(k, rank_IA.shape[1])
                    ),
                    dim=1,
                    keepdim=True,
                )[0]
            )
            / rank_IA.shape[0]
        ).item()
    # average one image corresponds to len(all_audo_feats) // len(all_img_feats) audio
    recall_results_IA["recall_random@1"] = 1
    _recall_at = 1
    for i in range(len(AB_answers) // len(BA_answers)):
        recall_results_IA["recall_random@1"] *= (len(AB_answers) - _recall_at - i) / (
            len(AB_answers) - i
        )

    recall_results_IA["recall_random@1"] = 1 - recall_results_IA["recall_random@1"]

    # convert to %
    recall_results_IA["recall_random@1"] *= 100
    recall_results_AI["recall_random@1"] *= 100

    for _k in ["recall@{}".format(r) for r in recall_at]:
        recall_results_IA[_k] *= 100
        recall_results_AI[_k] *= 100
        recall_results_mean[_k] = (recall_results_IA[_k] + recall_results_AI[_k]) / 2.0

    return recall_results_AI, recall_results_IA, recall_results_mean
