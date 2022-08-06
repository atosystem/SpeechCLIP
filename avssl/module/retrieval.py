from typing import Tuple

import torch


def mutualRetrieval(
    score_per_A: torch.Tensor,
    score_per_B: torch.Tensor,
    AB_answers: torch.Tensor,
    BA_answers: torch.Tensor,
    recall_at: list,
    modality_A_title: str = "audio",
    modality_B_title: str = "image",
) -> Tuple[dict, dict, dict]:
    """mutualRetrieval
    A to B and B to A retrieval


    Args:
        score_per_A (torch.Tensor): tensor shape = ( #modalityA_samples, #modalityB)
        score_per_B (torch.Tensor): tensor shape = ( #modalityB, #modalityA_samples)
        AB_answers (torch.Tensor): tensor shape = ( #modalityA_samples,) : list of the golden answer (pair ID) for each instance of madailty A
        BA_answers (torch.Tensor): tensor shape = ( #modalityB_samples,) : list of the golden answer (pair ID) for each instance of madailty B
        modality_A_title (str): the name for modality A
        modality_B_title (str): the name for modality B

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

    # AB : A -> B, BA: B -> A
    rank_AB = BA_answers.reshape(1, -1).repeat(AB_answers.shape[0], 1)
    rank_BA = AB_answers.reshape(1, -1).repeat(BA_answers.shape[0], 1)

    assert rank_AB.shape == score_per_A.shape, (
        rank_AB.shape,
        score_per_A.shape,
    )
    assert rank_BA.shape == score_per_B.shape, (
        rank_BA.shape,
        score_per_B.shape,
    )

    for r in range(AB_answers.shape[0]):
        rank_AB[r, :] = rank_AB[r, score_per_A[r, :]]

    for r in range(BA_answers.shape[0]):
        rank_BA[r, :] = rank_BA[r, score_per_B[r, :]]

    rank_AB = rank_AB == AB_answers.unsqueeze(-1)
    rank_BA = rank_BA == BA_answers.unsqueeze(-1)

    recall_results_AB = {}
    recall_results_BA = {}
    recall_results_mean = {}

    # AB (A to B)
    for k in recall_at:
        if k > rank_AB.shape[1]:
            print(
                "recall@{} is not eligible for #{} {} samples".format(
                    k, rank_AB.shape[1], modality_B_title
                )
            )
        recall_results_AB["recall@{}".format(k)] = (
            torch.sum(
                torch.max(
                    rank_AB[:, : min(k, rank_AB.shape[1])].reshape(
                        rank_AB.shape[0], min(k, rank_AB.shape[1])
                    ),
                    dim=1,
                    keepdim=True,
                )[0]
            )
            / rank_AB.shape[0]
        ).item()

    # BBA (B to A)
    for k in recall_at:
        if k > rank_BA.shape[1]:
            print(
                "recall@{} is not eligible for #{} {} samples".format(
                    k, rank_BA.shape[1], modality_A_title
                )
            )
        recall_results_BA["recall@{}".format(k)] = (
            torch.sum(
                torch.max(
                    rank_BA[:, : min(k, rank_BA.shape[1])].reshape(
                        rank_BA.shape[0], min(k, rank_BA.shape[1])
                    ),
                    dim=1,
                    keepdim=True,
                )[0]
            )
            / rank_BA.shape[0]
        ).item()

    for _k in ["recall@{}".format(r) for r in recall_at]:
        recall_results_BA[_k] *= 100
        recall_results_AB[_k] *= 100
        recall_results_mean[_k] = (recall_results_BA[_k] + recall_results_AB[_k]) / 2.0

    return recall_results_AB, recall_results_BA, recall_results_mean
