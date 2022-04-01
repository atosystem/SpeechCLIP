from typing import Tuple

import torch


def audioImageRetrieval(
    score_per_audio: torch.Tensor,
    score_per_image: torch.Tensor,
    AI_answers: torch.Tensor,
    IA_answers: torch.Tensor,
    recall_at: list,
) -> Tuple[dict, dict]:
    """Audio Image Retrieval Code

    Args:
        score_per_audio (torch.Tensor): tensor shape = (#audioSamples,#imageSamples)
        score_per_image (torch.Tensor): tensor shape = (#imageSamples,#audioSamples)
        AI_answers (torch.Tensor): tensor shape = (#audioSamples,)
        IA_answers (torch.Tensor): tensor shape = (#imageSamples,)
        recall_at (list): recall at ... ex : [1,5,10]

    Return:
        Tuple( dict, dict) : recall_results_AI, recall_results_IA
    """

    assert len(score_per_audio.shape) == 2
    assert len(score_per_image.shape) == 2
    assert len(AI_answers.shape) == 1
    assert len(IA_answers.shape) == 1

    assert score_per_audio.shape == (
        len(AI_answers),
        len(IA_answers),
    ), "{} , {}".format(score_per_audio.shape, (len(AI_answers), len(IA_answers)))
    assert score_per_image.shape == (
        len(IA_answers),
        len(AI_answers),
    ), "{} , {}".format(score_per_image.shape, (len(IA_answers), len(AI_answers)))

    score_per_audio = torch.argsort(score_per_audio, dim=1, descending=True).cpu()
    score_per_image = torch.argsort(score_per_image, dim=1, descending=True).cpu()

    # AI : Audio -> Image, IA: Image -> Audio
    rank_AI = IA_answers.reshape(1, -1).repeat(AI_answers.shape[0], 1)
    rank_IA = AI_answers.reshape(1, -1).repeat(IA_answers.shape[0], 1)

    assert rank_AI.shape == score_per_audio.shape, (
        rank_AI.shape,
        score_per_audio.shape,
    )
    assert rank_IA.shape == score_per_image.shape, (
        rank_IA.shape,
        score_per_image.shape,
    )

    for r in range(AI_answers.shape[0]):
        rank_AI[r, :] = rank_AI[r, score_per_audio[r, :]]

    for r in range(IA_answers.shape[0]):
        rank_IA[r, :] = rank_IA[r, score_per_image[r, :]]

    rank_AI = rank_AI == AI_answers.unsqueeze(-1)
    rank_IA = rank_IA == IA_answers.unsqueeze(-1)

    recall_results_AI = {}
    recall_results_IA = {}
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
        )
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
        )
    # average one image corresponds to len(all_audo_feats) // len(all_img_feats) audio
    recall_results_IA["recall_random@1"] = 1
    _recall_at = 1
    for i in range(len(AI_answers) // len(IA_answers)):
        recall_results_IA["recall_random@1"] *= (len(AI_answers) - _recall_at - i) / (
            len(AI_answers) - i
        )

    recall_results_IA["recall_random@1"] = 1 - recall_results_IA["recall_random@1"]

    return recall_results_AI, recall_results_IA
