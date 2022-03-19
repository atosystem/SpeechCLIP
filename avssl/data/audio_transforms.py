import torch
import numpy as np


def random_crop_max_length(
    audio: torch.Tensor, max_len: int, orig_len: int = 1000000000
) -> torch.Tensor:
    audio_len = min(len(audio), orig_len)
    if audio_len < max_len or max_len < 0:
        return audio

    offset = np.random.randint(audio_len - max_len)
    return audio[offset : offset + max_len]
