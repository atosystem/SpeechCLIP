import numpy as np
import torch


def random_crop_max_length(
    audio: torch.Tensor, max_len: int, orig_len: int = 1000000000
) -> torch.Tensor:
    """Randomly crop an audio feature sequence into max_len.

    Args:
        audio (torch.Tensor): Audio features (T, *)
        max_len (int): Maximum length.
        orig_len (int, optional): Original length of audio. Defaults to 1000000000.

    Returns:
        torch.Tensor: Cropped audio features.
    """
    audio_len = min(len(audio), orig_len)
    if audio_len <= max_len or max_len < 0:
        return audio[:audio_len]

    offset = np.random.randint(audio_len - max_len)
    return audio[offset : offset + max_len]
