import torch
from avssl.data import random_crop_max_length


def test_audio_transform():
    wav = torch.randn((10000,), dtype=torch.float)

    out_1 = random_crop_max_length(wav, 1000)
    out_2 = random_crop_max_length(wav, 1000, 100)

    assert out_1.shape == (1000,)
    assert out_2.shape == (100,)
