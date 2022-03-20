import pytest
import torch
from avssl.module import S3prlSpeechEncoder

device = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_TOLERANCE_DISTANCE = 1e-5
MAX_TOLERANCE_FRAME_ERROR = 2


def check_distance(feat_1, feat_2):
    return (feat_1 - feat_2).abs().mean() < MAX_TOLERANCE_DISTANCE


def try_model(model, model_name):
    with torch.no_grad():
        wav_len = [80000 - 97 * i for i in range(8)]
        wav = [torch.randn(l, dtype=torch.float).to(device) for l in wav_len]
        feat_all, feat_len = model(wav, wav_len)
        feat_hid, _ = model(wav, wav_len, "hidden_states")
        max_hidden = len(feat_hid)
        feat_layers_2, _ = model(wav, wav_len, [2, max_hidden - 1])

        assert isinstance(feat_all, dict)
        assert isinstance(feat_hid, (tuple, list))
        assert isinstance(feat_hid[0], torch.Tensor)
        assert feat_hid[0].shape[0] == 8
        assert feat_hid[0].shape[-1] == model.out_dim
        assert check_distance(feat_layers_2[0], feat_hid[2])
        assert check_distance(feat_layers_2[1], feat_hid[max_hidden - 1])

        if model_name in {"apc"}:
            assert model.downsample_rate == 160
        if model_name in {"hubert", "distilhubert"}:
            assert model.downsample_rate == 320

        for h in range(len(feat_hid)):
            assert (
                abs(feat_hid[h].shape[1] - feat_len.max().item())
                <= MAX_TOLERANCE_FRAME_ERROR
            )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["apc", "hubert"])
def test_speech_encoder(name):
    # With pre-trained weights
    model = S3prlSpeechEncoder(name, pretrained=True, trainable=False, device=device)
    model.eval()
    try_model(model, name)
    del model

    # Re-initialize pre-trained model
    model = S3prlSpeechEncoder(name, pretrained=False, trainable=False, device=device)
    model.eval()
    try_model(model, name)
    del model
