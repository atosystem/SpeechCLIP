import torch
import numpy as np
from avssl.module import MeanPoolingLayer


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_pooling():
    with torch.no_grad():
        x = torch.arange(6, dtype=torch.float, device=device).reshape(1, 6, 1)
        x = x.expand(4, -1, 32)
        x_len = torch.LongTensor([6, 4, 2, 1]).to(device)

        assert x.shape == (4, 6, 32)

        # Test MeanPoolingLayer
        model = MeanPoolingLayer().to(device)
        out = model(x, x_len)
        out_red = out.mean(-1).cpu().tolist()
        out_gt = [2.5, 1.5, 0.5, 0.0]

        assert out.shape == (4, 32)
        assert np.isclose(out_red, out_gt).all()
        del model

        model = MeanPoolingLayer(32, 64).to(device)
        out = model(x)

        assert out.shape == (4, 64)
        del model
