import torch
import numpy as np
from avssl.module import MeanPoolingLayer, AttentativePoolingLayer


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


def test_attentative_pooling_msk():
    with torch.no_grad():
        BATCH_SZ = 4
        modA_seqLen = 16
        modB_seqLen = 32
        modA_dim = 3
        modB_dim = 5

        # test both modality given
        modA_seq_lens = torch.tensor([x for x in range(1, BATCH_SZ + 1)]).unsqueeze(-1)
        modB_seq_lens = torch.tensor([x for x in range(2, BATCH_SZ + 2)]).unsqueeze(-1)

        # Test AttentativePoolingLayer
        model = AttentativePoolingLayer(
            dim_A=modA_dim,
            dim_B=modB_dim,
        ).to(device)

        mask = model.generate_input_msk(
            input_A_lens=modA_seq_lens,
            input_B_lens=modB_seq_lens,
            max_Alen=modA_seqLen,
            max_Blen=modB_seqLen,
        )

        # test mask
        assert mask.shape == (BATCH_SZ, modA_seqLen, modB_seqLen)

        for _b in range(BATCH_SZ):
            assert mask[_b, 0, 0].item() == 0
            assert torch.sum((mask[_b, 0, :] == 0)) == modB_seq_lens[_b]
            assert torch.sum((mask[_b, :, 0] == 0)) == modA_seq_lens[_b]

            assert torch.sum(
                (mask[_b, 0, modB_seq_lens[_b] :] == float("-inf"))
            ) == max(0, modB_seqLen - modB_seq_lens[_b])
            assert torch.sum(
                (mask[_b, modA_seq_lens[_b] :, 0] == float("-inf"))
            ) == max(0, modA_seqLen - modA_seq_lens[_b])

        del mask

        # test both modality A given, B not given
        modA_seq_lens = torch.tensor([x for x in range(1, BATCH_SZ + 1)]).unsqueeze(-1)

        # Test AttentativePoolingLayer
        model = AttentativePoolingLayer(
            dim_A=modA_dim,
            dim_B=modB_dim,
        ).to(device)

        mask = model.generate_input_msk(
            input_A_lens=modA_seq_lens,
            max_Alen=modA_seqLen,
        )

        # test mask
        assert mask.shape == (BATCH_SZ, modA_seqLen, 1)

        for _b in range(BATCH_SZ):
            assert mask[_b, 0, 0].item() == 0
            assert torch.sum((mask[_b, :, 0] == 0)) == modA_seq_lens[_b]
            assert torch.sum(
                (mask[_b, modA_seq_lens[_b] :, 0] == float("-inf"))
            ) == max(0, modA_seqLen - modA_seq_lens[_b])

        del model
        del mask


def test_attentative_pooling():
    with torch.no_grad():
        BATCH_SZ = 4
        modA_seqLen = 16
        modB_seqLen = 32
        modA_dim = 3
        modB_dim = 5

        modA_seq_lens = torch.tensor([x for x in range(1, BATCH_SZ + 1)]).unsqueeze(-1)
        modB_seq_lens = torch.tensor([x for x in range(2, BATCH_SZ + 2)]).unsqueeze(-1)

        tensor_A = torch.rand(
            (BATCH_SZ, modA_dim, modA_seqLen), dtype=torch.float, device=device
        )
        tensor_B = torch.rand(
            (BATCH_SZ, modB_dim, modB_seqLen), dtype=torch.float, device=device
        )

        # Test AttentativePoolingLayer
        model = AttentativePoolingLayer(
            dim_A=modA_dim,
            dim_B=modB_dim,
        ).to(device)

        mask = model.generate_input_msk(
            input_A_lens=modA_seq_lens,
            input_B_lens=modB_seq_lens,
            max_Alen=modA_seqLen,
            max_Blen=modB_seqLen,
        )

        out_A, out_B = model(tensor_A, tensor_B, intput_msk=mask)

        assert out_A.shape == (BATCH_SZ, modA_dim)
        assert out_B.shape == (BATCH_SZ, modB_dim)

        del mask
        del tensor_B

        mask = model.generate_input_msk(
            input_A_lens=modA_seq_lens,
            max_Alen=modA_seqLen,
        )

        tensor_B = torch.rand((BATCH_SZ, modB_dim, 1), dtype=torch.float, device=device)

        out_A, out_B = model(tensor_A, tensor_B, intput_msk=mask)

        assert out_A.shape == (BATCH_SZ, modA_dim)
        assert out_B.shape == (BATCH_SZ, modB_dim)

        del mask
        del model
