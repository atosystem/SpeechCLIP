from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class MeanPoolingLayer(nn.Module):
    def __init__(
        self,
        in_dim: int = 0,
        out_dim: int = 0,
        bias: bool = True,
        pre_proj: bool = True,
        post_proj: bool = True,
    ):
        """Mean pooling layer with linear layers.

        Args:
            in_dim (int, optional): Input dimension. Defaults to 0.
            out_dim (int, optional): Output dimension. Defaults to 0.
            bias (bool, optional): Linear layer bias. Defaults to True.
            pre_proj (bool, optional): Pre-projection layer. Defaults to True.
            post_proj (bool, optional): Post-projection layer. Defaults to True.
        """
        super().__init__()

        self.pre_proj = None
        self.post_proj = None

        if in_dim > 0 and out_dim > 0:
            if pre_proj:
                self.pre_proj = nn.Linear(in_dim, out_dim, bias=bias)
            if post_proj:
                self.post_proj = nn.Linear(
                    in_dim if not pre_proj else out_dim, out_dim, bias=bias
                )

    def forward(self, x: torch.Tensor, x_len: torch.Tensor = None) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): Input features. (B, T, D)
            x_len (torch.Tensor): Feature lengths. (B, )

        Returns:
            torch.Tensor: Mean pooled features.
        """
        if self.pre_proj is not None:
            x = self.pre_proj(x)

        if x_len is not None:
            x = [x[b, : x_len[b]].mean(0) for b in range(len(x))]
            x = torch.stack(x, dim=0)
        else:
            x = x.mean(1)

        if self.post_proj is not None:
            x = self.post_proj(x)

        return x


class AttentivePoolingLayer(nn.Module):
    def __init__(self, dim_A: int, dim_B: int, degraded: bool = False) -> None:
        """Attentative Pooling

        Args:
            dim_A (int): dimension for modality A
            dim_B (int): dimension for modality B
        """
        super().__init__()

        self.dim_A = dim_A
        self.dim_B = dim_B
        self.degraded = degraded

        if not degraded:
            # learnable
            self.U = torch.nn.Parameter(torch.randn(self.dim_A, self.dim_B))
            self.U.requires_grad = True
        else:
            # not learnabble
            assert self.dim_A == self.dim_B
            self.U = torch.nn.Parameter(torch.eye(self.dim_A))
            self.U.requires_grad = False

        self.softmaxLayer = torch.nn.Softmax(dim=-1)

    def generate_input_msk(
        self,
        input_A_lens: torch.Tensor = None,
        input_B_lens: torch.Tensor = None,
        max_Alen: int = 1,
        max_Blen: int = 1,
    ) -> torch.Tensor:
        """Generate input mask for pooling

        Args:
            input_A_lens (torch.Tensor, optional): lengths for modality A, shape: (bsz,1). Defaults to None.
            input_B_lens (torch.Tensor, optional): lengths for modality B, shape: (bsz,1). Defaults to None.
            max_Alen (int): max input len for modality A
            max_Blen (int): max input len for modality B


        Returns:
            torch.Tensor: input mask, shape: ( bsz, max_Aseqlen , max_Bseqlen )
        """

        if input_A_lens is None and input_B_lens is None:
            raise ValueError("input_A_lens and input_B_lens cannot both be None")

        if input_A_lens is not None and input_B_lens is not None:
            assert (
                input_A_lens.shape[0] == input_B_lens.shape[0]
            ), "input_A_lens and input_B_lens must have same bsz, but got {} and {} instead".format(
                input_A_lens.shape[0], input_B_lens.shape[0]
            )

        if input_A_lens is not None:
            bsz = input_A_lens.shape[0]
            device = input_A_lens.device
        else:
            bsz = input_B_lens.shape[0]
            device = input_B_lens.device

        msk = torch.zeros((bsz, max_Alen, max_Blen), device=device, dtype=float)

        for _b in range(bsz):
            if input_A_lens is not None:
                assert (
                    not input_A_lens[_b] == 0
                ), "Modality A has 0 length on {}".format(_b)
                # assert not input_A_lens[_b] > max_Alen, "Modality A has length > max_Alen on {}, {}>{}".format(_b,input_A_lens[_b],max_Alen)

                msk[_b, input_A_lens[_b] :, :] = float("-inf")

            if input_B_lens is not None:
                assert (
                    not input_B_lens[_b] == 0
                ), "Modality B has 0 length on {}".format(_b)
                # assert not input_B_lens[_b] > max_Blen, "Modality B has length > max_Blen on {}, {}>{}".format(_b,input_B_lens[_b],max_Blen)

                msk[_b, :, input_B_lens[_b] :] = float("-inf")

        return msk

    def batch_forward(
        self,
        input_A: torch.Tensor,
        input_B: torch.Tensor,
        intput_msk: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Attentive poolilng with A and B (can be different in batch dimension)
        Assume modality B has fixed size

        Args:
            input_A (torch.Tensor): input features for modality A, shape: (bsz_A,dim,seq_len)
            input_B (torch.Tensor): input features for modality B, shape: (bsz_B,dim,seq_len)
            intput_msk (torch.Tensor,Optional): input features mask for modality A,B , shape: (bsz_A, seq_lenA, seq_lenB)

            mask: 0 for on and -inf for off
                if one of the dimension (seq_lenA or seq_lenB) has seq_len of 1, it will be auto broadcast to the input tensor shape

        Returns:
            Tuple[ torch.Tensor,torch.Tensor ]: (bsz_A,bsz_B,dimA), (bsz_A,bsz_B,dimB)
        """

        assert len(input_A.shape) == 3, "input_A.shape must be (bsz_A,dim,seq_len)"
        assert len(input_B.shape) == 3, "input_B.shape must be (bsz_B,dim,seq_len)"

        if intput_msk is not None:
            assert (
                input_A.shape[0] == intput_msk.shape[0]
            ), "input and intput_msk must have same bsz, but got {} and {} instead".format(
                input_A.shape[0], intput_msk.shape[0]
            )
            # repeat mask for modality A
            if intput_msk.shape[1] == 1:
                intput_msk = intput_msk.repeat(1, input_A.shape[1], 1)

            # repeat mask for modality B
            if intput_msk.shape[2] == 1:
                intput_msk = intput_msk.repeat(1, 1, input_B.shape[2])

        _align = torch.matmul(input_A.permute(0, 2, 1), self.U)
        _align = torch.einsum("acd,bdf->abcf", [_align, input_B])
        _align = torch.tanh(_align)

        # _align.shape: bsz_A, bsz_B, seq_len_A, seq_len_B

        # add mask on _align
        if intput_msk is not None:
            intput_msk = intput_msk.unsqueeze(1).repeat(1, _align.shape[1], 1, 1)
            assert _align.shape == intput_msk.shape, "{},{}".format(
                _align.shape, intput_msk.shape
            )
            intput_msk = intput_msk.to(_align.device)
            intput_msk = intput_msk.type_as(_align)

            _align = _align + intput_msk

        _align = _align.reshape(-1, input_A.shape[2], input_B.shape[2])

        _scoreA, _ = torch.max(_align, dim=2)
        _scoreB, _ = torch.max(_align, dim=1)

        del _align

        # _scoreA.shape: bsz_A*bsz_B, seq_len_B
        # _scoreB.shape: bsz_A*bsz_B, seq_len_A
        assert _scoreA.shape == (input_A.shape[0] * input_B.shape[0], input_A.shape[2])
        assert _scoreB.shape == (input_A.shape[0] * input_B.shape[0], input_B.shape[2])

        _scoreA = F.softmax(_scoreA, dim=-1)
        _scoreB = F.softmax(_scoreB, dim=-1)

        _scoreA = _scoreA.reshape(input_A.shape[0], input_B.shape[0], input_A.shape[2])
        _scoreB = _scoreB.reshape(input_A.shape[0], input_B.shape[0], input_B.shape[2])

        output_A = torch.matmul(
            input_A.unsqueeze(1).repeat(1, input_B.shape[0], 1, 1),
            _scoreA.unsqueeze(-1),
        )
        output_B = torch.matmul(
            input_B.unsqueeze(0).repeat(input_A.shape[0], 1, 1, 1),
            _scoreB.unsqueeze(-1),
        )

        del _scoreA, _scoreB

        output_A = output_A.reshape(
            input_A.shape[0], input_B.shape[0], input_A.shape[1]
        )
        output_B = output_B.reshape(
            input_A.shape[0], input_B.shape[0], input_B.shape[1]
        )

        # (bsz_A,bsz_B,dimA), (bsz_A,bsz_B,dimB)
        return output_A, output_B

    def cal_batch_embedding(
        self,
        input_A: torch.Tensor,
        input_B: torch.Tensor,
        intput_msk: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate Embedding in Batch

        Assume that instance in modality B is represented by one vector for each

        Args:
            input_A (torch.Tensor): input features for modality A, shape: (bsz,dim,seq_len)
            input_B (torch.Tensor): input features for modality B, shape: (dim, total_data_pairs_count)
                                    len_B is the total number of data pairs in the dataset

            intput_msk (torch.Tensor, optional): input features mask for modality A,B , shape: (bsz, seq_lenA, 1). Defaults to None.

        Returns:
            torch.Tensor,torch.Tensor ]: modelity A's pooled representation (B is omitted, since it is the same after attentive pooling)
        """

        assert len(input_A.shape) == 3, "input_A.shape must be (bsz,dim,seq_len)"
        assert (
            len(input_B.shape) == 2
        ), "input_B.shape must be (dim,total_data_pairs_count)"

        if intput_msk is not None:
            assert (
                input_A.shape[0] == intput_msk.shape[0]
            ), "input and intput_msk must have same bsz, but got {} and {} instead".format(
                input_A.shape[0], input_B.shape[0]
            )

        _align = torch.matmul(self.U, input_B)
        _align = torch.matmul(input_A.permute(0, 2, 1), _align)
        _align = torch.tanh(_align)

        # _align.shape: bsz, seq_len_A, #imageSamples
        assert _align.shape == (
            input_A.shape[0],
            input_A.shape[2],
            input_B.shape[1],
        ), "{} {}".format(
            _align.shape, (input_A.shape[0], input_A.shape[2], input_B.shape[1])
        )

        # add mask on _align
        if intput_msk is not None:
            assert _align.shape[:2] == intput_msk.shape[:2], "{},{}".format(
                _align.shape, intput_msk.shape
            )
            assert intput_msk.shape[2] == 1
            intput_msk = intput_msk.repeat(1, 1, _align.shape[2])

            intput_msk = intput_msk.to(_align.device)
            intput_msk = intput_msk.type_as(_align)

            _align = _align + intput_msk

        _score = F.softmax(_align, dim=1)
        output_A = torch.matmul(input_A, _score)

        assert output_A.shape == (
            input_A.shape[0],
            input_A.shape[1],
            input_B.shape[1],
        ), "{} {}".format(
            output_A.shape, (input_A.shape[0], input_A.shape[1], input_B.shape[1])
        )

        return output_A

    def forward(
        self,
        input_A: torch.Tensor,
        input_B: torch.Tensor,
        intput_msk: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input_A (torch.Tensor): input features for modality A, shape: (bsz,dim,seq_len)
            input_B (torch.Tensor): input features for modality B, shape: (bsz,dim,seq_len)
            intput_msk (torch.Tensor,Optional): input features mask for modality A,B , shape: (bsz, seq_lenA, seq_lenB)

            mask: 0 for on and -inf for off
                if one of the dimension (seq_lenA or seq_lenB) has seq_len of 1, it will be auto broadcast to the input tensor shape

        Returns:
            Tuple[ torch.Tensor,torch.Tensor ]: (bsz,dimA), (bsz,dimB)
        """

        assert len(input_A.shape) == 3, "input_A.shape must be (bsz,dim,seq_len)"
        assert len(input_B.shape) == 3, "input_B.shape must be (bsz,dim,seq_len)"

        assert (
            input_A.shape[0] == input_B.shape[0]
        ), "input_A and input_B must have same bsz, but got {} and {} instead".format(
            input_A.shape[0], input_B.shape[0]
        )

        if intput_msk is not None:
            assert (
                input_A.shape[0] == intput_msk.shape[0]
            ), "input and intput_msk must have same bsz, but got {} and {} instead".format(
                input_A.shape[0], input_B.shape[0]
            )
            # repeat mask for modality A
            if intput_msk.shape[1] == 1:
                intput_msk = intput_msk.repeat(1, input_A.shape[1], 1)

            # repeat mask for modality B
            if intput_msk.shape[2] == 1:
                intput_msk = intput_msk.repeat(1, 1, input_B.shape[2])

        _align = torch.matmul(input_A.permute(0, 2, 1), self.U)
        _align = torch.matmul(_align, input_B)
        _align = torch.tanh(_align)

        # _align.shape: bsz, seq_len_A, seq_len_B

        # add mask on _align
        if intput_msk is not None:
            assert _align.shape == intput_msk.shape, "{},{}".format(
                _align.shape, intput_msk.shape
            )
            intput_msk = intput_msk.to(_align.device)
            intput_msk = intput_msk.type_as(_align)

            _align = _align + intput_msk

        _scoreA, _ = torch.max(_align, dim=2)
        _scoreB, _ = torch.max(_align, dim=1)

        # _scoreA.shape: bsz, seq_len_B
        # _scoreB.shape: bsz, seq_len_A
        assert _scoreA.shape == (input_A.shape[0], input_A.shape[2])
        assert _scoreB.shape == (input_B.shape[0], input_B.shape[2])

        _scoreA = F.softmax(_scoreA, dim=-1)
        _scoreB = F.softmax(_scoreB, dim=-1)

        _scoreA = _scoreA.unsqueeze(-1)
        _scoreB = _scoreB.unsqueeze(-1)

        output_A = torch.matmul(input_A, _scoreA).squeeze()
        output_B = torch.matmul(input_B, _scoreB).squeeze()

        return output_A, output_B
