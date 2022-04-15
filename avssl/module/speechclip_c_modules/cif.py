from turtle import forward

import numpy
import torch
from torch import nn


class AlphaNetwork(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.feat_dim,
                out_channels=self.feat_dim,
                kernel_size=3,
                stride=1,
                padding="same",
                dilation=1,
                groups=1,
                bias=True,
            ),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.feat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        assert x.shape[2] == self.feat_dim
        x = x.permute(0, 2, 1)
        # x (bsz, hid_dim, seq_len,)
        x = self.conv_layer(x)
        x = x.permute(0, 2, 1)
        # x (bsz, seq_len, hid_dim)
        x = self.output_layer(x)
        return x


class CIF(nn.Module):
    def __init__(
        self,
        audio_feat_dim,
        beta=1.0,
        scaling_stragety=False,
        cal_quantity_loss=False,
        tail_handling=False,
    ):
        super().__init__()
        self.audio_feat_dim = audio_feat_dim

        # threshold (default=1.0)
        self.beta = beta
        self.weight_network = AlphaNetwork(feat_dim=self.audio_feat_dim)

        self.scaling_stragety = scaling_stragety
        self.cal_quantity_loss = cal_quantity_loss
        self.tail_handling = tail_handling

        if self.cal_quantity_loss:
            self.quantity_loss_criteria = nn.L1Loss()

    def forward(
        self, encoder_outputs, encoder_lens=None, target_length=None, paddingTensor=None
    ):
        device = encoder_outputs.device
        # encoder_outputs = (bsz,seq_len,hid_dim)
        assert encoder_outputs.shape[2] == self.audio_feat_dim
        bsz, seq_len = encoder_outputs.shape[:2]

        if encoder_lens is not None:
            assert encoder_lens.shape[0] == bsz

        alphas = self.weight_network(encoder_outputs)
        assert alphas.shape == (bsz, seq_len, 1), alphas.shape

        if self.training:
            # training
            if target_length is not None and self.scaling_stragety:
                # scaling strategy
                assert target_length.shape[0] == bsz
                target_length = target_length.view(bsz, 1)
                # alphas = encoder_lens.view(bsz,1).repeat(1,seq_len) * alphas
                alphas = (
                    alphas
                    / torch.sum(alphas.view(bsz, seq_len), dim=-1, keepdim=True)
                    * target_length
                )

        if self.cal_quantity_loss:
            assert (
                target_length is not None
            ), "target_length cannot be None to calculate quantity_loss"
            quantity_loss = self.quantity_loss_criteria(
                torch.sum(alphas.view(bsz, seq_len), dim=-1), target_length
            )

        output_c = []
        max_c_len = 0
        for bsz_i in range(bsz):
            alpha = alphas[bsz_i, :, :]
            h = encoder_outputs[bsz_i, :, :]
            alpha_accum = [torch.zeros((1,)).to(device)]
            h_accum = [torch.zeros((self.audio_feat_dim,)).to(device)]
            c = []

            assert alpha.shape == (seq_len, 1)
            assert h.shape == (seq_len, self.audio_feat_dim)

            for u in range(1, seq_len + 1):
                if encoder_lens is not None:
                    if u > encoder_lens[bsz_i]:
                        break
                # u : current timestep (start from 1 to seq_len)
                alpha_u = alpha[u - 1]
                h_u = h[u - 1, :]

                # alpha^a_u = alpha^a_(u-1) + alpha_u
                alpha_accum_u = alpha_accum[u - 1] + alpha_u
                alpha_accum.append(alpha_accum_u)

                assert len(alpha_accum)

                if alpha_accum_u < self.beta:
                    # no boundary is located
                    # h^a_u = h^a_(u-1) + alpha_u * h_u
                    h_accum.append(h_accum[u - 1] + alpha_u * h_u)
                else:
                    # boundart located
                    # divide alpha into 2 parts : alpha_u1 and alpha_u2

                    alpha_u1 = 1 - alpha_accum[u - 1]

                    c.append(h_accum[u - 1] + alpha_u1 * h_u)

                    alpha_u2 = alpha_u - alpha_u1
                    alpha_accum_u = alpha_u2
                    alpha_accum[-1] = alpha_accum_u
                    h_accum.append(alpha_u2 * h_u)

            if self.tail_handling and not self.training:
                # add additional firing if residual weight > 0.5
                if alpha_accum[-1] > 0.5:
                    c.append(h_accum[-1])

            c = torch.stack(c)
            max_c_len = max(max_c_len, c.shape[0])
            output_c.append(c)

        if paddingTensor is None:
            paddingTensor = torch.zeros(
                self.audio_feat_dim,
            ).to(device)

        len_tensor = []
        for i in range(len(output_c)):
            assert max_c_len >= output_c[i].shape[0], "{} {}".format(
                max_c_len, output_c[i].shape[0]
            )
            len_tensor.append((output_c[i].shape[0]))
            output_c[i] = torch.cat(
                [
                    output_c[i],
                    paddingTensor.view(1, self.audio_feat_dim).repeat(
                        max_c_len - output_c[i].shape[0], 1
                    ),
                ],
                dim=0,
            )
            assert output_c[i].shape == (max_c_len, self.audio_feat_dim)

        output_c = torch.stack(output_c, dim=0)

        assert output_c.shape == (bsz, max_c_len, self.audio_feat_dim)
        len_tensor = torch.tensor(len_tensor).to(device)

        if self.cal_quantity_loss:
            return output_c, len_tensor, quantity_loss
        else:
            return output_c, len_tensor


if __name__ == "__main__":
    bsz = 8
    audio_dim = 512
    seq_len = 70
    cif = CIF(
        audio_feat_dim=512,
        beta=1.0,
        scaling_stragety=False,
        cal_quantity_loss=True,
        tail_handling=False,
    )

    audio_input = torch.randn(bsz, seq_len, audio_dim)
    audio_input_lens = torch.randint(1, seq_len, (bsz,))

    audio_input = audio_input.cuda()
    audio_input_lens = audio_input_lens.cuda()
    cif = cif.cuda()

    output_c, q_loss = cif(
        encoder_outputs=audio_input, encoder_lens=None, target_length=audio_input_lens
    )

    q_loss.backward()

    print(q_loss)
    for x in output_c:
        print(x.shape)
