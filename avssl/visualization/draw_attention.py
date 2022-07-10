import json
import os
import pickle

import librosa
import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

from ..model import (
    KeywordCascadedSpeechClip,
    KeywordCascadedSpeechClip_ProjVQ_Cosine,
    KWClip_GeneralTransformer,
)


class WordSegment:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word


def loadAudio(path):
    waveform, _ = librosa.load(path, sr=16_000)
    audio = torch.FloatTensor(waveform)
    return audio


"""
# mymodel = KeywordCascadedSpeechClip.load_from_checkpoint("/work/twsezjg982/atosytem/audio-visual-ssl/exp/kw_lr_1e-4_heads_4_keyword_1/epoch=87-step=27544-val_recall_mean_1=28.0200.ckpt")

# mymodel = KeywordCascadedSpeechClip.load_from_checkpoint("/work/twsezjg982/atosytem/audio-visual-ssl/exp/kw_lr_1e-4_heads_8_keyword_1/epoch=76-step=24101-val_recall_mean_1=29.4100.ckpt")

# mymodel = KeywordCascadedSpeechClip.load_from_checkpoint(
#     "/home/twsezjg982/kw4_head8/epoch=30-step=9703-val_recall_mean_1=27.9100.ckpt"
# )

# mymodel = KeywordCascadedSpeechClip.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_lr_1e-4_heads_8_keyword_1/epoch=76-step=24101-val_recall_mean_1=29.4100.ckpt"
# )


# mymodel = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8/epoch=161-step=2591-val_recall_mean_1=5.2600.ckpt"
# )

# mymodel = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8_bsz_64_weightedSum/epoch=48-step=22931-val_recall_mean_1=6.1000.ckpt"
# )
# mymodel = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/cosineVq_kw_8_bsz_64_weightedSum_vqTemp_fixed_0.1/epoch=46-step=21995-val_recall_mean_1=6.7500.ckpt"
# )

# mymodel = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/cosineVq_kw_8_bsz_64_weightedSum_fixedtmp_0.1_div_per_frame_0.5/epoch=100-step=47267-val_recall_mean_1=5.6400.ckpt"
# )

# mymodel = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/cosineVq_kw_8_bsz_64_weightedSum_fixedtmp_0.1_posEmb_50k/epoch=99-step=46799-val_recall_mean_1=9.9800.ckpt"
# )
# mymodel = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/cosineVq_kw_8_bsz_64_weightedSum_fixedtmp_0.1_div_per_kw_0.5_frame_0.5/epoch=81-step=38375-val_recall_mean_1=4.7200.ckpt"
# )
# mymodel = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/cosineVq_kw_8_bsz_64_weightedSum_fixedtmp_0.1_div_per_kw_0.5/epoch=98-step=46331-val_recall_mean_1=7.3100.ckpt"
# )

# mymodel = KeywordCascadedSpeechClip.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_8kw_1head/epoch=29-step=9390-val_recall_mean_1=30.7600.ckpt"
# )
"""
mymodel = KWClip_GeneralTransformer.load_from_checkpoint(
    "/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_integrated_c1.0_p1.0_L14_HuLarge_1024_1024_768_s3prlNormed/epoch=17-step=39869-val_recall_mean_10=36.6634.ckpt"
    # "/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_cm_L14_HuLarge_1024_1024_768_s3prlNormed/epoch=12-step=28794-val_recall_mean_10=36.1455.ckpt"
)


forced_alignment_fps = [
    "m007089117jn22tdu1n4f-3RKNTXVS3MYN3YATNC3CUY8JHYDA48_146855_777697.json",
    "m071506418gb9vo0w5xq3-3R3YRB5GRF3MKW482IUU726DDJBUA2_20254_138769.json",
    "m101ooe1o3phic-338JKRMM26ZHFAYO2JNVONHNKUAHAS_329806_801148.json",
    "m101ooe1o3phic-37TD41K0AH9UYY92XRVLYWEDT8DCSB_542388_562226.json",
    "m101ooe1o3phic-3GGAI1SQEVYR03WXDUZZMNF71JOMC2_376372_738274.json",
    "m142hce9axjif8-39RP059MEHT8QY4RZU34B39PJMMBMU_329806_805261.json",
    "m15u8m5wo61bch-3OS46CRSLFZLA9H5HVNNNJ6AVBF6V6_146855_777748.json",
    "m1hmx7x9x79uz0-3EG49X351UCDEYM0HAO5B43VBNY6XM_542388_563294.json",
    "m1yqqk4y9sb8nj-3NLZY2D53PPBOMOVI0ZF6FPZ8B9LQH_542388_555551.json",
    "m253dwe1ylfsrg-3C8HJ7UOP7U48W7758J7XLGSA7YMZ1_146855_778486.json",
    "m26qs044w4aqo0-3X65QVEQI0N7ULECDIYKMA0I4YNCLM_20254_829250.json",
    "m2f7cvoresy1mn-324G5B4FB38OL8UUX84Q84GG0BE079_376372_827928.json",
    "m2ogmlp7avprkd-39L1G8WVWQR6REEPRF509SNAZ6O319_376372_828192.json",
    "m3eqbvx6tg3uix-3AQF3RZ558IWECVFHGIJ5X3O7IT6FM_376372_826795.json",
    "m3g9fiffe9vbh5-39K0FND3AHF37OZZTG38GSPJ4TIMA8_376372_826513.json",
    "m3mu2kfdnj4txh-3XIQGXAUMC8WIY050KTXHI3R70G7XP_329806_805648.json",
    "m3o81ftkkenshp-3HFNH7HEMHEV2UAU792ZTNKR9ZLQGL_20254_134131.json",
    "m3ww5rws3x0c66-352YTHGROVD2DLI7TID4BKVNV4Y4HV_329806_801097.json",
    "ma2y8v1t30alr-3E47SOBEYQW54K66SF17UU8QNHGICB_146855_782320.json",
    "me5j6sdpvwis3-3RKNTXVS3MYN3YATNC3CUY8JEB7A4P_146855_785122.json",
    "mmrmd49dz5fe7-3D3VGR7TA0FUKJD6P9KFFJ5N477R3P_20254_128848.json",
    "mo0p4oczywrvw-3RSDURM96AM6RI5PSOT5662HYBWEY9_542388_558623.json",
    "mr2dhrdsb7sjj-3WI0P0II61SS2BF3IYQRSSRZSCNRDV_329806_804010.json",
    "ms7s6lss73vb7-3IXQG4FA2TYTJELXALZZ6H35ILM9BV_20254_136153.json",
    "mwyct2ysfgia9-3OCHAWUVGOKKDD7PJEI6LN8DF84XKN_542388_556310.json",
]

audio_fps = [x.replace(".json", ".wav") for x in forced_alignment_fps]
forced_alignment_fps = [
    f"/home/twsezjg982/dataset/COCO_KSplit_test/force_align/result_force_aligned_trimed/{x}"
    for x in forced_alignment_fps
]
audio_fps = [
    f"/home/twsezjg982/dataset/COCO_KSplit_test/wavs_trim/{x}" for x in audio_fps
]

# forced_alignment_fps = [f"/home/twsezjg982/dataset/COCO_KSplit_test/force_align/result_force_aligned/{x}" for x in forced_alignment_fps]
# audio_fps = [f"/home/twsezjg982/dataset/COCO_KSplit_test/wavs/{x}" for x in audio_fps]

"""
audio_fps = [
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-0.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-1.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-2.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-3.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-4.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-5.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-6.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-7.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-8.wav",
    "/work/twsezjg982/dataset/SNIPS/test/Aditi-snips-test-9.wav",
]

forced_alignment_fps = [
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-0.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-1.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-2.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-3.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-4.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-5.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-6.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-7.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-8.json",
    "/work/twsezjg982/dataset/SNIPS/force_alignment/results/Aditi-snips-test-9.json",
]


# audio_fps = [
#     "/work/twsezjg982/dataset/flickr/flickr_audio/wavs_with_no_silence/756004341_1a816df714_0.wav",
#     "/work/twsezjg982/dataset/flickr/flickr_audio/wavs_with_no_silence/756004341_1a816df714_1.wav",
#     "/work/twsezjg982/dataset/flickr/flickr_audio/wavs_with_no_silence/756004341_1a816df714_2.wav",
#     "/work/twsezjg982/dataset/flickr/flickr_audio/wavs_with_no_silence/756004341_1a816df714_3.wav",
#     "/work/twsezjg982/dataset/flickr/flickr_audio/wavs_with_no_silence/756004341_1a816df714_4.wav",
# ]

# forced_alignment_fps = [
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_no_silence/756004341_1a816df714_0.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_no_silence/756004341_1a816df714_1.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_no_silence/756004341_1a816df714_2.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_no_silence/756004341_1a816df714_3.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_no_silence/756004341_1a816df714_4.json",
# ]
# audio_fps = [
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_0.wav",
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_1.wav",
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_2.wav",
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_3.wav",
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_4.wav",
# ]

# forced_alignment_fps = [
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_0.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_1.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_2.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_3.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_4.json",
# ]

# audio_fps = [
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_0.wav",
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_1.wav",
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_2.wav",
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_3.wav",
#     "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_4.wav",
# ]

# forced_alignment_fps = [
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_0.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_1.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_2.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_3.json",
#     "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_4.json",
# ]
"""


def draw_plot_old(
    waveform,
    attention_weights,
    alignment_data,
    output_image_path,
    top1_kw_text=None,
    ent_per_Kw=None,
):
    # attention_weights:  num_heads, keyword_num, src_len (audio_len)
    n_heads = attention_weights.shape[0]
    n_keywords = attention_weights.shape[1]
    # print(attention_weights.shape)
    # scaling
    attention_weights = (
        attention_weights
        / torch.max(attention_weights, dim=-1, keepdim=True).values
        * 0.8
    )

    print("attention_weights length", len(attention_weights))
    print("wavform length", len(waveform))

    word_alignments = list(
        filter(lambda x: x["name"] == "words", alignment_data["tiers"])
    )[0]

    word_segments = list(map(lambda x: WordSegment(*x), word_alignments["entries"]))

    print(f"nheads={n_heads}, nkw={n_keywords}")

    fig, axs = plt.subplots(
        nrows=n_heads,
        ncols=n_keywords,
        figsize=(15 * n_keywords, 3 * n_heads + 0.5 * (n_heads - 1)),
        sharex=True,
    )

    all_ent_over_time = np.zeros(n_keywords)
    all_sth_over_time = np.zeros(n_keywords)
    all_codeEnt = np.zeros(n_keywords)

    for head_i in range(n_heads):
        if n_heads == 1:
            ax_row = axs
        else:
            ax_row = axs[head_i]
        for kw_i in range(n_keywords):
            if n_keywords == 1:
                ax = ax_row
            else:
                ax = ax_row[kw_i]
            ax.plot(waveform)
            ratio = waveform.size(0) / alignment_data["xmax"]

            for word in word_segments:
                x0 = ratio * word.start
                x1 = ratio * word.end
                if not word.word == "":
                    ax.axvspan(x0, x1, alpha=1, color="red", fill=False)
                    ax.annotate(f"{word.word}", (x0, 0.8))
                # ax.annotate(f"{word.score:.2f}", (x0, 0.8))

            # print("word x1",x1)
            for _x, _weight in enumerate(attention_weights[head_i, kw_i]):
                x0 = 320 * _x
                x1 = 320 * (_x + 1)

                ax.axvspan(x0, x1, alpha=min(1.0, _weight.item()), color="green")
                # ax.annotate(f"{_weight:.2f}", (x0, 0.8))
                # ax.annotate(f"{word.score:.2f}", (x0, 0.8))
            normed_prob = attention_weights[head_i, kw_i]
            normed_prob = normed_prob / torch.sum(normed_prob)
            ent = -torch.mean(normed_prob * torch.log(normed_prob))

            smoothness = F.mse_loss(normed_prob[:-1], normed_prob[1:]).item()
            all_ent_over_time[kw_i] = ent.item()
            all_sth_over_time[kw_i] = smoothness
            all_codeEnt[kw_i] = ent_per_Kw[kw_i].item()
            # print(len(waveform))
            # print("weight x1",x1)
            # exit(1)
            # for seg in segments:
            #     if seg.label != "|":
            # ax.annotate(seg.label, (seg.start * ratio, 0.9))
            xticks = ax.get_xticks()
            plt.xticks(xticks, xticks / 16_000)
            ax.set_xlabel("time [second]")
            ax.set_yticks([])
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlim(0, waveform.size(-1))
            if top1_kw_text is None:
                ax.set_title(f"Kw#{kw_i}, Head#{head_i}")
            else:
                if not ent_per_Kw is None:
                    ax.set_title(
                        f"Kw#{kw_i}, Head#{head_i}, kw='{top1_kw_text[kw_i]}', ent={ent.item():2f}, codeEnt={ent_per_Kw[kw_i].item():2f}, sth:{smoothness:2f}"
                    )
                else:
                    ax.set_title(
                        f"Kw#{kw_i}, Head#{head_i}, kw='{top1_kw_text[kw_i]}', ent={ent.item():2f}"
                    )
            description_text = ""
            description_text += "min(ent): kw#{}\n".format(np.argmin(all_ent_over_time))
            description_text += "max(ent): kw#{}\n".format(np.argmax(all_sth_over_time))
            description_text += "min(ent): kw#{}\n".format(np.argmin(all_codeEnt))
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # # place a text box in upper left in axes coords
            # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            #         verticalalignment='top', bbox=props)

    plt.text(0.5, 0.5, description_text, fontsize=14, horizontalalignment="right")
    plt.savefig(output_image_path)
    plt.clf()


def draw_plot(
    waveform,
    attention_weights,
    alignment_data,
    output_image_path,
    top_kw_text=None,
    ent_per_Kw=None,
):
    # print(top_kw_text)
    # exit(1)
    # attention_weights:  num_heads, keyword_num, src_len (audio_len)
    n_heads = attention_weights.shape[0]
    n_keywords = attention_weights.shape[1]
    # print(attention_weights.shape)
    # scaling
    # attention_weights = torch.softmax(attention_weights / 0.05,dim=-1)
    attention_weights = (
        attention_weights
        / torch.max(attention_weights, dim=-1, keepdim=True).values
        * 1
    )
    attention_weights = attention_weights.squeeze()
    # print(attention_weights.shape)
    max_att_kw = torch.argmax(attention_weights, dim=0)
    for i in range(len(max_att_kw) - 2):
        if max_att_kw[i] == max_att_kw[i + 2] and max_att_kw[i] != max_att_kw[i + 1]:
            max_att_kw[i + 1] = max_att_kw[i]
    # print(max_att_kw)

    print("attention_weights length", len(attention_weights))
    print("wavform length", len(waveform))

    word_alignments = list(
        filter(lambda x: x["name"] == "words", alignment_data["tiers"])
    )[0]

    word_segments = list(map(lambda x: WordSegment(*x), word_alignments["entries"]))
    word_list = [x.word for x in word_segments]

    print(f"nheads={n_heads}, nkw={n_keywords}")
    mpl.style.use("default")
    ratio = waveform.size(0) / alignment_data["xmax"]

    fig, axs = plt.subplots(
        nrows=n_keywords + 1,
        ncols=2,
        figsize=(15 * 2, 0.4 * (n_keywords + 1) + 0.001 * (n_keywords + 1 - 1)),
        sharex=True,
    )
    for kw_i in range(n_keywords + 1):
        ax_row = axs[kw_i]
        ax_row[1].axis("off")
        # ax_row[1].get_xaxis().set_visible(False)
        # ax_row[1].get_yaxis().set_visible(False)
        # ax_row[1].spines['top'].set_visible(False)
        # ax_row[1].spines['right'].set_visible(False)
        # ax_row[1].spines['bottom'].set_visible(False)
        # ax_row[1].spines['left'].set_visible(False)
    # plt.draw()
    for kw_i in range(n_keywords + 1):
        if n_keywords == 1:
            ax_row = axs
        else:
            ax_row = axs[kw_i]
        ax = ax_row[0]
        # ax_row[1].get_xaxis().set_visible(False)
        # ax_row[1].get_yaxis().set_visible(False)
        # ax_row[1].spines['top'].set_visible(False)
        # ax_row[1].spines['right'].set_visible(False)
        # ax_row[1].spines['bottom'].set_visible(False)
        # ax_row[1].spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if kw_i == 0:
            ax.plot(waveform, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            for word in word_segments:
                x0 = ratio * word.start
                x1 = ratio * word.end
                if not word.word == "":
                    # ax.axvspan(x0, x1, alpha=1, color="red", fill=False)
                    ax.axvline(x0, color="red", linestyle="dashed", lw=0.5)
                    ax.axvline(x1, color="red", linestyle="dashed", lw=0.5)
                    # print(len(word.word))
                    # print((x0+x1)/2)

                    # ax.annotate(f"{word.word}", ((x0+x1)/2 - len(word.word) * 350, 0))
                    ax.annotate(
                        f"{word.word}",
                        ((x0 + x1) / 2, 0),
                        ha="center",
                        va="center",
                        fontsize=14,
                    )
            # ax.annotate(f"{word.score:.2f}", (x0, 0.8))
            continue
        else:
            ax.text(
                0.1,
                (1 / (n_keywords + 1) - 0.024) * (n_keywords + 1 - kw_i) + 0.045,
                f"kw#{ kw_i-1}",
                transform=plt.gcf().transFigure,
                fontsize=15,
            )
            ax_row[1].text(
                0.48,
                (1 / (n_keywords + 1) - 0.024) * (n_keywords + 1 - kw_i) + 0.045,
                ", ".join(
                    [
                        r"$\bf{" + x + "}$" if x in word_list else x
                        for x in top_kw_text[kw_i - 1][:6]
                    ]
                ),
                transform=plt.gcf().transFigure,
                zorder=1000,
                fontsize=15,
            )
            kw_i -= 1
            for word in word_segments:
                x0 = ratio * word.start
                x1 = ratio * word.end
                if not word.word == "":
                    ax.axvline(x0, color="red", linestyle="dashed", lw=0.5)
                    ax.axvline(x1, color="red", linestyle="dashed", lw=0.5)
                    # ax.axvspan(x0, x1, alpha=1, color="red", fill=False)
                    # print(len(word.word))
                    # print((x0+x1)/2)
                    # ax.annotate(f"{word.word}", ((x0+x1)/2 - len(word.word) * 350, 0.5))

        # print("word x1",x1)
        for _x, _weight in enumerate(attention_weights[kw_i]):
            x0 = 320 * _x
            x1 = 320 * (_x + 1)
            _weight = _weight.item()
            # print(attention_weights[_kw_i,_x].item())
            # if attention_weights[_kw_i,_x].item() > 0.001:
            if _weight > 0.001:
                # _weight = attention_weights[_kw_i,_x].item()
                ax.axvspan(x0, x1, alpha=min(1.0, _weight), color=f"C{kw_i}")

    # all_ent_over_time = np.zeros(n_keywords)
    # all_sth_over_time = np.zeros(n_keywords)
    # all_codeEnt = np.zeros(n_keywords)

    # for head_i in range(n_heads):
    #     if n_heads == 1:
    #         ax_row = axs
    #     else:
    #         ax_row = axs[head_i]
    #     for kw_i in range(n_keywords):
    #         if n_keywords == 1:
    #             ax = ax_row
    #         else:
    #             ax = ax_row[kw_i]
    #         ax.plot(waveform)
    #         ratio = waveform.size(0) / alignment_data["xmax"]

    #         for word in word_segments:
    #             x0 = ratio * word.start
    #             x1 = ratio * word.end
    #             if not word.word == "":
    #                 ax.axvspan(x0, x1, alpha=1, color="red", fill=False)
    #                 ax.annotate(f"{word.word}", (x0, 0.8))
    #             # ax.annotate(f"{word.score:.2f}", (x0, 0.8))

    #         # print("word x1",x1)
    #         for _x, _weight in enumerate(attention_weights[head_i, kw_i]):
    #             x0 = 320 * _x
    #             x1 = 320 * (_x + 1)

    #             ax.axvspan(x0, x1, alpha=min(1.0, _weight.item()), color="green")
    #             # ax.annotate(f"{_weight:.2f}", (x0, 0.8))
    #             # ax.annotate(f"{word.score:.2f}", (x0, 0.8))
    #         normed_prob = attention_weights[head_i, kw_i]
    #         normed_prob = normed_prob / torch.sum(normed_prob)
    #         ent = -torch.mean(normed_prob * torch.log(normed_prob))

    #         smoothness = F.mse_loss(normed_prob[:-1], normed_prob[1:]).item()
    #         all_ent_over_time[kw_i] = ent.item()
    #         all_sth_over_time[kw_i] = smoothness
    #         all_codeEnt[kw_i] = ent_per_Kw[kw_i].item()
    #         # print(len(waveform))
    #         # print("weight x1",x1)
    #         # exit(1)
    #         # for seg in segments:
    #         #     if seg.label != "|":
    #         # ax.annotate(seg.label, (seg.start * ratio, 0.9))
    #         xticks = ax.get_xticks()
    #         plt.xticks(xticks, xticks / 16_000)
    #         ax.set_xlabel("time [second]")
    #         ax.set_yticks([])
    #         ax.set_ylim(-1.0, 1.0)
    #         ax.set_xlim(0, waveform.size(-1))
    #         if top1_kw_text is None:
    #             ax.set_title(f"Kw#{kw_i}, Head#{head_i}")
    #         else:
    #             if not ent_per_Kw is None:
    #                 ax.set_title(
    #                     f"Kw#{kw_i}, Head#{head_i}, kw='{top1_kw_text[kw_i]}', ent={ent.item():2f}, codeEnt={ent_per_Kw[kw_i].item():2f}, sth:{smoothness:2f}"
    #                 )
    #             else:
    #                 ax.set_title(
    #                     f"Kw#{kw_i}, Head#{head_i}, kw='{top1_kw_text[kw_i]}', ent={ent.item():2f}"
    #                 )
    #         description_text = ""
    #         description_text += "min(ent): kw#{}\n".format(np.argmin(all_ent_over_time))
    #         description_text += "max(ent): kw#{}\n".format(np.argmax(all_sth_over_time))
    #         description_text += "min(ent): kw#{}\n".format(np.argmin(all_codeEnt))
    #         # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    #         # # place a text box in upper left in axes coords
    #         # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    #         #         verticalalignment='top', bbox=props)

    # plt.text(0.5, 0.5, description_text, fontsize=14, horizontalalignment="right")
    # from matplotlib.patches import Patch
    # legend_elements = [  Patch(edgecolor=f'C{i}', label=f'keyword#{i}',
    #                             facecolor=f'C{i}') for i in range(n_keywords)]
    # # for i in range(n_keywords):
    # #     legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
    # #                     Line2D([0], [0], marker='o', color='w', label=f'keyword#{}',
    # #                             markerfacecolor='g', markersize=15),
    # #                     Patch(facecolor='orange', edgecolor='r',
    # #                             label='Color Patch')]

    # ax.legend(handles=legend_elements, loc='right')

    plt.xlim([0, waveform.size(0)])
    # plt.tight_layout()
    plt.tight_layout(pad=10)

    # plt.subplots_adjust(right=.25)
    # plt.subplots_adjust(left=.25)
    plt.savefig(output_image_path)
    plt.clf()


def process_file(audio_fp, forced_alignment_fp):
    mymodel.eval()
    audio_tensor = loadAudio(audio_fp)
    with open(forced_alignment_fp, "r") as f:
        alignment_data = json.load(f)
    # with open("tmp_pkl.pkl","wb") as f:
    #     with torch.no_grad():
    #         x = mymodel.get_attention_weights(
    #             wav=[audio_tensor]
    #         )
    #         pickle.dump(x,f)
    #         print(x[1])
    #         exit(1)

    with torch.no_grad():
        cls_weights, top1_kw_text, ent_per_Kw = mymodel.get_attention_weights(
            wav=[audio_tensor]
        )
    # with open("tmp_pkl.pkl","rb") as f:
    #     cls_weights, top1_kw_text, ent_per_Kw = pickle.load(f)

    num_head, keyword_num = cls_weights[0].shape[:2]
    # (bsz,num_head, keyword_num ,source_L)

    print(torch.sum(cls_weights[0]))

    draw_plot(
        waveform=audio_tensor,
        attention_weights=cls_weights[0][:, :, keyword_num:],
        alignment_data=alignment_data,
        output_image_path="{}.pdf".format(
            os.path.basename(audio_fp).replace(".wav", "")
        ),
        top_kw_text=top1_kw_text[0],
        ent_per_Kw=ent_per_Kw,
    )
    # exit(1)


for _audio_fp, _alignment_fp in zip(audio_fps, forced_alignment_fps):
    print(_audio_fp)
    process_file(_audio_fp, _alignment_fp)
