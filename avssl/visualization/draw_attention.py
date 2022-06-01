import json
import os

import librosa
import torch
from matplotlib import pyplot as plt

from ..model import KeywordCascadedSpeechClip, KeywordCascadedSpeechClip_ProjVQ_Cosine


class WordSegment:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word


def loadAudio(path):
    waveform, _ = librosa.load(path, sr=16_000)
    audio = torch.FloatTensor(waveform)
    return audio


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

mymodel = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(
    "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8_bsz_64_weightedSum/epoch=48-step=22931-val_recall_mean_1=6.1000.ckpt"
)

# mymodel = KeywordCascadedSpeechClip.load_from_checkpoint(
#     "/work/twsezjg982/atosystem/audio-visual-ssl/exp/kw_8kw_1head/epoch=29-step=9390-val_recall_mean_1=30.7600.ckpt"
# )

audio_fps = [
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_0.wav",
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_1.wav",
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_2.wav",
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_3.wav",
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence1/1000268201_693b08cb0e_4.wav",
]

forced_alignment_fps = [
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_0.json",
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_1.json",
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_2.json",
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_3.json",
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1000268201_693b08cb0e_4.json",
]

audio_fps = [
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_0.wav",
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_1.wav",
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_2.wav",
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_3.wav",
    "/work/twsezjg982/dataset/flickr/force_alignment/test_data_no_silence/1001773457_577c3a7d70_4.wav",
]

forced_alignment_fps = [
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_0.json",
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_1.json",
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_2.json",
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_3.json",
    "/work/twsezjg982/dataset/flickr/force_alignment/results_test_data_no_silence/1001773457_577c3a7d70_4.json",
]


def draw_plot(
    waveform, attention_weights, alignment_data, output_image_path, top1_kw_text=None
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
                ax.set_title(f"Kw#{kw_i}, Head#{head_i}, kw='{top1_kw_text[kw_i]}'")

    plt.savefig(output_image_path)
    plt.clf()


def process_file(audio_fp, forced_alignment_fp):
    mymodel.eval()
    audio_tensor = loadAudio(audio_fp)
    with open(forced_alignment_fp, "r") as f:
        alignment_data = json.load(f)
    with torch.no_grad():
        cls_weights, top1_kw_text = mymodel.get_attention_weights(wav=[audio_tensor])

    num_head, keyword_num = cls_weights[0].shape[:2]
    # (bsz,num_head, keyword_num ,source_L)

    print(torch.sum(cls_weights[0]))

    draw_plot(
        waveform=audio_tensor,
        attention_weights=cls_weights[0][:, :, keyword_num:],
        alignment_data=alignment_data,
        output_image_path="{}.png".format(
            os.path.basename(audio_fp).replace(".wav", "")
        ),
        top1_kw_text=top1_kw_text[0],
    )


for _audio_fp, _alignment_fp in zip(audio_fps, forced_alignment_fps):
    print(_audio_fp)
    process_file(_audio_fp, _alignment_fp)
