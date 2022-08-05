import json
import os

import clip
import numpy as np
import tqdm

DATASET_DIR = "<SpokenCOCO_root_dir>"


if __name__ == "__main__":
    with open(os.path.join(DATASET_DIR, "SpokenCOCO_train.json"), "r") as f:
        _data = json.load(f)["data"]
    captions = []
    for _entry in _data:
        for _cap in _entry["captions"]:
            captions.append(_cap["text"])

    with open(os.path.join(DATASET_DIR, "SpokenCOCO_val.json"), "r") as f:
        _data = json.load(f)["data"]
    for _entry in _data:
        for _cap in _entry["captions"]:
            captions.append(_cap["text"])

    lens = [len(x.strip().split()) for x in captions]
    print("max of lens:", max(lens))

    tokens = clip.tokenize(texts=captions, context_length=77, truncate=False)

    tokens = tokens.flatten().numpy()

    unique, counts = np.unique(tokens, return_counts=True)

    result_arr = np.asarray((unique, counts)).T

    print("Sort by frequencies")
    print(result_arr[result_arr[:, 1].argsort()[::-1]])
    np.savetxt(
        "text_clip_vocab_usage_byfreq.txt", result_arr[result_arr[:, 1].argsort()[::-1]]
    )
    np.save(
        "text_clip_vocab_usage_byfreq.npy", result_arr[result_arr[:, 1].argsort()[::-1]]
    )

    # np.save(
    #     "flickr_token_selected_idx_byfreq.npy",
    #     result_arr[result_arr[:, 1].argsort()[::-1]][:, 0],
    # )

    np.savetxt("text_clip_vocab_usage_byID.txt", result_arr[result_arr[:, 0].argsort()])
    np.save("text_clip_vocab_usage_byID.npy", result_arr[result_arr[:, 0].argsort()])

    print(result_arr.shape)
