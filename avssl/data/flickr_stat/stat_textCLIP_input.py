import os
import re

import clip
import numpy as np
import tqdm

DATASET_DIR = "<dataset_dir_of_Flickr8k>"

# TXT_FILE = "Flickr8k.lemma.token.txt"
TXT_FILE = "Flickr8k.token.txt"
# TXT_FILE = "captions.txt"

if __name__ == "__main__":
    with open(os.path.join(DATASET_DIR, TXT_FILE), "r") as f:
        _data = f.readlines()

    captions = []
    for i, _line in enumerate(tqdm.tqdm(_data)):
        _line = _line.strip()
        if i == "image,caption":
            continue
        _out = re.split("#[0-9]", _line)
        assert len(_out) == 2, _line
        _imgID, _caption = re.split("#[0-9]", _line)
        _caption = _caption.strip()
        if _caption[-1] == ".":
            _caption = _caption[:-1].strip()
        captions.append(_caption)

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
