from typing import Tuple


def collate_image_captions(batch: Tuple):
    if len(batch[0]) == 2:
        # no id given
        audio_list, audio_len_list, image_list = [], [], []
        for audio, image in batch:
            audio_list.append(audio)
            audio_len_list.append(len(audio))
            image_list.append(image)

        return audio_list, audio_len_list, image_list

    elif len(batch[0]) == 3:
        # id given
        audio_list, audio_len_list, image_list, id_list = [], [], [], []
        for audio, image, id in batch:
            audio_list.append(audio)
            audio_len_list.append(len(audio))
            image_list.append(image)
            id_list.append(id)

        return audio_list, audio_len_list, image_list, id_list

    else:
        raise NotImplementedError("Data format no implemented in collator")

    # # general code
    # return_batch = [ [] for _ in batch[0] ]
    # for _data in batch:
    #     for i, feat in enumerate(_data):
    #         return_batch[i].append(feat)
    # return return_batch


def collate_general(batch: Tuple):
    keysInBatch = list(batch[0].keys())
    if "wav" in keysInBatch:
        keysInBatch.append("wav_len")
    return_dict = {k: [] for k in keysInBatch}
    for _row in batch:
        for _key in keysInBatch:
            if _key == "wav_len":
                return_dict[_key].append(len(_row["wav"]))
            else:
                return_dict[_key].append(_row[_key])

    return return_dict
