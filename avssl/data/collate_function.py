from typing import Tuple


def collate_image_captions(batch: Tuple):
    audio_list, audio_len_list, image_list = [], [], []
    for audio, image in batch:
        audio_list.append(audio)
        audio_len_list.append(len(audio))
        image_list.append(image)

    return audio_list, audio_len_list, image_list
