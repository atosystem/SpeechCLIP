from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_general(batch: Tuple) -> dict:
    """collate function for general purpose

    Args:
        batch (Tuple): batch data

    Returns:
        dict: output collated data
    """
    keysInBatch = list(batch[0].keys())
    if "wav" in keysInBatch and isinstance(batch[0]["wav"], torch.Tensor):
        keysInBatch.append("wav_len")
    return_dict = {k: [] for k in keysInBatch}
    for _row in batch:
        for _key in keysInBatch:
            if _key == "wav_len":
                return_dict[_key].append(len(_row["wav"]))
            else:
                return_dict[_key].append(_row[_key])

    for key in return_dict:
        if isinstance(return_dict[key][0], torch.Tensor):
            if key == "wav":
                return_dict[key] = pad_sequence(return_dict[key], batch_first=True)
            else:
                return_dict[key] = torch.stack(return_dict[key], dim=0)
        else:
            return_dict[key] = torch.LongTensor(return_dict[key])

    return return_dict
