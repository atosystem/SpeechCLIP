import torch


def get_keypadding_mask(max_length: int, data_lens: torch.Tensor) -> torch.Tensor:
    """Create keypadding mask for attention layers

    Args:
        max_length (int): the max sequence length of the batch
        audio_len (torch.Tensor): the lens for each data in the batch, shape = (bsz,)

    Returns:
        torch.Tensor: key_padding_mask, bool Tensor, True for padding
    """
    bsz = data_lens.size(0)
    key_padding_mask = torch.ones([bsz, max_length])
    for mask, len in zip(key_padding_mask, data_lens):
        mask[:len] = 0.0
    key_padding_mask = key_padding_mask.type_as(data_lens).bool()

    return key_padding_mask
